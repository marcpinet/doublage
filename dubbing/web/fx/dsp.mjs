// Pure-DSP core for the voice FX pitch engine: YIN pitch detection + streaming TD-PSOLA
// (formant-preserving pitch shift / autotune). No Web Audio API in here, so the exact same
// code runs inside the AudioWorklet AND in Node for the objective regression test
// (scripts/test_fx_dsp.mjs): a known input pitch must come out shifted/snapped within ±5 cents.
//
// TD-PSOLA refresher (timescale 1:1):
//   - analysis epochs a_k spaced STRICTLY one period T apart on the input (re-anchoring them
//     per frame creates jittery marks -> audible subharmonics, so we never do that);
//   - output marks o_j spaced T/ratio apart;
//   - each output mark copies a Hann-windowed 2T grain centered on the NEAREST analysis epoch.
//   Pitch up (ratio>1): output marks are denser, some epochs are reused. Down: some skipped.
//   Unvoiced audio (consonants, breaths) passes through dry with short crossfades.

export const FMIN = 70;     // lowest tracked f0 (Hz)
export const FMAX = 500;    // highest tracked f0 (Hz)
export const YIN_THRESHOLD = 0.14;

// ---------------------------------------------------------------- YIN (CMNDF + parabolic interp)

// frame: Float32/64Array. Returns { f0, dip }: the best pitch candidate and the CMNDF value at
// its lag (lower dip = more confidently voiced). Always returns a candidate (global minimum when
// no dip crosses the hard threshold) so callers can apply their own voicing policy/hysteresis.
export function yinPitchDetailed(frame, sr, fmin = FMIN, fmax = FMAX) {
  const tauMin = Math.max(2, Math.floor(sr / fmax));
  const tauMax = Math.floor(sr / fmin);
  const n = frame.length;
  const W = n - tauMax;             // constant integration window
  if (W < tauMax || tauMax <= tauMin + 2) return { f0: 0, dip: 1 };

  const d = new Float64Array(tauMax + 1);
  for (let tau = 1; tau <= tauMax; tau++) {
    let sum = 0;
    for (let j = 0; j < W; j++) {
      const diff = frame[j] - frame[j + tau];
      sum += diff * diff;
    }
    d[tau] = sum;
  }
  // Cumulative-mean-normalized difference
  const dn = new Float64Array(tauMax + 1);
  dn[0] = 1;
  let cum = 0;
  for (let tau = 1; tau <= tauMax; tau++) {
    cum += d[tau];
    dn[tau] = cum > 0 ? (d[tau] * tau) / cum : 1;
  }
  // First dip under the hard threshold (walked to its local minimum), else the global minimum
  let tau = -1;
  for (let t = tauMin; t <= tauMax; t++) {
    if (dn[t] < YIN_THRESHOLD) {
      while (t + 1 <= tauMax && dn[t + 1] < dn[t]) t++;
      tau = t;
      break;
    }
  }
  if (tau < 0) {
    let best = tauMin;
    for (let t = tauMin + 1; t <= tauMax; t++) if (dn[t] < dn[best]) best = t;
    tau = best;
  }
  // Parabolic interpolation around the minimum for sub-sample precision
  let betterTau = tau;
  if (tau > 1 && tau < tauMax) {
    const s0 = dn[tau - 1], s1 = dn[tau], s2 = dn[tau + 1];
    const denom = 2 * (2 * s1 - s2 - s0);
    if (Math.abs(denom) > 1e-12) betterTau = tau + (s2 - s0) / denom;
  }
  return { f0: sr / betterTau, dip: dn[tau] };
}

// Strict voiced-or-zero variant (used by the regression tests to MEASURE pitch).
export function yinPitch(frame, sr, fmin = FMIN, fmax = FMAX, threshold = YIN_THRESHOLD) {
  const { f0, dip } = yinPitchDetailed(frame, sr, fmin, fmax);
  return dip < threshold ? f0 : 0;
}

// ------------------------------------------------------------------------------- note snapping

// Nearest chromatic note (A=440 grid) in log-frequency space.
export function snapChromatic(f0) {
  if (!(f0 > 0)) return f0;
  const n = Math.round(12 * Math.log2(f0 / 440));
  return 440 * Math.pow(2, n / 12);
}

// --------------------------------------------------------------------------- streaming TD-PSOLA

function analysisWin(sr) {
  // YIN (on the /2-decimated signal) needs >= 2*tauMax full-rate samples; add headroom.
  return 2 * Math.ceil(sr / FMIN) + 256;
}

// Fixed algorithmic latency in samples, reported so renders can re-align the output:
// half an analysis window (pitch is known up to the center of the last analyzed frame)
// + two max periods (pull() lags the synthesis frontier so late grains never land in
// already-emitted ring slots) + one pitch hop of margin.
export function psolaLatency(sr) {
  const win = analysisWin(sr);
  return (win >> 1) + 2 * Math.ceil(sr / FMIN) + 256;
}

// opts: { mode: "shift" | "autotune", semitones: number, strength: 0..1 }
export class PsolaStream {
  constructor(sr, opts = {}) {
    this.sr = sr;
    this.mode = opts.mode || "shift";
    this.semitones = opts.semitones || 0;
    this.strength = opts.strength == null ? 1 : opts.strength;

    this.win = analysisWin(sr);
    this.hop = 256;                       // pitch-track hop (~5.3 ms @48k)
    this.tauMax = Math.ceil(sr / FMIN);
    this.latency = psolaLatency(sr);

    const cap = 1 << 18;                  // ring capacity (~5.5 s @48k) — plenty of slack
    this.capMask = cap - 1;
    this.inRing = new Float32Array(cap);
    this.outRing = new Float32Array(cap);

    this.inAbs = 0;       // absolute count of samples pushed
    this.aFront = 0;      // input analyzed so far (next pitch frame starts here)
    this.sFront = 0;      // dry/env path advanced up to this absolute time
    this.emitAbs = 0;     // samples emitted to the caller

    this.f0Track = [];    // f0 at frame centers: f0Track[i] = f0 at time i*hop + win/2

    this.epochs = [];     // recent analysis epochs (absolute times, strictly T apart)
    this.nextEpoch = -1;  // next epoch to lay (-1 = waiting for a voiced anchor)
    this.nextOut = -1;    // next output mark
    this.voiced = false;
    this.smoothedRatio = 1;

    this.xfade = Math.max(32, Math.floor(sr * 0.005));
    this.voicedEnv = 0;
    this.envTarget = 0;
  }

  setParams(p) {
    if (p.mode) this.mode = p.mode;
    if (p.semitones != null) this.semitones = p.semitones;
    if (p.strength != null) this.strength = p.strength;
  }

  _in(t) { return (t >= 0 && t < this.inAbs) ? this.inRing[t & this.capMask] : 0; }

  _f0At(t) {
    const i = Math.round((t - this.win / 2) / this.hop);
    if (i < 0) return 0;
    if (i >= this.f0Track.length) return this.f0Track.length ? this.f0Track[this.f0Track.length - 1] : 0;
    return this.f0Track[i];
  }

  _periodAt(t) {
    const f0 = this._f0At(t);
    if (!(f0 > 0)) return 0;
    return Math.max(this.sr / FMAX, Math.min(this.sr / FMIN, this.sr / f0));
  }

  _ratioFor(f0) {
    let r;
    if (this.mode === "autotune") {
      const target = snapChromatic(f0);
      r = f0 > 0 ? Math.pow(target / f0, this.strength) : 1;
    } else {
      r = Math.pow(2, this.semitones / 12);
    }
    if (!(r > 0)) r = 1;
    return Math.min(2, Math.max(0.5, r));
  }

  push(block) {
    for (let i = 0; i < block.length; i++) {
      this.inRing[this.inAbs & this.capMask] = block[i];
      this.inAbs++;
    }
    this._analyze();
    this._synthesize();
  }

  _analyze() {
    while (this.aFront + this.win <= this.inAbs) {
      // Decimate by 2 for YIN (4x cheaper; parabolic interp keeps precision well under 5 cents)
      const half = this.win >> 1;
      const dec = this._dec || (this._dec = new Float64Array(half));
      for (let j = 0; j < half; j++) {
        const t = this.aFront + 2 * j;
        dec[j] = (this._in(t) + this._in(t + 1)) * 0.5;
      }
      // Voicing with hysteresis + hangover. Real mic speech is messy: a single strict
      // threshold only locks onto firmly-phonated voice, so the shift applied to nothing
      // unless the speaker FORCED a deep/high voice. Strict to turn ON (no shifting noise),
      // permissive to STAY on, and short gaps are bridged with the last known pitch.
      const det = yinPitchDetailed(dec, this.sr / 2);
      const thr = this._vOn ? 0.42 : 0.18;
      if (det.f0 > 0 && det.dip < thr) {
        this._vOn = true; this._vHang = 3; this._vLastF0 = det.f0;
        this.f0Track.push(det.f0);
      } else if (this._vHang > 0) {
        this._vHang--;
        this.f0Track.push(this._vLastF0 || 0);
      } else {
        this._vOn = false;
        this.f0Track.push(0);
      }
      this.aFront += this.hop;
    }
  }

  _placeGrain(epoch, outPos, T, env) {
    const half = Math.round(T);
    const gain = env / Math.max(0.5, this.smoothedRatio);  // Hann OLA at hop T/ratio sums to ~ratio
    for (let k = -half; k < half; k++) {
      const w = 0.5 - 0.5 * Math.cos(Math.PI * (k + half) / half);
      this.outRing[(outPos + k) & this.capMask] += this._in(epoch + k) * w * gain;
    }
  }

  _synthesize() {
    // Output can be rendered up to: pitch known (center of the last analyzed frame) AND
    // grain right-halves available (inAbs - tauMax).
    const pitchKnown = this.aFront - this.hop + (this.win >> 1);
    const safe = Math.min(pitchKnown, this.inAbs - this.tauMax);

    while (this.sFront < safe) {
      const t = this.sFront;
      const f0 = this._f0At(t);

      if (f0 > 0 && !this.voiced) {
        this.voiced = true;
        this.envTarget = 1;
        if (this.nextEpoch < t) { this.nextEpoch = t; this.epochs.length = 0; }
        if (this.nextOut < t) this.nextOut = t;
      } else if (!(f0 > 0) && this.voiced) {
        this.voiced = false;
        this.envTarget = 0;
      }

      // Epoch train: strictly period-spaced while voiced. Marks are kept as FLOATS and only
      // rounded when a grain is placed: rounding the spacing itself quantizes the ratio
      // (e.g. +2 st at 300 Hz came out 5.5 cents flat from 160/143 vs 160/142.54).
      while (this.voiced && this.nextEpoch <= t) {
        this.epochs.push(this.nextEpoch);
        const T = this._periodAt(this.nextEpoch) || this.sr / 200;
        this.nextEpoch += Math.max(8, T);
        if (this.epochs.length > 64) this.epochs.splice(0, this.epochs.length - 64);
      }

      // Output marks due now: copy the grain of the nearest epoch (writes up to one period
      // ahead of t and up to two behind — pull() lags by 2*tauMax so nothing already-emitted
      // ever receives a late grain).
      if (this.voiced && this.nextOut < t - this.tauMax) this.nextOut = t;
      while (this.voiced && this.nextOut <= t && this.epochs.length) {
        let best = this.epochs[this.epochs.length - 1], bd = Infinity;
        for (let i = this.epochs.length - 1; i >= 0; i--) {
          const d = Math.abs(this.epochs[i] - this.nextOut);
          if (d < bd) { bd = d; best = this.epochs[i]; } else if (this.epochs[i] < this.nextOut - bd) break;
        }
        const T = this._periodAt(best) || this.sr / 200;
        const ratio = this._ratioFor(this._f0At(best));
        // Autotune must retune INSTANTLY (the snap IS the effect — smoothing turns it into an
        // inaudible glide on speech); fixed shifts keep a little smoothing against f0 jitter.
        const alpha = this.mode === "autotune" ? 1 : 0.15;
        this.smoothedRatio = (1 - alpha) * this.smoothedRatio + alpha * ratio;
        this._placeGrain(Math.round(best), Math.round(this.nextOut), T, Math.min(1, this.voicedEnv + 0.5));
        this.nextOut += Math.max(8, T / this.smoothedRatio);
      }

      // Dry path with voiced/unvoiced crossfade envelope
      const step = 1 / this.xfade;
      this.voicedEnv += (this.envTarget > this.voicedEnv ? step : -step);
      this.voicedEnv = Math.min(1, Math.max(0, this.voicedEnv));
      this.outRing[t & this.capMask] += this._in(t) * (1 - this.voicedEnv);

      this.sFront++;
    }
  }

  // Pull processed samples into out; returns count written. Output sample i corresponds to
  // input sample i; the first `latency` samples are warm-up — callers trim via this.latency.
  pull(out) {
    // Lag two max periods behind sFront: grains can still be laid up to 2*tauMax behind it.
    const frontier = this.sFront - 2 * this.tauMax;
    const avail = Math.max(0, Math.min(out.length, frontier - this.emitAbs));
    for (let i = 0; i < avail; i++) {
      const slot = this.emitAbs & this.capMask;
      out[i] = this.outRing[slot];
      this.outRing[slot] = 0;            // slot is clean for its next lap
      this.emitAbs++;
    }
    return avail;
  }

  // Flush remaining tail (offline use): pad input with silence so synthesis catches up.
  flush() {
    this.push(new Float32Array(this.win + 4 * this.tauMax));
  }
}
