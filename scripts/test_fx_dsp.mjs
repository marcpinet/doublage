// Objective regression test for the voice-FX pitch engine (dubbing/web/fx/dsp.mjs).
// Run: node scripts/test_fx_dsp.mjs
//
// Methodology (the only validation that has ever held up on this project): feed a known
// pitch in, MEASURE what comes out. A synthetic voiced signal (harmonic-rich glottal-like
// tone) goes through the streaming PSOLA; the output pitch is measured frame-by-frame with
// the same YIN detector and must land within ±5 cents of the expected value. Levels and
// alignment are measured too — no claims, only numbers.

import { PsolaStream, yinPitch, psolaLatency, snapChromatic } from "../dubbing/web/fx/dsp.mjs";

const SR = 48000;
let failures = 0;

function check(label, ok, detail) {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? "  (" + detail + ")" : ""}`);
  if (!ok) failures++;
}

// Voiced test signal: f0 + decaying harmonics (vowel-ish), gentle fade-in/out.
function makeVoice(f0, seconds, sr = SR) {
  const n = Math.floor(seconds * sr);
  const x = new Float32Array(n);
  const H = 10;
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    let s = 0;
    for (let h = 1; h <= H; h++) s += Math.sin(2 * Math.PI * f0 * h * t + h * 1.7) / (h * h * 0.6 + h);
    const fade = Math.min(1, i / (0.02 * sr), (n - 1 - i) / (0.02 * sr));
    x[i] = 0.35 * s * fade;
  }
  return x;
}

function runPsola(x, opts) {
  const ps = new PsolaStream(SR, opts);
  const out = new Float32Array(x.length + ps.latency + SR);
  let written = 0;
  const pullBuf = new Float32Array(4096);
  for (let pos = 0; pos < x.length; pos += 4096) {
    ps.push(x.subarray(pos, Math.min(x.length, pos + 4096)));
    let got;
    while ((got = ps.pull(pullBuf)) > 0) { out.set(pullBuf.subarray(0, got), written); written += got; }
  }
  ps.flush();
  let got;
  while ((got = ps.pull(pullBuf)) > 0) { out.set(pullBuf.subarray(0, got), written); written += got; }
  return out.subarray(0, written);
}

// Median f0 over the steady middle of a signal, measured with the same YIN.
function measureF0(x, sr = SR) {
  const win = 2 * Math.ceil(sr / 70) + 256;
  const hop = 512;
  const f0s = [];
  const start = Math.floor(x.length * 0.25), end = Math.floor(x.length * 0.75);
  for (let pos = start; pos + win < end; pos += hop) {
    const f = yinPitch(x.subarray(pos, pos + win), sr);
    if (f > 0) f0s.push(f);
  }
  f0s.sort((a, b) => a - b);
  return { f0: f0s.length ? f0s[f0s.length >> 1] : 0, voicedFrames: f0s.length };
}

const cents = (a, b) => 1200 * Math.log2(a / b);
const rms = (x, a = 0, b = x.length) => {
  let s = 0;
  for (let i = a; i < b; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, b - a));
};
const db = (r) => 20 * Math.log10(Math.max(1e-12, r));

console.log("== 1. YIN detector sanity ==");
for (const f of [85, 150, 220, 392]) {
  const x = makeVoice(f, 1.0);
  const m = measureF0(x);
  const c = Math.abs(cents(m.f0, f));
  check(`detects ${f} Hz`, c < 3, `measured ${m.f0.toFixed(2)} Hz, ${c.toFixed(2)} cents off`);
}

console.log("== 2. Pitch shift accuracy (target: ±5 cents) ==");
for (const [f0, st] of [[150, -3], [150, +3], [110, -5], [110, +4], [220, -2], [300, +2], [150, -7]]) {
  const x = makeVoice(f0, 1.2);
  const y = runPsola(x, { mode: "shift", semitones: st });
  const expected = f0 * Math.pow(2, st / 12);
  const m = measureF0(y);
  const c = m.f0 > 0 ? cents(m.f0, expected) : NaN;
  check(`${f0} Hz ${st > 0 ? "+" : ""}${st} st -> ${expected.toFixed(1)} Hz`,
        Math.abs(c) < 5, `measured ${m.f0.toFixed(2)} Hz, ${c.toFixed(2)} cents off, ${m.voicedFrames} voiced frames`);
}

console.log("== 3. Autotune snapping (target: ±5 cents of the nearest note) ==");
for (const offCents of [-40, +35, +20]) {
  const base = 196.0;                            // G3
  const f0 = base * Math.pow(2, offCents / 1200); // detuned input
  const target = snapChromatic(f0);
  const x = makeVoice(f0, 1.2);
  const y = runPsola(x, { mode: "autotune", strength: 1.0 });
  const m = measureF0(y);
  const c = m.f0 > 0 ? cents(m.f0, target) : NaN;
  check(`G3 ${offCents > 0 ? "+" : ""}${offCents}c (${f0.toFixed(2)} Hz) snaps to ${target.toFixed(2)} Hz`,
        Math.abs(c) < 5, `measured ${m.f0.toFixed(2)} Hz, ${c.toFixed(2)} cents off`);
}

console.log("== 4. Level preservation (voiced region within ±3 dB) ==");
for (const st of [-4, 0, +4]) {
  const x = makeVoice(140, 1.2);
  const y = runPsola(x, { mode: "shift", semitones: st });
  const a = Math.floor(x.length * 0.3), b = Math.floor(x.length * 0.7);
  const dIn = db(rms(x, a, b));
  const dOut = db(rms(y, a, b));
  check(`${st > 0 ? "+" : ""}${st} st level`, Math.abs(dOut - dIn) < 3,
        `in ${dIn.toFixed(1)} dB, out ${dOut.toFixed(1)} dB`);
}

console.log("== 5. Output alignment (greedy pull is 1:1 with the input — dubbing sync) ==");
{
  // 300 ms of silence then voice: in greedy-pull mode (what the offline take render uses),
  // output sample i IS input sample i — the onset must not move by more than 15 ms.
  const pre = Math.floor(0.3 * SR);
  const v = makeVoice(160, 0.8);
  const x = new Float32Array(pre + v.length);
  x.set(v, pre);
  const y = runPsola(x, { mode: "shift", semitones: -3 });
  const thr = 0.02;
  let onIn = 0, onOut = 0;
  while (onIn < x.length && Math.abs(x[onIn]) < thr) onIn++;
  while (onOut < y.length && Math.abs(y[onOut]) < thr) onOut++;
  const drift = (onOut - onIn) / SR * 1000;
  check("onset drift", Math.abs(drift) < 15,
        `${drift.toFixed(1)} ms (worklet warm-up latency, realtime only: ${(psolaLatency(SR) / SR * 1000).toFixed(1)} ms)`);
}

console.log("== 6. Hygiene: no NaN, unvoiced passthrough ==");
{
  const x = makeVoice(150, 1.0);
  const y = runPsola(x, { mode: "shift", semitones: 3 });
  let nan = 0;
  for (let i = 0; i < y.length; i++) if (!Number.isFinite(y[i])) nan++;
  check("no NaN/Inf in output", nan === 0, `${nan} bad samples`);

  // White noise is unvoiced: it must pass through ~unchanged (no pitch grains).
  const rng = (() => { let s = 1234567; return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x40000000 - 1); })();
  const noise = new Float32Array(SR);
  for (let i = 0; i < noise.length; i++) noise[i] = 0.1 * rng();
  const ny = runPsola(noise, { mode: "shift", semitones: 5 });
  const dIn = db(rms(noise, 10000, 38000));
  const dOut = db(rms(ny, 10000, 38000));
  check("unvoiced (noise) passes through", Math.abs(dOut - dIn) < 2, `in ${dIn.toFixed(1)} dB, out ${dOut.toFixed(1)} dB`);
}

console.log("== 7. Real-mic robustness: noisy voice must still be shifted (voicing hysteresis) ==");
{
  // Voice + broadband noise (~10 dB SNR). With a single strict voicing threshold the shifter
  // only engaged on firmly-phonated voice ("works only when I force a deep voice IRL") — the
  // hysteresis must keep the engine locked on, so the output pitch IS the shifted pitch.
  const rng = (() => { let s = 987654; return () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x40000000 - 1); })();
  const x = makeVoice(130, 1.2);
  for (let i = 0; i < x.length; i++) x[i] += 0.035 * rng();
  const y = runPsola(x, { mode: "shift", semitones: -4 });
  const expected = 130 * Math.pow(2, -4 / 12);
  const m = measureF0(y);
  const c = m.f0 > 0 ? cents(m.f0, expected) : NaN;
  check(`noisy 130 Hz -4 st -> ${expected.toFixed(1)} Hz`, Math.abs(c) < 8,
        `measured ${m.f0.toFixed(2)} Hz, ${c.toFixed(2)} cents off, ${m.voicedFrames} voiced frames`);
}

console.log(failures === 0 ? "\nAll FX DSP tests passed." : `\n${failures} test(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
