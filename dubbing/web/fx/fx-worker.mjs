// Offline pitch processing for recorded takes (module Worker, keeps the UI thread free).
// Greedy pull => output sample i corresponds to input sample i (validated to 0.0 ms onset
// drift by scripts/test_fx_dsp.mjs), so the take's dubbing timing is untouched.

import { PsolaStream } from "./dsp.mjs?v=1";

self.onmessage = (e) => {
  const { id, samples, sr, mode, semitones, strength, dryMix, wetMix } = e.data;
  try {
    const ps = new PsolaStream(sr, { mode, semitones, strength });
    const out = new Float32Array(samples.length);
    const buf = new Float32Array(8192);
    let written = 0;
    const drain = () => {
      let got;
      while (written < out.length && (got = ps.pull(buf)) > 0) {
        const n = Math.min(got, out.length - written);
        out.set(buf.subarray(0, n), written);
        written += n;
      }
    };
    for (let pos = 0; pos < samples.length; pos += 8192) {
      ps.push(samples.subarray(pos, Math.min(samples.length, pos + 8192)));
      drain();
    }
    ps.flush();
    drain();
    const dry = dryMix == null ? 0 : dryMix;
    const wet = wetMix == null ? 1 : wetMix;
    if (dry !== 0 || wet !== 1) {
      for (let i = 0; i < out.length; i++) out[i] = samples[i] * dry + out[i] * wet;
    }
    self.postMessage({ id, ok: true, samples: out }, [out.buffer]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
