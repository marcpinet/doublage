// AudioWorklet wrapper around the PSOLA engine — REALTIME MONITORING ONLY (the "hear myself"
// test before recording). Recorded takes are processed offline by fx-worker.mjs instead, where
// the greedy pull keeps the output aligned 1:1 with the input (no latency to compensate).
//
// Here the engine's warm-up (~51 ms @48k) shows up as plain monitoring latency: we emit
// silence until the FIFO holds a full warm-up of audio, then consume steadily. Production
// jitter is bounded by the pitch hop (~5 ms), far below the warm-up target, so the queue
// never starves mid-stream (no glitches).

import { PsolaStream, psolaLatency } from "./dsp.mjs?v=1";

class FxPitchProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = (options && options.processorOptions) || {};
    this.ps = new PsolaStream(sampleRate, o);
    const cap = 1 << 16;
    this.qMask = cap - 1;
    this.queue = new Float32Array(cap);
    this.qr = 0;
    this.qw = 0;
    this.started = false;
    this.target = psolaLatency(sampleRate) + 512;
    this.tmp = new Float32Array(4096);
    this.port.onmessage = (e) => { if (e.data) this.ps.setParams(e.data); };
  }

  process(inputs, outputs) {
    const inp = inputs[0] && inputs[0][0];
    const out = outputs[0] && outputs[0][0];
    if (!out) return true;
    if (inp && inp.length) this.ps.push(inp);

    let got;
    while ((got = this.ps.pull(this.tmp)) > 0) {
      for (let i = 0; i < got; i++) { this.queue[this.qw & this.qMask] = this.tmp[i]; this.qw++; }
    }

    const avail = this.qw - this.qr;
    if (!this.started && avail >= this.target) this.started = true;
    if (this.started && avail >= out.length) {
      for (let i = 0; i < out.length; i++) { out[i] = this.queue[this.qr & this.qMask]; this.qr++; }
    } else {
      out.fill(0);
    }
    // Copy mono to any extra output channels
    const chans = outputs[0];
    for (let c = 1; c < chans.length; c++) chans[c].set(out);
    return true;
  }
}

registerProcessor("fx-pitch", FxPitchProcessor);
