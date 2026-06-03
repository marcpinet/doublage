from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .models import Line, Word


log = logging.getLogger(__name__)

# Characters allowed by the MMS_FA dictionary after normalization (latin + FR diacritics).
_KEEP_RE = re.compile(r"[^a-zàâäéèêëïîôöùûüç']")

# wav2vec2 (MMS_FA) model cached at the process level: loading costs a few seconds,
# we avoid it for repeated re-alignments (e.g. on every recording in the editor).
_MODEL_CACHE: dict = {}


def _load_model(device: Optional[str]):
    import torch
    from torchaudio.pipelines import MMS_FA as B

    key = str(device or "auto")
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cached = (B, dev, B.get_model().to(dev), B.get_tokenizer(), B.get_aligner(), B.get_dict())
        _MODEL_CACHE[key] = cached
    return cached


def clear_model_cache() -> None:
    """Clear the MMS_FA model cache and free VRAM."""
    from . import gpu

    for k in list(_MODEL_CACHE.keys()):
        try:
            del _MODEL_CACHE[k]
        except Exception:
            pass
    _MODEL_CACHE.clear()
    gpu.empty_cache()


def align_words(
    audio_path: Path,
    lines: list[Line],
    device: Optional[str] = None,
    preloaded: tuple | None = None,
    min_seg_sec: float = 0.2,
    collapse_ratio: float = 0.5,
) -> None:
    """Word-by-word forced alignment (torchaudio MMS_FA, wav2vec2).

    Fills ``line.words`` (a list of :class:`~dubbing.models.Word`) with timings
    in ABSOLUTE seconds (scene scale, like ``line.start_sec``). Aligned on the
    ``vocals`` stem preferably. Resilient: a failed line leaves ``words=[]``.

    No extra dependency: MMS_FA ships with torchaudio. ctc-forced-aligner
    is deliberately avoided (it imports the flashlight decoder, which is absent on Windows).
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    B, dev, model, tokenizer, aligner, vocab = _load_model(device)  # cached: loaded once per process

    if preloaded is not None:
        data, sr = preloaded
    else:
        data, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")
    full = data.mean(axis=1).astype("float32")  # mono
    total_dur = len(full) / sr

    def norm(word: str) -> str:
        w = _KEEP_RE.sub("", word.lower())
        return "".join(ch for ch in w if ch in vocab)

    n_aligned = 0
    for ln in lines:
        ln.words = []
        ln.align_text = ln.text  # marker: we attempted to align THIS text (even if alignment fails)
        text = (ln.text or "").strip()
        raw_tokens = [t for t in text.split() if norm(t)]
        if not raw_tokens:
            continue
        s, e = float(ln.start_sec), float(ln.end_sec)
        a, b = max(0.0, s - 0.25), min(total_dur, e + 0.25)
        seg = torch.from_numpy(full[int(a * sr):int(b * sr)]).float()
        if seg.numel() < int(min_seg_sec * sr):  # too short to align (default < 0.2 s)
            continue
        if sr != B.sample_rate:
            seg = torchaudio.functional.resample(seg, sr, B.sample_rate)
        seg = seg.unsqueeze(0).to(dev)
        try:
            with torch.inference_mode():
                emission, _ = model(seg)
            token_spans = aligner(emission[0], tokenizer([norm(t) for t in raw_tokens]))
            ratio = seg.size(1) / emission.size(1) / B.sample_rate
        except Exception as ex:
            log.warning("word alignment failed for %s: %s", ln.id, ex)
            continue
        for spans, raw in zip(token_spans, raw_tokens):
            ln.words.append(Word(
                text=raw,
                start_sec=round(a + spans[0].start * ratio, 3),
                end_sec=round(a + spans[-1].end * ratio, 3),
            ))
        # Guard: if the audio is not in the same language as the text (e.g. dubbing a
        # Japanese-original anime subtitled in French), the MMS_FA forcing "succeeds" but
        # crams all the words into a tiny fraction of the line. We detect this
        # (words covering < 50% of the line's duration) and discard that alignment:
        # the rhythmo band then falls back to even spacing (readable), instead of a wrong karaoke.
        if len(ln.words) >= 4:
            span = ln.words[-1].end_sec - ln.words[0].start_sec
            line_dur = e - s
            if line_dur > 0 and span < collapse_ratio * line_dur:
                log.warning(
                    "alignment collapsed for %s (%d words span %.2fs of %.2fs) -> dropping",
                    ln.id, len(ln.words), span, line_dur,
                )
                ln.words = []
        if ln.words:
            n_aligned += 1

    log.info("Word alignment: %d/%d lines kept (forced-aligned)", n_aligned, len(lines))
