from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .models import Line


log = logging.getLogger(__name__)

# Pipeline language codes (ISO-639-2, e.g. "fre") → Whisper codes (ISO-639-1, e.g. "fr").
# Any unknown code → None → Whisper auto-detects the language.
_LANG_MAP = {
    "fre": "fr", "fra": "fr", "eng": "en", "jpn": "ja", "spa": "es", "deu": "de",
    "ger": "de", "ita": "it", "por": "pt", "rus": "ru", "kor": "ko", "chi": "zh",
    "zho": "zh", "ara": "ar", "nld": "nl", "dut": "nl", "pol": "pl", "tur": "tr",
}

_MODEL_CACHE: dict = {}


def clear_model_cache() -> None:
    """Clear the Whisper model cache and free VRAM."""
    from . import gpu

    for k in list(_MODEL_CACHE.keys()):
        try:
            del _MODEL_CACHE[k]
        except Exception:
            pass
    _MODEL_CACHE.clear()
    gpu.empty_cache()


def _whisper_lang(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    code = language.strip().lower()
    if code in _LANG_MAP:
        return _LANG_MAP[code]
    if len(code) == 2:
        return code  # already an ISO-639-1 code
    return None  # unknown → auto-detection


def transcribe(audio_path: Path, language: Optional[str] = None, device: Optional[str] = None,
               model_name: str = "large-v3") -> list[Line]:
    """Generate subtitles by transcribing the audio with OpenAI Whisper (large-v3, locally).

    Each Whisper segment becomes a :class:`~dubbing.models.Line` (text + time bounds, in
    CLIP-LOCAL seconds: the audio passed in is already cropped to the chosen window). The precise
    word-by-word (karaoke) timings are (re)computed afterwards by ``align_words``; here we keep only
    the text and the segment bounds. Diarization, downstream, assigns each line to a character.
    """
    import torch
    import whisper

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (model_name, dev)
    model = _MODEL_CACHE.get(key)
    if model is None:
        log.info("Loading Whisper %s on %s (first run downloads the model, ~3 GB)…", model_name, dev)
        model = whisper.load_model(model_name, device=dev)
        _MODEL_CACHE[key] = model

    wl = _whisper_lang(language)
    try:  # an unknown code would crash Whisper → fall back to auto-detection
        from whisper.tokenizer import LANGUAGES
        if wl and wl not in LANGUAGES:
            wl = None
    except Exception:
        pass
    log.info("Transcribing %s with Whisper (language=%s)…", audio_path.name, wl or "auto-detect")
    result = model.transcribe(
        str(audio_path),
        language=wl,
        # True = a DTW pass over the cross-attentions re-derives each segment's start/end from its
        # first/last word — far tighter than the decoder's timestamp tokens, which drift and linger
        # into silence. Precise line bounds matter downstream: diarization assigns characters by
        # temporal overlap, MMS word alignment only searches the line window ±0.25 s, and OWS cuts
        # the original audio exactly on line bounds. The per-word timings themselves are discarded
        # (align_words/MMS stays the karaoke source of truth). ~10-25% slower; uses triton on CUDA
        # (this is what triton-windows + run.bat's MSVC env exist for), numba CPU fallback otherwise.
        word_timestamps=True,
        fp16=(dev == "cuda"),
        verbose=False,
    )

    lines: list[Line] = []
    last_end = 0.0
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, float(seg.get("start", last_end)))
        end = float(seg.get("end", start))
        if end <= start:
            end = start + 0.3  # safeguard: zero-length segment
        lines.append(Line(
            id=f"line_{len(lines):04d}",
            start_sec=round(start, 3),
            end_sec=round(end, 3),
            text=text,
            needs_review=True,
        ))
        last_end = end

    log.info("Whisper produced %d subtitle line(s)", len(lines))
    return lines
