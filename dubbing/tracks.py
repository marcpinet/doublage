from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}

FRENCH_LANG_ALIASES = {"fre", "fra", "fr"}
ENGLISH_LANG_ALIASES = {"eng", "en"}

LANGUAGE_ISO_MAP = {
    "fre": "fr", "fra": "fr",
    "eng": "en",
    "ger": "de", "deu": "de",
    "spa": "es",
    "ita": "it",
    "jpn": "ja",
    "por": "pt",
    "rus": "ru",
    "chi": "zh", "zho": "zh",
}

QUEBEC_KEYWORDS = ["VFQ", "QUEBEC", "QUÉBEC", "CANADIAN", "CANADIEN", "CANADA"]
PREFERRED_FRENCH_KEYWORDS = ["VFF", "FRANCAIS", "FRANÇAIS", "FRENCH", "PARISIAN"]


@dataclass
class Track:
    index: int
    codec_type: str
    codec_name: str
    language: Optional[str]
    title: Optional[str]
    forced: bool
    hearing_impaired: bool
    default: bool
    num_frames: int
    num_bytes: int


def normalize_language(code: Optional[str]) -> str:
    if not code:
        return "und"
    code = code.lower()
    if code in LANGUAGE_ISO_MAP:
        return LANGUAGE_ISO_MAP[code]
    return code[:2] if len(code) >= 2 else code


def language_aliases(code: str) -> set[str]:
    code = code.lower()
    if code in FRENCH_LANG_ALIASES:
        return FRENCH_LANG_ALIASES
    if code in ENGLISH_LANG_ALIASES:
        return ENGLISH_LANG_ALIASES
    return {code}


def probe_streams(path: Path) -> list[Track]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    data = json.loads(result.stdout)
    tracks: list[Track] = []
    for s in data.get("streams", []):
        tags = s.get("tags", {}) or {}
        disp = s.get("disposition", {}) or {}
        tracks.append(
            Track(
                index=s["index"],
                codec_type=s.get("codec_type", ""),
                codec_name=s.get("codec_name", ""),
                language=(tags.get("language") or "").lower() or None,
                title=tags.get("title"),
                forced=bool(disp.get("forced", 0)),
                hearing_impaired=bool(disp.get("hearing_impaired", 0)),
                default=bool(disp.get("default", 0)),
                num_frames=int(tags.get("NUMBER_OF_FRAMES", 0) or 0),
                num_bytes=int(tags.get("NUMBER_OF_BYTES", 0) or 0),
            )
        )
    return tracks


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return float(result.stdout.strip())


def _title_upper(track: Track) -> str:
    return (track.title or "").upper()


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(k in text for k in keywords)


def select_video_track(tracks: list[Track]) -> Track:
    candidates = [t for t in tracks if t.codec_type == "video" and t.codec_name not in {"mjpeg", "png"}]
    if not candidates:
        raise ValueError("No video track found")
    candidates.sort(key=lambda t: (t.default, t.num_frames), reverse=True)
    return candidates[0]


def _filter_by_language(tracks: list[Track], language: str) -> list[Track]:
    aliases = language_aliases(language)
    matches = [t for t in tracks if t.language in aliases]
    return matches if matches else tracks


def _score_french_preference(track: Track) -> int:
    title = _title_upper(track)
    if "VFF" in title:
        return 100
    if _has_any(title, ["FRANCAIS", "FRANÇAIS", "FRENCH", "PARISIAN"]):
        return 50
    return 0


def select_audio_track(tracks: list[Track], language: str = "fre") -> Track:
    candidates = [t for t in tracks if t.codec_type == "audio"]
    if not candidates:
        raise ValueError("No audio track found")

    candidates = _filter_by_language(candidates, language)

    is_french_request = language.lower() in FRENCH_LANG_ALIASES
    if is_french_request:
        non_quebec = [t for t in candidates if not _has_any(_title_upper(t), QUEBEC_KEYWORDS)]
        if non_quebec:
            candidates = non_quebec

    def score(t: Track) -> tuple[int, int, int]:
        pref = _score_french_preference(t) if is_french_request else 0
        return (pref, int(t.default), t.num_bytes)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def select_subtitle_track(tracks: list[Track], language: str = "fre") -> Track:
    candidates = [
        t for t in tracks
        if t.codec_type == "subtitle" and t.codec_name in TEXT_SUBTITLE_CODECS
    ]
    if not candidates:
        raise ValueError("No text subtitle track found (only image-based subs not supported in v1)")

    candidates = _filter_by_language(candidates, language)

    non_forced = [t for t in candidates if not t.forced]
    if non_forced:
        candidates = non_forced

    non_sdh = [t for t in candidates if not t.hearing_impaired and "SDH" not in _title_upper(t)]
    if non_sdh:
        candidates = non_sdh

    is_french_request = language.lower() in FRENCH_LANG_ALIASES
    if is_french_request:
        non_quebec = [t for t in candidates if not _has_any(_title_upper(t), QUEBEC_KEYWORDS)]
        if non_quebec:
            candidates = non_quebec

    def score(t: Track) -> tuple[int, int, int]:
        pref = _score_french_preference(t) if is_french_request else 0
        return (pref, int(t.default), t.num_frames)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def format_track_table(tracks: list[Track]) -> str:
    rows = []
    rows.append(
        f"{'idx':>3}  {'type':9s} {'codec':10s} {'lang':5s} {'title':24s} {'default':7s} {'forced':6s} {'sdh':3s}"
    )
    rows.append("-" * 80)
    for t in tracks:
        rows.append(
            f"{t.index:>3}  {t.codec_type:9s} {t.codec_name:10s} "
            f"{(t.language or '-'):5s} {(t.title or '-')[:24]:24s} "
            f"{('yes' if t.default else 'no'):7s} "
            f"{('yes' if t.forced else 'no'):6s} "
            f"{('yes' if t.hearing_impaired else 'no'):3s}"
        )
    return "\n".join(rows)
