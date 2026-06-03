from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pysubs2

from .models import Line


log = logging.getLogger(__name__)


_WHITESPACE_RE = re.compile(r"\s+")
# Dialogue dashes at the start of a line (- – —), optionally preceded by spaces.
_DIALOGUE_DASH_RE = re.compile(r"^\s*[-–—]\s*")
# A DIALOGUE dash = a dash FOLLOWED by a space ("- Hi"). It's the only reliable signal of a
# speaker change. A dash with no space ("-Hi", a hyphenation, a range like "3-4") is not one.
_DIALOGUE_DASH_START_RE = re.compile(r"^\s*[-–—]\s")
_ASS_TAGS_RE = re.compile(r"\{[^}]*\}")


def _clean_text(raw: str) -> str:
    text = _ASS_TAGS_RE.sub("", raw)
    text = text.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _split_cue(raw: str) -> list[str]:
    """Split a subtitle cue into separate dialogue lines (best effort).

    A .srt may pack two characters into ONE subtitle cue. The only RELIABLE signal of a
    speaker change is the **dialogue dash** ("- …"). A line break (``\\N``) on its own is
    almost always just a display line break WITHIN a single sentence (e.g. "You can do\\Nwhatever
    you want with it?") — splitting on it would wrongly break the sentence apart. So we split ONLY
    on dialogue dashes; line breaks without a dash are joined back together.

    - We split on line breaks, then regroup: a chunk starting with a dialogue dash starts a new
      line; otherwise it extends the previous one (display line break).
    - Unhandled case (accepted limitation): two speakers with no dash or clear separator → a single
      line, and the host will split it in the editor ("split" button)."""
    raw = raw.replace(r"\h", " ")
    parts = re.split(r"\\N|\\n|\n|\r", raw)
    turns: list[str] = []
    for p in parts:
        starts_turn = bool(_DIALOGUE_DASH_START_RE.match(p))
        clean = _clean_text(_DIALOGUE_DASH_RE.sub("", p))
        if not clean:
            continue
        if starts_turn or not turns:
            turns.append(clean)  # new speaker (dialogue dash) or the very first chunk
        else:
            turns[-1] = (turns[-1] + " " + clean).strip()  # plain display line break → joined back
    return turns


def parse_subtitles(
    path: Path,
    clip_start_sec: float = 0.0,
    clip_end_sec: Optional[float] = None,
    split_cues: bool = True,
) -> list[Line]:
    """Return dialogue lines inside [clip_start_sec, clip_end_sec], re-timed to clip start.

    When ``split_cues`` is True (default), a subtitle cue holding several dialogue turns
    (split by line breaks / dialogue dashes) is broken into one :class:`Line` per turn,
    sharing the cue's time window proportionally to each turn's text length."""
    subs = pysubs2.load(str(path))
    clip_len = (clip_end_sec - clip_start_sec) if clip_end_sec is not None else None
    lines: list[Line] = []
    counter = 0
    dropped = 0
    n_split = 0
    for ev in subs:
        if ev.is_comment:
            continue
        ev_start = ev.start / 1000.0
        ev_end = ev.end / 1000.0
        # keep only events overlapping the crop window
        if ev_end <= clip_start_sec:
            continue
        if clip_end_sec is not None and ev_start >= clip_end_sec:
            dropped += 1
            continue
        # shift into clip-local time (0 == start of the crop) and clamp to the clip
        start_sec = max(0.0, ev_start - clip_start_sec)
        end_sec = ev_end - clip_start_sec
        if clip_len is not None:
            end_sec = min(end_sec, clip_len)
        if end_sec <= start_sec:
            continue

        raw = ev.text if ev.text else ev.plaintext
        turns = _split_cue(raw) if split_cues else [_clean_text(ev.plaintext or ev.text)]
        turns = [t for t in turns if t]
        if not turns:
            continue
        if len(turns) > 1:
            n_split += 1

        # distribute the cue's time window proportionally to each line's text length
        total_chars = sum(len(t) for t in turns) or 1
        span = end_sec - start_sec
        acc = start_sec
        for ti, t in enumerate(turns):
            seg = span * (len(t) / total_chars)
            t_start = acc
            t_end = end_sec if ti == len(turns) - 1 else acc + seg
            acc = t_end
            if t_end <= t_start:
                continue
            lines.append(Line(
                id=f"line_{counter:04d}",
                start_sec=round(t_start, 3),
                end_sec=round(t_end, 3),
                text=t,
                character_id=None,
                source_cluster=None,
                confidence=None,
                needs_review=True,
            ))
            counter += 1
    if dropped:
        log.warning("Dropped %d subtitle events past clip end", dropped)
    if n_split:
        log.info("Split %d multi-speaker subtitle cue(s) into separate lines", n_split)
    log.info(
        "Parsed %d dialogue lines from %s (window %.2fs -> %s)",
        len(lines), path.name, clip_start_sec,
        f"{clip_end_sec:.2f}s" if clip_end_sec is not None else "end",
    )
    return lines
