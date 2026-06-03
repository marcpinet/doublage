from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .diarize import DiarSegment
from .models import Character, Line


log = logging.getLogger(__name__)


CONFIDENCE_THRESHOLD = 0.5
CLOSE_CALL_MARGIN = 0.1


COLOR_PALETTE = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b",
    "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
    "#f97316", "#a855f7", "#14b8a6", "#eab308",
]


def _index_segments_by_cluster(segments: list[DiarSegment]) -> dict[str, list[DiarSegment]]:
    out: dict[str, list[DiarSegment]] = defaultdict(list)
    for s in segments:
        out[s.cluster_id].append(s)
    return dict(out)


def _overlap_with_cluster(line: Line, segs: list[DiarSegment]) -> float:
    total = 0.0
    for s in segs:
        total += max(0.0, min(line.end_sec, s.end_sec) - max(line.start_sec, s.start_sec))
    return total


def align_lines_to_clusters(
    lines: list[Line],
    diar_segments: list[DiarSegment],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    close_call_margin: float = CLOSE_CALL_MARGIN,
) -> None:
    by_cluster = _index_segments_by_cluster(diar_segments)
    for line in lines:
        line_dur = max(line.end_sec - line.start_sec, 1e-6)

        scores = {cid: _overlap_with_cluster(line, segs) for cid, segs in by_cluster.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        if not ranked or ranked[0][1] <= 0.0:
            line.source_cluster = None
            line.confidence = 0.0
            line.needs_review = True
            continue

        top_cid, top_overlap = ranked[0]
        second_overlap = ranked[1][1] if len(ranked) > 1 else 0.0

        confidence = top_overlap / line_dur
        margin = (top_overlap - second_overlap) / line_dur

        line.source_cluster = top_cid
        line.confidence = round(confidence, 3)
        line.needs_review = confidence < confidence_threshold or margin < close_call_margin


def build_characters_from_segments(segments: list[DiarSegment]) -> list[Character]:
    unique = sorted({s.cluster_id for s in segments})
    chars: list[Character] = []
    for i, cid in enumerate(unique):
        chars.append(
            Character(
                id=f"char_{i}",
                name=f"Unknown {i}",
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                cluster_ids=[cid],
            )
        )
    return chars


def apply_line_clusters(
    lines: list[Line],
    line_clusters: dict[str, dict],
    margin_threshold: float = CLOSE_CALL_MARGIN,
) -> None:
    """Apply the result of ``diarize.diarize_by_lines`` to each line.

    ``line_clusters``: ``{line_id: {"cluster": "emb_N"|None, "margin": float}}``.
    The margin (cosine-similarity gap to the second speaker) is used as ``confidence``;
    ``needs_review`` is raised when the margin is low (ambiguous assignment)."""
    for line in lines:
        info = line_clusters.get(line.id)
        if not info or info.get("cluster") is None:
            line.source_cluster = None
            line.confidence = 0.0
            line.needs_review = True
            continue
        margin = float(info.get("margin", 0.0))
        line.source_cluster = info["cluster"]
        line.confidence = round(margin, 3)
        line.needs_review = margin < margin_threshold


def build_characters_from_line_clusters(lines: list[Line]) -> list[Character]:
    """Build the list of characters from the clusters assigned to the lines."""
    unique = sorted({l.source_cluster for l in lines if l.source_cluster})
    chars: list[Character] = []
    for i, cid in enumerate(unique):
        chars.append(
            Character(
                id=f"char_{i}",
                name=f"Unknown {i}",
                color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                cluster_ids=[cid],
            )
        )
    return chars


def resolve_character_ids(lines: list[Line], characters: list[Character]) -> None:
    cluster_to_char: dict[str, str] = {}
    for ch in characters:
        for cid in ch.cluster_ids:
            cluster_to_char[cid] = ch.id
    for line in lines:
        if line.source_cluster is None:
            line.character_id = None
        else:
            line.character_id = cluster_to_char.get(line.source_cluster)


def merge_consecutive_lines(lines: list[Line], max_gap_sec: float = 1.0) -> list[Line]:
    """Merge consecutive lines of the SAME character separated by a short
    silence (<= ``max_gap_sec``) into a single continuous line.

    A monologue that a subtitler split into several cues becomes a single
    line again: one recording (one countdown), one rhythmo line, and
    above all no more audio overlap at editing time (two cues of the same sentence
    are no longer two distinct takes stepping on each other).

    Lines without a character (``character_id`` None) are never merged
    (we cannot guarantee the same speaker). Preserves the order, re-indexes the ids."""
    merged: list[Line] = []
    cur: Optional[Line] = None
    for ln in lines:
        if (
            cur is not None
            and ln.character_id is not None
            and ln.character_id == cur.character_id
            and (ln.start_sec - cur.end_sec) <= max_gap_sec
        ):
            cur.end_sec = ln.end_sec
            cur.text = (cur.text + " " + ln.text).strip()
            if ln.words:  # absolute timings already set -> simple in-order concatenation
                cur.words = (cur.words or []) + ln.words
            if ln.confidence is not None:
                cur.confidence = ln.confidence if cur.confidence is None else min(cur.confidence, ln.confidence)
            cur.needs_review = bool(cur.needs_review or ln.needs_review)
        else:
            cur = ln
            merged.append(cur)
    for i, ln in enumerate(merged):
        ln.id = f"line_{i:04d}"
    return merged
