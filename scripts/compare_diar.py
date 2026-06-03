from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dubbing.align import (
    align_lines_to_clusters,
    build_characters_from_segments,
    resolve_character_ids,
)
from dubbing.diarize import diarize
from dubbing.subtitles import parse_subtitles


logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

VOCALS = ROOT / "out" / "audio" / "vocals.wav"
SUBS = ROOT / "out" / "subtitles" / "selected.srt"
CLIP_DURATION = 144.0


def run_variant(label: str, **diar_kwargs) -> None:
    lines = parse_subtitles(SUBS, clip_start_sec=0.0, clip_end_sec=CLIP_DURATION)
    segments = diarize(VOCALS, **diar_kwargs)
    characters = build_characters_from_segments(segments)
    align_lines_to_clusters(lines, segments)
    resolve_character_ids(lines, characters)

    print(f"\n{'='*72}")
    print(f"VARIANT: {label}")
    print(f"  diar segments: {len(segments)}  clusters: {len({s.cluster_id for s in segments})}")
    counts = Counter(ln.character_id for ln in lines)
    review = sum(1 for ln in lines if ln.needs_review)
    print(f"  lines per character: {dict(counts)}  needs_review: {review}/{len(lines)}")
    avg_conf = sum(ln.confidence or 0 for ln in lines) / len(lines)
    print(f"  avg confidence: {avg_conf:.2f}")

    print(f"  attributions:")
    for ln in lines:
        flag = " *" if ln.needs_review else "  "
        cid = ln.character_id or "----"
        print(f"   {flag}{ln.start_sec:6.2f}-{ln.end_sec:6.2f} {cid} c={ln.confidence:.2f}  {ln.text[:50]}")


if not VOCALS.exists():
    print(f"ERROR: {VOCALS} not found. Run the main pipeline first.")
    sys.exit(1)

run_variant("auto (no constraint)")
run_variant("num_speakers=2", num_speakers=2)
run_variant("num_speakers=3", num_speakers=3)
run_variant("num_speakers=4", num_speakers=4)
run_variant("min=2 max=4", min_speakers=2, max_speakers=4)
