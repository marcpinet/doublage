from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .align import (
    align_lines_to_clusters,
    apply_line_clusters,
    build_characters_from_line_clusters,
    build_characters_from_segments,
    resolve_character_ids,
)
from . import align_words as align_words_mod
from . import diarize as diarize_mod
from . import gpu
from . import transcribe as transcribe_mod
from .align_words import align_words
from .bundle import write_bundle
from .diarize import diarize, diarize_by_lines
from .extract import extract_audio, extract_subtitles, extract_video
from .models import BUNDLE_VERSION, Assets, Bundle, Source
from .separate import separate_voice
from .subtitles import parse_subtitles
from .tracks import (
    Track,
    format_track_table,
    normalize_language,
    probe_duration,
    probe_streams,
    select_audio_track,
    select_subtitle_track,
    select_video_track,
)


log = logging.getLogger("dubbing")


def hms_to_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid time format (expected SS, MM:SS or HH:MM:SS): {value!r}")


def _resolve_track(tracks: list[Track], index: int, expected_type: str) -> Track:
    match = next((t for t in tracks if t.index == index), None)
    if match is None:
        raise ValueError(f"Track index {index} not present in input")
    if match.codec_type != expected_type:
        raise ValueError(
            f"Track index {index} is {match.codec_type!r}, expected {expected_type!r}"
        )
    return match


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dubbing.pipeline",
        description="Dubbing preprocessing pipeline (Block A).",
    )
    p.add_argument("--input", required=True, type=Path, help="Input video file (.mkv or .mp4)")
    p.add_argument("--output", type=Path, help="Output project directory (required unless --list-tracks)")
    p.add_argument("--start", default=None, help="Start time (SS, MM:SS, or HH:MM:SS)")
    p.add_argument("--end", default=None, help="End time (SS, MM:SS, or HH:MM:SS)")
    p.add_argument("--language", default="fre", help="Preferred audio/subtitle language code (default: fre)")
    p.add_argument("--audio-track", type=int, default=None, help="Force a specific audio track index")
    p.add_argument("--subtitle-track", type=int, default=None, help="Force a specific subtitle track index")
    p.add_argument("--video-track", type=int, default=None, help="Force a specific video track index")
    p.add_argument("--list-tracks", action="store_true", help="List streams in the input and exit")
    p.add_argument("--skip-separation", action="store_true", help="Skip M1 (Demucs source separation)")
    p.add_argument("--skip-diarization", action="store_true", help="Skip M2 (pyannote diarization + alignment)")
    p.add_argument("--skip-alignment", action="store_true", help="Skip word-level forced alignment (karaoke timings)")
    p.add_argument("--auto-subs", action="store_true", help="Generate subtitles automatically with Whisper large-v3 (ignore the embedded subtitle track)")
    p.add_argument("--no-split-cues", action="store_true", help="Do not split multi-speaker subtitle cues (\\N / dialogue dashes)")
    p.add_argument("--device", default=None, help="Device for Demucs/pyannote (cuda/cpu/mps). Default: auto.")
    p.add_argument("--model", default="htdemucs_ft", help="Demucs model name (default: htdemucs_ft)")
    p.add_argument("--diar-model", default="pyannote/speaker-diarization-3.1", help="pyannote model id")
    p.add_argument("--num-speakers", type=int, default=None, help="Force exact number of speakers")
    p.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers")
    p.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Skip stages whose output already exists (extraction, separation, subtitle extraction)")
    # Advanced tuning (sane defaults; expose for fine control on tricky content).
    p.add_argument("--diar-short-line", type=float, default=0.6,
                   help="Lines shorter than this (sec) are widened around their center for embedding diarization (default: 0.6)")
    p.add_argument("--diar-short-pad", type=float, default=0.4,
                   help="Half-width (sec) of the window used for short lines in embedding diarization (default: 0.4)")
    p.add_argument("--align-min-seg", type=float, default=0.2,
                   help="Minimum segment length (sec) to attempt word alignment (default: 0.2)")
    p.add_argument("--align-collapse-ratio", type=float, default=0.5,
                   help="Drop a word alignment if it spans less than this fraction of the line duration (default: 0.5)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p


def run_pipeline(args: argparse.Namespace) -> int:
    input_path: Path = args.input
    if not input_path.exists():
        log.error("Input file does not exist: %s", input_path)
        return 1

    log.info("Probing streams of %s", input_path.name)
    tracks = probe_streams(input_path)

    if args.list_tracks:
        print(format_track_table(tracks))
        return 0

    if args.output is None:
        log.error("--output is required (unless --list-tracks)")
        return 1

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    video_track = (
        _resolve_track(tracks, args.video_track, "video")
        if args.video_track is not None
        else select_video_track(tracks)
    )
    audio_track = (
        _resolve_track(tracks, args.audio_track, "audio")
        if args.audio_track is not None
        else select_audio_track(tracks, args.language)
    )
    # In --auto-subs mode, we generate subtitles with Whisper: the embedded subtitle track is
    # ignored entirely (no need to select / extract one).
    subtitle_track = None
    if not args.auto_subs:
        subtitle_track = (
            _resolve_track(tracks, args.subtitle_track, "subtitle")
            if args.subtitle_track is not None
            else select_subtitle_track(tracks, args.language)
        )

    log.info(
        "Selected video    [%d] codec=%s",
        video_track.index, video_track.codec_name,
    )
    log.info(
        "Selected audio    [%d] codec=%s lang=%s title=%s",
        audio_track.index, audio_track.codec_name,
        audio_track.language or "-", audio_track.title or "-",
    )
    if subtitle_track is not None:
        log.info(
            "Selected subtitle [%d] codec=%s lang=%s title=%s",
            subtitle_track.index, subtitle_track.codec_name,
            subtitle_track.language or "-", subtitle_track.title or "-",
        )
    else:
        log.info("Subtitles         : auto-generated with Whisper (embedded track ignored)")

    start_sec = hms_to_seconds(args.start) if args.start else 0.0
    if args.end:
        end_sec: Optional[float] = hms_to_seconds(args.end)
        duration_sec = end_sec - start_sec
    else:
        end_sec = None
        duration_sec = probe_duration(input_path) - start_sec

    if duration_sec <= 0:
        log.error("Computed duration is non-positive (%.2f sec)", duration_sec)
        return 1

    log.info("Time range: %.2fs to %s (duration %.2fs)",
             start_sec, f"{end_sec:.2f}s" if end_sec is not None else "end-of-file", duration_sec)

    video_out = output_dir / "video" / "source_muted.mp4"
    audio_out = output_dir / "audio" / "original.wav"

    resume = bool(getattr(args, "resume", False))

    def _exists(p: Path) -> bool:
        try:
            return p.exists() and p.stat().st_size > 0
        except OSError:
            return False

    skip_video = resume and _exists(video_out)
    skip_audio = resume and _exists(audio_out)
    if skip_video:
        log.info("resume: video exists, skipping")
    if skip_audio:
        log.info("resume: audio exists, skipping")

    # extract_video / extract_audio are blocking, independent subprocess calls:
    # we run them in parallel (subprocess releases the GIL).
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        if not skip_video:
            log.info("Extracting video    -> %s", video_out.relative_to(output_dir))
            futures.append(pool.submit(
                extract_video, input_path, video_out, video_track.index, start_sec, duration_sec))
        if not skip_audio:
            log.info("Extracting audio    -> %s", audio_out.relative_to(output_dir))
            futures.append(pool.submit(
                extract_audio, input_path, audio_out, audio_track.index, start_sec, duration_sec))
        for fut in futures:
            fut.result()  # propagate any exceptions

    lines = []
    if not args.auto_subs:
        subs_out = output_dir / "subtitles" / "selected.srt"
        if resume and _exists(subs_out):
            log.info("resume: subtitles exist, skipping extraction")
        else:
            log.info("Extracting subtitle -> %s", subs_out.relative_to(output_dir))
            extract_subtitles(input_path, subs_out, subtitle_track.index)
        log.info("Parsing subtitles")
        # the .srt holds the whole track (absolute times); window + offset to the crop here
        lines = parse_subtitles(subs_out, clip_start_sec=start_sec, clip_end_sec=start_sec + duration_sec,
                                split_cues=not args.no_split_cues)

    assets = Assets()
    vocals_path: Optional[Path] = None
    if args.skip_separation:
        log.info("Skipping source separation (--skip-separation)")
    else:
        audio_dir = output_dir / "audio"
        existing_vocals = audio_dir / "vocals.wav"
        existing_no_vocals = audio_dir / "no_vocals.wav"
        if resume and _exists(existing_vocals) and _exists(existing_no_vocals):
            log.info("resume: stems exist, skipping")
            vocals_path, no_vocals_path = existing_vocals, existing_no_vocals
        else:
            log.info("Running Demucs source separation (model=%s, device=%s)",
                     args.model, args.device or "auto")
            vocals_path, no_vocals_path = separate_voice(
                audio_out, audio_dir,
                model=args.model, device=args.device,
            )
        assets.original_vocals = vocals_path.relative_to(output_dir).as_posix()
        assets.background_track = no_vocals_path.relative_to(output_dir).as_posix()
        log.info("Separated stems: %s, %s",
                 assets.original_vocals, assets.background_track)
        # separate_voice frees its own model; we observe the residual VRAM.
        gpu.log_mem("after separation")

    # --auto-subs: we transcribe (Whisper) AFTER separation, preferably on the isolated vocals
    # for a cleaner transcription, otherwise on the raw audio. Diarization (below) then assigns
    # each line to a character; align_words refines the word-level timings.
    if args.auto_subs:
        from .transcribe import transcribe
        asr_input = vocals_path if vocals_path is not None else audio_out
        log.info("Auto-generating subtitles with Whisper (input=%s)", asr_input.name)
        lines = transcribe(asr_input, language=(audio_track.language or args.language), device=args.device)
        # One-shot pipeline: we evict the Whisper model (~3 GB) to reduce the VRAM peak.
        transcribe_mod.clear_model_cache()
        gpu.empty_cache()
        gpu.log_mem("after transcription")

    characters = []
    if args.skip_diarization:
        log.info("Skipping diarization (--skip-diarization)")
    else:
        diar_input = vocals_path if vocals_path is not None else audio_out
        # P3: we decode the diarization audio only once and reuse it
        # for BOTH diarization AND word-level alignment (same file in practice).
        import soundfile as sf

        diar_data, diar_sr = sf.read(str(diar_input), always_2d=True, dtype="float32")
        diar_preloaded = (diar_data, diar_sr)
        # Per-line embeddings when the number of voices is known (much more robust).
        # Fall back to pyannote overlap if --num-speakers is absent or on failure.
        use_embeddings = args.num_speakers is not None
        if use_embeddings:
            log.info("Running per-line embedding diarization (input=%s, speakers=%d)",
                     diar_input.name, args.num_speakers)
            try:
                line_clusters = diarize_by_lines(
                    diar_input, lines, num_speakers=args.num_speakers, device=args.device,
                    preloaded=diar_preloaded,
                    short_line_sec=args.diar_short_line, short_line_pad=args.diar_short_pad,
                )
                apply_line_clusters(lines, line_clusters)
                characters = build_characters_from_line_clusters(lines)
                resolve_character_ids(lines, characters)
                log.info("Diarized by per-line embeddings (%d speakers)", args.num_speakers)
            except Exception as e:
                log.warning("Embedding diarization failed (%s); falling back to overlap", e)
                use_embeddings = False
        if not use_embeddings:
            log.info("Running diarization (model=%s, input=%s)",
                     args.diar_model, diar_input.name)
            segments = diarize(
                diar_input,
                model=args.diar_model,
                device=args.device,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                preloaded=diar_preloaded,
            )
            characters = build_characters_from_segments(segments)
            align_lines_to_clusters(lines, segments)
            resolve_character_ids(lines, characters)
        # One-shot pipeline: we evict the pyannote models to reduce the VRAM peak.
        diarize_mod.clear_model_cache()
        gpu.empty_cache()
        gpu.log_mem("after diarization")
        n_review = sum(1 for ln in lines if ln.needs_review)
        log.info(
            "Aligned %d lines to %d character(s); %d flagged needs_review",
            len(lines), len(characters), n_review,
        )

        # Word-level forced alignment (for the karaoke rhythmo band). Best-effort.
        if not args.skip_alignment:
            align_input = vocals_path if vocals_path is not None else audio_out
            # Reuse the already-decoded audio only if the path is identical.
            align_preloaded = diar_preloaded if align_input == diar_input else None
            try:
                align_words(align_input, lines, device=args.device, preloaded=align_preloaded,
                            min_seg_sec=args.align_min_seg, collapse_ratio=args.align_collapse_ratio)
            except Exception as e:
                log.warning("Word alignment failed (%s); lines keep no words", e)
            finally:
                align_words_mod.clear_model_cache()
                gpu.empty_cache()
                gpu.log_mem("after alignment")
        # Free the shared decoded audio.
        del diar_data
        diar_preloaded = None

    bundle = Bundle(
        version=BUNDLE_VERSION,
        source=Source(
            video_path=video_out.relative_to(output_dir).as_posix(),
            original_audio_path=audio_out.relative_to(output_dir).as_posix(),
            duration_sec=round(duration_sec, 3),
            language=normalize_language(audio_track.language or args.language),
        ),
        assets=assets,
        characters=characters,
        lines=lines,
    )

    bundle_path = write_bundle(bundle, output_dir)
    log.info("Wrote bundle: %s (%d lines)", bundle_path, len(lines))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130
    except Exception as e:
        log.error("%s: %s", type(e).__name__, e)
        if args.verbose:
            log.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
