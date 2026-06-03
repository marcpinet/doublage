from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def _build_seek_args(start_sec: Optional[float], duration_sec: Optional[float]) -> list[str]:
    args: list[str] = []
    if start_sec is not None and start_sec > 0:
        args += ["-ss", f"{start_sec:.3f}"]
    if duration_sec is not None and duration_sec > 0:
        args += ["-t", f"{duration_sec:.3f}"]
    return args


def _run_ffmpeg(cmd: list[str]) -> None:
    log.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tail = (result.stderr or "")[-1000:]
        raise RuntimeError(
            f"ffmpeg exited with code {result.returncode}.\n--- last stderr ---\n{tail}"
        )


def extract_video(
    input_path: Path,
    output_path: Path,
    video_index: int,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
) -> None:
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        *_build_seek_args(start_sec, duration_sec),
        "-i", str(input_path),
        "-map", f"0:{video_index}",
        "-an", "-sn", "-dn",
        "-map_chapters", "-1",
        "-map_metadata", "-1",
        "-c:v", "copy",
        str(output_path),
    ]
    _run_ffmpeg(cmd)


def extract_audio(
    input_path: Path,
    output_path: Path,
    audio_index: int,
    start_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    sample_rate: int = 48000,
    channels: int = 2,
) -> None:
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        *_build_seek_args(start_sec, duration_sec),
        "-i", str(input_path),
        "-map", f"0:{audio_index}",
        "-vn", "-sn", "-dn",
        "-map_chapters", "-1",
        "-map_metadata", "-1",
        "-c:a", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(output_path),
    ]
    _run_ffmpeg(cmd)


def extract_subtitles(
    input_path: Path,
    output_path: Path,
    subtitle_index: int,
) -> None:
    # No -ss/-t: ffmpeg ignores input seeking on subtitle streams, so we window in parse_subtitles().
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-map", f"0:{subtitle_index}",
        "-c:s", "srt",
        str(output_path),
    ]
    _run_ffmpeg(cmd)
