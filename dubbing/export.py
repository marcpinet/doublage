from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


from .ffmpeg_gpu import decode_args, video_encode_args

log = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    voice_gain_db: float = 0.0
    bg_gain_db: float = -3.0
    audio_bitrate: str = "192k"
    transcode_video: bool = False
    video_crf: int = 20


def _gain_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def export_mix(
    video_path: Path,
    background_path: Path,
    recording_path: Path,
    output_path: Path,
    options: Optional[ExportOptions] = None,
) -> Path:
    _check_ffmpeg()
    opts = options or ExportOptions()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for p in (video_path, background_path, recording_path):
        if not p.exists():
            raise FileNotFoundError(p)

    voice_lin = _gain_to_linear(opts.voice_gain_db)
    bg_lin = _gain_to_linear(opts.bg_gain_db)

    filter_complex = (
        f"[1:a]volume={bg_lin:.4f}[bg];"
        f"[2:a]volume={voice_lin:.4f}[voc];"
        f"[bg][voc]amix=inputs=2:duration=longest:normalize=0[mix]"
    )

    cmd = [
        "ffmpeg", "-y",
        *(decode_args() if opts.transcode_video else []),
        "-i", str(video_path),
        "-i", str(background_path),
        "-i", str(recording_path),
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[mix]",
    ]
    if opts.transcode_video:
        cmd += video_encode_args(opts.video_crf, cpu_preset="medium")
    else:
        cmd += ["-c:v", "copy"]
    cmd += [
        "-c:a", "aac", "-b:a", opts.audio_bitrate,
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]

    log.info("export: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tail = (result.stderr or "")[-1500:]
        raise RuntimeError(f"ffmpeg export failed (code {result.returncode}):\n{tail}")

    return output_path


def next_export_name(exports_dir: Path, prefix: str = "dub", ext: str = "mp4") -> str:
    exports_dir.mkdir(parents=True, exist_ok=True)
    # max(existing numbers) + 1, NOT count + 1: if dub_001 was deleted, count+1 would
    # produce dub_002 again and silently overwrite it.
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.{re.escape(ext)}$")
    nums = [int(m.group(1)) for p in exports_dir.glob(f"{prefix}_*.{ext}")
            if (m := pat.match(p.name))]
    return f"{prefix}_{max(nums, default=0) + 1:03d}.{ext}"
