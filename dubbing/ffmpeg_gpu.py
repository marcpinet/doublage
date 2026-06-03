"""GPU acceleration (NVIDIA NVENC) for ffmpeg encoding, with automatic CPU fallback.

Every site that re-encodes video (distribution, editing, export, preview) goes
through ``video_encode_args()`` / ``decode_args()``: if ffmpeg can encode in
``h264_nvenc`` on this machine, we encode (and decode) on the GPU; otherwise we fall
back to ``libx264`` (CPU), exactly as before. The ``-c:v copy`` commands (extraction)
and audio are not affected.
"""
from __future__ import annotations

import logging
import subprocess
from functools import lru_cache

log = logging.getLogger(__name__)

# NVENC preset (p1=fast … p7=quality) approximating a libx264 preset (speed↔quality)
_NVENC_PRESET = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p3", "faster": "p4",
    "fast": "p4", "medium": "p5", "slow": "p6", "slower": "p7", "veryslow": "p7",
}


def _probe_nvenc_once() -> tuple[bool, str]:
    """A real h264_nvenc encoding attempt. Returns (ok, last stderr).

    Probes at 256x256 + ``-pix_fmt yuv420p``: NVENC has a MINIMUM dimension (~145x49),
    an image that is too small (128x128) fails with "Frame Dimension less than the minimum
    supported value" and gave a FALSE negative -> everything fell back to CPU."""
    try:
        test = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        return test.returncode == 0, (test.stderr or "").strip()[-300:]
    except Exception as e:
        return False, str(e)


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True if ffmpeg can ACTUALLY encode H.264 on the NVIDIA GPU.

    Verified by a real mini-encode, not just by the ``-encoders`` list (which may
    advertise h264_nvenc without a usable GPU/driver). A failure is retried once to
    absorb a transient false negative (driver cold-start, GPU momentarily busy).
    Memoized: runs only once per process."""
    try:
        listed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        if "h264_nvenc" not in (listed.stdout or ""):
            log.info("ffmpeg: h264_nvenc not in build -> CPU encoding (libx264)")
            return False
        ok, err = _probe_nvenc_once()
        if not ok:
            ok, err = _probe_nvenc_once()  # 2nd attempt: absorbs a transient failure
        log.info("ffmpeg: NVENC %s", "available -> GPU encoding" if ok
                 else f"not usable -> CPU encoding (libx264). Details: {err}")
        return ok
    except Exception as e:
        log.info("ffmpeg: NVENC detection failed (%s) -> CPU encoding", e)
        return False


def decode_args() -> list[str]:
    """Options to place BEFORE the first ``-i`` to decode on the GPU.

    ``-hwaccel cuda`` is just a hint: ffmpeg falls back on its own to software
    decoding for codecs the GPU cannot decode, so it is risk-free."""
    return ["-hwaccel", "cuda"] if nvenc_available() else []


def video_encode_args(crf: int = 23, cpu_preset: str = "medium") -> list[str]:
    """``-c:v …`` args to encode the video: GPU (h264_nvenc) if available, else libx264.

    ``crf`` = quality (scale ~0-51, lower = better); NVENC uses it via ``-cq``."""
    if nvenc_available():
        return [
            "-c:v", "h264_nvenc",
            "-preset", _NVENC_PRESET.get(cpu_preset, "p5"),
            "-rc", "vbr", "-cq", str(crf), "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    return ["-c:v", "libx264", "-preset", cpu_preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
