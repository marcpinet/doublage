from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


from .ffmpeg_gpu import decode_args, video_encode_args

log = logging.getLogger("dubbing.distribute")

BUNDLE_FILENAME = "project.json"


@dataclass
class DistributeOptions:
    video_crf: int = 23
    max_height: int = 720
    audio_bitrate: str = "128k"
    include_vocals: bool = True


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def _run_ffmpeg(cmd: list[str]) -> None:
    log.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        tail = (result.stderr or "")[-1500:]
        raise RuntimeError(f"ffmpeg exited with code {result.returncode}.\n--- last stderr ---\n{tail}")


def _sig(path: Path) -> str:
    """Cheap fingerprint of a source file: size + mtime. Changes if the file is re-encoded."""
    st = path.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def _transcode_video(src: Path, dst: Path, opts: DistributeOptions) -> None:
    vf = f"scale=-2:'min({opts.max_height},ih)'"
    cmd = [
        "ffmpeg", "-y",
        *decode_args(),
        "-i", str(src),
        "-an", "-sn", "-dn",
        "-map", "0:v:0",
        "-vf", vf,
        *video_encode_args(opts.video_crf, cpu_preset="veryfast"),
        "-movflags", "+faststart",
        str(dst),
    ]
    _run_ffmpeg(cmd)


def _transcode_audio(src: Path, dst: Path, opts: DistributeOptions) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn", "-sn", "-dn",
        "-c:a", "aac", "-b:a", opts.audio_bitrate, "-ac", "2",
        "-movflags", "+faststart",
        str(dst),
    ]
    _run_ffmpeg(cmd)


def build_distribution(
    bundle_dir: Path,
    out_zip: Optional[Path] = None,
    options: Optional[DistributeOptions] = None,
    on_step: Optional[Callable[[str], None]] = None,
) -> Path:
    _check_ffmpeg()
    opts = options or DistributeOptions()
    bundle_dir = bundle_dir.resolve()

    def step(m: str) -> None:
        log.info("%s", m)
        if on_step:
            on_step(m)

    bundle_path = bundle_dir / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    src = bundle.get("source", {})
    assets = bundle.get("assets", {}) or {}

    video_rel = src.get("video_path")
    bg_rel = assets.get("background_track")
    vocals_rel = assets.get("original_vocals")

    if not video_rel:
        raise RuntimeError("bundle has no source.video_path")
    video_src = bundle_dir / video_rel
    if not video_src.exists():
        raise FileNotFoundError(video_src)
    if not bg_rel:
        raise RuntimeError("bundle has no assets.background_track (run source separation first)")
    bg_src = bundle_dir / bg_rel
    if not bg_src.exists():
        raise FileNotFoundError(bg_src)

    if out_zip is None:
        out_zip = Path(tempfile.gettempdir()) / f"dist_{bundle_dir.name}.zip"

    use_vocals = bool(opts.include_vocals and vocals_rel and (bundle_dir / vocals_rel).exists())

    # Transcoded-media cache: the dozens-of-seconds H.264/AAC encode is the slow part, and it only
    # depends on the SOURCE files + encode options — NOT on edits to project.json (characters,
    # lines, timing). So we cache the encoded media keyed by a fingerprint and, on a hit, skip the
    # transcode and just repackage with the CURRENT project.json. Lives under the bundle so it
    # survives reuse of the same scene; invalidated automatically if a source file changes.
    cache_dir = bundle_dir / ".dist_cache"
    cache_key = {
        "v": 1,
        "video": _sig(video_src),
        "bg": _sig(bg_src),
        "vocals": (_sig(bundle_dir / vocals_rel) if use_vocals else None),
        "opts": {
            "crf": opts.video_crf, "max_height": opts.max_height,
            "audio_bitrate": opts.audio_bitrate, "include_vocals": use_vocals,
        },
    }
    cache_meta = cache_dir / "cache.json"
    c_video = cache_dir / "play.mp4"
    c_bg = cache_dir / "bg.m4a"
    c_vocals = cache_dir / "vocals.m4a"

    def _cache_valid() -> bool:
        try:
            if not (cache_meta.exists() and c_video.exists() and c_bg.exists()):
                return False
            if use_vocals and not c_vocals.exists():
                return False
            return json.loads(cache_meta.read_text(encoding="utf-8")) == cache_key
        except Exception:
            return False

    cache_hit = _cache_valid()
    if cache_hit:
        step("Reusing cached transcode (sources unchanged) — skipping re-encode.")
    else:
        # Rebuild the cache atomically: encode into a temp dir, then swap into place.
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dub_enc_", dir=str(cache_dir)) as enc:
            enc = Path(enc)
            step("Transcoding video to H.264 (browser-playable)…")
            _transcode_video(video_src, enc / "play.mp4", opts)
            step("Transcoding background audio to AAC…")
            _transcode_audio(bg_src, enc / "bg.m4a", opts)
            if use_vocals:
                step("Transcoding reference vocals to AAC…")
                _transcode_audio(bundle_dir / vocals_rel, enc / "vocals.m4a", opts)
            os.replace(enc / "play.mp4", c_video)
            os.replace(enc / "bg.m4a", c_bg)
            if use_vocals:
                os.replace(enc / "vocals.m4a", c_vocals)
            else:
                c_vocals.unlink(missing_ok=True)
            cache_meta.write_text(json.dumps(cache_key, ensure_ascii=False), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="dub_dist_") as tmp:
        tmp = Path(tmp)
        (tmp / "video").mkdir()
        (tmp / "audio").mkdir()

        shutil.copyfile(c_video, tmp / "video" / "play.mp4")
        shutil.copyfile(c_bg, tmp / "audio" / "bg.m4a")

        dist = json.loads(json.dumps(bundle))
        dist["source"]["video_path"] = "video/play.mp4"
        dist["assets"]["background_track"] = "audio/bg.m4a"
        dist["source"].pop("original_audio_path", None)

        if use_vocals:
            shutil.copyfile(c_vocals, tmp / "audio" / "vocals.m4a")
            dist["assets"]["original_vocals"] = "audio/vocals.m4a"
        else:
            dist["assets"].pop("original_vocals", None)

        dist["distribution"] = True
        (tmp / BUNDLE_FILENAME).write_text(
            json.dumps(dist, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        step("Packaging distribution bundle…")
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(tmp.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(tmp).as_posix())

    size_mb = out_zip.stat().st_size / 1024 / 1024
    step(f"Distribution bundle ready: {out_zip.name} ({size_mb:.1f} MB)")
    return out_zip
