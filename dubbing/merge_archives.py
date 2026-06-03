from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


from .ffmpeg_gpu import decode_args, video_encode_args

log = logging.getLogger("dubbing.merge")


@dataclass
class TakeRef:
    user: str
    line_id: Optional[str]
    take_n: int
    offset_ms: int
    file_path: Path
    character_id: Optional[str] = None
    gain_db: float = 0.0
    crop_start_ms: int = 0
    crop_end_ms: Optional[int] = None  # None = up to the end of the recording
    latency_ms: int = 0  # capture latency to compensate (placed earlier by this much)


def _extract_archive(archive: Path, dest: Path) -> tuple[dict, list[TakeRef]]:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            extracted = (dest / member).resolve()
            try:
                extracted.relative_to(dest.resolve())
            except ValueError:
                raise RuntimeError(f"unsafe path in archive: {member}")
        zf.extractall(dest)
    manifest_path = dest / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"archive {archive} has no manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    takes = []
    for t in manifest.get("takes", []):
        fp = dest / t["file_name"]
        if not fp.exists():
            log.warning("missing take file %s in %s", t["file_name"], archive.name)
            continue
        takes.append(
            TakeRef(
                user=manifest["user"],
                line_id=t.get("line_id"),
                take_n=int(t["take_n"]),
                offset_ms=int(t.get("offset_ms", 0)),
                file_path=fp,
                character_id=t.get("character_id"),
                gain_db=float(t.get("gain_db", 0.0)),
                crop_start_ms=int(t.get("crop_start_ms", 0) or 0),
                crop_end_ms=(int(t["crop_end_ms"]) if t.get("crop_end_ms") not in (None, "") else None),
                latency_ms=int(t.get("latency_ms", 0) or 0),
            )
        )
    return manifest, takes


def _gain_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _dub_windows(all_takes, lines_by_id, lines, pad_ms):
    """Intervals [start,end] (seconds) where a line is dubbed → the original audio
    must give way to the background (no_vocals) + the take. Pro = the take's line;
    Live/session (without line_id) = all lines of the take's character.
    Padding ±pad_ms then merging of the intervals that overlap."""
    pad = pad_ms / 1000.0
    wins: list[tuple[float, float]] = []
    for take in all_takes:
        if take.line_id and take.line_id in lines_by_id:
            ln = lines_by_id[take.line_id]
            wins.append((float(ln["start_sec"]), float(ln["end_sec"])))
        elif take.character_id:
            for ln in lines:
                if ln.get("character_id") == take.character_id:
                    wins.append((float(ln["start_sec"]), float(ln["end_sec"])))
    if not wins:
        return []
    wins = sorted((max(0.0, s - pad), e + pad) for s, e in wins)
    merged = [list(wins[0])]
    for s, e in wins[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def merge(
    bundle_dir: Path,
    archives: list[Path],
    output: Path,
    bg_gain_db: float = -3.0,
    voice_gain_db: float = 0.0,
    transcode_video: bool = False,
    audio_mode: str = "separated",
    duck_pad_ms: int = 60,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    bundle_path = bundle_dir / "project.json"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    video_path = (bundle_dir / bundle["source"]["video_path"]).resolve()
    bg_rel = bundle.get("assets", {}).get("background_track")
    if not bg_rel:
        raise RuntimeError("bundle has no background_track (run M1 source separation first)")
    bg_path = (bundle_dir / bg_rel).resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not bg_path.exists():
        raise FileNotFoundError(bg_path)

    ows = audio_mode == "ows"
    orig_path = None
    if ows:
        orig_rel = bundle.get("source", {}).get("original_audio_path")
        if not orig_rel:
            raise RuntimeError("OWS mode needs source.original_audio_path (re-preprocess the scene)")
        orig_path = (bundle_dir / orig_rel).resolve()
        if not orig_path.exists():
            raise FileNotFoundError(orig_path)

    lines = bundle.get("lines", [])
    lines_by_id = {ln["id"]: ln for ln in lines}

    with tempfile.TemporaryDirectory(prefix="dub_merge_") as tmpd:
        tmpd = Path(tmpd)
        all_takes: list[TakeRef] = []
        for arc in archives:
            sub = tmpd / arc.stem
            _, takes = _extract_archive(arc, sub)
            log.info("archive %s: %d take(s) by %s", arc.name, len(takes), takes[0].user if takes else "?")
            all_takes.extend(takes)

        if not all_takes:
            raise RuntimeError("no usable takes across the provided archives")

        bg_lin = _gain_to_linear(bg_gain_db)
        voice_lin = _gain_to_linear(voice_gain_db)

        filters: list[str] = []
        amix_pieces: list[str] = []

        if ows:
            # Base = original audio everywhere, EXCEPT during the dubbed windows where we
            # switch to no_vocals (music/SFX without the original voice) + the take.
            windows = _dub_windows(all_takes, lines_by_id, lines, duck_pad_ms)
            win_expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in windows) or "0"
            log.info("OWS: %d dub window(s)", len(windows))
            inputs = [str(video_path), str(orig_path), str(bg_path)]
            # original: present everywhere, cut out during the windows
            filters.append(f"[1:a]volume={bg_lin:.4f},volume=0:enable='{win_expr}'[orig]")
            # no_vocals: muted everywhere, except during the windows (inverse enable).
            # lt(...,1) and not 1-(...): the windows are merged today, but if two ever
            # overlapped, the sum would be 2 and 1-2=-1 is truthy for ffmpeg -> both
            # [orig] and [nov] muted at once (audio dropout).
            filters.append(f"[2:a]volume={bg_lin:.4f},volume=0:enable='lt({win_expr},1)'[nov]")
            amix_pieces = ["[orig]", "[nov]"]
            take_base_idx = 3
        else:
            inputs = [str(video_path), str(bg_path)]
            filters.append(f"[1:a]volume={bg_lin:.4f}[bg]")
            amix_pieces = ["[bg]"]
            take_base_idx = 2

        for i, take in enumerate(all_takes):
            inputs.append(str(take.file_path))
            input_idx = take_base_idx + i
            if take.line_id and take.line_id in lines_by_id:
                line = lines_by_id[take.line_id]
                base_delay_ms = int(line["start_sec"] * 1000) + take.offset_ms
            else:
                base_delay_ms = take.offset_ms
            # Compensate the recording capture latency (Bluetooth headsets etc.): the audio is late by
            # latency_ms, so the dub must land EARLIER by that much. Two ways, combined:
            #  1) pull the placement earlier (reduce adelay) — works when there's delay to give back;
            #  2) for the leftover that can't fit (e.g. a Live take already starts at t=0, base_delay≈0),
            #     TRIM that many ms off the HEAD of the recording instead, so the voice still shifts up.
            # Without (2), a Live take's latency was silently lost (0 − 307 → max(0,…) → 0).
            lat = max(0, int(take.latency_ms))
            from_delay = min(lat, max(0, base_delay_ms))   # part absorbed by shifting the placement
            base_delay_ms = max(0, base_delay_ms - from_delay)
            latency_head_trim_ms = lat - from_delay        # remainder trimmed from the take's head
            # Per-take gain (set by the host in the collection mixer) compounds with the
            # global voice gain: total linear = voice_lin * 10^(take.gain_db/20).
            take_lin = voice_lin * _gain_to_linear(take.gain_db)
            lbl = f"v{i}"
            # IMPORTANT: we do NOT auto-trim leading/trailing silence. Players record while WATCHING
            # the video, so any silence at the head of a take is intentional timing (the subtitle cue
            # is often early vs the on-screen mouth) — stripping it would pull the voice EARLIER and
            # desync it. The recording's first sample is placed exactly at base_delay, preserving the
            # timing the player performed. (The host can still crop precisely on the timeline.)
            # Crop (host edge-trim in the timeline): keep only [crop_start, crop_end] of the raw
            # recording, applied BEFORE adelay; asetpts resets timestamps so adelay positions from 0.
            # The latency-head-trim (leftover capture latency that adelay couldn't absorb) is added to
            # the crop start, so a Live take's voice is pulled earlier by trimming its first ms.
            crop = ""
            cs = max(0, int(take.crop_start_ms)) + max(0, int(latency_head_trim_ms))
            ce = take.crop_end_ms
            if cs > 0 or ce is not None:
                a = f"atrim=start={cs / 1000:.3f}"
                if ce is not None and int(ce) > cs:
                    a += f":end={int(ce) / 1000:.3f}"
                crop = f"{a},asetpts=PTS-STARTPTS,"
            # Force the take to AUDIBLE stereo. Some mics record true mono, and some record stereo
            # with sound on a SINGLE channel (mono mic into a stereo container) — both play in one ear
            # only. `aformat=stereo` alone doesn't fix the one-sided-stereo case. So we first downmix
            # to mono (`pan=mono|c0=.5*c0+.5*c1`, which keeps whichever channel has the signal) then
            # expand that mono back to both L+R, guaranteeing centered, both-ears audio.
            filters.append(
                f"[{input_idx}:a]aresample=44100,"
                f"pan=mono|c0=0.5*c0+0.5*c1,aformat=channel_layouts=stereo,"
                f"{crop}adelay={base_delay_ms}|{base_delay_ms},volume={take_lin:.4f}[{lbl}]"
            )
            amix_pieces.append(f"[{lbl}]")

        filters.append(
            "".join(amix_pieces)
            + f"amix=inputs={len(amix_pieces)}:duration=longest:normalize=0[mix]"
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y"]
        if transcode_video:
            cmd += decode_args()  # GPU decoding of the input video (input 0)
        for inp in inputs:
            cmd += ["-i", inp]
        cmd += ["-filter_complex", ";".join(filters)]
        cmd += ["-map", "0:v:0", "-map", "[mix]"]
        if transcode_video:
            cmd += video_encode_args(20, cpu_preset="medium")
        else:
            cmd += ["-c:v", "copy"]
        cmd += [
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-shortest",
            str(output),
        ]

        log.info("ffmpeg: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            tail = (result.stderr or "")[-2000:]
            raise RuntimeError(f"ffmpeg failed (code {result.returncode}):\n{tail}")

    return output


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dubbing.merge_archives",
        description="Merge per-user submission archives into a final dubbed video.",
    )
    p.add_argument("--bundle", required=True, type=Path, help="Bundle directory (project.json at root)")
    p.add_argument("--archives", required=True, nargs="+", type=Path,
                   help="Per-user submission ZIP archives (or one mega-zip)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--bg-gain-db", type=float, default=-3.0)
    p.add_argument("--voice-gain-db", type=float, default=0.0)
    p.add_argument("--transcode-video", action="store_true",
                   help="Re-encode video to H.264 (fixes playback if HEVC unsupported)")
    p.add_argument("--audio-mode", choices=["separated", "ows"], default="separated",
                   help="separated: Demucs background everywhere. ows: original audio except during the dubbed lines")
    return p


def _flatten_megazip(megazip: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(megazip) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".zip"):
                continue
            out = dest / Path(member).name
            with zf.open(member) as src, open(out, "wb") as f:
                shutil.copyfileobj(src, f)
            extracted.append(out)
    return extracted


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    archives: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="dub_megaunpack_") as tmpd:
        tmpd = Path(tmpd)
        for arc in args.archives:
            if not arc.exists():
                log.error("archive not found: %s", arc); return 1
            try:
                with zipfile.ZipFile(arc) as zf:
                    members = zf.namelist()
                inner_zips = [m for m in members if m.lower().endswith(".zip")]
                has_manifest = any(m.endswith("manifest.json") for m in members)
                if inner_zips and not has_manifest:
                    log.info("megazip detected: %s -> %d inner submissions", arc.name, len(inner_zips))
                    archives.extend(_flatten_megazip(arc, tmpd / arc.stem))
                else:
                    archives.append(arc)
            except zipfile.BadZipFile:
                log.error("not a zip: %s", arc); return 1

        try:
            output = merge(
                bundle_dir=args.bundle.resolve(),
                archives=archives,
                output=args.output.resolve(),
                bg_gain_db=args.bg_gain_db,
                voice_gain_db=args.voice_gain_db,
                transcode_video=args.transcode_video,
                audio_mode=args.audio_mode,
            )
        except FileNotFoundError as e:
            log.error("file not found: %s", e); return 1
        except RuntimeError as e:
            log.error("%s", e); return 1

    log.info("wrote %s (%.1f MB)", output, output.stat().st_size / 1024 / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
