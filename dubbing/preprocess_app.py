from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
import shutil
import string
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .ffmpeg_gpu import decode_args, video_encode_args
from .tracks import (
    format_track_table,
    probe_duration,
    probe_streams,
    select_audio_track,
    select_subtitle_track,
)


log = logging.getLogger("dubbing.preprocess")

WEB_DIR = Path(__file__).parent / "web"

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts"}

jobs: dict[str, dict] = {}
preview_jobs: dict[str, dict] = {}


async def _iter_proc_lines(proc):
    """Yield output lines from a subprocess, split on both '\\n' AND '\\r'.

    We must NOT use StreamReader.readline(): it raises LimitOverrunError once a single
    "line" exceeds its 64 KiB buffer. ffmpeg's progress meter (frame=… time=…) rewrites
    the same line with carriage returns and NEVER emits '\\n', so on a long encode the
    accumulated bytes blow past that limit and crash the reader. Reading raw chunks and
    splitting on \\r/\\n ourselves has no separator-size limit.
    """
    buf = bytearray()
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        buf.extend(chunk)
        # Split on CR or LF so ffmpeg progress updates (\r) surface as their own lines, but treat a
        # CRLF pair ('\r\n', the default on Windows logging) as a SINGLE break so we don't emit a
        # spurious empty line after every normal log line.
        start = 0
        i = 0
        n = len(buf)
        # Don't process a trailing '\r' yet: it might be the first half of a '\r\n' split across reads.
        limit = n - 1 if (n and buf[n - 1] == 0x0D) else n
        while i < limit:
            b = buf[i]
            if b == 0x0A or b == 0x0D:
                yield bytes(buf[start:i])
                # swallow the '\n' of a '\r\n' pair
                if b == 0x0D and i + 1 < n and buf[i + 1] == 0x0A:
                    i += 1
                start = i + 1
            i += 1
        del buf[:start]
        # Safety valve: if a chunk really has no separator at all, don't grow unbounded.
        if len(buf) > 65536:
            yield bytes(buf)
            buf.clear()
    if buf:
        yield bytes(buf)


def _default_browse_path() -> Path:
    home = Path.home()
    for candidate in ("Videos", "Movies", "Media", "Documents"):
        p = home / candidate
        if p.exists() and p.is_dir():
            return p
    return home


def _validate_video_path(path_str: str) -> Path:
    if not path_str:
        raise HTTPException(400, "missing path")
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise HTTPException(404, f"file not found: {p}")
    if not p.is_file():
        raise HTTPException(400, "not a file")
    if p.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported extension {p.suffix}")
    return p


def create_app() -> FastAPI:
    app = FastAPI(title="Dubbing Preprocess")

    @app.get("/")
    def root():
        return FileResponse(WEB_DIR / "preprocess.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/api/probe")
    def probe(path: str, language: str = "fre"):
        p = _validate_video_path(path)
        try:
            tracks = probe_streams(p)
            duration = probe_duration(p)
        except Exception as e:
            raise HTTPException(500, f"ffprobe failed: {e}")

        def _suggest(fn):
            try:
                return fn(tracks, language).index
            except Exception:
                return None

        return {
            "path": str(p),
            "duration_sec": duration,
            "suggested_audio": _suggest(select_audio_track),
            "suggested_subtitle": _suggest(select_subtitle_track),
            "tracks": [
                {
                    "index": t.index,
                    "codec_type": t.codec_type,
                    "codec_name": t.codec_name,
                    "language": t.language,
                    "title": t.title,
                    "forced": t.forced,
                    "hearing_impaired": t.hearing_impaired,
                    "default": t.default,
                    "num_frames": t.num_frames,
                }
                for t in tracks
            ],
        }

    @app.get("/api/preview")
    def preview(path: str):
        p = _validate_video_path(path)
        return FileResponse(p, media_type="video/mp4" if p.suffix.lower() == ".mp4" else None)

    @app.post("/api/pick-file")
    def pick_file():
        import json as _json
        import subprocess as _sp
        exts = " ".join("*" + e for e in sorted(VIDEO_EXTS))
        code = (
            "import tkinter as tk\n"
            "from tkinter import filedialog\n"
            "import json, sys\n"
            "root = tk.Tk()\n"
            "root.withdraw()\n"
            "root.attributes('-topmost', True)\n"
            f"p = filedialog.askopenfilename(title='Choose video file', filetypes=[('Video files', '{exts}'), ('All files', '*.*')])\n"
            "root.destroy()\n"
            "print(json.dumps({'path': p or None}))\n"
        )
        kwargs = {"capture_output": True, "text": True, "timeout": 600}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000
        try:
            res = _sp.run([sys.executable, "-c", code], **kwargs)
        except _sp.TimeoutExpired:
            raise HTTPException(408, "file picker timeout")
        if res.returncode != 0:
            raise HTTPException(500, f"file picker failed: {(res.stderr or '')[:400]}")
        try:
            line = (res.stdout or "").strip().splitlines()[-1]
            return _json.loads(line)
        except Exception:
            raise HTTPException(500, "could not parse picker output")

    @app.post("/api/generate-preview")
    async def generate_preview(request: Request):
        body = await request.json()
        input_path = _validate_video_path(body.get("path", ""))
        audio_track = body.get("audio_track")

        job_id = secrets.token_urlsafe(6)
        tmp_dir = Path(tempfile.gettempdir()) / "dubbing-preview"
        tmp_dir.mkdir(exist_ok=True)
        out_path = tmp_dir / f"preview_{job_id}.mp4"

        # Preview fallback: ACTUALLY re-encode to H.264 720p (not -c:v copy, which kept
        # the HEVC and made the preview unplayable in browsers without HEVC support).
        cmd = ["ffmpeg", "-y", *decode_args(), "-i", str(input_path)]
        cmd += ["-map", "0:v:0"]
        if audio_track is not None and str(audio_track) != "":
            cmd += ["-map", f"0:{int(audio_track)}"]
        else:
            cmd += ["-map", "0:a:0"]
        cmd += [
            "-vf", "scale=-2:'min(720,ih)'",
            *video_encode_args(28, cpu_preset="veryfast"),
            "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-movflags", "+faststart",
            str(out_path),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )
        except FileNotFoundError:
            raise HTTPException(500, "ffmpeg not found on PATH")

        preview_jobs[job_id] = {
            "status": "running",
            "path": str(out_path),
            "process": proc,
            "error": None,
        }

        async def wait():
            assert proc.stdout is not None
            tail_buf = []
            async for line in _iter_proc_lines(proc):
                try:
                    s = line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    s = ""
                tail_buf.append(s)
                if len(tail_buf) > 30:
                    tail_buf.pop(0)
            rc = await proc.wait()
            # The job may have been cleaned up (host left / cropped) while encoding: use .get().
            j = preview_jobs.get(job_id)
            if j is None:
                return
            if rc == 0:
                j["status"] = "done"
            else:
                j["status"] = "failed"
                j["error"] = "\n".join(tail_buf[-10:])

        asyncio.create_task(wait())
        log.info("Preview job %s started", job_id)
        return {"job_id": job_id}

    @app.post("/api/preview-proxy/{job_id}/cleanup")
    def preview_cleanup(job_id: str):
        """Purge a preview proxy: kill it if still encoding, delete its temp file.

        Called by the host UI when it leaves the source screen or after the bundle is
        built (the cropped scene supersedes the preview), so transcoded previews don't
        accumulate in the temp dir."""
        job = preview_jobs.pop(job_id, None)
        if job is None:
            return {"ok": True}
        proc = job.get("process")
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        try:
            p = Path(job.get("path", ""))
            if p.name:
                p.unlink(missing_ok=True)
        except OSError:
            pass
        return {"ok": True}

    @app.get("/api/preview-proxy/{job_id}/status")
    def preview_status(job_id: str):
        job = preview_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "preview job not found")
        return {"status": job["status"], "error": job.get("error")}

    @app.get("/api/preview-proxy/{job_id}")
    def preview_file(job_id: str):
        job = preview_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "preview job not found")
        if job["status"] == "running":
            raise HTTPException(425, "still generating")
        if job["status"] != "done":
            raise HTTPException(500, job.get("error") or "preview failed")
        return FileResponse(job["path"], media_type="video/mp4")

    @app.post("/api/process")
    async def process(request: Request):
        body = await request.json()
        input_path = _validate_video_path(body.get("input", ""))
        output_dir = Path(body.get("output", "")).expanduser().resolve()
        if not output_dir.name:
            raise HTTPException(400, "missing output dir")
        start = (body.get("start") or "").strip() or None
        end = (body.get("end") or "").strip() or None
        language = (body.get("language") or "fre").strip()
        num_speakers = body.get("num_speakers")
        skip_separation = bool(body.get("skip_separation"))
        skip_diarization = bool(body.get("skip_diarization"))
        auto_subs = bool(body.get("auto_subs"))
        audio_track = body.get("audio_track")
        subtitle_track = body.get("subtitle_track")

        cmd = [
            sys.executable, "-m", "dubbing.pipeline",
            "--input", str(input_path),
            "--output", str(output_dir),
            "--language", language,
            "--verbose",
        ]
        if start: cmd += ["--start", start]
        if end: cmd += ["--end", end]
        if num_speakers is not None and num_speakers != "":
            cmd += ["--num-speakers", str(int(num_speakers))]
        if audio_track is not None and audio_track != "":
            cmd += ["--audio-track", str(int(audio_track))]
        if subtitle_track is not None and subtitle_track != "" and not auto_subs:
            cmd += ["--subtitle-track", str(int(subtitle_track))]
        if skip_separation: cmd += ["--skip-separation"]
        if skip_diarization: cmd += ["--skip-diarization"]
        if auto_subs: cmd += ["--auto-subs"]

        job_id = secrets.token_urlsafe(6)
        log_q: asyncio.Queue = asyncio.Queue()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        jobs[job_id] = {
            "status": "running",
            "process": proc,
            "log_queue": log_q,
            "cmd": cmd,
            "output_dir": str(output_dir),
            "returncode": None,
        }

        async def consume():
            assert proc.stdout is not None
            async for line in _iter_proc_lines(proc):
                try:
                    s = line.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    s = repr(line)
                await log_q.put(s)
            rc = await proc.wait()
            jobs[job_id]["returncode"] = rc
            jobs[job_id]["status"] = "done" if rc == 0 else "failed"
            await log_q.put(None)

        asyncio.create_task(consume())
        log.info("Started job %s cmd: %s", job_id, " ".join(cmd))
        return {"job_id": job_id, "cmd": cmd}

    @app.get("/api/process/{job_id}")
    def process_status(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return {
            "status": job["status"],
            "returncode": job["returncode"],
            "output_dir": job["output_dir"],
        }

    @app.get("/api/process/{job_id}/log")
    async def stream_log(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        q: asyncio.Queue = job["log_queue"]

        async def gen():
            while True:
                line = await q.get()
                if line is None:
                    yield f"event: end\ndata: {job['status']}\n\n"
                    break
                safe = line.replace("\n", " ")
                yield f"data: {safe}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    @app.post("/api/process/{job_id}/cancel")
    def cancel(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        proc = job["process"]
        if proc.returncode is None:
            try: proc.terminate()
            except ProcessLookupError: pass
        return {"ok": True}

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dubbing.preprocess_app", description="Local preprocess UI (admin tool)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    return p


def main(argv=None) -> int:
    import uvicorn
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    app = create_app()
    log.info("Preprocess UI at http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
