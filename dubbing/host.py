from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import preprocess_app
from .distribute import build_distribution
from .export import next_export_name
from .merge_archives import merge as merge_archives


log = logging.getLogger("dubbing.host")

# ES modules (the voice-FX worklet/worker under web/fx/) are refused by browsers unless served
# with a JavaScript MIME type. Python's mimetypes can be broken by the OS (the Windows registry
# maps .mjs to text/plain), so pin the mapping explicitly.
import mimetypes  # noqa: E402
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

WEB_DIR = Path(__file__).parent / "web"
BUNDLE_FILENAME = "project.json"
DEFAULT_RELAY = "http://127.0.0.1:21826"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Recently used scenes, so the host can reuse them after a restart.
SCENES_REGISTRY = Path.home() / ".doublage" / "scenes.json"


@dataclass
class HostConfig:
    bundle_dir: Optional[Path]
    relay_url: str


host_jobs: dict[str, dict] = {}


def _load_registry() -> list[dict]:
    try:
        data = json.loads(SCENES_REGISTRY.read_text(encoding="utf-8"))
        return [e for e in data if isinstance(e, dict) and e.get("dir")] if isinstance(data, list) else []
    except Exception:
        return []


def _save_registry(entries: list[dict]) -> None:
    try:
        SCENES_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        SCENES_REGISTRY.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("could not save scenes registry: %s", e)


def _scene_meta(bundle_dir: Path) -> dict:
    """Best-effort light metadata read from a bundle's project.json."""
    info = {"lines": 0, "characters": 0, "duration_sec": 0}
    try:
        d = json.loads((bundle_dir / BUNDLE_FILENAME).read_text(encoding="utf-8"))
        info["lines"] = len(d.get("lines") or [])
        info["characters"] = len(d.get("characters") or [])
        info["duration_sec"] = int((d.get("source") or {}).get("duration_sec", 0) or 0)
    except Exception:
        pass
    return info


def _remember_scene(bundle_dir: Optional[Path]) -> None:
    """Upsert a scene into the registry and bump it to the top (most recent)."""
    if bundle_dir is None:
        return
    try:
        p = Path(bundle_dir).resolve()
    except Exception:
        return
    if not (p / BUNDLE_FILENAME).exists():
        return
    key = str(p)
    entries = [e for e in _load_registry() if e.get("dir") != key]
    entries.insert(0, {"dir": key, "name": p.name, "last_used": time.time(), **_scene_meta(p)})
    _save_registry(entries[:50])


OUTPUT_BUNDLES_DIRNAME = "output_bundles"


def _discover_bundle_dirs() -> list[Path]:
    """Bundle dirs to auto-offer: children of ./output_bundles plus, for backward
    compatibility, children of the working dir itself (older bundles lived there)."""
    found: list[Path] = []
    seen: set = set()
    for root in (Path.cwd() / OUTPUT_BUNDLES_DIRNAME, Path.cwd()):
        try:
            for child in root.iterdir():
                if child.is_dir() and (child / BUNDLE_FILENAME).exists():
                    r = child.resolve()
                    if r not in seen:
                        seen.add(r)
                        found.append(child)
        except Exception:
            pass
    return found


# --- Automatic karaoke re-alignment -----------------------------------------------------------
# Same characters as align_words.py (MMS_FA Latin dictionary + French diacritics).
_ALIGN_KEEP_RE = re.compile(r"[^a-zàâäéèêëïîôöùûüç']")


def _alignable_tokens(text: str) -> list[str]:
    return [t for t in (text or "").split() if _ALIGN_KEEP_RE.sub("", t.lower())]


def _line_needs_realign(line: dict) -> bool:
    """True if the line has alignable text whose word-level alignment is out of date
    (never aligned, or subtitle edited since the last alignment)."""
    text = line.get("text") or ""
    if len(_alignable_tokens(text)) < 2:
        return False  # nothing to align (karaoke impossible) → we never touch it
    return (line.get("align_text") or None) != text


# project.json is written by both the save endpoint and the realign thread: serialize all
# read-modify-write through one lock per bundle so concurrent saves can't clobber each other.
_bundle_locks: dict[str, threading.Lock] = {}
_bundle_locks_guard = threading.Lock()


def _bundle_lock(bundle_dir: Path) -> threading.Lock:
    key = str(bundle_dir.resolve())
    with _bundle_locks_guard:
        return _bundle_locks.setdefault(key, threading.Lock())


def _realign_stale(bundle_dir: Path) -> int:
    """Re-aligns ONLY the lines whose current text has not yet been aligned, then rewrites the
    bundle. Returns the number of re-aligned lines (0 if there is nothing to do → no model is
    loaded). The align_text marker avoids endlessly retrying a given text (useful in multilingual
    cases where alignment fails: we don't retry it on every save)."""
    from dataclasses import asdict
    from .align_words import align_words
    from .models import Line

    bp = bundle_dir / BUNDLE_FILENAME
    lock = _bundle_lock(bundle_dir)
    with lock:
        data = json.loads(bp.read_text(encoding="utf-8"))
    stale = [l for l in data.get("lines", []) if _line_needs_realign(l)]
    if not stale:
        return 0
    assets = data.get("assets") or {}
    src = data.get("source") or {}
    rel = assets.get("original_vocals") or src.get("original_audio_path")
    if not rel:
        raise RuntimeError("bundle has no vocals/original audio to align against")
    audio = bundle_dir / rel
    if not audio.exists():
        raise FileNotFoundError(f"audio not found: {audio}")
    lines = [Line(id=l["id"], start_sec=float(l["start_sec"]), end_sec=float(l["end_sec"]),
                  text=l.get("text") or "") for l in stale]
    align_words(audio, lines)  # device auto (cuda if available); fills .words and .align_text
    by_id = {ln.id: ln for ln in lines}
    # Alignment takes seconds: the bundle may have been saved again meanwhile. Re-read it and
    # graft the timings only onto lines whose text is still the one we aligned — never write
    # back our (possibly stale) snapshot wholesale.
    n = 0
    with lock:
        data = json.loads(bp.read_text(encoding="utf-8"))
        for l in data.get("lines", []):
            ln = by_id.get(l.get("id"))
            if ln is not None and (l.get("text") or "") == ln.text:
                l["words"] = [asdict(w) for w in ln.words]
                l["align_text"] = ln.align_text
                n += 1
        bp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("Auto re-aligned %d line(s) with stale/absent karaoke timings", n)
    return n


def create_app(config: HostConfig) -> FastAPI:
    app = FastAPI(title="Doublage Host")
    app.state.config = config

    def bundle_path() -> Path:
        if config.bundle_dir is None:
            raise HTTPException(400, "no bundle directory configured (start with --bundle <dir>)")
        return config.bundle_dir / BUNDLE_FILENAME

    @app.get("/api/capabilities")
    def capabilities():
        bp = config.bundle_dir / BUNDLE_FILENAME if config.bundle_dir else None
        has_bundle = bool(bp and bp.exists())
        return {
            "is_host_backend": True,
            "can_preprocess": True,
            "relay_url": config.relay_url,
            "has_bundle": has_bundle,
            "bundle_name": config.bundle_dir.name if (has_bundle and config.bundle_dir) else None,
        }

    @app.post("/api/use-bundle")
    async def use_bundle(request: Request):
        body = await request.json()
        d = (body.get("dir") or "").strip()
        if not d:
            raise HTTPException(400, "missing dir")
        p = Path(d).expanduser().resolve()
        if not (p / BUNDLE_FILENAME).exists():
            raise HTTPException(400, f"no {BUNDLE_FILENAME} in {p}")
        config.bundle_dir = p
        _remember_scene(p)
        log.info("Active bundle set to %s", p)
        return {"ok": True, "bundle_name": p.name}

    @app.get("/api/recent-scenes")
    def recent_scenes():
        active = str(config.bundle_dir.resolve()) if config.bundle_dir else None
        registry = _load_registry()
        scenes: dict[str, dict] = {}  # keyed by resolved dir path
        # registry: scenes used before (may live anywhere)
        kept = []
        for e in registry:
            p = Path(e["dir"])
            if not (p / BUNDLE_FILENAME).exists():
                continue  # prune folders that are gone
            kept.append(e)
            scenes[str(p.resolve())] = {"dir": e["dir"], "name": e.get("name") or p.name, "last_used": e.get("last_used", 0)}
        if len(kept) != len(registry):
            _save_registry(kept)
        # auto-discover bundles under ./output_bundles (and legacy ./out_* at the root)
        for child in _discover_bundle_dirs():
            key = str(child.resolve())
            scenes.setdefault(key, {"dir": key, "name": child.name, "last_used": 0})
        out = [{**s, "active": key == active, **_scene_meta(Path(s["dir"]))} for key, s in scenes.items()]
        out.sort(key=lambda s: s["name"].lower())
        out.sort(key=lambda s: s["last_used"], reverse=True)
        return {"scenes": out, "active_dir": active}

    @app.post("/api/forget-scene")
    async def forget_scene(request: Request):
        body = await request.json()
        d = (body.get("dir") or "").strip()
        if not d:
            raise HTTPException(400, "missing dir")
        try:
            key = str(Path(d).expanduser().resolve())
        except Exception:
            key = d
        _save_registry([e for e in _load_registry() if e.get("dir") not in (d, key)])
        return {"ok": True}

    def _listed_scene_dirs() -> set[Path]:
        """Bundle dirs the UI offers (registry + discovered), used to bound deletion."""
        dirs: set[Path] = set()
        for e in _load_registry():
            try:
                p = Path(e["dir"]).resolve()
                if (p / BUNDLE_FILENAME).exists():
                    dirs.add(p)
            except Exception:
                pass
        for child in _discover_bundle_dirs():
            dirs.add(child.resolve())
        return dirs

    @app.post("/api/delete-scene")
    async def delete_scene(request: Request):
        body = await request.json()
        d = (body.get("dir") or "").strip()
        if not d:
            raise HTTPException(400, "missing dir")
        try:
            p = Path(d).expanduser().resolve()
        except Exception:
            raise HTTPException(400, "bad dir")
        if not (p / BUNDLE_FILENAME).exists():
            raise HTTPException(400, "not a bundle directory")
        if p not in _listed_scene_dirs():
            raise HTTPException(403, "refused: not a listed scene")
        if config.bundle_dir and config.bundle_dir.resolve() == p:
            raise HTTPException(409, "scene is the active bundle")
        shutil.rmtree(p, ignore_errors=True)
        _save_registry([e for e in _load_registry()
                        if e.get("dir") and Path(e["dir"]).expanduser().resolve() != p])
        log.info("Deleted scene %s", p)
        return {"ok": True}

    @app.get("/api/bundle")
    def get_bundle():
        bp = bundle_path()
        if not bp.exists():
            raise HTTPException(404, "no local bundle (preprocess one first)")
        with open(bp, "r", encoding="utf-8") as f:
            return JSONResponse(json.load(f))

    @app.put("/api/bundle")
    async def put_bundle(request: Request):
        data = await request.json()
        if not isinstance(data, dict) or "lines" not in data or "characters" not in data:
            raise HTTPException(400, "invalid bundle payload")
        bp = bundle_path()

        def save_and_realign() -> int:
            with _bundle_lock(bp.parent):
                with open(bp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.write("\n")
            log.info("Local bundle saved (%d lines, %d characters)",
                     len(data.get("lines", [])), len(data.get("characters", [])))
            # Automatic karaoke re-alignment of lines whose text has changed (best-effort:
            # it must NEVER make the save fail, even if torch is absent or the audio is missing).
            try:
                return _realign_stale(bp.parent)
            except Exception as e:
                log.warning("auto re-align skipped: %s", e)
                return 0

        realigned = await asyncio.to_thread(save_and_realign)
        return {"ok": True, "realigned": realigned}

    @app.post("/api/host")
    async def host_room(request: Request):
        if config.bundle_dir is None or not (config.bundle_dir / BUNDLE_FILENAME).exists():
            raise HTTPException(400, "no local bundle to host (preprocess one first)")
        _remember_scene(config.bundle_dir)
        body = await request.json()
        name = (body.get("name") or "").strip()
        if not (1 <= len(name) <= 32):
            raise HTTPException(400, "name must be 1-32 chars")
        cfg = body.get("config") or {}

        job_id = secrets.token_urlsafe(6)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        host_jobs[job_id] = {"status": "running", "queue": queue, "result": None, "error": None}

        def emit(line: str) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, line)

        def work() -> dict:
            tmp_zip = Path(config.bundle_dir).parent / f".dist_{config.bundle_dir.name}_{job_id}.zip"
            try:
                emit("Building browser-playable distribution bundle…")
                build_distribution(config.bundle_dir, out_zip=tmp_zip, on_step=emit)

                emit("Creating room on relay…")
                r = requests.post(
                    f"{config.relay_url}/api/rooms",
                    json={"name": name, "config": cfg},
                    timeout=15,
                )
                r.raise_for_status()
                room = r.json()
                code, token = room["code"], room["token"]
                emit(f"Room created: {code}. Uploading bundle…")

                with open(tmp_zip, "rb") as f:
                    up = requests.post(
                        f"{config.relay_url}/api/rooms/{code}/bundle",
                        data=f,
                        headers={"X-Room-Token": token, "Content-Type": "application/zip"},
                        timeout=600,
                    )
                up.raise_for_status()
                emit("Bundle uploaded. Room is ready.")
                return {
                    "code": code,
                    "token": token,
                    "mid": room["mid"],
                    "relay_url": config.relay_url,
                }
            finally:
                tmp_zip.unlink(missing_ok=True)

        async def run():
            try:
                result = await asyncio.to_thread(work)
                host_jobs[job_id]["result"] = result
                host_jobs[job_id]["status"] = "done"
            except requests.RequestException as e:
                host_jobs[job_id]["error"] = f"relay unreachable or rejected: {e}"
                host_jobs[job_id]["status"] = "failed"
            except Exception as e:
                host_jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
                host_jobs[job_id]["status"] = "failed"
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(run())
        return {"job_id": job_id}

    @app.get("/api/host/{job_id}/events")
    async def host_events(job_id: str):
        job = host_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        queue: asyncio.Queue = job["queue"]

        async def gen():
            while True:
                line = await queue.get()
                if line is None:
                    if job["status"] == "done":
                        yield f"event: done\ndata: {json.dumps(job['result'], ensure_ascii=False)}\n\n"
                    else:
                        yield f"event: error\ndata: {json.dumps(job.get('error') or 'failed')}\n\n"
                    break
                yield f"data: {json.dumps(line, ensure_ascii=False)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    @app.post("/api/assemble")
    async def assemble(request: Request):
        if config.bundle_dir is None or not (config.bundle_dir / BUNDLE_FILENAME).exists():
            raise HTTPException(400, "no local bundle")
        body = await request.json()
        code = (body.get("code") or "").strip()
        token = (body.get("token") or "").strip()
        relay = (body.get("relay_url") or config.relay_url).rstrip("/")
        bg_gain = float(body.get("bg_gain_db", -3.0))
        voice_gain = float(body.get("voice_gain_db", 0.0))
        transcode = bool(body.get("transcode_video", False))
        container = "mkv" if str(body.get("container", "mp4")).lower() == "mkv" else "mp4"
        audio_mode = "ows" if str(body.get("audio_mode", "separated")).lower() == "ows" else "separated"
        # Per-character latency correction (ms) from the montage UI — a manual tweak on TOP of each
        # take's own measured latency_ms, in case the auto-calibration was off. {character_id: ms}.
        latency_adjust = body.get("latency_adjust") or {}
        if not isinstance(latency_adjust, dict):
            latency_adjust = {}
        # Montage edits accumulated locally in the timeline (drag/crop/gain), flushed only at Render
        # so each action stays instant/offline. List of {rel, offset_ms?, gain_db?, crop_*}. We apply
        # them here as overrides on top of the relay's take metadata, keyed by rel.
        edits_by_rel: dict[str, dict] = {}
        for e in (body.get("edits") or []):
            if isinstance(e, dict) and e.get("rel"):
                edits_by_rel[str(e["rel"])] = e
        # Split pieces: extra virtual takes sharing a source file (rel) but with their own crop/offset.
        splits = [s for s in (body.get("splits") or []) if isinstance(s, dict) and s.get("rel")]
        if not code or not token:
            raise HTTPException(400, "missing code/token")

        def work():
            headers = {"X-Room-Token": token}
            r = requests.get(f"{relay}/api/rooms/{code}/takes", headers=headers, timeout=30)
            r.raise_for_status()
            takes = [t for t in r.json().get("takes", []) if t.get("active")]
            if not takes:
                raise RuntimeError("no active takes to assemble")
            with tempfile.TemporaryDirectory(prefix="dub_collect_") as tmp:
                tmp = Path(tmp)
                tdir = tmp / "takes"
                tdir.mkdir()
                manifest = {"user": "room", "bundle_slug": config.bundle_dir.name, "takes": []}
                downloaded: dict[str, str] = {}  # rel -> local file_name (so splits reuse the file)
                for t in takes:
                    safe = t["rel"].replace("/", "__")
                    with requests.get(f"{relay}/api/rooms/{code}/takes/file", params={"rel": t["rel"]},
                                      headers=headers, timeout=300, stream=True) as fr:
                        fr.raise_for_status()
                        with open(tdir / safe, "wb") as fout:
                            for chunk in fr.iter_content(chunk_size=1 << 16):
                                fout.write(chunk)
                    downloaded[t.get("rel")] = safe
                    # Apply the host's local montage edits (sent only at Render) as overrides.
                    ov = edits_by_rel.get(t.get("rel"), {})
                    def _pick(field, default):
                        return ov[field] if field in ov else t.get(field, default)
                    cid = t.get("character_id")
                    # Total latency to compensate at merge = the take's own measured latency
                    # (per-player calibration, applied by default) + the host's per-character tweak.
                    try:
                        adj = int(latency_adjust.get(cid, 0) or 0) if cid else 0
                    except (TypeError, ValueError):
                        adj = 0
                    lat = int(t.get("latency_ms", 0) or 0) + adj
                    ce_val = _pick("crop_end_ms", None)
                    manifest["takes"].append({
                        "line_id": t.get("line_id"),
                        "character_id": cid,
                        "take_n": t.get("take_n", 1),
                        "offset_ms": int(_pick("offset_ms", 0) or 0),
                        "gain_db": float(_pick("gain_db", 0.0) or 0.0),
                        "latency_ms": lat,
                        "crop_start_ms": int(_pick("crop_start_ms", 0) or 0),
                        "crop_end_ms": (int(ce_val) if ce_val not in (None, "") else None),
                        "file_name": f"takes/{safe}",
                    })
                # Split pieces → additional manifest takes pointing to the SAME downloaded file.
                for s in splits:
                    safe = downloaded.get(s.get("rel"))
                    if not safe:
                        continue  # the source file wasn't among the active takes — skip
                    cid = s.get("character_id")
                    try:
                        adj = int(latency_adjust.get(cid, 0) or 0) if cid else 0
                    except (TypeError, ValueError):
                        adj = 0
                    ce_val = s.get("crop_end_ms")
                    manifest["takes"].append({
                        "line_id": s.get("line_id"),
                        "character_id": cid,
                        "take_n": 1,
                        "offset_ms": int(s.get("offset_ms", 0) or 0),
                        "gain_db": float(s.get("gain_db", 0.0) or 0.0),
                        "latency_ms": int(s.get("latency_ms", 0) or 0) + adj,
                        "crop_start_ms": int(s.get("crop_start_ms", 0) or 0),
                        "crop_end_ms": (int(ce_val) if ce_val not in (None, "") else None),
                        "file_name": f"takes/{safe}",
                    })
                (tmp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
                sub = tmp / "submission.zip"
                with zipfile.ZipFile(sub, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(tmp / "manifest.json", "manifest.json")
                    for f in tdir.iterdir():
                        zf.write(f, f"takes/{f.name}")
                exports = config.bundle_dir / "exports"
                name = next_export_name(exports, ext=container)
                out = exports / name
                merge_archives(config.bundle_dir, [sub], out, bg_gain_db=bg_gain, voice_gain_db=voice_gain, transcode_video=transcode, audio_mode=audio_mode)
                return {"name": name, "rel": f"exports/{name}", "bytes": out.stat().st_size, "takes": len(takes)}

        try:
            return await asyncio.to_thread(work)
        except requests.RequestException as e:
            raise HTTPException(502, f"relay unreachable: {e}")
        except (RuntimeError, FileNotFoundError) as e:
            raise HTTPException(500, str(e))

    @app.get("/api/download")
    def download(name: str):
        if config.bundle_dir is None:
            raise HTTPException(400, "no bundle")
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(400, "bad name")
        p = config.bundle_dir / "exports" / name
        if not p.exists():
            raise HTTPException(404, "not found")
        return FileResponse(p, filename=name, media_type="application/octet-stream")

    @app.get("/assets/{path:path}")
    def serve_asset(path: str):
        if config.bundle_dir is None:
            raise HTTPException(404, "no active bundle")
        base = config.bundle_dir.resolve()
        target = (base / path).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise HTTPException(403, "forbidden")
        if not target.exists() or not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    app.mount("/pp", preprocess_app.create_app())
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def root():
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__PUBLIC_URL__", os.environ.get("PUBLIC_URL", "").rstrip("/")))

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dubbing.host", description="Doublage host app (local)")
    p.add_argument("--bundle", type=Path, default=None, help="Local preprocessed bundle directory")
    p.add_argument("--relay", default=os.environ.get("RELAY_URL") or DEFAULT_RELAY, help="Relay base URL (env: RELAY_URL)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p


def main(argv=None) -> int:
    import uvicorn

    _load_dotenv(Path.cwd() / ".env")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    bundle_dir = args.bundle.resolve() if args.bundle else None
    if bundle_dir:
        _remember_scene(bundle_dir)
    config = HostConfig(bundle_dir=bundle_dir, relay_url=args.relay.rstrip("/"))
    app = create_app(config)
    log.info("Host app: bundle=%s relay=%s", bundle_dir or "(none)", config.relay_url)
    log.info("Serving at http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
