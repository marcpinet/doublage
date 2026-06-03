from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import shutil
import time
import zipfile
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware


log = logging.getLogger("relay")

# ES modules (the voice-FX worklet/worker under web/fx/) are refused by browsers unless served
# with a JavaScript MIME type. Python's mimetypes can be broken by the OS (the Windows registry
# maps .mjs to text/plain), so pin the mapping explicitly.
import mimetypes  # noqa: E402
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
CODE_LEN = 5
BUNDLE_FILENAME = "project.json"
VALID_MODES = ("live", "pro")

# When a non-host member disconnects, keep them (and their claimed character) for this long so a
# refresh or brief blip doesn't drop them. If they haven't reconnected by then, the ghost member is
# removed and its character freed so others can claim it. The host is exempt (host leaving = the
# room is purged by the empty-room grace / idle TTL instead).
MEMBER_DISCONNECT_GRACE_SEC = 20


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 21826
    storage_dir: Path = ROOT / "relay_storage"
    web_dir: Path = REPO_ROOT / "dubbing" / "web"
    room_ttl_min: int = 240
    empty_room_grace_min: int = 5
    max_room_mb: int = 800
    max_take_mb: int = 64
    max_rooms: int = 50
    max_members_per_room: int = 24
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv(REPO_ROOT / ".env")
        _load_dotenv(ROOT / ".env")
        origins_raw = os.environ.get("ALLOWED_ORIGINS", "*").strip()
        origins = ["*"] if origins_raw in ("", "*") else [o.strip() for o in origins_raw.split(",") if o.strip()]
        return cls(
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "21826")),
            storage_dir=Path(os.environ.get("STORAGE_DIR", str(ROOT / "relay_storage"))).expanduser(),
            web_dir=Path(os.environ.get("WEB_DIR", str(REPO_ROOT / "dubbing" / "web"))).expanduser(),
            room_ttl_min=int(os.environ.get("ROOM_TTL_MIN", "240")),
            empty_room_grace_min=int(os.environ.get("EMPTY_ROOM_GRACE_MIN", "5")),
            max_room_mb=int(os.environ.get("MAX_ROOM_MB", "800")),
            max_take_mb=int(os.environ.get("MAX_TAKE_MB", "64")),
            max_rooms=int(os.environ.get("MAX_ROOMS", "50")),
            max_members_per_room=int(os.environ.get("MAX_MEMBERS_PER_ROOM", "24")),
            allowed_origins=origins,
        )


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class Member:
    token: str
    mid: str
    name: str
    is_host: bool
    character_id: Optional[str] = None
    connected: bool = False
    done: bool = False
    joined_at: float = field(default_factory=time.time)


@dataclass
class Room:
    code: str
    host_token: str
    config: dict
    phase: str = "lobby"
    bundle_ready: bool = False
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    members: dict[str, Member] = field(default_factory=dict)
    sockets: dict[str, WebSocket] = field(default_factory=dict)
    transport: dict = field(default_factory=dict)
    ever_connected: bool = False
    empty_since: Optional[float] = None

    def touch(self) -> None:
        self.last_active = time.time()


class Relay:
    def __init__(self, config: Config):
        self.config = config
        self.rooms: dict[str, Room] = {}
        self.public_dir = config.storage_dir / "public"
        self.takes_dir = config.storage_dir / "takes"
        self.public_dir.mkdir(parents=True, exist_ok=True)
        self.takes_dir.mkdir(parents=True, exist_ok=True)


    def _new_code(self) -> str:
        for _ in range(1000):
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))
            if code not in self.rooms:
                return code
        raise RuntimeError("could not allocate a unique room code")

    def create_room(self, host_name: str, config: Optional[dict]) -> tuple[Room, Member]:
        if len(self.rooms) >= self.config.max_rooms:
            raise HTTPException(503, "too many active rooms, try again later")
        code = self._new_code()
        host_token = secrets.token_urlsafe(24)
        room = Room(
            code=code,
            host_token=host_token,
            config=normalize_config(config),
        )
        host = Member(
            token=host_token,
            mid=secrets.token_hex(4),
            name=host_name,
            is_host=True,
        )
        room.members[host_token] = host
        self.rooms[code] = room
        log.info("Room %s created by host %r", code, host_name)
        return room, host

    def get_room(self, code: str) -> Room:
        room = self.rooms.get((code or "").upper())
        if room is None:
            raise HTTPException(404, "room not found")
        return room

    def member_of(self, room: Room, token: str) -> Member:
        m = room.members.get(token or "")
        if m is None:
            raise HTTPException(403, "invalid room token")
        return m

    def require_host(self, room: Room, token: str) -> Member:
        m = self.member_of(room, token)
        if not m.is_host:
            raise HTTPException(403, "host only")
        return m

    def join_room(self, code: str, name: str) -> tuple[Room, Member]:
        room = self.get_room(code)
        if len(room.members) >= self.config.max_members_per_room:
            raise HTTPException(403, "room is full")
        token = secrets.token_urlsafe(24)
        member = Member(token=token, mid=secrets.token_hex(4), name=name, is_host=False)
        room.members[token] = member
        room.touch()
        log.info("Room %s joined by %r", room.code, name)
        return room, member

    def close_room(self, room: Room) -> None:
        self.rooms.pop(room.code, None)
        shutil.rmtree(self.public_dir / room.code, ignore_errors=True)
        shutil.rmtree(self.takes_dir / room.code, ignore_errors=True)
        log.info("Room %s closed", room.code)

    def cleanup_rooms(self) -> list[str]:
        """Purge rooms that everyone has left (after a short grace) or that hit the idle TTL.

        Closing a room frees its space on disk (bundle under public/<code> and all
        recordings under takes/<code>) and drops it from memory. The "everyone left"
        path only fires once a room has actually been occupied (ever_connected), so a
        freshly created room isn't purged before the host connects; the grace absorbs
        brief disconnects/refreshes (e.g. the host reloading during montage)."""
        now = time.time()
        ttl = self.config.room_ttl_min * 60
        grace = self.config.empty_room_grace_min * 60
        closed: list[str] = []
        for room in list(self.rooms.values()):
            connected = any(m.connected for m in room.members.values())
            if connected:
                room.empty_since = None
            elif room.empty_since is None:
                room.empty_since = now
            empty_purge = room.ever_connected and room.empty_since is not None and (now - room.empty_since) >= grace
            idle_purge = (now - room.last_active) >= ttl
            if empty_purge or idle_purge:
                self.close_room(room)
                closed.append(room.code)
                log.info("Room %s purged (%s)", room.code, "everyone left" if empty_purge else "idle TTL")
        return closed


    async def broadcast_state(self, room: Room) -> None:
        msg = json.dumps({"type": "state", "room": room_state(room)}, ensure_ascii=False)
        dead: list[str] = []
        for token, ws in list(room.sockets.items()):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(token)
        for token in dead:
            room.sockets.pop(token, None)
            m = room.members.get(token)
            if m:
                m.connected = False

    async def send_to(self, room: Room, token: str, payload: dict) -> None:
        ws = room.sockets.get(token)
        if ws is None:
            return
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            room.sockets.pop(token, None)

    async def broadcast_event(self, room: Room, payload: dict, exclude: Optional[str] = None) -> None:
        msg = json.dumps(payload, ensure_ascii=False)
        for token, ws in list(room.sockets.items()):
            if token == exclude:
                continue
            try:
                await ws.send_text(msg)
            except Exception:
                room.sockets.pop(token, None)


def normalize_config(config: Optional[dict]) -> dict:
    config = config or {}
    mode = config.get("mode", "live")
    if mode not in VALID_MODES:
        mode = "live"
    if mode == "live":
        take_limit = 1
    else:
        take_limit = config.get("take_limit", 3)
        if take_limit in ("", None, "unlimited", 0):
            take_limit = None
        else:
            try:
                take_limit = max(1, int(take_limit))
            except (TypeError, ValueError):
                take_limit = 3
    return {"mode": mode, "take_limit": take_limit}


def room_state(room: Room) -> dict:
    return {
        "code": room.code,
        "phase": room.phase,
        "config": room.config,
        "bundle_ready": room.bundle_ready,
        "members": [
            {
                "mid": m.mid,
                "name": m.name,
                "is_host": m.is_host,
                "character_id": m.character_id,
                "connected": m.connected,
                "done": m.done,
            }
            for m in room.members.values()
        ],
        "transport": room.transport or None,
    }


def _safe_extract_zip(zip_path: Path, target_dir: Path, max_bytes: Optional[int] = None) -> None:
    target_resolved = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        # Zip bomb guard: the upload is capped on its COMPRESSED size, so check the declared
        # uncompressed total before extracting anything. The media is already compressed
        # (H.264/AAC), a legit bundle barely expands — 2x headroom is generous.
        if max_bytes is not None:
            total = sum(i.file_size for i in zf.infolist())
            if total > max_bytes * 2:
                raise HTTPException(413, "bundle zip expands too large")
        for member in zf.namelist():
            extracted = (target_dir / member).resolve()
            try:
                extracted.relative_to(target_resolved)
            except ValueError:
                raise HTTPException(400, f"unsafe path in zip: {member}")
        zf.extractall(target_dir)
    entries = [p for p in target_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and not (target_dir / BUNDLE_FILENAME).exists():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(target_dir / item.name))
        inner.rmdir()


def _take_target(base: Path, rel: str) -> Path:
    base_r = base.resolve()
    target = (base / rel).resolve()
    target.relative_to(base_r)
    return target


def _set_take_active(base: Path, rel: str) -> bool:
    target = _take_target(base, rel)
    meta_path = Path(str(target) + ".json")
    if not meta_path.exists():
        return False
    for sib in target.parent.glob("*.json"):
        try:
            m = json.loads(sib.read_text(encoding="utf-8"))
        except Exception:
            continue
        m["active"] = (sib.name == meta_path.name)
        sib.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return True


def _update_take_offset(base: Path, rel: str, offset_ms: int) -> bool:
    target = _take_target(base, rel)
    meta_path = Path(str(target) + ".json")
    if not meta_path.exists():
        return False
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    m["offset_ms"] = int(offset_ms)
    meta_path.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return True


def _update_take_crop(base: Path, rel: str, crop_start_ms: int,
                      crop_end_ms: Optional[int], offset_ms: Optional[int]) -> bool:
    target = _take_target(base, rel)
    meta_path = Path(str(target) + ".json")
    if not meta_path.exists():
        return False
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    cs = max(0, int(crop_start_ms))
    m["crop_start_ms"] = cs
    if crop_end_ms is None:
        m["crop_end_ms"] = None
    else:
        ce = int(crop_end_ms)
        if ce <= cs:
            ce = cs + 50  # keep at least a sliver
        m["crop_end_ms"] = ce
    if offset_ms is not None:
        m["offset_ms"] = int(offset_ms)
    meta_path.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return True


def _update_take_gain(base: Path, rel: str, gain_db: float) -> bool:
    target = _take_target(base, rel)
    meta_path = Path(str(target) + ".json")
    if not meta_path.exists():
        return False
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    m["gain_db"] = float(gain_db)
    meta_path.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return True


def _update_member_latency(base: Path, mid: str, latency_ms: int) -> int:
    """Set latency_ms on ALL of a member's takes (their takes live under base/<mid>/). Lets a player
    calibrate AFTER recording and have it applied retroactively. Returns the count updated."""
    member_dir = (base / mid)
    if not member_dir.exists():
        return 0
    lat = max(0, min(1000, int(latency_ms)))
    n = 0
    for meta_file in member_dir.rglob("*.json"):
        try:
            m = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        m["latency_ms"] = lat
        meta_file.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        n += 1
    return n


def create_app(config: Config) -> FastAPI:
    relay = Relay(config)

    # Per-IP sliding-window rate limit on the unauthenticated endpoints (room create/join):
    # without it, a scanner can enumerate the 5-char code space or mass-create rooms.
    rate_buckets: dict[str, deque] = {}

    def check_rate(request: Request, key: str, limit: int, window_sec: float = 60.0) -> None:
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "?")
        bucket = rate_buckets.setdefault(f"{key}:{ip}", deque())
        now = time.monotonic()
        while bucket and now - bucket[0] > window_sec:
            bucket.popleft()
        if len(bucket) >= limit:
            raise HTTPException(429, "too many requests, slow down")
        bucket.append(now)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def janitor():
            while True:
                try:
                    # rmtree on a slow SD card can block for a while: keep it off the event loop.
                    await asyncio.to_thread(relay.cleanup_rooms)
                    cutoff = time.monotonic() - 300
                    for k in [k for k, dq in rate_buckets.items() if not dq or dq[-1] < cutoff]:
                        rate_buckets.pop(k, None)
                except Exception as e:
                    log.warning("janitor error: %s", e)
                await asyncio.sleep(60)

        task = asyncio.create_task(janitor())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Doublage Relay", lifespan=lifespan)
    app.state.relay = relay

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    @app.get("/api/health")
    def health():
        return {"ok": True, "rooms": len(relay.rooms)}


    @app.post("/api/rooms")
    async def create_room(request: Request):
        check_rate(request, "create", limit=5)
        body = await request.json()
        name = (body.get("name") or "").strip()
        if not (1 <= len(name) <= 32):
            raise HTTPException(400, "name must be 1-32 chars")
        room, host = relay.create_room(name, body.get("config"))
        return {
            "code": room.code,
            "token": host.token,
            "mid": host.mid,
            "is_host": True,
            "room": room_state(room),
        }

    @app.post("/api/rooms/{code}/join")
    async def join_room(code: str, request: Request):
        check_rate(request, "join", limit=15)
        body = await request.json()
        name = (body.get("name") or "").strip()
        if not (1 <= len(name) <= 32):
            raise HTTPException(400, "name must be 1-32 chars")
        room, member = relay.join_room(code, name)
        await relay.broadcast_state(room)
        return {
            "code": room.code,
            "token": member.token,
            "mid": member.mid,
            "is_host": False,
            "room": room_state(room),
        }

    @app.get("/api/rooms/{code}")
    def get_room(code: str):
        room = relay.get_room(code)
        return {"room": room_state(room)}

    @app.post("/api/rooms/{code}/close")
    async def close_room(code: str, request: Request):
        room = relay.get_room(code)
        relay.require_host(room, request.headers.get("x-room-token", ""))
        await relay.broadcast_event(room, {"type": "closed"})
        await asyncio.to_thread(relay.close_room, room)
        return {"ok": True}


    @app.post("/api/rooms/{code}/bundle")
    async def upload_bundle(code: str, request: Request):
        room = relay.get_room(code)
        relay.require_host(room, request.headers.get("x-room-token", ""))

        target = relay.public_dir / room.code
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        tmp_path = relay.public_dir / f".upload_{room.code}_{secrets.token_hex(4)}.zip"

        max_bytes = config.max_room_mb * 1024 * 1024
        size = 0
        try:
            with open(tmp_path, "wb") as f:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        raise HTTPException(413, f"bundle exceeds {config.max_room_mb} MB")
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty upload")
            await asyncio.to_thread(_safe_extract_zip, tmp_path, target, max_bytes)
            if not (target / BUNDLE_FILENAME).exists():
                raise HTTPException(400, f"bundle zip missing {BUNDLE_FILENAME}")
        except HTTPException:
            shutil.rmtree(target, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(target, ignore_errors=True)
            raise HTTPException(500, f"bundle upload failed: {e}")
        finally:
            tmp_path.unlink(missing_ok=True)

        room.bundle_ready = True
        room.touch()
        await relay.broadcast_state(room)
        log.info("Room %s bundle uploaded (%.1f MB)", room.code, size / 1024 / 1024)
        return {"ok": True, "bytes": size}


    @app.post("/api/rooms/{code}/takes")
    async def upload_take(code: str, request: Request):
        room = relay.get_room(code)
        member = relay.member_of(room, request.headers.get("x-room-token", ""))
        # Cap BEFORE parsing the form: request.form() buffers the whole multipart body, so
        # an oversized upload must be rejected on its Content-Length (browsers always send it;
        # the chunked-read cap below catches anything that lies or omits it).
        max_take_bytes = config.max_take_mb * 1024 * 1024
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        if declared > max_take_bytes + 1024 * 1024:
            raise HTTPException(413, f"take exceeds {config.max_take_mb} MB")
        form = await request.form()
        upload = form.get("recording")
        if upload is None:
            raise HTTPException(400, "missing 'recording'")
        ext = "".join(c for c in str(form.get("ext", "webm")) if c.isalnum())[:6] or "webm"
        line_id = (str(form.get("line_id", "")).strip() or None)
        # line_id becomes a directory name (line_<id>); sanitize to safe chars so it can't escape
        # the takes dir (path traversal) — covers both real subtitle ids and synthetic free-take ids.
        if line_id is not None:
            line_id = "".join(c for c in line_id if c.isalnum() or c in "-_")[:48] or None
        character_id = (str(form.get("character_id", "")).strip() or None)
        if character_id is not None:
            character_id = "".join(c for c in character_id if c.isalnum() or c in "-_")[:48] or None
        try:
            offset_ms = int(form.get("offset_ms", 0))
        except (TypeError, ValueError):
            offset_ms = 0
        try:
            gain_db = max(-30.0, min(12.0, float(form.get("gain_db", 0))))
        except (TypeError, ValueError):
            gain_db = 0.0
        try:
            latency_ms = max(0, min(1000, int(float(form.get("latency_ms", 0)))))
        except (TypeError, ValueError):
            latency_ms = 0
        # Voice FX baked into the take by the player (display-only metadata for the host UI).
        fx = (str(form.get("fx", "")).strip() or None)
        if fx is not None:
            fx = "".join(c for c in fx if c.isalnum() or c in "-_")[:24] or None
        try:
            fx_intensity = max(0.0, min(1.0, float(form.get("fx_intensity", 0.5))))
        except (TypeError, ValueError):
            fx_intensity = 0.5
        kind = "line" if line_id else "session"

        member_dir = relay.takes_dir / room.code / member.mid
        if line_id:
            member_dir = member_dir / f"line_{line_id}"
        member_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(member_dir.glob(f"take_*.{ext}"))
        take_n = len(existing) + 1
        dest = member_dir / f"take_{take_n:03d}.{ext}"
        size = 0
        try:
            with open(dest, "wb") as f:
                while True:
                    chunk = await upload.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_take_bytes:
                        raise HTTPException(413, f"take exceeds {config.max_take_mb} MB")
                    f.write(chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        if size == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "empty upload")

        meta = {
            "mid": member.mid,
            "name": member.name,
            "character_id": character_id,
            "line_id": line_id,
            "kind": kind,
            "take_n": take_n,
            "offset_ms": offset_ms,
            "gain_db": gain_db,
            "latency_ms": latency_ms,
            "crop_start_ms": 0,
            "crop_end_ms": None,
            "fx": fx,
            "fx_intensity": fx_intensity if fx else None,
            "ext": ext,
            "rel": dest.relative_to(relay.takes_dir / room.code).as_posix(),
            "bytes": size,
            "active": True,
            "created_at": now_ms(),
        }
        (dest.with_suffix(dest.suffix + ".json")).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        _set_take_active(relay.takes_dir / room.code, meta["rel"])
        room.touch()
        await relay.broadcast_event(room, {"type": "take_uploaded", "mid": member.mid, "line_id": line_id})
        log.info("Room %s take by %s (%s, %d bytes)", room.code, member.name, kind, size)
        return {"ok": True, **meta}

    @app.get("/api/rooms/{code}/takes")
    def list_takes(code: str, request: Request):
        room = relay.get_room(code)
        relay.require_host(room, request.headers.get("x-room-token", ""))
        base = relay.takes_dir / room.code
        out = []
        if base.exists():
            for meta_file in sorted(base.rglob("*.json")):
                try:
                    out.append(json.loads(meta_file.read_text(encoding="utf-8")))
                except Exception:
                    continue
        return {"takes": out}

    @app.get("/api/rooms/{code}/takes/file")
    def download_take(code: str, rel: str, request: Request):
        room = relay.get_room(code)
        # Media elements (<audio>/<video>) can't set headers, so accept the host token
        # from the query string as well (same pattern as the WebSocket endpoint).
        token = request.headers.get("x-room-token", "") or request.query_params.get("token", "")
        relay.require_host(room, token)
        base = (relay.takes_dir / room.code).resolve()
        target = (base / rel).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise HTTPException(403, "forbidden")
        if not target.exists() or not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    def _take_owner_ok(rel: str, member: Member) -> bool:
        return member.is_host or (rel.split("/", 1)[0] == member.mid)

    @app.post("/api/rooms/{code}/takes/select")
    async def select_take(code: str, request: Request):
        room = relay.get_room(code)
        member = relay.member_of(room, request.headers.get("x-room-token", ""))
        body = await request.json()
        rel = (body.get("rel") or "").strip()
        if not rel or not _take_owner_ok(rel, member):
            raise HTTPException(403, "forbidden")
        try:
            ok = _set_take_active(relay.takes_dir / room.code, rel)
        except ValueError:
            raise HTTPException(403, "forbidden")
        if not ok:
            raise HTTPException(404, "take not found")
        room.touch()
        await relay.broadcast_event(room, {"type": "takes_changed", "mid": member.mid})
        return {"ok": True}

    @app.post("/api/rooms/{code}/takes/offset")
    async def offset_take(code: str, request: Request):
        room = relay.get_room(code)
        member = relay.member_of(room, request.headers.get("x-room-token", ""))
        body = await request.json()
        rel = (body.get("rel") or "").strip()
        try:
            offset_ms = int(body.get("offset_ms", 0))
        except (TypeError, ValueError):
            offset_ms = 0
        if not rel or not _take_owner_ok(rel, member):
            raise HTTPException(403, "forbidden")
        try:
            ok = _update_take_offset(relay.takes_dir / room.code, rel, offset_ms)
        except ValueError:
            raise HTTPException(403, "forbidden")
        if not ok:
            raise HTTPException(404, "take not found")
        room.touch()
        return {"ok": True}

    @app.post("/api/rooms/{code}/takes/gain")
    async def gain_take(code: str, request: Request):
        room = relay.get_room(code)
        member = relay.member_of(room, request.headers.get("x-room-token", ""))
        body = await request.json()
        rel = (body.get("rel") or "").strip()
        try:
            gain_db = max(-30.0, min(12.0, float(body.get("gain_db", 0.0))))
        except (TypeError, ValueError):
            gain_db = 0.0
        if not rel or not _take_owner_ok(rel, member):
            raise HTTPException(403, "forbidden")
        try:
            ok = _update_take_gain(relay.takes_dir / room.code, rel, gain_db)
        except ValueError:
            raise HTTPException(403, "forbidden")
        if not ok:
            raise HTTPException(404, "take not found")
        room.touch()
        await relay.broadcast_event(room, {"type": "takes_changed", "mid": member.mid})
        return {"ok": True}

    @app.post("/api/rooms/{code}/takes/latency")
    async def set_latency(code: str, request: Request):
        # A member sets their measured capture latency on ALL their takes (retroactive calibration).
        room = relay.get_room(code)
        member = relay.member_of(room, request.headers.get("x-room-token", ""))
        body = await request.json()
        try:
            latency_ms = max(0, min(1000, int(float(body.get("latency_ms", 0)))))
        except (TypeError, ValueError):
            latency_ms = 0
        n = _update_member_latency(relay.takes_dir / room.code, member.mid, latency_ms)
        room.touch()
        if n:
            await relay.broadcast_event(room, {"type": "takes_changed", "mid": member.mid})
        return {"ok": True, "updated": n}

    @app.post("/api/rooms/{code}/takes/crop")
    async def crop_take(code: str, request: Request):
        room = relay.get_room(code)
        member = relay.member_of(room, request.headers.get("x-room-token", ""))
        body = await request.json()
        rel = (body.get("rel") or "").strip()

        def _opt_int(key):
            v = body.get(key, None)
            if v is None or v == "":
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        crop_start_ms = _opt_int("crop_start_ms") or 0
        crop_end_ms = _opt_int("crop_end_ms")     # None = up to the end of the recording
        offset_ms = _opt_int("offset_ms")         # left-edge trims shift the offset too
        if not rel or not _take_owner_ok(rel, member):
            raise HTTPException(403, "forbidden")
        try:
            ok = _update_take_crop(relay.takes_dir / room.code, rel, crop_start_ms, crop_end_ms, offset_ms)
        except ValueError:
            raise HTTPException(403, "forbidden")
        if not ok:
            raise HTTPException(404, "take not found")
        room.touch()
        await relay.broadcast_event(room, {"type": "takes_changed", "mid": member.mid})
        return {"ok": True}

    @app.post("/api/rooms/{code}/takes/delete")
    async def delete_take(code: str, request: Request):
        room = relay.get_room(code)
        member = relay.member_of(room, request.headers.get("x-room-token", ""))
        body = await request.json()
        rel = (body.get("rel") or "").strip()
        if not rel or not _take_owner_ok(rel, member):
            raise HTTPException(403, "forbidden")
        try:
            target = _take_target(relay.takes_dir / room.code, rel)
        except ValueError:
            raise HTTPException(403, "forbidden")
        meta_path = Path(str(target) + ".json")
        if not target.exists() and not meta_path.exists():
            raise HTTPException(404, "take not found")
        target.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        room.touch()
        await relay.broadcast_event(room, {"type": "takes_changed", "mid": member.mid})
        log.info("Room %s take deleted by %s (%s)", room.code, member.name, rel)
        return {"ok": True}


    @app.websocket("/ws/{code}")
    async def ws_room(ws: WebSocket, code: str, token: str = ""):
        room = relay.rooms.get((code or "").upper())
        if room is None:
            await ws.close(code=4404)
            return
        member = room.members.get(token)
        if member is None:
            await ws.close(code=4403)
            return

        await ws.accept()
        old = room.sockets.get(token)
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass
        room.sockets[token] = ws
        member.connected = True
        room.ever_connected = True
        room.empty_since = None
        room.touch()
        await ws.send_text(json.dumps({"type": "hello", "mid": member.mid, "is_host": member.is_host}))
        await relay.broadcast_state(room)

        try:
            while True:
                raw = await ws.receive_text()
                room.touch()
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                await handle_ws_message(relay, room, member, msg)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.debug("ws error room=%s: %s", room.code, e)
        finally:
            if room.sockets.get(token) is ws:
                room.sockets.pop(token, None)
                member.connected = False
                if not any(m.connected for m in room.members.values()):
                    room.empty_since = time.time()  # start the purge grace once the room is empty
                if room.code in relay.rooms:
                    await relay.broadcast_state(room)
                # Non-host: after a grace period, if they haven't reconnected (closed the tab for
                # good, vs a quick refresh), drop the ghost member so its character is freed and it
                # no longer blocks the lobby or the "everyone is done" check.
                if not member.is_host:
                    async def _prune_if_gone(tok=token, mid=member.mid):
                        try:
                            await asyncio.sleep(MEMBER_DISCONNECT_GRACE_SEC)
                            if room.code not in relay.rooms:
                                return
                            m = room.members.get(tok)
                            if m is None or m.connected:
                                return  # reconnected (or already gone) → keep them
                            room.members.pop(tok, None)
                            room.sockets.pop(tok, None)
                            log.info("Room %s: pruned ghost member %s (left, character freed)", room.code, mid)
                            await relay.broadcast_state(room)
                        except Exception as e:
                            log.debug("prune error room=%s: %s", room.code, e)
                    asyncio.create_task(_prune_if_gone())


    if config.web_dir.exists():
        app.mount("/static", StaticFiles(directory=config.web_dir), name="static")
    relay.public_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=relay.public_dir), name="assets")

    # Read + transform once at startup (it never changes while running): re-reading the
    # file on every GET / is a synchronous disk hit inside the event loop.
    index_file = config.web_dir / "index.html"
    index_html = (index_file.read_text(encoding="utf-8")
                  .replace("__PUBLIC_URL__", os.environ.get("PUBLIC_URL", "").rstrip("/"))
                  if index_file.exists() else None)

    @app.get("/")
    def root():
        if index_html is None:
            return JSONResponse({"error": "web UI not found", "web_dir": str(config.web_dir)}, status_code=500)
        return HTMLResponse(index_html)

    return app


async def handle_ws_message(relay: Relay, room: Room, member: Member, msg: dict) -> None:
    mtype = msg.get("type")

    if mtype == "ping":
        await relay.send_to(room, member.token, {"type": "pong", "t0": msg.get("t0"), "t1": now_ms()})
        return

    if mtype == "claim":
        character_id = msg.get("character_id")
        if character_id:
            for other in room.members.values():
                if other.token != member.token and other.character_id == character_id:
                    await relay.send_to(room, member.token, {
                        "type": "error", "msg": f"character already taken by {other.name}",
                    })
                    return
        member.character_id = character_id
        await relay.broadcast_state(room)
        return

    if mtype == "release":
        member.character_id = None
        await relay.broadcast_state(room)
        return

    if mtype == "rename":
        name = (msg.get("name") or "").strip()
        if 1 <= len(name) <= 32:
            member.name = name
            await relay.broadcast_state(room)
        return

    if mtype == "set_done":
        member.done = bool(msg.get("done"))
        await relay.broadcast_state(room)
        return

    if not member.is_host:
        return

    if mtype == "assign_all":
        # Host-only: randomly assign one character to every member (host included). The relay is
        # bundle-agnostic, so the host sends the character id list. Players can still change after
        # (normal claim/release is untouched). Extra members beyond the char count get none.
        char_ids = [str(c) for c in (msg.get("character_ids") or []) if str(c)]
        members = list(room.members.values())
        if char_ids and members:
            pool = list(char_ids)
            secrets.SystemRandom().shuffle(pool)
            for i, m in enumerate(members):
                m.character_id = pool[i] if i < len(pool) else None
            await relay.broadcast_state(room)
        return

    if mtype == "set_config":
        room.config = normalize_config(msg.get("config"))
        await relay.broadcast_state(room)
    elif mtype == "set_phase":
        phase = msg.get("phase")
        if phase in ("lobby", "preparing", "playing", "collecting", "done"):
            room.phase = phase
            await relay.broadcast_state(room)
    elif mtype == "transport":
        room.transport = msg.get("transport") or {}
        await relay.broadcast_event(room, {"type": "transport", "transport": room.transport})
    elif mtype == "kick":
        target_mid = msg.get("mid")
        victim = next((m for m in room.members.values() if m.mid == target_mid and not m.is_host), None)
        if victim:
            ws = room.sockets.pop(victim.token, None)
            room.members.pop(victim.token, None)
            if ws is not None:
                try:
                    await ws.close(code=4001)
                except Exception:
                    pass
            await relay.broadcast_state(room)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="relay", description="Doublage relay server")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--storage", type=Path, default=None)
    return p


def main(argv=None) -> int:
    import uvicorn

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    config = Config.from_env()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.storage:
        config.storage_dir = args.storage

    app = create_app(config)
    log.info("Relay storage=%s web=%s ttl=%dmin", config.storage_dir, config.web_dir, config.room_ttl_min)
    log.info("Serving on http://%s:%d", config.host, config.port)
    uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
