"""Dev smoke test for the relay hardening (run: python scripts/smoke_relay.py)."""
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from relay.server import Config, create_app

config = Config(storage_dir=Path(tempfile.mkdtemp()), max_take_mb=1, max_rooms=3)
client = TestClient(create_app(config))

# Room creation + join
r = client.post("/api/rooms", json={"name": "host"})
assert r.status_code == 200, r.text
room = r.json()
code, host_token = room["code"], room["token"]
r = client.post(f"/api/rooms/{code}/join", json={"name": "player"})
assert r.status_code == 200, r.text
player_token = r.json()["token"]
print("create/join: OK")

# Room cap (max_rooms=3, one already exists)
codes = [client.post("/api/rooms", json={"name": f"h{i}"}) for i in range(2)]
assert all(c.status_code == 200 for c in codes), [c.status_code for c in codes]
r = client.post("/api/rooms", json={"name": "overflow"})
assert r.status_code == 503, f"expected 503, got {r.status_code}"
print("max rooms cap: OK (503)")

# Rate limit on create (limit 5/min; 4 used above -> the 5th passes the limiter, the 6th gets 429)
client.post("/api/rooms", json={"name": "spam1"})
r = client.post("/api/rooms", json={"name": "spam2"})
assert r.status_code == 429, f"expected 429, got {r.status_code}"
print("create rate limit: OK (429)")

# Take upload within the limit (with voice-FX metadata)
small = io.BytesIO(b"x" * 1024)
r = client.post(
    f"/api/rooms/{code}/takes",
    headers={"x-room-token": player_token},
    files={"recording": ("take.webm", small, "audio/webm")},
    data={"ext": "webm", "line_id": "l1", "fx": "demon", "fx_intensity": "0.7"},
)
assert r.status_code == 200, r.text
assert r.json()["bytes"] == 1024
assert r.json()["fx"] == "demon" and r.json()["fx_intensity"] == 0.7, r.json()
print("small take upload (+fx meta): OK")

# Oversized take (max_take_mb=1) -> 413, nothing left on disk
big = io.BytesIO(b"x" * (2 * 1024 * 1024))
r = client.post(
    f"/api/rooms/{code}/takes",
    headers={"x-room-token": player_token},
    files={"recording": ("take.webm", big, "audio/webm")},
    data={"ext": "webm", "line_id": "l2"},
)
assert r.status_code == 413, f"expected 413, got {r.status_code}: {r.text}"
leftovers = list((config.storage_dir / "takes").rglob("*l2*"))
assert not any(p.is_file() for p in leftovers), leftovers
print("oversized take: OK (413, no partial file)")

# A player cannot touch another member's takes (rel prefix != their mid)
r2 = client.post(f"/api/rooms/{code}/join", json={"name": "p2"})
other_token = r2.json()["token"]
victim_rel = client.get(f"/api/rooms/{code}/takes", headers={"x-room-token": host_token}).json()["takes"][0]["rel"]
r = client.post(f"/api/rooms/{code}/takes/select", headers={"x-room-token": other_token}, json={"rel": victim_rel})
assert r.status_code == 403, f"expected 403, got {r.status_code}"
print("cross-member take access: OK (403)")

print("\nAll relay smoke tests passed.")
