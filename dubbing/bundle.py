from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import Bundle


BUNDLE_FILENAME = "project.json"


def write_bundle(bundle: Bundle, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / BUNDLE_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(bundle), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def read_bundle(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
