from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


BUNDLE_VERSION = "1.0"


@dataclass
class Source:
    video_path: str
    original_audio_path: str
    duration_sec: float
    language: str


@dataclass
class Assets:
    background_track: Optional[str] = None
    original_vocals: Optional[str] = None


@dataclass
class Character:
    id: str
    name: str
    color: str
    cluster_ids: list[str] = field(default_factory=list)


@dataclass
class Word:
    text: str
    start_sec: float
    end_sec: float


@dataclass
class Line:
    id: str
    start_sec: float
    end_sec: float
    text: str
    character_id: Optional[str] = None
    source_cluster: Optional[str] = None
    confidence: Optional[float] = None
    needs_review: bool = True
    words: list = field(default_factory=list)  # list[Word]; empty if word-level alignment is not available
    align_text: Optional[str] = None  # text against which words was (re)aligned; != text ⇒ alignment must be redone


@dataclass
class Bundle:
    version: str
    source: Source
    assets: Assets
    characters: list[Character] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
