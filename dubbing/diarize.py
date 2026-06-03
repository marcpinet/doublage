from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"

# Process-level caches: the host editor calls diarization repeatedly and
# reloading the pyannote models costs several seconds.
_PIPE_CACHE: dict = {}  # key: (model, device) -> Pipeline
_EMB_CACHE: dict = {}   # key: (embedding_model, device) -> embedding Model


def clear_model_cache() -> None:
    """Clear the diarization model caches and free VRAM.

    The one-shot pipeline evicts the caches between stages to reduce peak VRAM.
    """
    from . import gpu

    for cache in (_PIPE_CACHE, _EMB_CACHE):
        for k in list(cache.keys()):
            try:
                del cache[k]
            except Exception:
                pass
        cache.clear()
    gpu.empty_cache()


@dataclass
class DiarSegment:
    cluster_id: str
    start_sec: float
    end_sec: float


def _load_dotenv(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _resolve_hf_token() -> str:
    _load_dotenv(Path.cwd())
    tok = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError(
            "HuggingFace token not found.\n"
            "  - Set HUGGINGFACE_TOKEN env var, or\n"
            "  - Create a .env file at the project root with HUGGINGFACE_TOKEN=hf_...\n"
            "  (see .env.example)"
        )
    return tok


def _normalize_speaker_label(label: str) -> str:
    if label.startswith("SPEAKER_"):
        try:
            n = int(label.split("_", 1)[1])
            return f"spk_{n}"
        except ValueError:
            pass
    return label


def _load_audio_in_memory(path: Path):
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy()).contiguous()
    return {"waveform": waveform, "sample_rate": sr}


def diarize(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    preloaded: tuple | None = None,
) -> list[DiarSegment]:
    import torch
    from pyannote.audio import Pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    key = (model, device)
    pipeline = _PIPE_CACHE.get(key)
    if pipeline is None:
        token = _resolve_hf_token()
        log.info("Loading pyannote pipeline: %s", model)
        try:
            pipeline = Pipeline.from_pretrained(model, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(model, use_auth_token=token)

        if pipeline is None:
            raise RuntimeError(
                f"pyannote returned None for model {model!r}. "
                "Most likely the EULA hasn't been accepted on HuggingFace, or the token "
                "lacks read access."
            )
        pipeline.to(torch.device(device))
        _PIPE_CACHE[key] = pipeline

    log.info("Diarizing %s on device=%s", audio_path.name, device)

    if preloaded is not None:
        data, sr = preloaded
        waveform = torch.from_numpy(data.T.copy()).contiguous()
        audio_input = {"waveform": waveform, "sample_rate": sr}
    else:
        audio_input = _load_audio_in_memory(audio_path)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    output = pipeline(audio_input, **kwargs)

    annotation = getattr(output, "speaker_diarization", output)

    segments: list[DiarSegment] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            DiarSegment(
                cluster_id=_normalize_speaker_label(speaker),
                start_sec=float(segment.start),
                end_sec=float(segment.end),
            )
        )
    clusters = {s.cluster_id for s in segments}
    log.info(
        "Diarization produced %d segments across %d cluster(s)",
        len(segments), len(clusters),
    )
    return segments


EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def diarize_by_lines(
    audio_path: Path,
    lines: list,
    num_speakers: int,
    device: Optional[str] = None,
    embedding_model: str = EMBEDDING_MODEL,
    preloaded: tuple | None = None,
    short_line_sec: float = 0.6,
    short_line_pad: float = 0.4,
) -> dict[str, dict]:
    """Per-line embedding diarization.

    For each line we compute ONE speaker embedding (on the vocals stem), then
    cluster the embeddings into ``num_speakers`` groups. Far more robust than
    overlap for close-sounding vocals / short lines.

    Returns ``{line_id: {"cluster": "emb_N"|None, "margin": float}}`` where ``margin``
    is the cosine-similarity gap to the second-closest speaker (a confidence proxy).

    Loads the audio into memory (soundfile) and passes ``{"waveform","sample_rate"}`` to
    inference: torchcodec is broken on this machine, so ``Inference.crop(path,...)`` would crash.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from pyannote.audio import Inference, Model
    from sklearn.cluster import AgglomerativeClustering

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    key = (embedding_model, device)
    inf = _EMB_CACHE.get(key)
    if inf is None:
        token = _resolve_hf_token()
        log.info("Loading speaker-embedding model: %s on device=%s", embedding_model, device)
        try:
            model = Model.from_pretrained(embedding_model, token=token)
        except TypeError:
            model = Model.from_pretrained(embedding_model, use_auth_token=token)
        if model is None:
            raise RuntimeError(
                f"pyannote returned None for {embedding_model!r} "
                "(EULA not accepted on HuggingFace, or token without access)."
            )
        try:
            inf = Inference(model, window="whole", device=torch.device(device))
        except TypeError:
            inf = Inference(model, window="whole")
            try:
                inf.to(torch.device(device))
            except Exception:
                pass
        _EMB_CACHE[key] = inf

    if preloaded is not None:
        data, sr = preloaded
    else:
        data, sr = sf.read(str(audio_path), always_2d=True, dtype="float32")  # (T, C)
    wav = torch.from_numpy(data.T.copy()).contiguous()                    # (C, T)
    dur = wav.shape[1] / sr
    min_len = int(0.2 * sr)

    ids: list[str] = []
    embs: list = []
    for ln in lines:
        s, e = float(ln.start_sec), float(ln.end_sec)
        if e - s < short_line_sec:  # very short line: widen around the center
            mid = (s + e) / 2.0
            s, e = max(0.0, mid - short_line_pad), min(dur, mid + short_line_pad)
        e = min(e, dur)
        seg = wav[:, int(s * sr):int(e * sr)]
        emb = None
        if seg.shape[1] >= min_len:
            try:
                out = inf({"waveform": seg, "sample_rate": sr})
                out = getattr(out, "data", out)
                emb = np.asarray(out, dtype="float32").reshape(-1)
            except Exception as ex:
                log.warning("embedding failed for %s: %s", ln.id, ex)
        ids.append(ln.id)
        embs.append(emb)

    valid = [(i, v) for i, v in enumerate(embs) if v is not None]
    if len(valid) < num_speakers:
        raise RuntimeError(
            f"only {len(valid)} usable embeddings for {num_speakers} speakers"
        )

    X = np.stack([v for _, v in valid])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    n_clusters = min(num_speakers, len(valid))
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    labels = clustering.fit_predict(X)

    centroids: dict = {}
    for lab in set(labels):
        c = X[labels == lab].mean(axis=0)
        centroids[lab] = c / (np.linalg.norm(c) + 1e-9)

    result: dict[str, dict] = {lid: {"cluster": None, "margin": 0.0} for lid in ids}
    for pos, ((idx, _), lab) in enumerate(zip(valid, labels)):
        row = X[pos]
        sims = {k: float(row @ c) for k, c in centroids.items()}
        own = sims[lab]
        other = max((val for k, val in sims.items() if k != lab), default=0.0)
        result[ids[idx]] = {"cluster": f"emb_{int(lab)}", "margin": own - other}

    log.info(
        "Per-line diarization: %d/%d lines embedded across %d cluster(s)",
        len(valid), len(ids), n_clusters,
    )
    return result
