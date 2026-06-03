from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


DEFAULT_MODEL = "htdemucs_ft"


def separate_voice(
    input_audio: Path,
    audio_dir: Path,
    model: str = DEFAULT_MODEL,
    device: Optional[str] = None,
) -> tuple[Path, Path]:
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    audio_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading Demucs model=%s on device=%s", model, device)
    loaded = get_model(model)
    loaded.to(device)
    loaded.eval()

    target_sr = int(loaded.samplerate)
    target_channels = int(loaded.audio_channels)

    log.info("Loading audio (resample to %d Hz, %d ch)", target_sr, target_channels)
    audio_file = AudioFile(input_audio)
    wav = audio_file.read(streams=0, samplerate=target_sr, channels=target_channels)

    ref = wav.mean(0)
    # Epsilon: a silent (or DC-only) input has std=0 — without it the normalization
    # fills both stems with NaN and silently corrupts everything downstream.
    ref_std = ref.std() + 1e-8
    wav = (wav - ref.mean()) / ref_std
    mix = wav.unsqueeze(0).to(device)

    log.info("Applying model on %s", input_audio.name)
    with torch.no_grad():
        sources = apply_model(loaded, mix, split=True, overlap=0.25, progress=False)
    sources = sources * ref_std + ref.mean()
    sources = sources.squeeze(0).cpu()

    src_names = list(loaded.sources)
    if "vocals" not in src_names:
        raise RuntimeError(f"Model has no vocals stem. Sources: {src_names}")

    vocals_idx = src_names.index("vocals")
    vocals = sources[vocals_idx]
    no_vocals = sources.sum(dim=0) - vocals

    vocals_dst = audio_dir / "vocals.wav"
    no_vocals_dst = audio_dir / "no_vocals.wav"
    sf.write(str(vocals_dst), vocals.numpy().T, target_sr, subtype="PCM_16")
    sf.write(str(no_vocals_dst), no_vocals.numpy().T, target_sr, subtype="PCM_16")

    log.info("Wrote stems: %s, %s", vocals_dst.name, no_vocals_dst.name)

    # The Demucs model (~2 GB) is not cached anywhere: we explicitly release
    # everything holding tensors/the model on the GPU.
    from . import gpu

    del loaded, sources, mix, wav
    gpu.empty_cache()

    return vocals_dst, no_vocals_dst
