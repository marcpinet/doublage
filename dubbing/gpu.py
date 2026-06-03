from __future__ import annotations

import gc
import logging


log = logging.getLogger(__name__)


def empty_cache() -> None:
    """Frees GPU/VRAM memory if possible.

    Runs ``gc.collect()`` then, if torch is importable and CUDA is available,
    ``torch.cuda.empty_cache()``. Safe no-op if torch/cuda are absent.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def log_mem(tag: str) -> None:
    """Logs the allocated VRAM (in GB) at info level, prefixed by ``tag``.

    Silent if torch/cuda are absent.
    """
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            log.info("%s: %.2f GB VRAM allocated", tag, allocated)
    except Exception:
        pass
