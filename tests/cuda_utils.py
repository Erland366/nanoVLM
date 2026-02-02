from __future__ import annotations

import torch


def cuda_is_usable() -> bool:
    """Return True iff CUDA is available and the current device is supported by this PyTorch build."""
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
        return arch in supported
    except Exception:
        return False

