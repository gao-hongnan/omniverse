from __future__ import annotations

import torch

__all__ = ["get_device"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # auto select device
        device = f"cuda:{torch.cuda.current_device()}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return torch.device(device)
