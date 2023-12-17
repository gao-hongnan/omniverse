from __future__ import annotations

import torch

__all__ = ["get_device"]


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
