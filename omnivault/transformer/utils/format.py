from __future__ import annotations

from typing import List


def format_lr(lr_or_lrs: float | List[float], precision: int) -> str:
    format_str = f"%.{precision}f"
    if isinstance(lr_or_lrs, list):
        return ", ".join([format_str % lr for lr in lr_or_lrs])
    return format_str % lr_or_lrs
