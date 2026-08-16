from __future__ import annotations

from typing import Protocol


class DataStructure(Protocol):
    def __len__(self) -> int: ...


class TimingResult(Protocol):
    sizes: list[int]
    avg_times: list[float]
    median_times: list[float]
    best_times: list[float]
    worst_times: list[float]
