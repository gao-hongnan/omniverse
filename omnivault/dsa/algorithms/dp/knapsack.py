from __future__ import annotations

from collections.abc import Sequence

from ...core.errors import RectangularityViolation


def _validate_pair(weights: Sequence[int], values: Sequence[int]) -> None:
    if len(weights) != len(values):
        raise RectangularityViolation(f"weights/values length mismatch: {len(weights)} != {len(values)}")


def knapsack_01(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    _validate_pair(weights, values)
    if capacity <= 0:
        return 0
    table: list[int] = [0] * (capacity + 1)
    for weight, value in zip(weights, values, strict=True):
        for c in range(capacity, weight - 1, -1):
            candidate: int = table[c - weight] + value
            if candidate > table[c]:
                table[c] = candidate
    return table[capacity]


def knapsack_unbounded(weights: Sequence[int], values: Sequence[int], capacity: int) -> int:
    _validate_pair(weights, values)
    if capacity <= 0:
        return 0
    table: list[int] = [0] * (capacity + 1)
    for c in range(1, capacity + 1):
        best: int = table[c]
        for weight, value in zip(weights, values, strict=True):
            if weight <= c:
                candidate: int = table[c - weight] + value
                if candidate > best:
                    best = candidate
        table[c] = best
    return table[capacity]
