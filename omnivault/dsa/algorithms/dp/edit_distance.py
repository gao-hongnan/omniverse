from __future__ import annotations

from ...core.errors import RectangularityViolation


def levenshtein(a: str, b: str) -> int:
    m: int = len(a)
    n: int = len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    previous: list[int] = list(range(n + 1))
    current: list[int] = [0] * (n + 1)
    for i in range(1, m + 1):
        current[0] = i
        for j in range(1, n + 1):
            cost: int = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            )
        previous, current = current, previous
    return previous[n]


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise RectangularityViolation(f"hamming requires equal-length inputs: {len(a)} != {len(b)}")
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)
