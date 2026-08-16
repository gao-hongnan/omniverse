from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence

from ...core.types import ComparableT, ItemT


def lis(seq: Sequence[ComparableT]) -> int:
    tails: list[ComparableT] = []
    for value in seq:
        position = bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)


def lcs(a: Sequence[ItemT], b: Sequence[ItemT]) -> int:
    m: int = len(a)
    n: int = len(b)
    if m == 0 or n == 0:
        return 0
    previous: list[int] = [0] * (n + 1)
    current: list[int] = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous, current = current, previous
    return previous[n]
