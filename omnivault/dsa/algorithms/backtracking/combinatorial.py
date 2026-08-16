from __future__ import annotations

from collections.abc import Iterator, Sequence

from ...core.errors import EmptyContainer


def permutations(seq: Sequence[object]) -> Iterator[list[object]]:
    items = list(seq)
    n = len(items)
    used = [False] * n
    current: list[object] = []

    def backtrack() -> Iterator[list[object]]:
        if len(current) == n:
            yield list(current)
            return
        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            current.append(items[i])
            yield from backtrack()
            current.pop()
            used[i] = False

    yield from backtrack()


def combinations(seq: Sequence[object], k: int) -> Iterator[list[object]]:
    items = list(seq)
    n = len(items)
    if n == 0 and k > 0:
        raise EmptyContainer("cannot draw combinations of k>0 from empty sequence")
    if k < 0 or k > n:
        return

    current: list[object] = []

    def backtrack(start: int) -> Iterator[list[object]]:
        if len(current) == k:
            yield list(current)
            return
        remaining = k - len(current)
        upper = n - remaining
        for i in range(start, upper + 1):
            current.append(items[i])
            yield from backtrack(i + 1)
            current.pop()

    yield from backtrack(0)


def subsets(seq: Sequence[object]) -> Iterator[list[object]]:
    items = list(seq)
    n = len(items)
    current: list[object] = []

    def backtrack(start: int) -> Iterator[list[object]]:
        yield list(current)
        for i in range(start, n):
            current.append(items[i])
            yield from backtrack(i + 1)
            current.pop()

    yield from backtrack(0)
