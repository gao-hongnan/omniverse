from __future__ import annotations

from collections.abc import Sequence

from .points import Point


def _cross(o: Point[float], a: Point[float], b: Point[float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Sequence[Point[float]]) -> list[Point[float]]:
    unique = sorted({(p[0], p[1]) for p in points})
    if len(unique) <= 1:
        return [Point[float]([x, y]) for x, y in unique]
    if len(unique) == 2:
        return [Point[float]([x, y]) for x, y in unique]

    pts = [Point[float]([x, y]) for x, y in unique]

    lower: list[Point[float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[Point[float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]
