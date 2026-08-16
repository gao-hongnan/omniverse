from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from ..core.errors import EmptyContainer
from .points import Point


def _dist(a: Point[float], b: Point[float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _brute(pts: list[Point[float]]) -> tuple[Point[float], Point[float], float]:
    best_a, best_b = pts[0], pts[1]
    best = _dist(best_a, best_b)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = _dist(pts[i], pts[j])
            if d < best:
                best, best_a, best_b = d, pts[i], pts[j]
    return best_a, best_b, best


def _strip_closest(
    strip: list[Point[float]], best: tuple[Point[float], Point[float], float]
) -> tuple[Point[float], Point[float], float]:
    strip.sort(key=lambda p: p[1])
    a, b, d = best
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and (strip[j][1] - strip[i][1]) < d:
            cand = _dist(strip[i], strip[j])
            if cand < d:
                d, a, b = cand, strip[i], strip[j]
            j += 1
    return a, b, d


def _closest_recursive(
    sorted_x: list[Point[float]],
) -> tuple[Point[float], Point[float], float]:
    if len(sorted_x) <= 3:
        return _brute(sorted_x)
    mid = len(sorted_x) // 2
    mid_x = sorted_x[mid][0]
    left = _closest_recursive(sorted_x[:mid])
    right = _closest_recursive(sorted_x[mid:])
    best = left if left[2] < right[2] else right
    strip = [p for p in sorted_x if abs(p[0] - mid_x) < best[2]]
    return _strip_closest(strip, best)


def closest_pair(
    points: Sequence[Point[float]],
) -> tuple[Point[float], Point[float], float]:
    if len(points) < 2:
        raise EmptyContainer("closest_pair requires at least 2 points")
    sorted_x = sorted(points, key=lambda p: (p[0], p[1]))
    return _closest_recursive(sorted_x)
