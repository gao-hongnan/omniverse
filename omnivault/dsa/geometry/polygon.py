from __future__ import annotations

from collections.abc import Sequence

from ..core.errors import InvalidConfiguration
from .points import Point


def polygon_area(vertices: Sequence[Point[float]]) -> float:
    n = len(vertices)
    if n < 3:
        raise InvalidConfiguration(f"polygon requires at least 3 vertices, got {n}")
    total = 0.0
    for i in range(n):
        x1, y1 = vertices[i][0], vertices[i][1]
        x2, y2 = vertices[(i + 1) % n][0], vertices[(i + 1) % n][1]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0
