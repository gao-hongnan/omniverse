from __future__ import annotations

from .points import Point


def _orientation(p: Point[float], q: Point[float], r: Point[float]) -> int:
    value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if value > 0:
        return 1
    if value < 0:
        return 2
    return 0


def _on_segment(p: Point[float], q: Point[float], r: Point[float]) -> bool:
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def segments_intersect(p1: Point[float], p2: Point[float], p3: Point[float], p4: Point[float]) -> bool:
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    return bool(o4 == 0 and _on_segment(p3, p2, p4))
