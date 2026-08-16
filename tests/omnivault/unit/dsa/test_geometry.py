from __future__ import annotations

from math import isclose, sqrt

import pytest

from omnivault.dsa.core.errors import EmptyContainer, InvalidConfiguration
from omnivault.dsa.geometry import (
    Point,
    closest_pair,
    convex_hull,
    polygon_area,
    segments_intersect,
)


def _pt(x: float, y: float) -> Point[float]:
    return Point[float]([x, y])


class TestConvexHull:
    @pytest.mark.unit
    def test_square_with_interior_point(self) -> None:
        points = [_pt(0, 0), _pt(1, 1), _pt(2, 2), _pt(2, 0), _pt(0, 2)]
        hull = convex_hull(points)
        assert hull == [_pt(0, 0), _pt(2, 0), _pt(2, 2), _pt(0, 2)]

    @pytest.mark.unit
    def test_triangle(self) -> None:
        points = [_pt(0, 0), _pt(4, 0), _pt(2, 3)]
        hull = convex_hull(points)
        assert hull == [_pt(0, 0), _pt(4, 0), _pt(2, 3)]

    @pytest.mark.unit
    def test_collinear_points(self) -> None:
        points = [_pt(0, 0), _pt(1, 1), _pt(2, 2), _pt(3, 3)]
        hull = convex_hull(points)
        assert hull == [_pt(0, 0), _pt(3, 3)]

    @pytest.mark.unit
    def test_single_point(self) -> None:
        assert convex_hull([_pt(1, 2)]) == [_pt(1, 2)]

    @pytest.mark.unit
    def test_two_points(self) -> None:
        assert convex_hull([_pt(0, 0), _pt(1, 1)]) == [_pt(0, 0), _pt(1, 1)]

    @pytest.mark.unit
    def test_empty(self) -> None:
        assert convex_hull([]) == []


class TestClosestPair:
    @pytest.mark.unit
    def test_basic(self) -> None:
        points = [_pt(0, 0), _pt(1, 1), _pt(10, 10), _pt(5, 5)]
        a, b, d = closest_pair(points)
        assert {(a[0], a[1]), (b[0], b[1])} == {(0.0, 0.0), (1.0, 1.0)}
        assert isclose(d, sqrt(2.0))

    @pytest.mark.unit
    def test_two_points(self) -> None:
        a, b, d = closest_pair([_pt(0, 0), _pt(3, 4)])
        assert {(a[0], a[1]), (b[0], b[1])} == {(0.0, 0.0), (3.0, 4.0)}
        assert isclose(d, 5.0)

    @pytest.mark.unit
    def test_duplicate_points(self) -> None:
        _, _, d = closest_pair([_pt(2, 2), _pt(2, 2), _pt(7, 7)])
        assert isclose(d, 0.0)

    @pytest.mark.unit
    def test_too_few_raises(self) -> None:
        with pytest.raises(EmptyContainer):
            closest_pair([_pt(0, 0)])
        with pytest.raises(EmptyContainer):
            closest_pair([])


class TestPolygonArea:
    @pytest.mark.unit
    def test_rectangle(self) -> None:
        verts = [_pt(0, 0), _pt(4, 0), _pt(4, 3), _pt(0, 3)]
        assert isclose(polygon_area(verts), 12.0)

    @pytest.mark.unit
    def test_triangle(self) -> None:
        verts = [_pt(0, 0), _pt(4, 0), _pt(0, 3)]
        assert isclose(polygon_area(verts), 6.0)

    @pytest.mark.unit
    def test_clockwise_same_area(self) -> None:
        verts = [_pt(0, 0), _pt(0, 3), _pt(4, 3), _pt(4, 0)]
        assert isclose(polygon_area(verts), 12.0)

    @pytest.mark.unit
    def test_too_few_vertices_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            polygon_area([_pt(0, 0), _pt(1, 1)])


class TestSegmentsIntersect:
    @pytest.mark.unit
    def test_crossing_segments(self) -> None:
        assert segments_intersect(_pt(0, 0), _pt(2, 2), _pt(0, 2), _pt(2, 0)) is True

    @pytest.mark.unit
    def test_collinear_disjoint(self) -> None:
        assert segments_intersect(_pt(0, 0), _pt(1, 1), _pt(2, 2), _pt(3, 3)) is False

    @pytest.mark.unit
    def test_collinear_overlapping(self) -> None:
        assert segments_intersect(_pt(0, 0), _pt(2, 2), _pt(1, 1), _pt(3, 3)) is True

    @pytest.mark.unit
    def test_touching_at_endpoint(self) -> None:
        assert segments_intersect(_pt(0, 0), _pt(1, 1), _pt(1, 1), _pt(2, 0)) is True

    @pytest.mark.unit
    def test_parallel_non_collinear(self) -> None:
        assert segments_intersect(_pt(0, 0), _pt(2, 0), _pt(0, 1), _pt(2, 1)) is False
