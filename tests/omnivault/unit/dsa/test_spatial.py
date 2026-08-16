from __future__ import annotations

import pytest

from omnivault.dsa.core.errors import IncompatibleSketch
from omnivault.dsa.geometry import KDTree, Point, QuadTree, Rectangle


@pytest.mark.unit
class TestPoint:
    def test_two_d_instantiation(self) -> None:
        point: Point[float] = Point([1.0, 2.0])
        assert point.dimension == 2
        assert point[0] == 1.0
        assert point[1] == 2.0

    def test_three_d_instantiation(self) -> None:
        point: Point[float] = Point([1.0, 2.0, 3.0])
        assert point.dimension == 3
        assert point[2] == 3.0

    def test_setitem(self) -> None:
        point: Point[float] = Point([1.0, 2.0])
        point[0] = 5.0
        assert point[0] == 5.0

    def test_equality(self) -> None:
        point1: Point[float] = Point([1.0, 2.0])
        point2: Point[float] = Point([1.0, 2.0])
        point3: Point[float] = Point([1.0, 3.0])
        assert point1 == point2
        assert point1 != point3

    def test_equality_non_point(self) -> None:
        point: Point[float] = Point([1.0, 2.0])
        assert point != [1.0, 2.0]
        assert point != "not a point"

    def test_repr(self) -> None:
        point: Point[float] = Point([1.0, 2.0])
        assert repr(point) == "Point([1.0, 2.0])"

    def test_distance_squared_axis_aligned(self) -> None:
        a: Point[float] = Point([0.0, 0.0])
        b: Point[float] = Point([3.0, 4.0])
        assert a.distance_squared(b) == pytest.approx(25.0)

    def test_distance_axis_aligned(self) -> None:
        a: Point[float] = Point([0.0, 0.0])
        b: Point[float] = Point([3.0, 4.0])
        assert a.distance(b) == pytest.approx(5.0)

    def test_distance_same_point(self) -> None:
        a: Point[float] = Point([5.0, 5.0])
        b: Point[float] = Point([5.0, 5.0])
        assert a.distance_squared(b) == pytest.approx(0.0)
        assert a.distance(b) == pytest.approx(0.0)

    def test_distance_squared_dimension_mismatch_raises(self) -> None:
        a: Point[float] = Point([0.0, 0.0])
        b: Point[float] = Point([1.0, 2.0, 3.0])
        with pytest.raises(IncompatibleSketch, match="Points must have same dimension"):
            a.distance_squared(b)

    def test_distance_dimension_mismatch_raises(self) -> None:
        a: Point[float] = Point([0.0, 0.0])
        b: Point[float] = Point([1.0, 2.0, 3.0])
        with pytest.raises(IncompatibleSketch):
            a.distance(b)


@pytest.mark.unit
class TestRectangle:
    def test_instantiation(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        assert rect.dimension == 2
        assert rect.min_point == Point([0.0, 0.0])
        assert rect.max_point == Point([10.0, 10.0])

    def test_contains_interior_point(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        assert rect.contains(Point([5.0, 5.0])) is True

    def test_contains_boundary_point(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        assert rect.contains(Point([0.0, 0.0])) is True
        assert rect.contains(Point([10.0, 10.0])) is True
        assert rect.contains(Point([5.0, 0.0])) is True

    def test_contains_exterior_point(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        assert rect.contains(Point([11.0, 5.0])) is False
        assert rect.contains(Point([-1.0, 5.0])) is False

    def test_contains_dimension_mismatch_returns_false(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        result = rect.contains(Point([1.0, 2.0, 3.0]))
        assert result is False

    def test_dimension_mismatch_constructor_raises(self) -> None:
        with pytest.raises(IncompatibleSketch, match="Points must have same dimension"):
            Rectangle(Point([0.0, 0.0]), Point([1.0, 1.0, 1.0]))

    def test_intersects_overlapping(self) -> None:
        rect1: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        rect2: Rectangle[float] = Rectangle(Point([5.0, 5.0]), Point([15.0, 15.0]))
        assert rect1.intersects(rect2) is True
        assert rect2.intersects(rect1) is True

    def test_intersects_non_overlapping(self) -> None:
        rect1: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([5.0, 5.0]))
        rect2: Rectangle[float] = Rectangle(Point([10.0, 10.0]), Point([15.0, 15.0]))
        assert rect1.intersects(rect2) is False

    def test_intersects_dimension_mismatch_returns_false(self) -> None:
        rect1: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        rect2: Rectangle[float] = Rectangle(Point([0.0, 0.0, 0.0]), Point([5.0, 5.0, 5.0]))
        assert rect1.intersects(rect2) is False

    def test_area_2d(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([4.0, 5.0]))
        assert rect.area() == pytest.approx(20.0)

    def test_area_non_2d_raises(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0, 0.0]), Point([4.0, 5.0, 6.0]))
        with pytest.raises(IncompatibleSketch, match="Area calculation only supported for 2D rectangles"):
            rect.area()

    def test_repr(self) -> None:
        rect: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        assert "Rectangle" in repr(rect)


@pytest.mark.unit
class TestKDTree:
    def test_instantiation(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        assert tree.dimension == 2
        assert tree.size == 0
        assert tree.root is None

    def test_insert_single_point(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([1.0, 2.0]), "a")
        assert tree.size == 1
        assert tree.root is not None

    def test_insert_multiple_points(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([2.0, 3.0]), "a")
        tree.insert(Point([5.0, 4.0]), "b")
        tree.insert(Point([9.0, 6.0]), "c")
        assert tree.size == 3

    def test_search_exact_match(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([2.0, 3.0]), "a")
        tree.insert(Point([5.0, 4.0]), "b")
        assert tree.search(Point([5.0, 4.0])) == "b"

    def test_search_missing_returns_none(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([1.0, 1.0]), "x")
        assert tree.search(Point([9.0, 9.0])) is None

    def test_search_empty_tree_returns_none(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        assert tree.search(Point([5.0, 4.0])) is None

    def test_insert_dimension_mismatch_raises(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        with pytest.raises(IncompatibleSketch):
            tree.insert(Point([1.0, 2.0, 3.0]), "a")

    def test_nearest_neighbor_single_point(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([2.0, 3.0]), "a")
        result = tree.nearest_neighbor(Point([2.0, 3.0]))
        assert result is not None
        point, data = result
        assert point == Point([2.0, 3.0])
        assert data == "a"

    def test_nearest_neighbor_multiple_points(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([0.0, 0.0]), "a")
        tree.insert(Point([10.0, 10.0]), "b")
        result = tree.nearest_neighbor(Point([1.0, 1.0]))
        assert result is not None
        point, data = result
        assert data == "a"

    def test_nearest_neighbor_empty_tree_returns_none(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        result = tree.nearest_neighbor(Point([5.0, 5.0]))
        assert result is None

    def test_range_search_all_inside(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([1.0, 1.0]), "a")
        tree.insert(Point([2.0, 2.0]), "b")
        tree.insert(Point([3.0, 3.0]), "c")
        region: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([5.0, 5.0]))
        results = tree.range_search(region)
        assert len(results) == 3

    def test_range_search_partial(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([1.0, 1.0]), "a")
        tree.insert(Point([2.0, 2.0]), "b")
        tree.insert(Point([8.0, 8.0]), "c")
        region: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([5.0, 5.0]))
        results = tree.range_search(region)
        assert len(results) == 2
        assert ("a", "b") == tuple(sorted([data for _, data in results]))

    def test_range_search_empty_region(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([1.0, 1.0]), "a")
        region: Rectangle[float] = Rectangle(Point([10.0, 10.0]), Point([15.0, 15.0]))
        results = tree.range_search(region)
        assert len(results) == 0

    def test_k_nearest_neighbors(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([0.0, 0.0]), "a")
        tree.insert(Point([1.0, 0.0]), "b")
        tree.insert(Point([2.0, 0.0]), "c")
        tree.insert(Point([10.0, 10.0]), "d")
        results = tree.k_nearest_neighbors(Point([0.5, 0.0]), k=2)
        assert len(results) == 2

    def test_k_nearest_neighbors_k_zero(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([0.0, 0.0]), "a")
        results = tree.k_nearest_neighbors(Point([0.0, 0.0]), k=0)
        assert len(results) == 0

    def test_k_nearest_neighbors_k_exceeds_size(self) -> None:
        tree: KDTree[str, float] = KDTree(dimension=2)
        tree.insert(Point([0.0, 0.0]), "a")
        tree.insert(Point([1.0, 1.0]), "b")
        results = tree.k_nearest_neighbors(Point([0.0, 0.0]), k=10)
        assert len(results) == 2


@pytest.mark.unit
class TestQuadTree:
    def test_instantiation(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds, capacity=4)
        assert tree.size == 0

    def test_insert_single_point(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds, capacity=4)
        success = tree.insert(Point([1.0, 1.0]), "a")
        assert success is True
        assert tree.size == 1

    def test_insert_multiple_points(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds, capacity=4)
        assert tree.insert(Point([1.0, 1.0]), "a") is True
        assert tree.insert(Point([2.0, 2.0]), "b") is True
        assert tree.insert(Point([8.0, 8.0]), "c") is True
        assert tree.size == 3

    def test_insert_beyond_capacity_subdivides(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds, capacity=2)
        tree.insert(Point([1.0, 1.0]), "a")
        tree.insert(Point([2.0, 2.0]), "b")
        tree.insert(Point([8.0, 8.0]), "c")
        assert tree.size == 3

    def test_insert_outside_bounds_returns_false(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds)
        success = tree.insert(Point([15.0, 15.0]), "a")
        assert success is False
        assert tree.size == 0

    def test_insert_non_2d_raises(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds)
        with pytest.raises(IncompatibleSketch, match="QuadTree only supports 2D points"):
            tree.insert(Point([1.0, 2.0, 3.0]), "a")

    def test_query_range_all_inside(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds, capacity=2)
        tree.insert(Point([1.0, 1.0]), "a")
        tree.insert(Point([2.0, 2.0]), "b")
        tree.insert(Point([3.0, 3.0]), "c")
        query: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([5.0, 5.0]))
        results = tree.query_range(query)
        assert len(results) == 3

    def test_query_range_partial(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds, capacity=2)
        tree.insert(Point([1.0, 1.0]), "a")
        tree.insert(Point([2.0, 2.0]), "b")
        tree.insert(Point([8.0, 8.0]), "c")
        query: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([5.0, 5.0]))
        results = tree.query_range(query)
        assert len(results) == 2

    def test_query_range_empty(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds)
        tree.insert(Point([1.0, 1.0]), "a")
        query: Rectangle[float] = Rectangle(Point([5.0, 5.0]), Point([8.0, 8.0]))
        results = tree.query_range(query)
        assert len(results) == 0

    def test_query_circle_all_inside(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds, capacity=2)
        tree.insert(Point([5.0, 5.0]), "a")
        tree.insert(Point([5.5, 5.5]), "b")
        results = tree.query_circle(Point([5.0, 5.0]), radius=1.0)
        assert len(results) >= 1

    def test_query_circle_empty(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds)
        tree.insert(Point([1.0, 1.0]), "a")
        results = tree.query_circle(Point([8.0, 8.0]), radius=1.0)
        assert len(results) == 0

    def test_nearest_neighbor_empty_tree_returns_none(self) -> None:
        bounds: Rectangle[float] = Rectangle(Point([0.0, 0.0]), Point([10.0, 10.0]))
        tree: QuadTree[str, float] = QuadTree(boundary=bounds)
        result = tree.nearest_neighbor(Point([5.0, 5.0]))
        assert result is None
