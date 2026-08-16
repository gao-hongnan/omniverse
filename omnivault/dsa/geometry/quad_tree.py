from __future__ import annotations

from typing import Generic

from ..core.errors import IncompatibleSketch
from ..core.types import ItemT, NumericT
from .points import Point, Rectangle


class QuadTreeNode(Generic[ItemT, NumericT]):  # noqa: UP046
    def __init__(self, boundary: Rectangle[NumericT], capacity: int) -> None:
        self.boundary: Rectangle[NumericT] = boundary
        self.capacity = capacity
        self.points: list[tuple[Point[NumericT], ItemT]] = []
        self.children: list[QuadTreeNode[ItemT, NumericT]] | None = None
        self.divided = False

    def subdivide(self) -> None:
        if self.divided:
            return

        min_x = self.boundary.min_point[0]
        min_y = self.boundary.min_point[1]
        max_x = self.boundary.max_point[0]
        max_y = self.boundary.max_point[1]

        mid_x = type(min_x)((min_x + max_x) / 2)
        mid_y = type(min_y)((min_y + max_y) / 2)

        nw = Rectangle[NumericT](Point[NumericT]([min_x, mid_y]), Point[NumericT]([mid_x, max_y]))
        ne = Rectangle[NumericT](Point[NumericT]([mid_x, mid_y]), Point[NumericT]([max_x, max_y]))
        sw = Rectangle[NumericT](Point[NumericT]([min_x, min_y]), Point[NumericT]([mid_x, mid_y]))
        se = Rectangle[NumericT](Point[NumericT]([mid_x, min_y]), Point[NumericT]([max_x, mid_y]))

        self.children = [
            QuadTreeNode[ItemT, NumericT](nw, self.capacity),
            QuadTreeNode[ItemT, NumericT](ne, self.capacity),
            QuadTreeNode[ItemT, NumericT](sw, self.capacity),
            QuadTreeNode[ItemT, NumericT](se, self.capacity),
        ]
        self.divided = True


class QuadTree(Generic[ItemT, NumericT]):  # noqa: UP046
    def __init__(self, boundary: Rectangle[NumericT], capacity: int = 4) -> None:
        self.root: QuadTreeNode[ItemT, NumericT] = QuadTreeNode[ItemT, NumericT](boundary, capacity)
        self.size = 0

    def insert(self, point: Point[NumericT], data: ItemT) -> bool:
        if point.dimension != 2:
            raise IncompatibleSketch("QuadTree only supports 2D points")

        success = self._insert_recursive(self.root, point, data)
        if success:
            self.size += 1
        return success

    def _insert_recursive(self, node: QuadTreeNode[ItemT, NumericT], point: Point[NumericT], data: ItemT) -> bool:
        if not node.boundary.contains(point):
            return False

        if len(node.points) < node.capacity:
            node.points.append((point, data))
            return True

        if not node.divided:
            node.subdivide()

        return any(self._insert_recursive(child, point, data) for child in node.children) if node.children else False

    def query_range(self, region: Rectangle[NumericT]) -> list[tuple[Point[NumericT], ItemT]]:
        results: list[tuple[Point[NumericT], ItemT]] = []
        self._query_range_recursive(self.root, region, results)
        return results

    def _query_range_recursive(
        self,
        node: QuadTreeNode[ItemT, NumericT],
        region: Rectangle[NumericT],
        results: list[tuple[Point[NumericT], ItemT]],
    ) -> None:
        if not node.boundary.intersects(region):
            return

        for point, data in node.points:
            if region.contains(point):
                results.append((point, data))

        if node.divided and node.children:
            for child in node.children:
                self._query_range_recursive(child, region, results)

    def query_circle(self, center: Point[NumericT], radius: float) -> list[tuple[Point[NumericT], ItemT]]:
        radius_squared = radius * radius
        results: list[tuple[Point[NumericT], ItemT]] = []
        self._query_circle_recursive(self.root, center, radius_squared, results)
        return results

    def _query_circle_recursive(
        self,
        node: QuadTreeNode[ItemT, NumericT],
        center: Point[NumericT],
        radius_squared: float,
        results: list[tuple[Point[NumericT], ItemT]],
    ) -> None:
        min_x = node.boundary.min_point[0]
        min_y = node.boundary.min_point[1]
        max_x = node.boundary.max_point[0]
        max_y = node.boundary.max_point[1]

        closest_x = max(min_x, min(center[0], max_x))
        closest_y = max(min_y, min(center[1], max_y))

        distance_squared = (center[0] - closest_x) ** 2 + (center[1] - closest_y) ** 2

        if distance_squared > radius_squared:
            return

        for point, data in node.points:
            if point.distance_squared(center) <= radius_squared:
                results.append((point, data))

        if node.divided and node.children:
            for child in node.children:
                self._query_circle_recursive(child, center, radius_squared, results)

    def nearest_neighbor(self, query_point: Point[NumericT]) -> tuple[Point[NumericT], ItemT] | None:
        best_point: Point[NumericT] | None = None
        best_data: ItemT | None = None
        best_distance = float("inf")

        self._nearest_neighbor_recursive(self.root, query_point, best_point, best_data, best_distance)

        if best_point is not None and best_data is not None:
            return (best_point, best_data)
        return None

    def _nearest_neighbor_recursive(
        self,
        node: QuadTreeNode[ItemT, NumericT],
        query_point: Point[NumericT],
        best_point: Point[NumericT] | None,
        best_data: ItemT | None,
        best_distance: float,
    ) -> tuple[Point[NumericT] | None, ItemT | None, float]:
        min_x = node.boundary.min_point[0]
        min_y = node.boundary.min_point[1]
        max_x = node.boundary.max_point[0]
        max_y = node.boundary.max_point[1]

        closest_x = max(min_x, min(query_point[0], max_x))
        closest_y = max(min_y, min(query_point[1], max_y))

        distance_to_boundary = (query_point[0] - closest_x) ** 2 + (query_point[1] - closest_y) ** 2

        if distance_to_boundary >= best_distance:
            return best_point, best_data, best_distance

        for point, data in node.points:
            distance = query_point.distance_squared(point)
            if distance < best_distance:
                best_point = point
                best_data = data
                best_distance = distance

        if node.divided and node.children:
            for child in node.children:
                best_point, best_data, best_distance = self._nearest_neighbor_recursive(
                    child, query_point, best_point, best_data, best_distance
                )

        return best_point, best_data, best_distance
