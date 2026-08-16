from __future__ import annotations

from typing import Generic

from ..core.errors import IncompatibleSketch
from ..core.types import ItemT, NumericT
from .points import Point, Rectangle


class KDTreeNode(Generic[ItemT, NumericT]):  # noqa: UP046
    def __init__(
        self,
        point: Point[NumericT],
        data: ItemT,
        dimension: int,
        left: KDTreeNode[ItemT, NumericT] | None = None,
        right: KDTreeNode[ItemT, NumericT] | None = None,
    ) -> None:
        self.point: Point[NumericT] = point
        self.data = data
        self.dimension = dimension
        self.left: KDTreeNode[ItemT, NumericT] | None = left
        self.right: KDTreeNode[ItemT, NumericT] | None = right


class KDTree(Generic[ItemT, NumericT]):  # noqa: UP046
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.root: KDTreeNode[ItemT, NumericT] | None = None
        self.size = 0

    def insert(self, point: Point[NumericT], data: ItemT) -> None:
        if point.dimension != self.dimension:
            raise IncompatibleSketch(f"Point dimension {point.dimension} doesn't match tree dimension {self.dimension}")

        self.root = self._insert_recursive(self.root, point, data, 0)
        self.size += 1

    def _insert_recursive(
        self, node: KDTreeNode[ItemT, NumericT] | None, point: Point[NumericT], data: ItemT, depth: int
    ) -> KDTreeNode[ItemT, NumericT]:
        if node is None:
            return KDTreeNode[ItemT, NumericT](point, data, depth % self.dimension)

        axis = depth % self.dimension

        if point[axis] < node.point[axis]:
            node.left = self._insert_recursive(node.left, point, data, depth + 1)
        else:
            node.right = self._insert_recursive(node.right, point, data, depth + 1)

        return node

    def search(self, point: Point[NumericT]) -> ItemT | None:
        node = self._search_recursive(self.root, point, 0)
        return node.data if node else None

    def _search_recursive(
        self, node: KDTreeNode[ItemT, NumericT] | None, point: Point[NumericT], depth: int
    ) -> KDTreeNode[ItemT, NumericT] | None:
        if node is None:
            return None

        if node.point == point:
            return node

        axis = depth % self.dimension

        if point[axis] < node.point[axis]:
            return self._search_recursive(node.left, point, depth + 1)
        else:
            return self._search_recursive(node.right, point, depth + 1)

    def nearest_neighbor(self, query_point: Point[NumericT]) -> tuple[Point[NumericT], ItemT] | None:
        if self.root is None:
            return None

        best_node, _ = self._nearest_neighbor_recursive(self.root, query_point, 0, self.root, float("inf"))

        return (best_node.point, best_node.data)

    def _nearest_neighbor_recursive(
        self,
        node: KDTreeNode[ItemT, NumericT] | None,
        query_point: Point[NumericT],
        depth: int,
        best_node: KDTreeNode[ItemT, NumericT],
        best_distance: float,
    ) -> tuple[KDTreeNode[ItemT, NumericT], float]:
        if node is None:
            return best_node, best_distance

        distance = query_point.distance_squared(node.point)
        if distance < best_distance:
            best_node = node
            best_distance = distance

        axis = depth % self.dimension
        diff = query_point[axis] - node.point[axis]

        if diff < 0:
            best_node, best_distance = self._nearest_neighbor_recursive(
                node.left, query_point, depth + 1, best_node, best_distance
            )
            if diff * diff < best_distance:
                best_node, best_distance = self._nearest_neighbor_recursive(
                    node.right, query_point, depth + 1, best_node, best_distance
                )
        else:
            best_node, best_distance = self._nearest_neighbor_recursive(
                node.right, query_point, depth + 1, best_node, best_distance
            )
            if diff * diff < best_distance:
                best_node, best_distance = self._nearest_neighbor_recursive(
                    node.left, query_point, depth + 1, best_node, best_distance
                )

        return best_node, best_distance

    def range_search(self, region: Rectangle[NumericT]) -> list[tuple[Point[NumericT], ItemT]]:
        results: list[tuple[Point[NumericT], ItemT]] = []
        self._range_search_recursive(self.root, region, 0, results)
        return results

    def _range_search_recursive(
        self,
        node: KDTreeNode[ItemT, NumericT] | None,
        region: Rectangle[NumericT],
        depth: int,
        results: list[tuple[Point[NumericT], ItemT]],
    ) -> None:
        if node is None:
            return

        if region.contains(node.point):
            results.append((node.point, node.data))

        axis = depth % self.dimension

        if node.point[axis] >= region.min_point[axis]:
            self._range_search_recursive(node.left, region, depth + 1, results)

        if node.point[axis] <= region.max_point[axis]:
            self._range_search_recursive(node.right, region, depth + 1, results)

    def k_nearest_neighbors(self, query_point: Point[NumericT], k: int) -> list[tuple[Point[NumericT], ItemT]]:
        if k <= 0:
            return []

        heap: list[tuple[float, KDTreeNode[ItemT, NumericT]]] = []
        self._k_nearest_recursive(self.root, query_point, 0, k, heap)

        heap.sort(key=lambda x: x[0])
        return [(node.point, node.data) for _, node in heap]

    def _k_nearest_recursive(
        self,
        node: KDTreeNode[ItemT, NumericT] | None,
        query_point: Point[NumericT],
        depth: int,
        k: int,
        heap: list[tuple[float, KDTreeNode[ItemT, NumericT]]],
    ) -> None:
        if node is None:
            return

        distance = query_point.distance_squared(node.point)

        if len(heap) < k:
            heap.append((distance, node))
            heap.sort(key=lambda x: x[0], reverse=True)
        elif distance < heap[0][0]:
            heap[0] = (distance, node)
            heap.sort(key=lambda x: x[0], reverse=True)

        axis = depth % self.dimension
        diff = query_point[axis] - node.point[axis]

        if diff < 0:
            self._k_nearest_recursive(node.left, query_point, depth + 1, k, heap)
            if len(heap) < k or diff * diff < heap[0][0]:
                self._k_nearest_recursive(node.right, query_point, depth + 1, k, heap)
        else:
            self._k_nearest_recursive(node.right, query_point, depth + 1, k, heap)
            if len(heap) < k or diff * diff < heap[0][0]:
                self._k_nearest_recursive(node.left, query_point, depth + 1, k, heap)
