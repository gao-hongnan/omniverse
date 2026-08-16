from __future__ import annotations

from typing import Generic

from ..core.errors import CycleDetected
from ..core.types import ItemT, NumericT
from .points import Point, Rectangle


class RTree(Generic[ItemT, NumericT]):  # noqa: UP046
    def __init__(self, max_entries: int = 4) -> None:
        self.max_entries = max_entries
        self.root: RTreeNode[ItemT, NumericT] | None = None
        self.size = 0

    def insert(self, rectangle: Rectangle[NumericT], data: ItemT) -> None:
        if self.root is None:
            self.root = RTreeNode[ItemT, NumericT](is_leaf=True)

        self._insert_recursive(self.root, rectangle, data)
        self.size += 1

    def _insert_recursive(self, node: RTreeNode[ItemT, NumericT], rectangle: Rectangle[NumericT], data: ItemT) -> None:
        if node.is_leaf:
            node.entries.append((rectangle, data))
            if len(node.entries) > self.max_entries:
                self._split_node(node)
        else:
            best_child = self._choose_subtree(node, rectangle)
            self._insert_recursive(best_child, rectangle, data)

    def _choose_subtree(
        self, node: RTreeNode[ItemT, NumericT], rectangle: Rectangle[NumericT]
    ) -> RTreeNode[ItemT, NumericT]:
        if node.children is None or len(node.children) == 0:
            raise CycleDetected("Non-leaf node must have children")
        best_child = node.children[0]
        best_enlargement = float("inf")

        for child in node.children:
            if child.mbr is None:
                continue
            enlargement = self._calculate_enlargement(child.mbr, rectangle)
            if enlargement < best_enlargement:
                best_enlargement = enlargement
                best_child = child

        return best_child

    def _calculate_enlargement(self, mbr: Rectangle[NumericT], rectangle: Rectangle[NumericT]) -> NumericT:
        new_min = Point([min(mbr.min_point[i], rectangle.min_point[i]) for i in range(mbr.dimension)])
        new_max = Point([max(mbr.max_point[i], rectangle.max_point[i]) for i in range(mbr.dimension)])
        new_mbr = Rectangle(new_min, new_max)
        return new_mbr.area() - mbr.area()

    def _split_node(self, node: RTreeNode[ItemT, NumericT]) -> None:
        pass

    def query(self, region: Rectangle[NumericT]) -> list[ItemT]:
        if self.root is None:
            return []

        results: list[ItemT] = []
        self._query_recursive(self.root, region, results)
        return results

    def _query_recursive(
        self, node: RTreeNode[ItemT, NumericT], region: Rectangle[NumericT], results: list[ItemT]
    ) -> None:
        if node.mbr and not node.mbr.intersects(region):
            return

        if node.is_leaf:
            for rect, data in node.entries:
                if rect.intersects(region):
                    results.append(data)
        else:
            if node.children is not None:
                for child in node.children:
                    self._query_recursive(child, region, results)


class RTreeNode(Generic[ItemT, NumericT]):  # noqa: UP046
    def __init__(self, is_leaf: bool = False) -> None:
        self.is_leaf = is_leaf
        self.entries: list[tuple[Rectangle[NumericT], ItemT]] = []
        self.children: list[RTreeNode[ItemT, NumericT]] | None = None if is_leaf else []
        self.mbr: Rectangle[NumericT] | None = None

    def update_mbr(self) -> None:
        if not self.entries and not self.children:
            self.mbr = None
            return

        if self.is_leaf and self.entries:
            first_rect = self.entries[0][0]
            min_coords = first_rect.min_point.coordinates[:]
            max_coords = first_rect.max_point.coordinates[:]

            for rect, _ in self.entries[1:]:
                for i in range(first_rect.dimension):
                    min_coords[i] = min(min_coords[i], rect.min_point[i])
                    max_coords[i] = max(max_coords[i], rect.max_point[i])

            self.mbr = Rectangle(Point(min_coords), Point(max_coords))
        elif not self.is_leaf and self.children:
            for child in self.children:
                child.update_mbr()

            first_mbr = self.children[0].mbr
            if first_mbr is None:
                self.mbr = None
                return

            min_coords = first_mbr.min_point.coordinates[:]
            max_coords = first_mbr.max_point.coordinates[:]

            for child in self.children[1:]:
                if child.mbr:
                    for i in range(first_mbr.dimension):
                        min_coords[i] = min(min_coords[i], child.mbr.min_point[i])
                        max_coords[i] = max(max_coords[i], child.mbr.max_point[i])

            self.mbr = Rectangle(Point(min_coords), Point(max_coords))
