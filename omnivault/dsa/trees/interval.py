from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class IntervalNode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: float
    end: float
    value: Any
    max_end: float
    left: IntervalNode | None = None
    right: IntervalNode | None = None


type Interval = tuple[float, float, Any]


class IntervalTree:
    __slots__ = ("_root",)

    def __init__(self) -> None:
        self._root: IntervalNode | None = None

    def insert(self, start: float, end: float, value: Any) -> None:
        self._root = self._insert(self._root, start, end, value)

    def query_point(self, point: float) -> list[Interval]:
        results: list[Interval] = []
        self._collect_point(self._root, point, results)
        return results

    def query_range(self, start: float, end: float) -> list[Interval]:
        results: list[Interval] = []
        self._collect_range(self._root, start, end, results)
        return results

    def _insert(
        self,
        node: IntervalNode | None,
        start: float,
        end: float,
        value: Any,
    ) -> IntervalNode:
        if node is None:
            return IntervalNode(start=start, end=end, value=value, max_end=end)
        if start < node.start:
            node.left = self._insert(node.left, start, end, value)
        else:
            node.right = self._insert(node.right, start, end, value)
        node.max_end = max(node.max_end, end)
        return node

    def _collect_point(
        self,
        node: IntervalNode | None,
        point: float,
        out: list[Interval],
    ) -> None:
        if node is None or point > node.max_end:
            return
        if node.start <= point <= node.end:
            out.append((node.start, node.end, node.value))
        self._collect_point(node.left, point, out)
        if node.left is None or node.left.max_end >= point:
            self._collect_point(node.right, point, out)
        else:
            self._collect_point(node.right, point, out)

    def _collect_range(
        self,
        node: IntervalNode | None,
        start: float,
        end: float,
        out: list[Interval],
    ) -> None:
        if node is None or start > node.max_end:
            return
        if node.start <= end and node.end >= start:
            out.append((node.start, node.end, node.value))
        self._collect_range(node.left, start, end, out)
        if node.start <= end:
            self._collect_range(node.right, start, end, out)
