from __future__ import annotations

import math
from collections.abc import Iterator
from typing import cast

from pydantic import BaseModel

from ..core.errors import EmptyContainer, IncompatibleSketch
from ..core.types import Comparable


class FibonacciNode[ItemT, PriorityT: Comparable](BaseModel):
    key: ItemT
    priority: PriorityT
    degree: int = 0
    marked: bool = False
    parent: FibonacciNode[ItemT, PriorityT] | None = None
    child: FibonacciNode[ItemT, PriorityT] | None = None
    left: FibonacciNode[ItemT, PriorityT] | None = None
    right: FibonacciNode[ItemT, PriorityT] | None = None

    def __init__(self, key: ItemT, priority: PriorityT, **data: object) -> None:
        super().__init__(key=key, priority=priority, **data)
        self.left = self
        self.right = self


class FibonacciHeap[ItemT, PriorityT: Comparable]:
    __slots__ = ("_min_node", "_size")

    def __init__(self) -> None:
        self._min_node: FibonacciNode[ItemT, PriorityT] | None = None
        self._size: int = 0

    def is_empty(self) -> bool:
        return self._min_node is None

    def size(self) -> int:
        return self._size

    def min_priority(self) -> PriorityT:
        if self._min_node is None:
            raise EmptyContainer("Heap is empty")
        return self._min_node.priority

    def min_key(self) -> ItemT:
        if self._min_node is None:
            raise EmptyContainer("Heap is empty")
        return self._min_node.key

    def insert(self, key: ItemT, priority: PriorityT) -> FibonacciNode[ItemT, PriorityT]:
        node = FibonacciNode(key, priority)
        self._size += 1

        if self._min_node is None:
            self._min_node = node
        else:
            self._add_to_root_list(node)
            if priority < self._min_node.priority:
                self._min_node = node

        return node

    def extract_min(self) -> tuple[ItemT, PriorityT]:
        min_node = self._min_node
        if min_node is None:
            raise EmptyContainer("Heap is empty")

        if min_node.child is not None:
            child: FibonacciNode[ItemT, PriorityT] | None = min_node.child
            first_child = child
            while child is not None:
                next_child = child.right
                child.parent = None
                self._add_to_root_list(child)

                if next_child == first_child:
                    break
                child = next_child

        self._remove_from_root_list(min_node)

        if min_node == min_node.right:
            self._min_node = None
        else:
            self._min_node = min_node.right
            self._consolidate()

        self._size -= 1
        return min_node.key, min_node.priority

    def decrease_key(self, node: FibonacciNode[ItemT, PriorityT], new_priority: PriorityT) -> None:
        if new_priority > node.priority:
            raise IncompatibleSketch("New priority is greater than current priority")

        node.priority = new_priority
        parent = node.parent

        if parent is not None and node.priority < parent.priority:
            self._cut(node, parent)
            self._cascading_cut(parent)

        if self._min_node is not None and node.priority < self._min_node.priority:
            self._min_node = node

    def delete(self, node: FibonacciNode[ItemT, PriorityT]) -> None:
        negative_infinity = cast("PriorityT", float("-inf"))
        self.decrease_key(node, negative_infinity)
        self.extract_min()

    def merge(self, other: FibonacciHeap[ItemT, PriorityT]) -> FibonacciHeap[ItemT, PriorityT]:
        merged = FibonacciHeap[ItemT, PriorityT]()
        merged._size = self._size + other._size

        if self._min_node is None:
            merged._min_node = other._min_node
        elif other._min_node is None:
            merged._min_node = self._min_node
        else:
            merged._min_node = self._min_node

            if self._min_node and other._min_node:
                self_last = self._min_node.left
                other_last = other._min_node.left

                if self_last is not None and other_last is not None:
                    self._min_node.left = other_last
                    other_last.right = self._min_node
                    other._min_node.left = self_last
                    self_last.right = other._min_node

                if other._min_node.priority < self._min_node.priority:
                    merged._min_node = other._min_node

        return merged

    def _add_to_root_list(self, node: FibonacciNode[ItemT, PriorityT]) -> None:
        if self._min_node is None:
            self._min_node = node
            node.left = node
            node.right = node
        else:
            node.right = self._min_node.right
            node.left = self._min_node
            if self._min_node.right is not None:
                self._min_node.right.left = node
            self._min_node.right = node

    def _remove_from_root_list(self, node: FibonacciNode[ItemT, PriorityT]) -> None:
        if node.right == node:
            return

        if node.left is not None:
            node.left.right = node.right
        if node.right is not None:
            node.right.left = node.left

    def _consolidate(self) -> None:
        max_degree = int(math.log2(self._size)) + 1
        degree_table: list[FibonacciNode[ItemT, PriorityT] | None] = [None] * max_degree

        root_nodes: list[FibonacciNode[ItemT, PriorityT]] = []
        current = self._min_node

        if current is not None:
            first = current
            while True:
                root_nodes.append(current)
                assert current.right is not None
                current = current.right
                if current == first:
                    break

        for node in root_nodes:
            degree = node.degree

            while degree < len(degree_table) and degree_table[degree] is not None:
                other = degree_table[degree]
                assert other is not None

                if node.priority > other.priority:
                    node, other = other, node

                self._link(other, node)
                degree_table[degree] = None
                degree += 1

            degree_table[degree] = node

        self._min_node = None

        for slot in degree_table:
            if slot is not None:
                if self._min_node is None:
                    self._min_node = slot
                    slot.left = slot
                    slot.right = slot
                else:
                    self._add_to_root_list(slot)
                    if slot.priority < self._min_node.priority:
                        self._min_node = slot

    def _link(self, child: FibonacciNode[ItemT, PriorityT], parent: FibonacciNode[ItemT, PriorityT]) -> None:
        self._remove_from_root_list(child)
        child.parent = parent

        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            child_left = parent.child.left
            if child_left is not None:
                child.left = child_left
                child_left.right = child
            child.right = parent.child
            parent.child.left = child

        parent.degree += 1
        child.marked = False

    def _cut(self, node: FibonacciNode[ItemT, PriorityT], parent: FibonacciNode[ItemT, PriorityT]) -> None:
        parent.degree -= 1

        if parent.child == node:
            if node.right == node:
                parent.child = None
            else:
                parent.child = node.right

        if node.left is not None:
            node.left.right = node.right
        if node.right is not None:
            node.right.left = node.left

        self._add_to_root_list(node)
        node.parent = None
        node.marked = False

    def _cascading_cut(self, node: FibonacciNode[ItemT, PriorityT]) -> None:
        parent = node.parent
        if parent is not None:
            if not node.marked:
                node.marked = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __iter__(self) -> Iterator[tuple[ItemT, PriorityT]]:
        if self._min_node is None:
            return

        visited: set[int] = set()
        stack: list[FibonacciNode[ItemT, PriorityT]] = [self._min_node]

        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue

            visited.add(id(node))
            yield node.key, node.priority

            if node.child is not None:
                child = node.child
                first_child = child
                while True:
                    stack.append(child)
                    assert child.right is not None
                    child = child.right
                    if child == first_child:
                        break

            if node.right is not None and id(node.right) not in visited:
                current = node.right
                while current != self._min_node and id(current) not in visited:
                    stack.append(current)
                    if current.right is None:
                        break
                    current = current.right

    def _validate_heap_property(self) -> bool:
        if self._min_node is None:
            return True

        visited: set[int] = set()
        return self._validate_node(self._min_node, visited)

    def _validate_node(self, node: FibonacciNode[ItemT, PriorityT], visited: set[int]) -> bool:
        if id(node) in visited:
            return True

        visited.add(id(node))

        if node.child is not None:
            child = node.child
            first_child = child
            while True:
                if child.parent != node:
                    return False
                if child.priority < node.priority:
                    return False
                if not self._validate_node(child, visited):
                    return False
                assert child.right is not None
                child = child.right
                if child == first_child:
                    break

        return True

    def __repr__(self) -> str:
        if self.is_empty():
            return f"{self.__class__.__name__}()"

        return f"{self.__class__.__name__}(size={self._size}, min_priority={self.min_priority()!r})"
