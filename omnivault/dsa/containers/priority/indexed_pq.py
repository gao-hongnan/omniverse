from __future__ import annotations

from typing import Generic

from ...core.errors import EmptyContainer, InvalidConfiguration, KeyNotFound
from ...core.types import KeyT, PriorityT


class IndexedPriorityQueue(Generic[KeyT, PriorityT]):  # noqa: UP046
    __slots__ = ("_heap", "_position")

    def __init__(self) -> None:
        self._heap: list[tuple[KeyT, PriorityT]] = []
        self._position: dict[KeyT, int] = {}

    def __len__(self) -> int:
        return len(self._heap)

    def __contains__(self, key: object) -> bool:
        return key in self._position

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def insert(self, key: KeyT, priority: PriorityT) -> None:
        if key in self._position:
            raise InvalidConfiguration(f"Key {key!r} already present in IndexedPriorityQueue")
        self._heap.append((key, priority))
        idx = len(self._heap) - 1
        self._position[key] = idx
        self._sift_up(idx)

    def decrease_key(self, key: KeyT, new_priority: PriorityT) -> None:
        if key not in self._position:
            raise KeyNotFound(f"Key {key!r} not in IndexedPriorityQueue")
        idx = self._position[key]
        current_priority = self._heap[idx][1]
        if new_priority > current_priority:
            raise InvalidConfiguration(
                f"decrease_key requires new_priority <= current; got {new_priority!r} > {current_priority!r}"
            )
        self._heap[idx] = (key, new_priority)
        self._sift_up(idx)

    def pop_min(self) -> tuple[KeyT, PriorityT]:
        if not self._heap:
            raise EmptyContainer("pop_min from empty IndexedPriorityQueue")
        top = self._heap[0]
        last = self._heap.pop()
        del self._position[top[0]]
        if self._heap:
            self._heap[0] = last
            self._position[last[0]] = 0
            self._sift_down(0)
        return top

    def peek_min(self) -> tuple[KeyT, PriorityT]:
        if not self._heap:
            raise EmptyContainer("peek_min from empty IndexedPriorityQueue")
        return self._heap[0]

    def _sift_up(self, idx: int) -> None:
        while idx > 0:
            parent = (idx - 1) // 2
            if self._heap[idx][1] < self._heap[parent][1]:
                self._swap(idx, parent)
                idx = parent
            else:
                break

    def _sift_down(self, idx: int) -> None:
        n = len(self._heap)
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx
            if left < n and self._heap[left][1] < self._heap[smallest][1]:
                smallest = left
            if right < n and self._heap[right][1] < self._heap[smallest][1]:
                smallest = right
            if smallest == idx:
                break
            self._swap(idx, smallest)
            idx = smallest

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._position[self._heap[i][0]] = i
        self._position[self._heap[j][0]] = j
