from __future__ import annotations

import heapq
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from rich.repr import Result

from ....core.errors import EmptyContainer, InvalidConfiguration
from ....core.types import Comparable
from .base import AbstractPriorityQueue

if TYPE_CHECKING:
    from collections.abc import Iterator


class HeapType(Enum):
    MIN = "min"
    MAX = "max"


class PriorityQueueItem[ItemT, PriorityT: Comparable](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: ItemT
    priority: PriorityT
    insertion_order: int = 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PriorityQueueItem):
            return NotImplemented

        if self.priority != other.priority:
            return bool(self.priority < other.priority)
        return self.insertion_order < other.insertion_order

    def __le__(self, other: object) -> bool:
        if not isinstance(other, PriorityQueueItem):
            return NotImplemented
        return self < other or self == other

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, PriorityQueueItem):
            return NotImplemented
        return not self <= other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, PriorityQueueItem):
            return NotImplemented
        return not self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PriorityQueueItem):
            return NotImplemented
        return self.priority == other.priority and self.insertion_order == other.insertion_order


class _MaxHeapItem[ItemT, PriorityT: Comparable](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: PriorityQueueItem[ItemT, PriorityT]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _MaxHeapItem):
            return NotImplemented
        return self.item > other.item

    def __le__(self, other: object) -> bool:
        if not isinstance(other, _MaxHeapItem):
            return NotImplemented
        return self.item >= other.item

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _MaxHeapItem):
            return NotImplemented
        return self.item < other.item

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, _MaxHeapItem):
            return NotImplemented
        return self.item <= other.item

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaxHeapItem):
            return NotImplemented
        return self.item == other.item


class BinaryHeapPriorityQueue[ItemT, PriorityT: Comparable](AbstractPriorityQueue[ItemT, PriorityT]):
    def __init__(self, *, heap_type: HeapType = HeapType.MIN) -> None:
        self._heap: list[PriorityQueueItem[ItemT, PriorityT] | _MaxHeapItem[ItemT, PriorityT]] = []
        self._heap_type = heap_type
        self._insertion_counter = 0

    def enqueue(self, item: ItemT, priority: PriorityT) -> None:
        queue_item = PriorityQueueItem[ItemT, PriorityT](
            value=item, priority=priority, insertion_order=self._insertion_counter
        )
        self._insertion_counter += 1

        if self._heap_type == HeapType.MIN:
            heapq.heappush(self._heap, queue_item)
        else:
            heapq.heappush(self._heap, _MaxHeapItem[ItemT, PriorityT](item=queue_item))

    def dequeue(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("dequeue from empty priority queue")

        if self._heap_type == HeapType.MIN:
            item = heapq.heappop(self._heap)
            if isinstance(item, PriorityQueueItem):
                return item.value
        else:
            max_item = heapq.heappop(self._heap)
            if isinstance(max_item, _MaxHeapItem):
                return max_item.item.value

        raise TypeError("Unexpected item type in heap")

    def peek(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("peek from empty priority queue")

        if self._heap_type == HeapType.MIN:
            item = self._heap[0]
            if isinstance(item, PriorityQueueItem):
                return item.value
        else:
            max_item = self._heap[0]
            if isinstance(max_item, _MaxHeapItem):
                return max_item.item.value

        raise TypeError("Unexpected item type in heap")

    def peek_priority(self) -> PriorityT:
        if self.is_empty():
            raise EmptyContainer("peek from empty priority queue")

        if self._heap_type == HeapType.MIN:
            item = self._heap[0]
            if isinstance(item, PriorityQueueItem):
                return item.priority
        else:
            max_item = self._heap[0]
            if isinstance(max_item, _MaxHeapItem):
                return max_item.item.priority

        raise TypeError("Unexpected item type in heap")

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[ItemT]:
        for heap_item in self._heap:
            if self._heap_type == HeapType.MIN and isinstance(heap_item, PriorityQueueItem):
                yield heap_item.value
            elif self._heap_type == HeapType.MAX and isinstance(heap_item, _MaxHeapItem):
                yield heap_item.item.value

    def __contains__(self, item: object) -> bool:
        return any(value == item for value in self)

    def clear(self) -> None:
        self._heap.clear()
        self._insertion_counter = 0

    def to_list(self) -> list[ItemT]:
        return list(self)

    def __rich_repr__(self) -> Result:
        yield from self

    def change_priority(self, item: ItemT, new_priority: PriorityT) -> bool:
        for i, heap_item in enumerate(self._heap):
            current_item: PriorityQueueItem[ItemT, PriorityT] | None = None

            if self._heap_type == HeapType.MIN and isinstance(heap_item, PriorityQueueItem):
                current_item = heap_item
            elif self._heap_type == HeapType.MAX and isinstance(heap_item, _MaxHeapItem):
                current_item = heap_item.item

            if current_item and current_item.value == item:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                self.enqueue(item, new_priority)
                return True

        return False

    def merge(self, other: BinaryHeapPriorityQueue[ItemT, PriorityT]) -> None:
        if self._heap_type != other._heap_type:
            raise InvalidConfiguration("Cannot merge priority queues with different heap types")

        for heap_item in other._heap:
            if self._heap_type == HeapType.MIN and isinstance(heap_item, PriorityQueueItem):
                self.enqueue(heap_item.value, heap_item.priority)
            elif self._heap_type == HeapType.MAX and isinstance(heap_item, _MaxHeapItem):
                self.enqueue(heap_item.item.value, heap_item.item.priority)
