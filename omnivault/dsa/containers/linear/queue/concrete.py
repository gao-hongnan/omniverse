from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from rich.repr import Result

from ....core.errors import EmptyContainer
from .base import AbstractQueue

if TYPE_CHECKING:
    from collections.abc import Iterator


class ArrayQueue[ItemT](AbstractQueue[ItemT]):
    def __init__(self, items: list[ItemT] | None = None) -> None:
        self._items: list[ItemT] = [] if items is None else items.copy()
        self._front = 0

    def enqueue(self, item: ItemT) -> None:
        self._items.append(item)

    def dequeue(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("dequeue from empty queue")

        item = self._items[self._front]
        self._front += 1

        if self._front > len(self._items) // 2:
            self._items = self._items[self._front :]
            self._front = 0

        return item

    def peek(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("peek from empty queue")
        return self._items[self._front]

    def is_empty(self) -> bool:
        return self._front >= len(self._items)

    def __len__(self) -> int:
        return len(self._items) - self._front

    def __iter__(self) -> Iterator[ItemT]:
        return iter(self._items[self._front :])

    def __contains__(self, item: object) -> bool:
        return item in self._items[self._front :]

    def clear(self) -> None:
        self._items.clear()
        self._front = 0

    def to_list(self) -> list[ItemT]:
        return self._items[self._front :].copy()

    def __rich_repr__(self) -> Result:
        yield from self.to_list()


class _QueueNode[ItemT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: ItemT
    next: _QueueNode[ItemT] | None = None


class LinkedListQueue[ItemT](AbstractQueue[ItemT]):
    def __init__(self, items: list[ItemT] | None = None) -> None:
        self._front: _QueueNode[ItemT] | None = None
        self._rear: _QueueNode[ItemT] | None = None
        self._size = 0

        if items:
            self.extend(items)

    def enqueue(self, item: ItemT) -> None:
        node = _QueueNode[ItemT](value=item)

        if self._rear is None:
            self._front = self._rear = node
        else:
            self._rear.next = node
            self._rear = node

        self._size += 1

    def dequeue(self) -> ItemT:
        if self._front is None:
            raise EmptyContainer("dequeue from empty queue")

        item = self._front.value
        self._front = self._front.next

        if self._front is None:
            self._rear = None

        self._size -= 1
        return item

    def peek(self) -> ItemT:
        if self._front is None:
            raise EmptyContainer("peek from empty queue")
        return self._front.value

    def is_empty(self) -> bool:
        return self._front is None

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[ItemT]:
        current = self._front
        while current is not None:
            yield current.value
            current = current.next

    def __contains__(self, item: object) -> bool:
        return any(value == item for value in self)

    def clear(self) -> None:
        self._front = self._rear = None
        self._size = 0

    def to_list(self) -> list[ItemT]:
        return list(self)

    def __rich_repr__(self) -> Result:
        yield from self
