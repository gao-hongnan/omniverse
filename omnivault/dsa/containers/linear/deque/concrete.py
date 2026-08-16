from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from rich.repr import Result

from ....core.errors import EmptyContainer
from .base import AbstractDeque

if TYPE_CHECKING:
    from collections.abc import Iterator


class ArrayDeque[ItemT](AbstractDeque[ItemT]):
    def __init__(self, items: list[ItemT] | None = None, *, maxlen: int | None = None) -> None:
        self._items: list[ItemT] = [] if items is None else items.copy()
        self._maxlen = maxlen

        if maxlen is not None and len(self._items) > maxlen:
            self._items = self._items[-maxlen:]

    def add_front(self, item: ItemT) -> None:
        if self._maxlen is not None and len(self._items) >= self._maxlen:
            if self._maxlen == 0:
                return
            self._items.pop()

        self._items.insert(0, item)

    def add_rear(self, item: ItemT) -> None:
        if self._maxlen is not None and len(self._items) >= self._maxlen:
            if self._maxlen == 0:
                return
            self._items.pop(0)

        self._items.append(item)

    def remove_front(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("remove from empty deque")
        return self._items.pop(0)

    def remove_rear(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("remove from empty deque")
        return self._items.pop()

    def peek_front(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("peek from empty deque")
        return self._items[0]

    def peek_rear(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("peek from empty deque")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ItemT]:
        return iter(self._items)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def clear(self) -> None:
        self._items.clear()

    def to_list(self) -> list[ItemT]:
        return self._items.copy()

    def __rich_repr__(self) -> Result:
        yield from self._items

    def reverse(self) -> None:
        self._items.reverse()

    def rotate(self, n: int = 1) -> None:
        if not self._items:
            return

        length = len(self._items)
        n = n % length

        if n == 0:
            return

        self._items = self._items[-n:] + self._items[:-n]


class _DequeNode[ItemT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: ItemT
    next: _DequeNode[ItemT] | None = None
    prev: _DequeNode[ItemT] | None = None


class LinkedListDeque[ItemT](AbstractDeque[ItemT]):
    def __init__(self, items: list[ItemT] | None = None, *, maxlen: int | None = None) -> None:
        self._front: _DequeNode[ItemT] | None = None
        self._rear: _DequeNode[ItemT] | None = None
        self._size = 0
        self._maxlen = maxlen

        if items:
            for item in items:
                self.add_rear(item)

    def add_front(self, item: ItemT) -> None:
        if self._maxlen is not None and self._size >= self._maxlen:
            if self._maxlen == 0:
                return
            self.remove_rear()

        node = _DequeNode[ItemT](value=item)

        if self._front is None:
            self._front = self._rear = node
        else:
            node.next = self._front
            self._front.prev = node
            self._front = node

        self._size += 1

    def add_rear(self, item: ItemT) -> None:
        if self._maxlen is not None and self._size >= self._maxlen:
            if self._maxlen == 0:
                return
            self.remove_front()

        node = _DequeNode[ItemT](value=item)

        if self._rear is None:
            self._front = self._rear = node
        else:
            node.prev = self._rear
            self._rear.next = node
            self._rear = node

        self._size += 1

    def remove_front(self) -> ItemT:
        if self._front is None:
            raise EmptyContainer("remove from empty deque")

        item = self._front.value
        self._front = self._front.next

        if self._front is None:
            self._rear = None
        else:
            self._front.prev = None

        self._size -= 1
        return item

    def remove_rear(self) -> ItemT:
        if self._rear is None:
            raise EmptyContainer("remove from empty deque")

        item = self._rear.value
        self._rear = self._rear.prev

        if self._rear is None:
            self._front = None
        else:
            self._rear.next = None

        self._size -= 1
        return item

    def peek_front(self) -> ItemT:
        if self._front is None:
            raise EmptyContainer("peek from empty deque")
        return self._front.value

    def peek_rear(self) -> ItemT:
        if self._rear is None:
            raise EmptyContainer("peek from empty deque")
        return self._rear.value

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

    def reverse(self) -> None:
        if self._size <= 1:
            return

        current = self._front
        self._front, self._rear = self._rear, self._front

        while current is not None:
            current.next, current.prev = current.prev, current.next
            current = current.prev

    def rotate(self, n: int = 1) -> None:
        if self._size <= 1:
            return

        n = n % self._size

        if n == 0:
            return

        for _ in range(n):
            item = self.remove_rear()
            self.add_front(item)
