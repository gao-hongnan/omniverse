from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import overload

from rich.repr import Result

from ....core.errors import EmptyContainer
from .base import AbstractStack, _Node


class ArrayStack[ItemT](AbstractStack[ItemT]):
    """Stack with underlying data structure being a list. Note in our case the
    top of the stack is the end of the list. So if you push 1, 2, 3, the stack
    will be [1, 2, 3] where 3 is the top of the stack.
    """

    __slots__ = ("_values",)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, iterable: Iterable[ItemT]) -> None: ...

    def __init__(self, iterable: Iterable[ItemT] | None = None) -> None:
        self._values: list[ItemT] = []

        if iterable is not None:
            buffered = list(iterable)
            self._values.extend(buffered)

    @property
    def size(self) -> int:
        return self.__len__()

    def push(self, value: ItemT) -> None:
        self._values.append(value)

    def pop(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("pop from an empty stack")
        return self._values.pop()

    def peek(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("peek from an empty stack")
        return self._values[-1]

    def clear(self) -> None:
        self._values[:] = []

    def __iter__(self) -> Iterator[ItemT]:
        for index in range(len(self._values) - 1, -1, -1):
            yield self._values[index]

    def __len__(self) -> int:
        return self._values.__len__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayStack):
            return NotImplemented
        return self._values == other._values

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        if self.is_empty():
            return f"{self.__class__.__name__}([])"

        values_str = ", ".join(str(value) for value in self._values)
        return f"{self.__class__.__name__}([{values_str}])"

    def __repr__(self) -> str:
        return self.__str__()

    def __rich_repr__(self) -> Result:
        yield "values", self._values


class LinkedListStack[ItemT](AbstractStack[ItemT]):
    """Stack implementation using a singly linked list. The head of the list
    represents the top of the stack.
    """

    __slots__ = ("_head", "_size")

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, iterable: Iterable[ItemT]) -> None: ...

    def __init__(self, iterable: Iterable[ItemT] | None = None) -> None:
        self._head: _Node[ItemT] | None = None
        self._size: int = 0
        if iterable is not None:
            buffered: list[ItemT] = list(iterable)
            for value in buffered:
                self.push(value)

    @property
    def size(self) -> int:
        return self._size

    def push(self, value: ItemT) -> None:
        new_node = _Node[ItemT](value=value, next=self._head)
        self._head = new_node
        self._size += 1

    def pop(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("pop from an empty stack")

        assert self._head is not None
        value = self._head.value
        self._head = self._head.next
        self._size -= 1
        return value

    def peek(self) -> ItemT:
        if self.is_empty():
            raise EmptyContainer("peek from an empty stack")

        assert self._head is not None
        return self._head.value

    def clear(self) -> None:
        self._head = None
        self._size = 0

    def __iter__(self) -> Iterator[ItemT]:
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __len__(self) -> int:
        return self._size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinkedListStack):
            return NotImplemented

        if self._size != other._size:
            return False

        self_current = self._head
        other_current = other._head

        while self_current is not None and other_current is not None:
            if self_current.value != other_current.value:
                return False
            self_current = self_current.next
            other_current = other_current.next

        return True

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        if self.is_empty():
            return f"{self.__class__.__name__}([])"
        values_list_top_to_bottom = list(self)
        values_str = ", ".join(str(value) for value in reversed(values_list_top_to_bottom))
        return f"{self.__class__.__name__}([{values_str}])"

    def __repr__(self) -> str:
        return self.__str__()

    def __rich_repr__(self) -> Result:
        values_list_top_to_bottom = list(self)
        yield "values", list(reversed(values_list_top_to_bottom))
        yield "size", self._size
