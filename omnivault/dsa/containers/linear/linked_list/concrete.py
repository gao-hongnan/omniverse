from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from .base import AbstractLinkedList, DoublyNode, SinglyNode

if TYPE_CHECKING:
    from collections.abc import Sequence


class SinglyLinkedList[ItemT](AbstractLinkedList[ItemT, SinglyNode[ItemT]]):
    def _initialize_from_values(self) -> None:
        assert self._values is not None
        for value in self._values:
            self.append(value)

    def traverse(self) -> str:
        temp_node = self._head
        result_parts = []
        while temp_node is not None:
            result_parts.append(str(temp_node.value))
            result_parts.append(" -> ")
            temp_node = temp_node.next
        if result_parts:
            result_parts[-1] = " -> None"
        else:
            result_parts.append("None")
        return "".join(result_parts)

    def append(self, value: ItemT) -> None:
        new_node = SinglyNode[ItemT](value=value)
        if not self._head:
            self._head = new_node
            self._size += 1
            return

        current = self._head
        while current.next:
            current = current.next
        current.next = new_node
        self._size += 1

    def prepend(self, value: ItemT) -> None:
        new_node = SinglyNode[ItemT](value=value)
        new_node.next = self._head
        self._head = new_node
        self._size += 1

    def remove(self, value: ItemT) -> bool:
        if not self._head:
            return False

        if self._head.value == value:
            self._head = self._head.next
            self._size -= 1
            return True

        current = self._head
        while current.next:
            if current.next.value == value:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        return False

    def __iter__(self) -> Iterator[ItemT]:
        current = self._head
        while current:
            yield current.value
            current = current.next


class DoublyLinkedList[ItemT](AbstractLinkedList[ItemT, DoublyNode[ItemT]]):
    def __init__(self, values: Sequence[ItemT] | None = None) -> None:
        self._tail: DoublyNode[ItemT] | None = None
        super().__init__(values)

    def _initialize_from_values(self) -> None:
        assert self._values is not None
        for value in self._values:
            self.append(value)

    def traverse(self) -> str:
        temp_node = self._head
        result_parts = []
        while temp_node is not None:
            result_parts.append(str(temp_node.value))
            result_parts.append(" <-> ")
            temp_node = temp_node.next
        if result_parts:
            result_parts[-1] = " <-> None"
        else:
            result_parts.append("None")
        return "".join(result_parts)

    def append(self, value: ItemT) -> None:
        new_node = DoublyNode[ItemT](value=value)
        self._size += 1

        if not self._head:
            self._head = self._tail = new_node
            return

        assert self._tail is not None
        new_node.prev = self._tail
        self._tail.next = new_node
        self._tail = new_node

    def prepend(self, value: ItemT) -> None:
        new_node = DoublyNode[ItemT](value=value)
        self._size += 1

        if not self._head:
            self._head = self._tail = new_node
            return

        new_node.next = self._head
        self._head.prev = new_node
        self._head = new_node

    def remove(self, value: ItemT) -> bool:
        current = self._head
        while current:
            if current.value == value:
                if current.prev is None:
                    self._head = current.next
                else:
                    current.prev.next = current.next

                if current.next is None:
                    self._tail = current.prev
                else:
                    current.next.prev = current.prev

                self._size -= 1
                return True
            current = current.next
        return False

    def clear(self) -> None:
        super().clear()
        self._tail = None

    def __iter__(self) -> Iterator[ItemT]:
        current = self._head
        while current:
            yield current.value
            current = current.next

    def __reversed__(self) -> Iterator[ItemT]:
        current = self._tail
        while current:
            yield current.value
            current = current.prev

    @property
    def tail(self) -> DoublyNode[ItemT] | None:
        return self._tail
