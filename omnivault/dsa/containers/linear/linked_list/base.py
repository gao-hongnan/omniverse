from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class LinkedNode[ItemT](Protocol):
    value: ItemT


class SinglyNode[ItemT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: ItemT
    next: SinglyNode[ItemT] | None = None


class DoublyNode[ItemT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: ItemT
    next: DoublyNode[ItemT] | None = None
    prev: DoublyNode[ItemT] | None = None


@runtime_checkable
class LinkedListProtocol[ItemT](Protocol):
    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[ItemT]: ...

    def append(self, value: ItemT) -> None: ...

    def prepend(self, value: ItemT) -> None: ...

    def remove(self, value: ItemT) -> bool: ...

    def clear(self) -> None: ...


# ABC retained: shared default methods (__init__, clear, is_empty, size, __len__, __bool__, __repr__, head) per rules/python-typings.md exception clause
class AbstractLinkedList[ItemT, NodeT: LinkedNode[Any]](ABC):
    def __init__(self, values: Sequence[ItemT] | None = None) -> None:
        self._head: NodeT | None = None
        self._size: int = 0
        self._values = values

        if values is not None:
            self._initialize_from_values()

    @abstractmethod
    def append(self, value: ItemT) -> None: ...

    @abstractmethod
    def prepend(self, value: ItemT) -> None: ...

    @abstractmethod
    def remove(self, value: ItemT) -> bool: ...

    @abstractmethod
    def traverse(self) -> str: ...

    @abstractmethod
    def _initialize_from_values(self) -> None: ...

    def clear(self) -> None:
        self._head = None
        self._size = 0

    def is_empty(self) -> bool:
        return self._head is None

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return bool(self._size)

    def __repr__(self) -> str:
        return self.traverse()

    @abstractmethod
    def __iter__(self) -> Iterator[ItemT]: ...

    @property
    def head(self) -> NodeT | None:
        return self._head
