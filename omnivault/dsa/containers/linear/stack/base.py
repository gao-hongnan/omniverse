from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from pydantic import BaseModel


# ABC retained: shared default methods (is_empty, __bool__) per rules/python-typings.md exception clause
class AbstractStack[ItemT](ABC):
    """Stack inferface with LIFO/FILO semantics."""

    @property
    @abstractmethod
    def size(self) -> int: ...

    def is_empty(self) -> bool:
        return self.__len__() == 0

    @abstractmethod
    def push(self, value: ItemT) -> None: ...

    @abstractmethod
    def pop(self) -> ItemT: ...

    @abstractmethod
    def peek(self) -> ItemT: ...

    @abstractmethod
    def __iter__(self) -> Iterator[ItemT]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    def __bool__(self) -> bool:
        return not self.is_empty()


class _Node[ItemT](BaseModel):
    """Represents a node in the singly linked list."""

    value: ItemT
    next: _Node[ItemT] | None = None
