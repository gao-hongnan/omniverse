from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


# ABC retained: shared default methods (__bool__, __repr__, extend) per rules/python-typings.md exception clause
class AbstractQueue[ItemT](ABC):
    @abstractmethod
    def __init__(self, items: list[ItemT] | None = None) -> None: ...

    @abstractmethod
    def enqueue(self, item: ItemT) -> None: ...

    @abstractmethod
    def dequeue(self) -> ItemT: ...

    @abstractmethod
    def peek(self) -> ItemT: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[ItemT]: ...

    @abstractmethod
    def __contains__(self, item: object) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def to_list(self) -> list[ItemT]: ...

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        items = ", ".join(repr(item) for item in self.to_list())
        return f"{self.__class__.__name__}([{items}])"

    def extend(self, items: Iterable[ItemT]) -> None:
        for item in items:
            self.enqueue(item)
