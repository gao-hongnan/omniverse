from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


# ABC retained: shared default methods (__bool__, __repr__, extend_front, extend_rear, append, appendleft, pop, popleft) per rules/python-typings.md exception clause
class AbstractDeque[ItemT](ABC):
    @abstractmethod
    def __init__(self, items: list[ItemT] | None = None, *, maxlen: int | None = None) -> None: ...

    @abstractmethod
    def add_front(self, item: ItemT) -> None: ...

    @abstractmethod
    def add_rear(self, item: ItemT) -> None: ...

    @abstractmethod
    def remove_front(self) -> ItemT: ...

    @abstractmethod
    def remove_rear(self) -> ItemT: ...

    @abstractmethod
    def peek_front(self) -> ItemT: ...

    @abstractmethod
    def peek_rear(self) -> ItemT: ...

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

    def extend_front(self, items: Iterable[ItemT]) -> None:
        for item in items:
            self.add_front(item)

    def extend_rear(self, items: Iterable[ItemT]) -> None:
        for item in items:
            self.add_rear(item)

    def append(self, item: ItemT) -> None:
        self.add_rear(item)

    def appendleft(self, item: ItemT) -> None:
        self.add_front(item)

    def pop(self) -> ItemT:
        return self.remove_rear()

    def popleft(self) -> ItemT:
        return self.remove_front()
