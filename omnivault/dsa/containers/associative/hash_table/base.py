from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterator


# ABC retained: shared default methods (update, get_default, pop, __contains__, __getitem__, __setitem__, __delitem__, __repr__, __bool__) per rules/python-typings.md exception clause
class AbstractHashTable[KeyT, ValueT](ABC):
    @abstractmethod
    def put(self, key: KeyT, value: ValueT) -> None: ...

    @abstractmethod
    def get(self, key: KeyT) -> ValueT: ...

    @abstractmethod
    def remove(self, key: KeyT) -> ValueT: ...

    @abstractmethod
    def contains_key(self, key: KeyT) -> bool: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[KeyT]: ...

    @abstractmethod
    def keys(self) -> Iterator[KeyT]: ...

    @abstractmethod
    def values(self) -> Iterator[ValueT]: ...

    @abstractmethod
    def items(self) -> Iterator[tuple[KeyT, ValueT]]: ...

    @abstractmethod
    def clear(self) -> None: ...

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        items = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in self.items())
        return f"{self.__class__.__name__}({{{items}}})"

    def __getitem__(self, key: KeyT) -> ValueT:
        return self.get(key)

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        self.put(key, value)

    def __delitem__(self, key: KeyT) -> None:
        self.remove(key)

    def __contains__(self, key: object) -> bool:
        try:
            return self.contains_key(cast(Any, key))
        except TypeError, KeyError:
            return False

    def update(self, other: AbstractHashTable[KeyT, ValueT] | dict[KeyT, ValueT]) -> None:
        if isinstance(other, AbstractHashTable):
            for key, value in other.items():
                self.put(key, value)
        else:
            for key, value in other.items():
                self.put(key, value)

    def get_default(self, key: KeyT, default: ValueT) -> ValueT:
        try:
            return self.get(key)
        except KeyError:
            return default

    def pop(self, key: KeyT, default: ValueT | None = None) -> ValueT:
        try:
            return self.remove(key)
        except KeyError:
            if default is not None:
                return default
            raise
