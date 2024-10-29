from __future__ import annotations

from collections.abc import MutableSequence
from typing import Generic, Iterable, List, cast, overload

from omnivault._types._generic import T

# see `_collections_abc.py` for the definition of `MutableSequence`.
# https://github.com/python/cpython/blob/3.7/Modules/_collectionsmodule.c


class Deque(MutableSequence[T], Generic[T]):
    def __init__(self, iterable: Iterable[T] | None = None, maxlen: int | None = None):
        self.maxlen = maxlen
        self._data = list(iterable) if iterable is not None else []

        if self.maxlen is not None and len(self._data) > self.maxlen:
            self._data = self._data[-self.maxlen :]

    def append(self, item: T) -> None:
        if self.maxlen is None or len(self._data) < self.maxlen:
            self._data.append(item)
        else:
            self._data.pop(0)
            self._data.append(item)

    def appendleft(self, item: T) -> None:
        if self.maxlen is None or len(self._data) < self.maxlen:
            self._data.insert(0, item)
        else:
            self._data.pop()
            self._data.insert(0, item)

    def pop(self, index: int = -1) -> T:
        return self._data.pop(index)

    def popleft(self) -> T:
        return self._data.pop(0)

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[T]:
        ...

    def __getitem__(self, index: int | slice) -> T | List[T]:
        result = self._data[index]
        if isinstance(index, slice):
            return list(result) if isinstance(result, Iterable) else [result]
        return result

    @overload
    def __setitem__(self, index: int, value: T) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None:
        ...

    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        if isinstance(index, slice):
            self._data[index] = list(value) if hasattr(value, "__iter__") else [value]
        else:
            self._data[index] = cast(T, value)

    def __delitem__(self, index: int | slice) -> None:
        del self._data[index]

    def insert(self, index: int, value: T) -> None:
        if self.maxlen is None or len(self._data) < self.maxlen:
            self._data.insert(index, value)
        else:
            raise OverflowError("Deque is at its maximum size")


my_deque = Deque([1, 2, 3], maxlen=3)
my_deque.append(4)
print(my_deque)  # Output might be [2, 3, 4] if maxlen is reached
