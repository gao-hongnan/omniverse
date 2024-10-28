from __future__ import annotations

from collections.abc import MutableSequence
from typing import Generic, Iterable, SupportsIndex, List, overload

from omnivault._types._generic import T

# see `_collections_abc.py` for the definition of `MutableSequence`.


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

    def __getitem__(self, index: SupportsIndex) -> T:
        return self._data[index]

    def __setitem__(self, index: SupportsIndex, value: T) -> None:
        self._data[index] = value

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
