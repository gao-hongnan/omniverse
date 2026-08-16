from __future__ import annotations

from typing import Generic

from ...core.errors import EmptyContainer
from ...core.types import ComparableT


class MinMaxStack(Generic[ComparableT]):  # noqa: UP046
    def __init__(self) -> None:
        self._values: list[ComparableT] = []
        self._mins: list[ComparableT] = []
        self._maxs: list[ComparableT] = []

    def push(self, value: ComparableT) -> None:
        self._values.append(value)
        if not self._mins or value <= self._mins[-1]:
            self._mins.append(value)
        else:
            self._mins.append(self._mins[-1])
        if not self._maxs or value >= self._maxs[-1]:
            self._maxs.append(value)
        else:
            self._maxs.append(self._maxs[-1])

    def pop(self) -> ComparableT:
        if not self._values:
            raise EmptyContainer("pop from empty min-max stack")
        self._mins.pop()
        self._maxs.pop()
        return self._values.pop()

    def peek(self) -> ComparableT:
        if not self._values:
            raise EmptyContainer("peek on empty min-max stack")
        return self._values[-1]

    def min(self) -> ComparableT:
        if not self._mins:
            raise EmptyContainer("min on empty min-max stack")
        return self._mins[-1]

    def max(self) -> ComparableT:
        if not self._maxs:
            raise EmptyContainer("max on empty min-max stack")
        return self._maxs[-1]

    def is_empty(self) -> bool:
        return not self._values

    def __len__(self) -> int:
        return len(self._values)
