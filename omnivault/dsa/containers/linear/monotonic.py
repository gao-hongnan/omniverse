from __future__ import annotations

from typing import Generic, Literal

from ...core.errors import EmptyContainer, InvalidConfiguration
from ...core.types import ComparableT

type MonotonicMode = Literal["increasing", "decreasing"]


class MonotonicStack(Generic[ComparableT]):  # noqa: UP046
    def __init__(self, mode: MonotonicMode) -> None:
        if mode not in ("increasing", "decreasing"):
            raise InvalidConfiguration(f"Unknown mode: {mode}")
        self._mode: MonotonicMode = mode
        self._values: list[ComparableT] = []
        self._indices: list[int] = []

    def _violates(self, top: ComparableT, incoming: ComparableT) -> bool:
        if self._mode == "increasing":
            return top > incoming
        return top < incoming

    def push(self, value: ComparableT) -> None:
        while self._values and self._violates(self._values[-1], value):
            self._values.pop()
        self._values.append(value)

    def push_index(self, index: int, value: ComparableT) -> None:
        while self._values and self._violates(self._values[-1], value):
            self._values.pop()
            if self._indices:
                self._indices.pop()
        self._values.append(value)
        self._indices.append(index)

    def pop(self) -> ComparableT:
        if not self._values:
            raise EmptyContainer("pop from empty monotonic stack")
        if self._indices:
            self._indices.pop()
        return self._values.pop()

    def pop_index(self) -> int:
        if not self._indices:
            raise EmptyContainer("pop_index from empty monotonic stack")
        self._values.pop()
        return self._indices.pop()

    def peek(self) -> ComparableT:
        if not self._values:
            raise EmptyContainer("peek on empty monotonic stack")
        return self._values[-1]

    def peek_index(self) -> int:
        if not self._indices:
            raise EmptyContainer("peek_index on empty monotonic stack")
        return self._indices[-1]

    def is_empty(self) -> bool:
        return not self._values

    def __len__(self) -> int:
        return len(self._values)


class MonotonicDeque(Generic[ComparableT]):  # noqa: UP046
    def __init__(self, mode: MonotonicMode) -> None:
        if mode not in ("increasing", "decreasing"):
            raise InvalidConfiguration(f"Unknown mode: {mode}")
        self._mode: MonotonicMode = mode
        self._buffer: list[tuple[int, ComparableT]] = []

    def _violates(self, back: ComparableT, incoming: ComparableT) -> bool:
        if self._mode == "increasing":
            return back > incoming
        return back < incoming

    def push(self, index: int, value: ComparableT) -> None:
        while self._buffer and self._violates(self._buffer[-1][1], value):
            self._buffer.pop()
        self._buffer.append((index, value))

    def pop_front(self) -> tuple[int, ComparableT]:
        if not self._buffer:
            raise EmptyContainer("pop_front from empty monotonic deque")
        return self._buffer.pop(0)

    def front_index(self) -> int:
        if not self._buffer:
            raise EmptyContainer("front_index on empty monotonic deque")
        return self._buffer[0][0]

    def front_value(self) -> ComparableT:
        if not self._buffer:
            raise EmptyContainer("front_value on empty monotonic deque")
        return self._buffer[0][1]

    def is_empty(self) -> bool:
        return not self._buffer

    def __len__(self) -> int:
        return len(self._buffer)
