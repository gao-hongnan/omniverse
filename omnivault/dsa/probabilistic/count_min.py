from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from typing import Generic

from ..core.errors import IncompatibleSketch, InvalidProbability
from ..core.types import ItemT


class CountMinSketch(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_table", "_width", "_depth", "_total_count")

    def __init__(self, width: int, depth: int) -> None:
        if width <= 0 or depth <= 0:
            raise InvalidProbability("Width and depth must be positive")

        self._width = width
        self._depth = depth
        self._table = [[0] * width for _ in range(depth)]
        self._total_count = 0

    @classmethod
    def from_error_params(cls, epsilon: float, delta: float) -> CountMinSketch[ItemT]:
        if not 0 < epsilon < 1 or not 0 < delta < 1:
            raise InvalidProbability("Epsilon and delta must be between 0 and 1")

        width = int(math.ceil(math.e / epsilon))
        depth = int(math.ceil(math.log(1 / delta)))

        return cls(width, depth)

    def _hash_functions(self, item: ItemT) -> Iterator[int]:
        item_str = str(item)

        for i in range(self._depth):
            hash_input = f"{item_str}_{i}".encode()
            hash_value = int(hashlib.md5(hash_input, usedforsecurity=False).hexdigest(), 16)
            yield hash_value % self._width

    def add(self, item: ItemT, count: int = 1) -> None:
        if count < 0:
            raise InvalidProbability("Count must be non-negative")

        for i, hash_value in enumerate(self._hash_functions(item)):
            self._table[i][hash_value] += count

        self._total_count += count

    def estimate(self, item: ItemT) -> int:
        estimates = []

        for i, hash_value in enumerate(self._hash_functions(item)):
            estimates.append(self._table[i][hash_value])

        return min(estimates) if estimates else 0

    def remove(self, item: ItemT, count: int = 1) -> None:
        if count < 0:
            raise InvalidProbability("Count must be non-negative")

        current_estimate = self.estimate(item)
        actual_removal = min(count, current_estimate)

        for i, hash_value in enumerate(self._hash_functions(item)):
            self._table[i][hash_value] = max(0, self._table[i][hash_value] - actual_removal)

        self._total_count = max(0, self._total_count - actual_removal)

    @property
    def width(self) -> int:
        return self._width

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def total_count(self) -> int:
        return self._total_count

    def merge(self, other: CountMinSketch[ItemT]) -> CountMinSketch[ItemT]:
        if self._width != other._width or self._depth != other._depth:
            raise IncompatibleSketch("Sketches must have same dimensions")

        result: CountMinSketch[ItemT] = CountMinSketch(self._width, self._depth)

        for i in range(self._depth):
            for j in range(self._width):
                result._table[i][j] = self._table[i][j] + other._table[i][j]

        result._total_count = self._total_count + other._total_count

        return result

    def clear(self) -> None:
        self._table = [[0] * self._width for _ in range(self._depth)]
        self._total_count = 0

    def is_empty(self) -> bool:
        return self._total_count == 0

    def memory_usage(self) -> int:
        return self._width * self._depth * 8

    def __len__(self) -> int:
        return self._total_count

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(width={self._width}, depth={self._depth}, total_count={self._total_count})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CountMinSketch):
            return NotImplemented

        return self._width == other._width and self._depth == other._depth and self._table == other._table

    def __add__(self, other: CountMinSketch[ItemT]) -> CountMinSketch[ItemT]:
        return self.merge(other)
