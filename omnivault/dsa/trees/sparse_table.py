from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic

from ..core.errors import InvalidConfiguration
from ..core.types import ItemT


class SparseTable(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_log", "_n", "_op", "_table")

    def __init__(self, arr: Sequence[ItemT], op: Callable[[ItemT, ItemT], ItemT]) -> None:
        if len(arr) == 0:
            raise InvalidConfiguration("SparseTable requires a non-empty sequence")
        self._n = len(arr)
        self._op = op
        self._log: list[int] = [0] * (self._n + 1)
        for i in range(2, self._n + 1):
            self._log[i] = self._log[i // 2] + 1
        k_max = self._log[self._n] + 1
        self._table: list[list[ItemT]] = [list(arr)]
        for k in range(1, k_max):
            row_len = self._n - (1 << k) + 1
            if row_len <= 0:
                break
            prev = self._table[k - 1]
            offset = 1 << (k - 1)
            self._table.append([op(prev[i], prev[i + offset]) for i in range(row_len)])

    def query(self, left: int, right: int) -> ItemT:
        if left < 0 or right >= self._n or left > right:
            raise InvalidConfiguration(f"Invalid range [{left}, {right}] for length {self._n}")
        length = right - left + 1
        k = self._log[length]
        offset = 1 << k
        left_block = self._table[k][left]
        right_block = self._table[k][right - offset + 1]
        return self._op(left_block, right_block)
