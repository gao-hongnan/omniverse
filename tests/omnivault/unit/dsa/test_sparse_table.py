from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from omnivault.dsa.core.errors import InvalidConfiguration
from omnivault.dsa.trees.sparse_table import SparseTable


class TestSparseTable:
    @pytest.mark.unit
    def test_min_query(self) -> None:
        table = SparseTable([1, 3, 2, 7, 9, 11, 3, 4], op=min)
        assert table.query(2, 5) == 2

    @pytest.mark.unit
    def test_max_query(self) -> None:
        table = SparseTable([1, 3, 2, 7, 9, 11, 3, 4], op=max)
        assert table.query(0, 7) == 11

    @pytest.mark.unit
    def test_gcd_query_non_idempotent(self) -> None:
        table = SparseTable([12, 18, 24, 30, 36], op=math.gcd)
        assert table.query(0, 4) == 6
        assert table.query(1, 3) == 6

    @pytest.mark.unit
    def test_single_element_query(self) -> None:
        table = SparseTable([5], op=min)
        assert table.query(0, 0) == 5

    @pytest.mark.unit
    @pytest.mark.parametrize(("left", "right"), [(-1, 2), (0, 10), (3, 1)])
    def test_invalid_range_raises(self, left: int, right: int) -> None:
        table: SparseTable[int] = SparseTable([1, 2, 3, 4, 5], op=min)
        with pytest.raises(InvalidConfiguration):
            table.query(left, right)

    @pytest.mark.unit
    def test_empty_sequence_raises(self) -> None:
        op: Callable[[int, int], int] = min
        with pytest.raises(InvalidConfiguration):
            SparseTable([], op=op)
