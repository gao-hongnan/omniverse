from __future__ import annotations

import pytest

from omnivault.dsa.algorithms.combinatorics import (
    binomial,
    catalan,
    catalan_sequence,
    pascals_triangle,
    permutations_count,
)
from omnivault.dsa.core.errors import InvalidConfiguration


class TestBinomial:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("n", "k", "expected"),
        [(5, 2, 10), (10, 0, 1), (10, 10, 1), (6, 3, 20), (7, 4, 35), (0, 0, 1)],
    )
    def test_binomial_known_values(self, n: int, k: int, expected: int) -> None:
        assert binomial(n, k) == expected

    @pytest.mark.unit
    def test_binomial_k_greater_than_n_is_zero(self) -> None:
        assert binomial(3, 5) == 0

    @pytest.mark.unit
    @pytest.mark.parametrize(("n", "k"), [(-1, 0), (5, -1), (-2, -3)])
    def test_binomial_negative_raises(self, n: int, k: int) -> None:
        with pytest.raises(InvalidConfiguration):
            binomial(n, k)


class TestPermutationsCount:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("n", "k", "expected"),
        [(5, 2, 20), (5, 0, 1), (5, 5, 120), (6, 3, 120)],
    )
    def test_permutations_count(self, n: int, k: int, expected: int) -> None:
        assert permutations_count(n, k) == expected

    @pytest.mark.unit
    def test_permutations_count_k_gt_n_zero(self) -> None:
        assert permutations_count(3, 5) == 0

    @pytest.mark.unit
    def test_permutations_count_negative_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            permutations_count(-1, 2)


class TestPascalsTriangle:
    @pytest.mark.unit
    def test_pascals_triangle_four_rows(self) -> None:
        assert pascals_triangle(4) == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]

    @pytest.mark.unit
    def test_pascals_triangle_zero_rows(self) -> None:
        assert pascals_triangle(0) == []

    @pytest.mark.unit
    def test_pascals_triangle_negative_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            pascals_triangle(-1)


class TestCatalan:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("n", "expected"),
        [(0, 1), (1, 1), (2, 2), (3, 5), (4, 14), (5, 42), (6, 132)],
    )
    def test_catalan_known_values(self, n: int, expected: int) -> None:
        assert catalan(n) == expected

    @pytest.mark.unit
    def test_catalan_sequence_first_six(self) -> None:
        assert catalan_sequence(5) == [1, 1, 2, 5, 14, 42]

    @pytest.mark.unit
    def test_catalan_negative_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            catalan(-1)

    @pytest.mark.unit
    def test_catalan_sequence_negative_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            catalan_sequence(-1)
