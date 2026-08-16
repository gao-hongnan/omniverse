from __future__ import annotations

import pytest

from omnivault.dsa.algorithms.dp import (
    coin_change_count,
    coin_change_min_coins,
    knapsack_01,
    knapsack_unbounded,
    lcs,
    levenshtein,
    lis,
    matrix_chain_order,
)
from omnivault.dsa.algorithms.dp.edit_distance import hamming
from omnivault.dsa.core.errors import RectangularityViolation


@pytest.mark.unit
class TestSequences:
    @pytest.mark.parametrize(
        ("seq", "expected"),
        [
            ([10, 9, 2, 5, 3, 7, 101, 18], 4),
            ([0, 1, 0, 3, 2, 3], 4),
            ([7, 7, 7, 7], 1),
            ([], 0),
            ([5], 1),
        ],
    )
    def test_lis(self, seq: list[int], expected: int) -> None:
        assert lis(seq) == expected

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("ABCBDAB", "BDCAB", 4),
            ("AGGTAB", "GXTXAYB", 4),
            ("", "ABC", 0),
            ("ABC", "ABC", 3),
        ],
    )
    def test_lcs(self, a: str, b: str, expected: int) -> None:
        assert lcs(a, b) == expected


@pytest.mark.unit
class TestEditDistance:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("", "abc", 3),
            ("abc", "", 3),
            ("abc", "abc", 0),
        ],
    )
    def test_levenshtein(self, a: str, b: str, expected: int) -> None:
        assert levenshtein(a, b) == expected

    def test_hamming_equal_length(self) -> None:
        assert hamming("karolin", "kathrin") == 3
        assert hamming("1011101", "1001001") == 2

    def test_hamming_mismatched_length_raises(self) -> None:
        with pytest.raises(RectangularityViolation):
            hamming("abc", "abcd")


@pytest.mark.unit
class TestKnapsack:
    @pytest.mark.parametrize(
        ("weights", "values", "capacity", "expected"),
        [
            ([1, 3, 4, 5], [1, 4, 5, 7], 7, 9),
            ([2, 3, 4, 5], [3, 4, 5, 6], 5, 7),
            ([1], [10], 0, 0),
        ],
    )
    def test_knapsack_01(self, weights: list[int], values: list[int], capacity: int, expected: int) -> None:
        assert knapsack_01(weights, values, capacity) == expected

    @pytest.mark.parametrize(
        ("weights", "values", "capacity", "expected"),
        [
            ([1, 3, 4, 5], [10, 40, 50, 70], 8, 110),
            ([2, 3], [3, 4], 6, 9),
        ],
    )
    def test_knapsack_unbounded(self, weights: list[int], values: list[int], capacity: int, expected: int) -> None:
        assert knapsack_unbounded(weights, values, capacity) == expected

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(RectangularityViolation):
            knapsack_01([1, 2], [3], 5)
        with pytest.raises(RectangularityViolation):
            knapsack_unbounded([1, 2], [3], 5)


@pytest.mark.unit
class TestCoinChange:
    @pytest.mark.parametrize(
        ("coins", "amount", "expected"),
        [
            ([1, 2, 5], 5, 4),
            ([2], 3, 0),
            ([1], 0, 1),
        ],
    )
    def test_coin_change_count(self, coins: list[int], amount: int, expected: int) -> None:
        assert coin_change_count(coins, amount) == expected

    @pytest.mark.parametrize(
        ("coins", "amount", "expected"),
        [
            ([1, 2, 5], 11, 3),
            ([2], 3, -1),
            ([1], 0, 0),
            ([186, 419, 83, 408], 6249, 20),
        ],
    )
    def test_coin_change_min(self, coins: list[int], amount: int, expected: int) -> None:
        assert coin_change_min_coins(coins, amount) == expected


@pytest.mark.unit
class TestMatrixChain:
    @pytest.mark.parametrize(
        ("dims", "expected"),
        [
            ([40, 20, 30, 10, 30], 26000),
            ([10, 20, 30, 40, 30], 30000),
            ([10, 20, 30], 6000),
        ],
    )
    def test_matrix_chain_order(self, dims: list[int], expected: int) -> None:
        assert matrix_chain_order(dims) == expected
