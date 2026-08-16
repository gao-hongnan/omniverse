from __future__ import annotations

import pytest

from omnivault.dsa.algorithms.searching import (
    IterativeBinarySearchExactMatch,
    LinearSearchForLoop,
    LinearSearchRecursive,
    LinearSearchTailRecursive,
    LinearSearchWhileLoop,
    RecursiveBinarySearchExactMatch,
    Search,
    SearchContext,
)


class TestLinearSearchStrategies:
    @pytest.fixture
    def test_data(self) -> dict[str, list[int]]:
        return {
            "sorted_array": [1, 3, 5, 7, 9, 11, 13, 15],
            "unsorted_array": [5, 2, 8, 1, 9, 3, 7],
            "empty_array": [],
            "single_element": [42],
            "duplicates": [1, 3, 3, 5, 7, 7, 7, 9],
        }

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LinearSearchForLoop,
            LinearSearchWhileLoop,
            LinearSearchRecursive,
            LinearSearchTailRecursive,
        ],
    )
    def test_found_in_sorted_array(self, test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(test_data["sorted_array"], 7)
        assert result == 3

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LinearSearchForLoop,
            LinearSearchWhileLoop,
            LinearSearchRecursive,
            LinearSearchTailRecursive,
        ],
    )
    def test_not_found_in_sorted_array(self, test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(test_data["sorted_array"], 10)
        assert result == -1

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LinearSearchForLoop,
            LinearSearchWhileLoop,
            LinearSearchRecursive,
            LinearSearchTailRecursive,
        ],
    )
    def test_found_in_unsorted_array(self, test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(test_data["unsorted_array"], 8)
        assert result == 2

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LinearSearchForLoop,
            LinearSearchWhileLoop,
            LinearSearchRecursive,
            LinearSearchTailRecursive,
        ],
    )
    def test_empty_array(self, test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(test_data["empty_array"], 1)
        assert result == -1

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LinearSearchForLoop,
            LinearSearchWhileLoop,
            LinearSearchRecursive,
            LinearSearchTailRecursive,
        ],
    )
    def test_single_element_found(self, test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(test_data["single_element"], 42)
        assert result == 0

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LinearSearchForLoop,
            LinearSearchWhileLoop,
            LinearSearchRecursive,
            LinearSearchTailRecursive,
        ],
    )
    def test_single_element_not_found(self, test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(test_data["single_element"], 24)
        assert result == -1

    @pytest.mark.parametrize(
        "strategy_class",
        [
            LinearSearchForLoop,
            LinearSearchWhileLoop,
            LinearSearchRecursive,
            LinearSearchTailRecursive,
        ],
    )
    def test_first_occurrence_with_duplicates(
        self, test_data: dict[str, list[int]], strategy_class: type[Search]
    ) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(test_data["duplicates"], 7)
        assert result == 4  # First occurrence of 7


class TestBinarySearchStrategies:
    @pytest.fixture
    def sorted_test_data(self) -> dict[str, list[int]]:
        return {
            "sorted_array": [1, 3, 5, 7, 9, 11, 13, 15],
            "empty_array": [],
            "single_element": [42],
            "two_elements": [10, 20],
            "large_array": list(range(0, 1000, 2)),  # Even numbers 0 to 998
        }

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_found_in_middle(self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["sorted_array"], 7)
        assert result == 3

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_found_at_start(self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["sorted_array"], 1)
        assert result == 0

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_found_at_end(self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["sorted_array"], 15)
        assert result == 7

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_not_found(self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["sorted_array"], 10)
        assert result == -1

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_empty_array(self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["empty_array"], 1)
        assert result == -1

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_single_element_found(self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["single_element"], 42)
        assert result == 0

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_single_element_not_found(
        self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]
    ) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["single_element"], 24)
        assert result == -1

    @pytest.mark.parametrize(
        "strategy_class",
        [IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch],
    )
    def test_large_array(self, sorted_test_data: dict[str, list[int]], strategy_class: type[Search]) -> None:
        context = SearchContext(strategy_class())
        result = context.execute_search(sorted_test_data["large_array"], 500)
        assert result == 250
        result = context.execute_search(sorted_test_data["large_array"], 999)
        assert result == -1


class TestSearchContext:
    def test_strategy_switching(self) -> None:
        data = [1, 3, 5, 7, 9]
        target = 5

        context = SearchContext(LinearSearchForLoop())
        assert context.execute_search(data, target) == 2

        context.strategy = IterativeBinarySearchExactMatch()
        assert context.execute_search(data, target) == 2

        context.strategy = LinearSearchRecursive()
        assert context.execute_search(data, target) == 2

    def test_with_float_values(self) -> None:
        data = [1.1, 2.2, 3.3, 4.4, 5.5]
        context = SearchContext(LinearSearchForLoop())
        assert context.execute_search(data, 3.3) == 2
        assert context.execute_search(data, 3.0) == -1

    def test_with_mixed_int_float(self) -> None:
        data = [1, 2.5, 3, 4.5, 5]
        context = SearchContext(LinearSearchForLoop())
        assert context.execute_search(data, 2.5) == 1
        assert context.execute_search(data, 3) == 2
