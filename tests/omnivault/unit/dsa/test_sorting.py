from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from omnivault.dsa.algorithms.sorting import (
    bubble_sort,
    bucket_sort,
    cocktail_sort,
    counting_sort,
    heap_sort,
    insertion_sort,
    intro_sort,
    merge_sort,
    quick_sort,
    radix_sort,
    randomized_quick_sort,
    selection_sort,
    tim_sort,
)

if TYPE_CHECKING:
    pass


class TestSortingAlgorithms:
    @pytest.fixture
    def unsorted_integers(self) -> list[int]:
        return [64, 34, 25, 12, 22, 11, 90, 5]

    @pytest.fixture
    def sorted_integers(self) -> list[int]:
        return [5, 11, 12, 22, 25, 34, 64, 90]

    @pytest.fixture
    def reverse_sorted_integers(self) -> list[int]:
        return [90, 64, 34, 25, 22, 12, 11, 5]

    @pytest.fixture
    def duplicate_integers(self) -> list[int]:
        return [5, 2, 8, 2, 9, 1, 5, 5]

    @pytest.fixture
    def single_element(self) -> list[int]:
        return [42]

    @pytest.fixture
    def empty_list(self) -> list[int]:
        return []

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_comparison_sorts_basic(
        self, sort_func: object, unsorted_integers: list[int], sorted_integers: list[int]
    ) -> None:
        arr = unsorted_integers.copy()
        sort_func(arr)  # type: ignore
        assert arr == sorted_integers

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_comparison_sorts_already_sorted(self, sort_func: object, sorted_integers: list[int]) -> None:
        arr = sorted_integers.copy()
        expected = sorted_integers.copy()
        sort_func(arr)  # type: ignore
        assert arr == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_comparison_sorts_reverse_sorted(
        self, sort_func: object, reverse_sorted_integers: list[int], sorted_integers: list[int]
    ) -> None:
        arr = reverse_sorted_integers.copy()
        sort_func(arr)  # type: ignore
        assert arr == sorted_integers

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_comparison_sorts_with_duplicates(self, sort_func: object, duplicate_integers: list[int]) -> None:
        arr = duplicate_integers.copy()
        expected = sorted(duplicate_integers)
        sort_func(arr)  # type: ignore
        assert arr == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_comparison_sorts_single_element(self, sort_func: object, single_element: list[int]) -> None:
        arr = single_element.copy()
        expected = single_element.copy()
        sort_func(arr)  # type: ignore
        assert arr == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_comparison_sorts_empty_list(self, sort_func: object, empty_list: list[int]) -> None:
        arr = empty_list.copy()
        expected = empty_list.copy()
        sort_func(arr)  # type: ignore
        assert arr == expected


class TestSortingWithCustomKey:
    @pytest.fixture
    def string_list(self) -> list[str]:
        return ["banana", "apple", "cherry", "date"]

    @pytest.fixture
    def tuple_list(self) -> list[tuple[str, int]]:
        return [("alice", 25), ("bob", 30), ("charlie", 20), ("diana", 35)]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_sort_with_key_function(self, sort_func: object, string_list: list[str]) -> None:
        arr = string_list.copy()
        sort_func(arr, key=len)  # type: ignore

        lengths = [len(s) for s in arr]
        assert lengths == sorted(lengths)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            quick_sort,
            randomized_quick_sort,
            merge_sort,
            heap_sort,
            intro_sort,
            tim_sort,
            insertion_sort,
            selection_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_sort_tuples_by_second_element(self, sort_func: object, tuple_list: list[tuple[str, int]]) -> None:
        arr = tuple_list.copy()
        sort_func(arr, key=lambda x: x[1])  # type: ignore

        ages = [age for _, age in arr]
        assert ages == sorted(ages)

    @pytest.mark.unit
    def test_sort_strings_reverse_alphabetical(self, string_list: list[str]) -> None:
        arr = string_list.copy()
        quick_sort(arr, key=lambda x: x[::-1])

        reverse_strings = [s[::-1] for s in arr]
        assert reverse_strings == sorted(reverse_strings)


class TestSpecializedSorts:
    @pytest.mark.unit
    def test_radix_sort_basic(self) -> None:
        arr = [170, 45, 75, 90, 2, 802, 24, 66]
        expected = sorted(arr)

        radix_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_radix_sort_single_digit(self) -> None:
        arr = [5, 2, 8, 1, 9]
        expected = sorted(arr)

        radix_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_radix_sort_with_zeros(self) -> None:
        arr = [0, 5, 0, 2, 0, 8]
        expected = sorted(arr)

        radix_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_counting_sort_basic(self) -> None:
        arr = [4, 2, 2, 8, 3, 3, 1]
        expected = sorted(arr)

        counting_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_counting_sort_with_max_val(self) -> None:
        arr = [4, 2, 2, 8, 3, 3, 1]
        expected = sorted(arr)

        counting_sort(arr, max_val=10)

        assert arr == expected

    @pytest.mark.unit
    def test_counting_sort_negative_numbers(self) -> None:
        arr = [4, -2, 2, -8, 3, -3, 1]
        expected = sorted(arr)

        counting_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_bucket_sort_basic(self) -> None:
        arr = [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]
        expected = sorted(arr)

        bucket_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_bucket_sort_custom_bucket_count(self) -> None:
        arr = [0.5, 0.3, 0.7, 0.1, 0.9, 0.2]
        expected = sorted(arr)

        bucket_sort(arr, bucket_count=3)

        assert arr == expected

    @pytest.mark.unit
    def test_bucket_sort_single_element(self) -> None:
        arr = [0.5]
        expected = [0.5]

        bucket_sort(arr)

        assert arr == expected


class TestSortingStability:
    @pytest.fixture
    def stable_test_data(self) -> list[tuple[str, int]]:
        return [("a", 2), ("b", 1), ("c", 2), ("d", 1), ("e", 2)]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "sort_func",
        [
            merge_sort,
            tim_sort,
            insertion_sort,
            bubble_sort,
            cocktail_sort,
        ],
    )
    def test_stable_sorts_preserve_order(self, sort_func: object, stable_test_data: list[tuple[str, int]]) -> None:
        arr = stable_test_data.copy()
        sort_func(arr, key=lambda x: x[1])  # type: ignore

        groups: dict[int, list[str]] = {}
        for item in arr:
            key = item[1]
            if key not in groups:
                groups[key] = []
            groups[key].append(item[0])

        assert groups[1] == ["b", "d"]
        assert groups[2] == ["a", "c", "e"]


class TestSortingPerformanceCharacteristics:
    @pytest.mark.unit
    def test_quick_sort_random_data(self) -> None:
        random.seed(42)
        arr = [random.randint(1, 1000) for _ in range(100)]
        expected = sorted(arr)

        quick_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_merge_sort_large_dataset(self) -> None:
        arr = list(range(1000, 0, -1))
        expected = list(range(1, 1001))

        merge_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_heap_sort_nearly_sorted(self) -> None:
        arr = list(range(100))
        arr[10], arr[90] = arr[90], arr[10]
        expected = sorted(arr)

        heap_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_intro_sort_worst_case_quick_sort(self) -> None:
        arr = list(range(100))
        expected = sorted(arr)

        intro_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_tim_sort_partially_sorted(self) -> None:
        arr = list(range(50)) + list(range(100, 150)) + list(range(50, 100))
        expected = sorted(arr)

        tim_sort(arr)

        assert arr == expected


class TestSortingEdgeCases:
    @pytest.mark.unit
    def test_sort_with_negative_numbers(self) -> None:
        arr = [-5, -1, -10, 0, 3, -2]
        expected = sorted(arr)

        quick_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_sort_with_large_numbers(self) -> None:
        arr = [1000000, 999999, 1000001, 500000]
        expected = sorted(arr)

        merge_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_radix_sort_empty_list(self) -> None:
        arr: list[int] = []

        radix_sort(arr)

        assert arr == []

    @pytest.mark.unit
    def test_counting_sort_empty_list(self) -> None:
        arr: list[int] = []

        counting_sort(arr)

        assert arr == []

    @pytest.mark.unit
    def test_bucket_sort_empty_list(self) -> None:
        arr: list[float] = []

        bucket_sort(arr)

        assert arr == []

    @pytest.mark.unit
    def test_sort_identical_elements(self) -> None:
        arr = [5, 5, 5, 5, 5]
        expected = [5, 5, 5, 5, 5]

        heap_sort(arr)

        assert arr == expected


class TestSortingCorrectness:
    @pytest.mark.unit
    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_random_arrays_different_sizes(self, size: int) -> None:
        random.seed(42)
        arr = [random.randint(1, 100) for _ in range(size)]
        expected = sorted(arr)

        quick_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_floating_point_numbers(self) -> None:
        arr = [3.14, 2.71, 1.41, 0.57, 2.23]
        expected = sorted(arr)

        merge_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_string_sorting(self) -> None:
        arr = ["zebra", "apple", "banana", "cherry"]
        expected = sorted(arr)

        quick_sort(arr)

        assert arr == expected

    @pytest.mark.unit
    def test_custom_objects_sorting(self) -> None:
        class Person:
            def __init__(self, name: str, age: int) -> None:
                self.name = name
                self.age = age

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Person) and self.age == other.age

        people = [Person("Alice", 30), Person("Bob", 25), Person("Charlie", 35)]

        merge_sort(people, key=lambda p: p.age)

        assert people[0].age == 25
        assert people[1].age == 30
        assert people[2].age == 35
