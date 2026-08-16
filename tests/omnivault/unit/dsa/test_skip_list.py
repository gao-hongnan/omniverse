from __future__ import annotations

import random

import pytest

from omnivault.dsa.containers.associative.skip_list import SkipList
from omnivault.dsa.core.errors import EmptyContainer


class TestSkipList:
    @pytest.fixture
    def empty_skip_list(self) -> SkipList[int, str]:
        return SkipList[int, str]()

    @pytest.fixture
    def sample_skip_list(self) -> SkipList[int, str]:
        sl = SkipList[int, str]()
        data = [(50, "fifty"), (30, "thirty"), (70, "seventy"), (20, "twenty"), (40, "forty")]
        for key, value in data:
            sl.insert(key, value)
        return sl

    @pytest.fixture
    def large_skip_list(self) -> SkipList[int, int]:
        sl = SkipList[int, int]()
        random.seed(42)  # For reproducible tests
        for i in range(100):
            sl.insert(i, i * 2)
        return sl

    def test_skip_list_initialization(self, empty_skip_list: SkipList[int, str]) -> None:
        assert empty_skip_list.is_empty()
        assert len(empty_skip_list) == 0
        assert empty_skip_list.size() == 0
        assert not empty_skip_list
        assert empty_skip_list.height >= 1
        assert empty_skip_list.max_height == 17  # max_level + 1

    def test_skip_list_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            SkipList[int, str](max_level=0)

        with pytest.raises(ValueError):
            SkipList[int, str](probability=0.0)

        with pytest.raises(ValueError):
            SkipList[int, str](probability=1.0)

    def test_skip_list_single_element(self, empty_skip_list: SkipList[int, str]) -> None:
        empty_skip_list.insert(42, "answer")

        assert not empty_skip_list.is_empty()
        assert len(empty_skip_list) == 1
        assert empty_skip_list.size() == 1
        assert bool(empty_skip_list)

        assert empty_skip_list.search(42) == "answer"
        assert empty_skip_list.contains(42)
        assert empty_skip_list.min_key() == 42
        assert empty_skip_list.max_key() == 42

    def test_skip_list_insertion_and_search(self, sample_skip_list: SkipList[int, str]) -> None:
        assert sample_skip_list.search(50) == "fifty"
        assert sample_skip_list.search(30) == "thirty"
        assert sample_skip_list.search(70) == "seventy"
        assert sample_skip_list.search(20) == "twenty"
        assert sample_skip_list.search(40) == "forty"

        assert sample_skip_list.size() == 5

    def test_skip_list_insertion_update_existing(self, empty_skip_list: SkipList[int, str]) -> None:
        empty_skip_list.insert(10, "ten")
        empty_skip_list.insert(10, "TEN")

        assert empty_skip_list.search(10) == "TEN"
        assert empty_skip_list.size() == 1

    def test_skip_list_search_nonexistent(self, sample_skip_list: SkipList[int, str]) -> None:
        with pytest.raises(KeyError):
            sample_skip_list.search(100)

    def test_skip_list_deletion(self, sample_skip_list: SkipList[int, str]) -> None:
        deleted_value = sample_skip_list.delete(50)
        assert deleted_value == "fifty"
        assert not sample_skip_list.contains(50)
        assert sample_skip_list.size() == 4

    def test_skip_list_deletion_nonexistent(self, sample_skip_list: SkipList[int, str]) -> None:
        with pytest.raises(KeyError):
            sample_skip_list.delete(100)

    def test_skip_list_contains(self, sample_skip_list: SkipList[int, str]) -> None:
        assert sample_skip_list.contains(50)
        assert sample_skip_list.contains(20)
        assert not sample_skip_list.contains(100)
        assert not sample_skip_list.contains(-10)

    def test_skip_list_min_max_keys(self, sample_skip_list: SkipList[int, str]) -> None:
        assert sample_skip_list.min_key() == 20
        assert sample_skip_list.max_key() == 70

    def test_skip_list_min_max_empty(self, empty_skip_list: SkipList[int, str]) -> None:
        with pytest.raises(EmptyContainer):
            empty_skip_list.min_key()

        with pytest.raises(EmptyContainer):
            empty_skip_list.max_key()

    def test_skip_list_predecessor_successor(self, sample_skip_list: SkipList[int, str]) -> None:
        # Predecessor tests
        assert sample_skip_list.predecessor(25) == 20
        assert sample_skip_list.predecessor(50) == 40
        assert sample_skip_list.predecessor(15) is None

        # Successor tests
        assert sample_skip_list.successor(25) == 30
        assert sample_skip_list.successor(50) == 70
        assert sample_skip_list.successor(80) is None

    def test_skip_list_range_search(self, sample_skip_list: SkipList[int, str]) -> None:
        # Range that includes multiple elements
        results = list(sample_skip_list.range_search(25, 65))
        expected = [(30, "thirty"), (40, "forty"), (50, "fifty")]
        assert results == expected

        # Range with single element
        results = list(sample_skip_list.range_search(50, 50))
        assert results == [(50, "fifty")]

        # Range with no elements
        results = list(sample_skip_list.range_search(80, 90))
        assert results == []

        # Invalid range
        results = list(sample_skip_list.range_search(70, 30))
        assert results == []

    def test_skip_list_clear(self, sample_skip_list: SkipList[int, str]) -> None:
        sample_skip_list.clear()
        assert sample_skip_list.is_empty()
        assert sample_skip_list.size() == 0

    def test_skip_list_keys_values_items(self, sample_skip_list: SkipList[int, str]) -> None:
        keys = list(sample_skip_list.keys())
        assert keys == [20, 30, 40, 50, 70]

        values = list(sample_skip_list.values())
        assert values == ["twenty", "thirty", "forty", "fifty", "seventy"]

        items = list(sample_skip_list.items())
        expected_items = [(20, "twenty"), (30, "thirty"), (40, "forty"), (50, "fifty"), (70, "seventy")]
        assert items == expected_items

    def test_skip_list_magic_methods(self, sample_skip_list: SkipList[int, str]) -> None:
        # Contains
        assert 50 in sample_skip_list
        assert 100 not in sample_skip_list
        assert "invalid" not in sample_skip_list

        # Getitem
        assert sample_skip_list[50] == "fifty"

        # Setitem
        sample_skip_list[80] = "eighty"
        assert sample_skip_list.search(80) == "eighty"

        # Delitem
        del sample_skip_list[80]
        assert not sample_skip_list.contains(80)

    def test_skip_list_iteration(self, sample_skip_list: SkipList[int, str]) -> None:
        keys = list(sample_skip_list)
        assert keys == [20, 30, 40, 50, 70]

    def test_skip_list_equality(self) -> None:
        sl1 = SkipList[int, str]()
        sl2 = SkipList[int, str]()

        data = [(5, "five"), (3, "three"), (7, "seven")]
        for key, value in data:
            sl1.insert(key, value)
            sl2.insert(key, value)

        assert sl1 == sl2

        sl2.insert(9, "nine")
        assert sl1 != sl2

    def test_skip_list_get_default(self, sample_skip_list: SkipList[int, str]) -> None:
        assert sample_skip_list.get_default(50, "default") == "fifty"
        assert sample_skip_list.get_default(100, "default") == "default"

    def test_skip_list_pop_operations(self, sample_skip_list: SkipList[int, str]) -> None:
        value = sample_skip_list.pop(50)
        assert value == "fifty"
        assert not sample_skip_list.contains(50)

        with pytest.raises(KeyError):
            sample_skip_list.pop(100)

        default_value = sample_skip_list.pop(100, "default")
        assert default_value == "default"

    def test_skip_list_popitem(self, sample_skip_list: SkipList[int, str]) -> None:
        key, value = sample_skip_list.popitem()
        assert key == 70  # Should be max key
        assert value == "seventy"
        assert not sample_skip_list.contains(70)

    def test_skip_list_popitem_empty(self, empty_skip_list: SkipList[int, str]) -> None:
        with pytest.raises(EmptyContainer):
            empty_skip_list.popitem()

    def test_skip_list_setdefault(self, sample_skip_list: SkipList[int, str]) -> None:
        value = sample_skip_list.setdefault(50, "default")
        assert value == "fifty"

        value = sample_skip_list.setdefault(100, "hundred")
        assert value == "hundred"
        assert sample_skip_list.search(100) == "hundred"

    def test_skip_list_update_operations(self, empty_skip_list: SkipList[int, str]) -> None:
        other_sl = SkipList[int, str]()
        other_sl.insert(1, "one")
        other_sl.insert(2, "two")

        empty_skip_list.update(other_sl)
        assert empty_skip_list.size() == 2
        assert empty_skip_list.search(1) == "one"
        assert empty_skip_list.search(2) == "two"

        dict_data = {3: "three", 4: "four"}
        empty_skip_list.update(dict_data)
        assert empty_skip_list.size() == 4

    def test_skip_list_display_structure(self, sample_skip_list: SkipList[int, str]) -> None:
        structure = sample_skip_list.display_structure()
        assert isinstance(structure, str)
        assert "Level" in structure

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_skip_list_performance_scalability(self, size: int) -> None:
        sl = SkipList[int, int]()

        # Insert elements
        for i in range(size):
            sl.insert(i, i * 2)

        assert sl.size() == size

        # Search elements
        for i in range(0, size, 10):
            assert sl.search(i) == i * 2

        # Delete some elements
        for i in range(0, size, 10):
            sl.delete(i)

        remaining_size = size - len(range(0, size, 10))
        assert sl.size() == remaining_size

    def test_skip_list_ordered_insertion(self) -> None:
        sl = SkipList[int, int]()

        # Insert in ascending order
        for i in range(20):
            sl.insert(i, i)

        keys = list(sl.keys())
        assert keys == list(range(20))

    def test_skip_list_reverse_ordered_insertion(self) -> None:
        sl = SkipList[int, int]()

        # Insert in descending order
        for i in range(19, -1, -1):
            sl.insert(i, i)

        keys = list(sl.keys())
        assert keys == list(range(20))

    def test_skip_list_random_insertion(self, large_skip_list: SkipList[int, int]) -> None:
        # Keys should be in sorted order despite random insertion
        keys = list(large_skip_list.keys())
        assert keys == sorted(keys)
        assert len(keys) == 100

    def test_skip_list_height_property(self) -> None:
        sl = SkipList[int, int](max_level=10, probability=0.5)

        # Insert many elements
        for i in range(1000):
            sl.insert(i, i)

        # Height should be reasonable (logarithmic)
        assert sl.height <= 15  # Should be much less than 1000

    @pytest.mark.parametrize("probability", [0.25, 0.5, 0.75])
    def test_skip_list_different_probabilities(self, probability: float) -> None:
        sl = SkipList[int, int](probability=probability)

        # Insert elements
        for i in range(50):
            sl.insert(i, i)

        # Should maintain sorted order regardless of probability
        keys = list(sl.keys())
        assert keys == list(range(50))

    def test_skip_list_duplicate_handling(self) -> None:
        sl = SkipList[int, str]()

        # Insert same key multiple times
        sl.insert(5, "five")
        sl.insert(5, "FIVE")
        sl.insert(5, "Five")

        assert sl.size() == 1
        assert sl.search(5) == "Five"  # Should have the last value

    def test_skip_list_range_operations_edge_cases(self) -> None:
        sl = SkipList[int, int]()

        # Insert some data
        for i in [10, 20, 30, 40, 50]:
            sl.insert(i, i)

        # Range that goes beyond actual data
        results = list(sl.range_search(25, 100))
        expected = [(30, 30), (40, 40), (50, 50)]
        assert results == expected

        # Range that starts before actual data
        results = list(sl.range_search(0, 25))
        expected = [(10, 10), (20, 20)]
        assert results == expected

    def test_skip_list_stress_operations(self) -> None:
        sl = SkipList[int, int]()
        random.seed(123)

        operations = []

        # Mixed operations
        for _ in range(100):
            op = random.choice(["insert", "delete", "search"])
            key = random.randint(1, 50)

            if op == "insert":
                sl.insert(key, key * 2)
                operations.append(("insert", key))
            elif op == "delete":
                try:
                    sl.delete(key)
                    operations.append(("delete", key))
                except KeyError:
                    pass  # Key not found, that's okay
            elif op == "search":
                try:
                    value = sl.search(key)
                    assert value == key * 2
                except KeyError:
                    pass  # Key not found, that's okay

        # Verify that remaining elements are still sorted
        keys = list(sl.keys())
        assert keys == sorted(keys)
