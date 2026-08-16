from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from omnivault.dsa.containers.associative.hash_table import (
    AbstractHashTable,
    ChainingHashTable,
    OpenAddressingHashTable,
)
from omnivault.dsa.containers.associative.hash_table.concrete import ProbingStrategy

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.omnivault.unit.dsa.conftest import TestData


class TestHashTableImplementations:
    @pytest.fixture(
        params=[
            ChainingHashTable,
            lambda: OpenAddressingHashTable(probing_strategy=ProbingStrategy.LINEAR),
            lambda: OpenAddressingHashTable(probing_strategy=ProbingStrategy.QUADRATIC),
            lambda: OpenAddressingHashTable(probing_strategy=ProbingStrategy.DOUBLE_HASH),
        ]
    )
    def hash_table_factory(self, request: pytest.FixtureRequest) -> Callable[[], AbstractHashTable[Any, Any]]:
        return cast("Callable[[], AbstractHashTable[Any, Any]]", request.param)

    @pytest.fixture
    def empty_ht(self, hash_table_factory: Callable[[], AbstractHashTable[Any, Any]]) -> AbstractHashTable[str, int]:
        return hash_table_factory()

    @pytest.fixture
    def filled_ht(
        self, hash_table_factory: Callable[[], AbstractHashTable[Any, Any]], test_data: TestData
    ) -> AbstractHashTable[str, int]:
        ht = hash_table_factory()
        for i, string in enumerate(test_data.strings):
            ht.put(string, i)
        return ht

    @pytest.mark.unit
    def test_empty_hash_table_properties(self, empty_ht: AbstractHashTable[str, int]) -> None:
        assert empty_ht.is_empty()
        assert len(empty_ht) == 0
        assert not empty_ht
        assert list(empty_ht.keys()) == []
        assert list(empty_ht.values()) == []
        assert list(empty_ht.items()) == []

    @pytest.mark.unit
    def test_put_single_item(self, empty_ht: AbstractHashTable[str, int]) -> None:
        empty_ht.put("key", 42)
        assert not empty_ht.is_empty()
        assert len(empty_ht) == 1
        assert empty_ht.get("key") == 42
        assert empty_ht.contains_key("key")
        assert "key" in empty_ht

    @pytest.mark.unit
    def test_put_update_existing_key(self, empty_ht: AbstractHashTable[str, int]) -> None:
        empty_ht.put("key", 42)
        empty_ht.put("key", 84)

        assert len(empty_ht) == 1
        assert empty_ht.get("key") == 84

    @pytest.mark.unit
    def test_get_existing_key(self, filled_ht: AbstractHashTable[str, int], test_data: TestData) -> None:
        for i, string in enumerate(test_data.strings):
            assert filled_ht.get(string) == i

    @pytest.mark.unit
    def test_get_nonexistent_key_raises_error(self, empty_ht: AbstractHashTable[str, int]) -> None:
        with pytest.raises(KeyError):
            empty_ht.get("nonexistent")

    @pytest.mark.unit
    def test_remove_existing_key(self, filled_ht: AbstractHashTable[str, int], test_data: TestData) -> None:
        first_string = test_data.strings[0]
        removed_value = filled_ht.remove(first_string)

        assert removed_value == 0
        assert len(filled_ht) == len(test_data.strings) - 1
        assert not filled_ht.contains_key(first_string)

    @pytest.mark.unit
    def test_remove_nonexistent_key_raises_error(self, empty_ht: AbstractHashTable[str, int]) -> None:
        with pytest.raises(KeyError):
            empty_ht.remove("nonexistent")

    @pytest.mark.unit
    def test_contains_key_operations(self, filled_ht: AbstractHashTable[str, int], test_data: TestData) -> None:
        for string in test_data.strings:
            assert filled_ht.contains_key(string)
            assert string in filled_ht

        assert not filled_ht.contains_key("nonexistent")
        assert "nonexistent" not in filled_ht

    @pytest.mark.unit
    def test_clear_hash_table(self, filled_ht: AbstractHashTable[str, int]) -> None:
        filled_ht.clear()
        assert filled_ht.is_empty()
        assert len(filled_ht) == 0

    @pytest.mark.unit
    def test_keys_iterator(self, filled_ht: AbstractHashTable[str, int], test_data: TestData) -> None:
        keys = list(filled_ht.keys())
        assert len(keys) == len(test_data.strings)
        for string in test_data.strings:
            assert string in keys

    @pytest.mark.unit
    def test_values_iterator(self, filled_ht: AbstractHashTable[str, int], test_data: TestData) -> None:
        values = list(filled_ht.values())
        assert len(values) == len(test_data.strings)
        for i in range(len(test_data.strings)):
            assert i in values

    @pytest.mark.unit
    def test_items_iterator(self, filled_ht: AbstractHashTable[str, int], test_data: TestData) -> None:
        items = list(filled_ht.items())
        assert len(items) == len(test_data.strings)

        for i, string in enumerate(test_data.strings):
            assert (string, i) in items

    @pytest.mark.unit
    def test_dict_like_access(self, empty_ht: AbstractHashTable[str, int]) -> None:
        empty_ht["key1"] = 100
        empty_ht["key2"] = 200

        assert empty_ht["key1"] == 100
        assert empty_ht["key2"] == 200

        del empty_ht["key1"]
        assert not empty_ht.contains_key("key1")
        assert len(empty_ht) == 1

    @pytest.mark.unit
    def test_update_from_dict(self, empty_ht: AbstractHashTable[str, int]) -> None:
        data = {"a": 1, "b": 2, "c": 3}
        empty_ht.update(data)

        assert len(empty_ht) == 3
        for key, value in data.items():
            assert empty_ht.get(key) == value

    @pytest.mark.unit
    def test_update_from_hash_table(
        self, empty_ht: AbstractHashTable[str, int], hash_table_factory: Callable[[], AbstractHashTable[Any, Any]]
    ) -> None:
        other_ht = hash_table_factory()
        other_ht.put("x", 10)
        other_ht.put("y", 20)

        empty_ht.update(other_ht)

        assert len(empty_ht) == 2
        assert empty_ht.get("x") == 10
        assert empty_ht.get("y") == 20

    @pytest.mark.unit
    def test_get_default(self, empty_ht: AbstractHashTable[str, int]) -> None:
        empty_ht.put("existing", 42)

        assert empty_ht.get_default("existing", 0) == 42
        assert empty_ht.get_default("nonexistent", 99) == 99

    @pytest.mark.unit
    def test_pop_with_default(self, empty_ht: AbstractHashTable[str, int]) -> None:
        empty_ht.put("key", 42)

        assert empty_ht.pop("key", 0) == 42
        assert empty_ht.pop("nonexistent", 99) == 99

        with pytest.raises(KeyError):
            empty_ht.pop("another_nonexistent")

    @pytest.mark.unit
    def test_repr_output(self, hash_table_factory: Callable[[], AbstractHashTable[Any, Any]]) -> None:
        ht = hash_table_factory()
        ht.put("key", "value")
        repr_str = repr(ht)
        assert "key" in repr_str
        assert "value" in repr_str

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_large_hash_table_operations(
        self, hash_table_factory: Callable[[], AbstractHashTable[Any, Any]], size: int
    ) -> None:
        ht = hash_table_factory()

        for i in range(size):
            ht.put(f"key_{i}", i)

        assert len(ht) == size

        for i in range(size):
            assert ht.get(f"key_{i}") == i

        for i in range(0, size, 2):
            ht.remove(f"key_{i}")

        assert len(ht) == size // 2

    @pytest.mark.edge_case
    def test_hash_collision_handling(self, hash_table_factory: Callable[[], AbstractHashTable[Any, Any]]) -> None:
        ht = hash_table_factory()

        keys_with_same_hash = ["Aa", "BB"]

        for i, key in enumerate(keys_with_same_hash):
            ht.put(key, i)

        for i, key in enumerate(keys_with_same_hash):
            assert ht.get(key) == i

    @pytest.mark.benchmark
    def test_performance_with_string_keys(
        self, hash_table_factory: Callable[[], AbstractHashTable[Any, Any]], test_data: TestData
    ) -> None:
        ht = hash_table_factory()

        strings = test_data.strings * 20

        for string in strings:
            ht.put(string, len(string))

        for string in strings:
            assert ht.contains_key(string)


class TestChainingHashTableSpecific:
    @pytest.mark.unit
    def test_initial_capacity_and_load_factor(self) -> None:
        ht = ChainingHashTable[str, int](initial_capacity=8, max_load_factor=0.8)

        assert ht.capacity == 8
        assert ht.load_factor == 0.0

    @pytest.mark.unit
    def test_invalid_initial_capacity_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Initial capacity must be at least 1"):
            ChainingHashTable[str, int](initial_capacity=0)

    @pytest.mark.unit
    def test_invalid_load_factor_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Load factor must be between 0 and 1"):
            ChainingHashTable[str, int](max_load_factor=1.5)

    @pytest.mark.unit
    def test_resize_on_load_factor_exceeded(self) -> None:
        ht = ChainingHashTable[str, int](initial_capacity=4, max_load_factor=0.75)

        for i in range(3):
            ht.put(f"key_{i}", i)

        initial_capacity = ht.capacity

        ht.put("trigger_resize", 999)

        assert ht.capacity == initial_capacity * 2
        assert len(ht) == 4

        for i in range(3):
            assert ht.get(f"key_{i}") == i
        assert ht.get("trigger_resize") == 999


class TestOpenAddressingHashTableSpecific:
    @pytest.fixture(params=[ProbingStrategy.LINEAR, ProbingStrategy.QUADRATIC, ProbingStrategy.DOUBLE_HASH])
    def probing_strategy(self, request: pytest.FixtureRequest) -> ProbingStrategy:
        return cast("ProbingStrategy", request.param)

    @pytest.mark.unit
    def test_probing_strategy_property(self, probing_strategy: ProbingStrategy) -> None:
        ht = OpenAddressingHashTable[str, int](probing_strategy=probing_strategy)
        assert ht.probing_strategy == probing_strategy

    @pytest.mark.unit
    def test_invalid_initial_capacity_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Initial capacity must be at least 1"):
            OpenAddressingHashTable[str, int](initial_capacity=0)

    @pytest.mark.unit
    def test_invalid_load_factor_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Load factor must be between 0 and 1"):
            OpenAddressingHashTable[str, int](max_load_factor=1.5)

    @pytest.mark.unit
    def test_resize_on_load_factor_exceeded(self, probing_strategy: ProbingStrategy) -> None:
        ht = OpenAddressingHashTable[str, int](
            initial_capacity=4, max_load_factor=0.5, probing_strategy=probing_strategy
        )

        ht.put("key_1", 1)
        ht.put("key_2", 2)

        initial_capacity = ht.capacity

        ht.put("trigger_resize", 999)

        assert ht.capacity == initial_capacity * 2
        assert len(ht) == 3

        assert ht.get("key_1") == 1
        assert ht.get("key_2") == 2
        assert ht.get("trigger_resize") == 999

    @pytest.mark.unit
    def test_deleted_slots_handling(self, probing_strategy: ProbingStrategy) -> None:
        ht = OpenAddressingHashTable[str, int](initial_capacity=8, probing_strategy=probing_strategy)

        ht.put("key_1", 1)
        ht.put("key_2", 2)
        ht.put("key_3", 3)

        ht.remove("key_2")

        ht.put("key_4", 4)

        assert len(ht) == 3
        assert ht.get("key_1") == 1
        assert ht.get("key_3") == 3
        assert ht.get("key_4") == 4
        assert not ht.contains_key("key_2")
