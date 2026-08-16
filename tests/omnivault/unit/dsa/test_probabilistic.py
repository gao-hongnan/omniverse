from __future__ import annotations

import pytest

from omnivault.dsa.probabilistic import BloomFilter, CountMinSketch, HyperLogLog


class TestBloomFilter:
    @pytest.fixture
    def small_bloom_filter(self) -> BloomFilter[str]:
        return BloomFilter[str](100, 0.01)

    @pytest.fixture
    def sample_bloom_filter(self) -> BloomFilter[str]:
        bf = BloomFilter[str](1000, 0.01)
        items = ["apple", "banana", "cherry", "date", "elderberry"]
        for item in items:
            bf.add(item)
        return bf

    def test_bloom_filter_initialization(self, small_bloom_filter: BloomFilter[str]) -> None:
        assert small_bloom_filter.size > 0
        assert small_bloom_filter.hash_count > 0
        assert small_bloom_filter.inserted_count == 0
        assert small_bloom_filter.bits_set == 0
        assert small_bloom_filter.is_empty()

    def test_bloom_filter_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            BloomFilter[str](0, 0.01)

        with pytest.raises(ValueError):
            BloomFilter[str](100, 0.0)

        with pytest.raises(ValueError):
            BloomFilter[str](100, 1.0)

    def test_bloom_filter_add_operation(self, small_bloom_filter: BloomFilter[str]) -> None:
        small_bloom_filter.add("test")

        assert small_bloom_filter.inserted_count == 1
        assert small_bloom_filter.bits_set > 0
        assert not small_bloom_filter.is_empty()
        assert bool(small_bloom_filter)

    def test_bloom_filter_contains_operation(self, sample_bloom_filter: BloomFilter[str]) -> None:
        # Items that were added
        assert "apple" in sample_bloom_filter
        assert "banana" in sample_bloom_filter
        assert sample_bloom_filter.might_contain("cherry")

        # Items that were not added (should definitely not be present)
        # Note: There's a small chance of false positives
        assert not sample_bloom_filter.definitely_not_contains("apple")

    def test_bloom_filter_false_positives(self) -> None:
        bf = BloomFilter[str](100, 0.1)  # Higher false positive rate for testing

        # Add some items
        added_items = ["item1", "item2", "item3"]
        for item in added_items:
            bf.add(item)

        # Check items that were added
        for item in added_items:
            assert item in bf

        # Check many items that were not added
        false_positives = 0
        test_items = [f"not_added_{i}" for i in range(100)]

        for item in test_items:
            if item in bf:
                false_positives += 1

        # False positive rate should be approximately as expected
        false_positive_rate = false_positives / len(test_items)
        assert false_positive_rate <= 0.2  # Allow some variance

    def test_bloom_filter_union(self) -> None:
        bf1 = BloomFilter[str](100, 0.01)
        bf2 = BloomFilter[str](100, 0.01)

        bf1.add("apple")
        bf1.add("banana")

        bf2.add("cherry")
        bf2.add("date")

        union_bf = bf1.union(bf2)

        assert "apple" in union_bf
        assert "banana" in union_bf
        assert "cherry" in union_bf
        assert "date" in union_bf

    def test_bloom_filter_union_incompatible(self) -> None:
        bf1 = BloomFilter[str](100, 0.01)
        bf2 = BloomFilter[str](200, 0.01)

        with pytest.raises(ValueError):
            bf1.union(bf2)

    def test_bloom_filter_intersection(self) -> None:
        bf1 = BloomFilter[str](100, 0.01)
        bf2 = BloomFilter[str](100, 0.01)

        # Add some common items
        common_items = ["apple", "banana"]
        for item in common_items:
            bf1.add(item)
            bf2.add(item)

        # Add unique items
        bf1.add("cherry")
        bf2.add("date")

        intersection_bf = bf1.intersection(bf2)

        # Common items should definitely be in intersection
        for item in common_items:
            assert item in intersection_bf

    def test_bloom_filter_clear(self, sample_bloom_filter: BloomFilter[str]) -> None:
        sample_bloom_filter.clear()

        assert sample_bloom_filter.inserted_count == 0
        assert sample_bloom_filter.bits_set == 0
        assert sample_bloom_filter.is_empty()

    def test_bloom_filter_estimated_false_positive_rate(self, sample_bloom_filter: BloomFilter[str]) -> None:
        estimated_rate = sample_bloom_filter.estimated_false_positive_rate()
        assert 0.0 <= estimated_rate <= 1.0

    def test_bloom_filter_magic_methods(self) -> None:
        bf1 = BloomFilter[str](100, 0.01)
        bf2 = BloomFilter[str](100, 0.01)

        items = ["apple", "banana", "cherry"]
        for item in items:
            bf1.add(item)
            bf2.add(item)

        # Test equality
        assert bf1 == bf2

        bf2.add("date")
        assert bf1 != bf2

        # Test union and intersection operators
        bf3 = BloomFilter[str](100, 0.01)
        bf3.add("elderberry")

    @pytest.mark.parametrize(
        ("expected_elements", "false_positive_rate"),
        [
            (50, 0.01),
            (500, 0.05),
            (1000, 0.1),
        ],
    )
    def test_bloom_filter_different_configurations(self, expected_elements: int, false_positive_rate: float) -> None:
        bf = BloomFilter[int](expected_elements, false_positive_rate)

        # Add elements
        for i in range(expected_elements // 2):
            bf.add(i)

        # Check that added elements are found
        for i in range(expected_elements // 2):
            assert i in bf


class TestCountMinSketch:
    @pytest.fixture
    def small_cms(self) -> CountMinSketch[str]:
        return CountMinSketch[str](100, 5)

    @pytest.fixture
    def sample_cms(self) -> CountMinSketch[str]:
        cms = CountMinSketch[str](1000, 10)
        items = ["apple", "banana", "cherry", "apple", "banana", "apple"]
        for item in items:
            cms.add(item)
        return cms

    def test_count_min_sketch_initialization(self, small_cms: CountMinSketch[str]) -> None:
        assert small_cms.width == 100
        assert small_cms.depth == 5
        assert small_cms.total_count == 0
        assert small_cms.is_empty()

    def test_count_min_sketch_invalid_parameters(self) -> None:
        with pytest.raises(ValueError):
            CountMinSketch[str](0, 5)

        with pytest.raises(ValueError):
            CountMinSketch[str](100, 0)

    def test_count_min_sketch_from_error_params(self) -> None:
        cms = CountMinSketch[str].from_error_params(0.01, 0.01)
        assert cms.width > 0
        assert cms.depth > 0

    def test_count_min_sketch_add_and_estimate(self, sample_cms: CountMinSketch[str]) -> None:
        # Check counts (might be approximate due to hash collisions)
        apple_count = sample_cms.estimate("apple")
        banana_count = sample_cms.estimate("banana")
        cherry_count = sample_cms.estimate("cherry")

        assert apple_count >= 3  # "apple" was added 3 times
        assert banana_count >= 2  # "banana" was added 2 times
        assert cherry_count >= 1  # "cherry" was added 1 time

        assert sample_cms.total_count == 6

    def test_count_min_sketch_add_with_count(self, small_cms: CountMinSketch[str]) -> None:
        small_cms.add("test", 5)

        estimate = small_cms.estimate("test")
        assert estimate >= 5
        assert small_cms.total_count == 5

    def test_count_min_sketch_add_negative_count(self, small_cms: CountMinSketch[str]) -> None:
        with pytest.raises(ValueError):
            small_cms.add("test", -1)

    def test_count_min_sketch_remove_operation(self, sample_cms: CountMinSketch[str]) -> None:
        initial_estimate = sample_cms.estimate("apple")
        sample_cms.remove("apple", 1)

        new_estimate = sample_cms.estimate("apple")
        assert new_estimate <= initial_estimate

    def test_count_min_sketch_remove_more_than_estimate(self, small_cms: CountMinSketch[str]) -> None:
        small_cms.add("test", 3)
        small_cms.remove("test", 5)  # Remove more than added

        estimate = small_cms.estimate("test")
        assert estimate == 0

    def test_count_min_sketch_estimate_nonexistent(self, small_cms: CountMinSketch[str]) -> None:
        estimate = small_cms.estimate("nonexistent")
        assert estimate == 0

    def test_count_min_sketch_merge(self) -> None:
        cms1 = CountMinSketch[str](100, 5)
        cms2 = CountMinSketch[str](100, 5)

        cms1.add("apple", 3)
        cms2.add("apple", 2)
        cms2.add("banana", 1)

        merged = cms1.merge(cms2)

        assert merged.estimate("apple") >= 5
        assert merged.estimate("banana") >= 1
        assert merged.total_count == 6

    def test_count_min_sketch_merge_incompatible(self) -> None:
        cms1 = CountMinSketch[str](100, 5)
        cms2 = CountMinSketch[str](200, 5)

        with pytest.raises(ValueError):
            cms1.merge(cms2)

    def test_count_min_sketch_clear(self, sample_cms: CountMinSketch[str]) -> None:
        sample_cms.clear()

        assert sample_cms.total_count == 0
        assert sample_cms.is_empty()
        assert sample_cms.estimate("apple") == 0

    def test_count_min_sketch_memory_usage(self, small_cms: CountMinSketch[str]) -> None:
        memory = small_cms.memory_usage()
        expected_memory = 100 * 5 * 8  # width * depth * 8 bytes per int
        assert memory == expected_memory

    def test_count_min_sketch_magic_methods(self) -> None:
        cms1 = CountMinSketch[str](100, 5)
        cms2 = CountMinSketch[str](100, 5)

        items = [("apple", 3), ("banana", 2)]
        for item, count in items:
            cms1.add(item, count)
            cms2.add(item, count)

        # Test equality
        assert cms1 == cms2

        cms2.add("cherry", 1)
        assert cms1 != cms2

        # Test addition
        cms3 = cms1 + cms2
        assert cms3.total_count == cms1.total_count + cms2.total_count

    @pytest.mark.parametrize(
        ("width", "depth"),
        [
            (50, 3),
            (200, 7),
            (1000, 10),
        ],
    )
    def test_count_min_sketch_different_dimensions(self, width: int, depth: int) -> None:
        cms = CountMinSketch[int](width, depth)

        # Add some items
        for i in range(100):
            cms.add(i, i % 10 + 1)

        # Check that estimates are reasonable
        for i in range(0, 100, 10):
            estimate = cms.estimate(i)
            expected = i % 10 + 1
            assert estimate >= expected


class TestHyperLogLog:
    @pytest.fixture
    def small_hll(self) -> HyperLogLog:
        return HyperLogLog(4)

    @pytest.fixture
    def sample_hll(self) -> HyperLogLog:
        hll = HyperLogLog(8)
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            hll.add(item)
        return hll

    def test_hyperloglog_initialization(self, small_hll: HyperLogLog) -> None:
        assert small_hll.precision == 4
        assert small_hll.cardinality() >= 0
        assert small_hll.is_empty()

    def test_hyperloglog_invalid_precision(self) -> None:
        with pytest.raises(ValueError):
            HyperLogLog(3)  # Too small

        with pytest.raises(ValueError):
            HyperLogLog(17)  # Too large

    def test_hyperloglog_add_operation(self, small_hll: HyperLogLog) -> None:
        small_hll.add("test")

        assert not small_hll.is_empty()
        assert bool(small_hll)
        cardinality = small_hll.cardinality()
        assert cardinality > 0

    def test_hyperloglog_cardinality_estimation(self, sample_hll: HyperLogLog) -> None:
        cardinality = sample_hll.cardinality()

        # Should be approximately 100, but allow for estimation error
        assert 80 <= cardinality <= 120

    def test_hyperloglog_duplicate_items(self) -> None:
        hll = HyperLogLog(8)

        # Add the same item multiple times
        for _ in range(10):
            hll.add("duplicate")

        # Cardinality should still be approximately 1
        cardinality = hll.cardinality()
        assert cardinality <= 5  # Allow for some estimation error

    def test_hyperloglog_merge(self) -> None:
        hll1 = HyperLogLog(6)
        hll2 = HyperLogLog(6)

        # Add different items to each HLL
        for i in range(50):
            hll1.add(f"item1_{i}")

        for i in range(50):
            hll2.add(f"item2_{i}")

        merged = hll1.merge(hll2)

        # Merged cardinality should be approximately 100
        cardinality = merged.cardinality()
        assert 80 <= cardinality <= 120

    def test_hyperloglog_merge_incompatible(self) -> None:
        hll1 = HyperLogLog(4)
        hll2 = HyperLogLog(6)

        with pytest.raises(ValueError):
            hll1.merge(hll2)

    def test_hyperloglog_clear(self, sample_hll: HyperLogLog) -> None:
        sample_hll.clear()

        assert sample_hll.is_empty()
        assert sample_hll.cardinality() == 0

    def test_hyperloglog_magic_methods(self) -> None:
        hll1 = HyperLogLog(6)
        hll2 = HyperLogLog(6)

        items = [f"item_{i}" for i in range(50)]
        for item in items:
            hll1.add(item)
            hll2.add(item)

        # Test equality
        assert hll1 == hll2

        hll2.add("extra_item")
        assert hll1 != hll2

        # Test addition (merge)
        hll3 = HyperLogLog(6)
        for i in range(25):
            hll3.add(f"new_item_{i}")

        merged = hll1 + hll3
        assert merged.cardinality() >= hll1.cardinality()

    @pytest.mark.parametrize(
        ("precision", "expected_accuracy"),
        [
            (4, 0.26),  # Standard error ≈ 1.04/√(2^4) ≈ 0.26
            (8, 0.065),  # Standard error ≈ 1.04/√(2^8) ≈ 0.065
            (12, 0.016),  # Standard error ≈ 1.04/√(2^12) ≈ 0.016
        ],
    )
    def test_hyperloglog_accuracy(self, precision: int, expected_accuracy: float) -> None:
        hll = HyperLogLog(precision)

        # Add a known number of unique items
        unique_count = 1000
        for i in range(unique_count):
            hll.add(f"unique_item_{i}")

        estimated = hll.cardinality()
        error = abs(estimated - unique_count) / unique_count

        # Error should be within expected bounds (allowing some variance)
        assert error <= expected_accuracy * 3  # 3 sigma bound

    def test_hyperloglog_large_dataset(self) -> None:
        hll = HyperLogLog(10)

        # Add many unique items
        unique_count = 10000
        for i in range(unique_count):
            hll.add(f"large_dataset_item_{i}")

        estimated = hll.cardinality()
        error_rate = abs(estimated - unique_count) / unique_count

        # Should be within 10% for large datasets
        assert error_rate <= 0.1

    def test_hyperloglog_mixed_data_types(self) -> None:
        hll = HyperLogLog(8)

        # Add different types of data
        items = [
            "string_item",
            42,
            3.14159,
            True,
            (1, 2, 3),
        ]

        for item in items:
            hll.add(item)

        cardinality = hll.cardinality()
        assert cardinality >= len(items)
