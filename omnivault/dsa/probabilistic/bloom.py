from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from typing import Generic, cast

from ..core.errors import IncompatibleSketch, InvalidProbability
from ..core.types import ItemT


class BloomFilter(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_bit_array", "_size", "_hash_count", "_inserted_elements")

    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01) -> None:
        if expected_elements <= 0:
            raise InvalidProbability("Expected elements must be positive")
        if not 0 < false_positive_rate < 1:
            raise InvalidProbability("False positive rate must be between 0 and 1")

        self._size = self._optimal_size(expected_elements, false_positive_rate)
        self._hash_count = self._optimal_hash_count(self._size, expected_elements)
        self._bit_array = [False] * self._size
        self._inserted_elements = 0

    @staticmethod
    def _optimal_size(expected_elements: int, false_positive_rate: float) -> int:
        return int(-expected_elements * math.log(false_positive_rate) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_hash_count(bit_array_size: int, expected_elements: int) -> int:
        return max(1, int(bit_array_size / expected_elements * math.log(2)))

    def _hash_functions(self, item: ItemT) -> Iterator[int]:
        item_bytes = str(item).encode("utf-8")

        hash1 = int(hashlib.md5(item_bytes, usedforsecurity=False).hexdigest(), 16)
        hash2 = int(hashlib.sha1(item_bytes, usedforsecurity=False).hexdigest(), 16)

        for i in range(self._hash_count):
            yield (hash1 + i * hash2) % self._size

    def add(self, item: ItemT) -> None:
        for hash_value in self._hash_functions(item):
            self._bit_array[hash_value] = True
        self._inserted_elements += 1

    def __contains__(self, item: object) -> bool:
        if getattr(item, "__hash__", None) is None:
            return False

        return all(self._bit_array[hash_value] for hash_value in self._hash_functions(cast(ItemT, item)))

    def might_contain(self, item: ItemT) -> bool:
        return item in self

    def definitely_not_contains(self, item: ItemT) -> bool:
        return item not in self

    @property
    def size(self) -> int:
        return self._size

    @property
    def hash_count(self) -> int:
        return self._hash_count

    @property
    def inserted_count(self) -> int:
        return self._inserted_elements

    @property
    def bits_set(self) -> int:
        return sum(self._bit_array)

    def estimated_false_positive_rate(self) -> float:
        if self._inserted_elements == 0:
            return 0.0

        proportion_set = self.bits_set / self._size
        return proportion_set**self._hash_count

    def union(self, other: BloomFilter[ItemT]) -> BloomFilter[ItemT]:
        if self._size != other._size or self._hash_count != other._hash_count:
            raise IncompatibleSketch("Bloom filters must have same size and hash count")

        result = BloomFilter.__new__(BloomFilter)
        result._size = self._size
        result._hash_count = self._hash_count
        result._bit_array = [a or b for a, b in zip(self._bit_array, other._bit_array, strict=True)]
        result._inserted_elements = self._inserted_elements + other._inserted_elements

        return result

    def intersection(self, other: BloomFilter[ItemT]) -> BloomFilter[ItemT]:
        if self._size != other._size or self._hash_count != other._hash_count:
            raise IncompatibleSketch("Bloom filters must have same size and hash count")

        result = BloomFilter.__new__(BloomFilter)
        result._size = self._size
        result._hash_count = self._hash_count
        result._bit_array = [a and b for a, b in zip(self._bit_array, other._bit_array, strict=True)]
        result._inserted_elements = min(self._inserted_elements, other._inserted_elements)

        return result

    def clear(self) -> None:
        self._bit_array = [False] * self._size
        self._inserted_elements = 0

    def is_empty(self) -> bool:
        return self._inserted_elements == 0

    def __len__(self) -> int:
        return self._inserted_elements

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self._size}, "
            f"hash_count={self._hash_count}, inserted={self._inserted_elements})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BloomFilter):
            return NotImplemented

        return (
            self._size == other._size and self._hash_count == other._hash_count and self._bit_array == other._bit_array
        )

    def __or__(self, other: BloomFilter[ItemT]) -> BloomFilter[ItemT]:
        return self.union(other)

    def __and__(self, other: BloomFilter[ItemT]) -> BloomFilter[ItemT]:
        return self.intersection(other)
