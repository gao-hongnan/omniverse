from __future__ import annotations

import hashlib
import math

from ..core.errors import IncompatibleSketch, InvalidProbability


class HyperLogLog:
    __slots__ = ("_buckets", "_bucket_count", "_alpha")

    def __init__(self, precision: int = 4) -> None:
        if not 4 <= precision <= 16:
            raise InvalidProbability("Precision must be between 4 and 16")

        self._bucket_count = 1 << precision
        self._buckets = [0] * self._bucket_count

        if precision == 4:
            self._alpha = 0.673
        elif precision == 5:
            self._alpha = 0.697
        elif precision == 6:
            self._alpha = 0.709
        else:
            self._alpha = 0.7213 / (1 + 1.079 / self._bucket_count)

    def _hash(self, item: object) -> int:
        return int(hashlib.md5(str(item).encode("utf-8"), usedforsecurity=False).hexdigest(), 16)

    def _leading_zeros(self, value: int) -> int:
        if value == 0:
            return 32

        count = 0
        while (value & 0x80000000) == 0:
            count += 1
            value <<= 1

        return count + 1

    def add(self, item: object) -> None:
        hash_value = self._hash(item)

        bucket_index = hash_value & (self._bucket_count - 1)
        remaining_bits = hash_value >> int(math.log2(self._bucket_count))

        leading_zeros = self._leading_zeros(remaining_bits)

        self._buckets[bucket_index] = max(self._buckets[bucket_index], leading_zeros)

    def cardinality(self) -> int:
        raw_estimate = self._alpha * (self._bucket_count**2) / sum(2 ** (-bucket) for bucket in self._buckets)

        if raw_estimate <= 2.5 * self._bucket_count:
            zeros = self._buckets.count(0)
            if zeros != 0:
                return int(self._bucket_count * math.log(self._bucket_count / zeros))

        if raw_estimate <= (1 / 30) * (1 << 32):
            return int(raw_estimate)
        else:
            return int(-1 * (1 << 32) * math.log(1 - raw_estimate / (1 << 32)))

    def merge(self, other: HyperLogLog) -> HyperLogLog:
        if self._bucket_count != other._bucket_count:
            raise IncompatibleSketch("HyperLogLog instances must have same bucket count")

        precision = int(math.log2(self._bucket_count))
        result = HyperLogLog(precision)

        for i in range(self._bucket_count):
            result._buckets[i] = max(self._buckets[i], other._buckets[i])

        return result

    def clear(self) -> None:
        self._buckets = [0] * self._bucket_count

    def is_empty(self) -> bool:
        return all(bucket == 0 for bucket in self._buckets)

    @property
    def precision(self) -> int:
        return int(math.log2(self._bucket_count))

    def __len__(self) -> int:
        return self.cardinality()

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(precision={self.precision}, cardinality={self.cardinality()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperLogLog):
            return NotImplemented

        return self._buckets == other._buckets

    def __add__(self, other: HyperLogLog) -> HyperLogLog:
        return self.merge(other)
