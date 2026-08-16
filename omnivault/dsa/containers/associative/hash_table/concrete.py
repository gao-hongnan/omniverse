from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict
from rich.repr import Result

from ....core.errors import InvalidConfiguration, KeyNotFound
from .base import AbstractHashTable

if TYPE_CHECKING:
    from collections.abc import Iterator


class ProbingStrategy(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    DOUBLE_HASH = "double_hash"


class _KeyValuePair[KeyT, ValueT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: KeyT
    value: ValueT
    is_deleted: bool = False


class _Sentinel:
    def __repr__(self) -> str:
        return "DELETED"


DELETED = _Sentinel()


class ChainingHashTable[KeyT, ValueT](AbstractHashTable[KeyT, ValueT]):
    def __init__(self, initial_capacity: int = 16, max_load_factor: float = 0.75) -> None:
        if initial_capacity < 1:
            raise InvalidConfiguration("Initial capacity must be at least 1")
        if not 0 < max_load_factor <= 1:
            raise InvalidConfiguration("Load factor must be between 0 and 1")

        self._capacity = initial_capacity
        self._max_load_factor = max_load_factor
        self._size = 0
        self._buckets: list[list[_KeyValuePair[KeyT, ValueT]]] = [[] for _ in range(self._capacity)]

    def _hash(self, key: KeyT) -> int:
        return hash(key) % self._capacity

    def _should_resize(self) -> bool:
        return self._size / self._capacity >= self._max_load_factor

    def _resize(self) -> None:
        old_buckets = self._buckets
        self._capacity *= 2
        self._size = 0
        self._buckets = [[] for _ in range(self._capacity)]

        for bucket in old_buckets:
            for pair in bucket:
                self.put(pair.key, pair.value)

    def put(self, key: KeyT, value: ValueT) -> None:
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]

        for pair in bucket:
            if pair.key == key:
                pair.value = value
                return

        if self._should_resize():
            self._resize()
            bucket_index = self._hash(key)
            bucket = self._buckets[bucket_index]

        bucket.append(_KeyValuePair[KeyT, ValueT](key=key, value=value))
        self._size += 1

    def get(self, key: KeyT) -> ValueT:
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]

        for pair in bucket:
            if pair.key == key:
                return pair.value

        raise KeyNotFound(f"Key not found: {key}")

    def remove(self, key: KeyT) -> ValueT:
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]

        for i, pair in enumerate(bucket):
            if pair.key == key:
                removed_value = pair.value
                bucket.pop(i)
                self._size -= 1
                return removed_value

        raise KeyNotFound(f"Key not found: {key}")

    def contains_key(self, key: KeyT) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[KeyT]:
        for bucket in self._buckets:
            for pair in bucket:
                yield pair.key

    def keys(self) -> Iterator[KeyT]:
        return iter(self)

    def values(self) -> Iterator[ValueT]:
        for bucket in self._buckets:
            for pair in bucket:
                yield pair.value

    def items(self) -> Iterator[tuple[KeyT, ValueT]]:
        for bucket in self._buckets:
            for pair in bucket:
                yield (pair.key, pair.value)

    def clear(self) -> None:
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0

    def __rich_repr__(self) -> Result:
        yield from self.items()

    @property
    def load_factor(self) -> float:
        return self._size / self._capacity

    @property
    def capacity(self) -> int:
        return self._capacity


class OpenAddressingHashTable[KeyT, ValueT](AbstractHashTable[KeyT, ValueT]):
    def __init__(
        self,
        initial_capacity: int = 16,
        max_load_factor: float = 0.5,
        probing_strategy: ProbingStrategy = ProbingStrategy.LINEAR,
    ) -> None:
        if initial_capacity < 1:
            raise InvalidConfiguration("Initial capacity must be at least 1")
        if not 0 < max_load_factor <= 1:
            raise InvalidConfiguration("Load factor must be between 0 and 1")

        self._capacity = initial_capacity
        self._max_load_factor = max_load_factor
        self._probing_strategy = probing_strategy
        self._size = 0
        self._table: list[_KeyValuePair[KeyT, ValueT] | _Sentinel | None] = [None] * self._capacity

    def _hash(self, key: KeyT) -> int:
        return hash(key) % self._capacity

    def _hash2(self, key: KeyT) -> int:
        return 1 + 2 * (hash(key) % max(1, self._capacity // 2))

    def _probe(self, key: KeyT, attempt: int) -> int:
        if self._probing_strategy == ProbingStrategy.LINEAR:
            return (self._hash(key) + attempt) % self._capacity
        elif self._probing_strategy == ProbingStrategy.QUADRATIC:
            return (self._hash(key) + attempt * (attempt + 1) // 2) % self._capacity
        elif self._probing_strategy == ProbingStrategy.DOUBLE_HASH:
            return (self._hash(key) + attempt * self._hash2(key)) % self._capacity
        else:
            raise InvalidConfiguration(f"Unknown probing strategy: {self._probing_strategy}")

    def _should_resize(self) -> bool:
        return self._size / self._capacity >= self._max_load_factor

    def _resize(self) -> None:
        old_table = self._table
        self._capacity *= 2
        self._size = 0
        self._table = [None] * self._capacity

        for entry in old_table:
            if isinstance(entry, _KeyValuePair):
                self.put(entry.key, entry.value)

    def _find_slot_for_insertion(self, key: KeyT) -> tuple[int, bool] | None:
        first_deleted: int | None = None

        for attempt in range(self._capacity):
            index = self._probe(key, attempt)
            entry = self._table[index]

            if entry is None:
                return (index if first_deleted is None else first_deleted), False
            elif entry is DELETED:
                if first_deleted is None:
                    first_deleted = index
            elif isinstance(entry, _KeyValuePair) and entry.key == key:
                return index, True

        return None if first_deleted is None else (first_deleted, False)

    def _find_slot_for_search(self, key: KeyT) -> tuple[int, bool]:
        for attempt in range(self._capacity):
            index = self._probe(key, attempt)
            entry = self._table[index]

            if entry is None:
                return index, False
            elif isinstance(entry, _KeyValuePair) and entry.key == key:
                return index, True

        return 0, False

    def put(self, key: KeyT, value: ValueT) -> None:
        slot = self._find_slot_for_insertion(key)

        if slot is not None and slot[1]:
            entry = self._table[slot[0]]
            if isinstance(entry, _KeyValuePair):
                entry.value = value
            return

        while slot is None or self._should_resize():
            self._resize()
            slot = self._find_slot_for_insertion(key)

        self._table[slot[0]] = _KeyValuePair[KeyT, ValueT](key=key, value=value)
        self._size += 1

    def get(self, key: KeyT) -> ValueT:
        index, found = self._find_slot_for_search(key)

        if found:
            entry = self._table[index]
            if isinstance(entry, _KeyValuePair):
                return entry.value

        raise KeyNotFound(f"Key not found: {key}")

    def remove(self, key: KeyT) -> ValueT:
        index, found = self._find_slot_for_search(key)

        if found:
            entry = self._table[index]
            if isinstance(entry, _KeyValuePair):
                removed_value = entry.value
                self._table[index] = DELETED
                self._size -= 1
                return removed_value

        raise KeyNotFound(f"Key not found: {key}")

    def contains_key(self, key: KeyT) -> bool:
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def is_empty(self) -> bool:
        return self._size == 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[KeyT]:
        for entry in self._table:
            if isinstance(entry, _KeyValuePair):
                yield entry.key

    def keys(self) -> Iterator[KeyT]:
        return iter(self)

    def values(self) -> Iterator[ValueT]:
        for entry in self._table:
            if isinstance(entry, _KeyValuePair):
                yield entry.value

    def items(self) -> Iterator[tuple[KeyT, ValueT]]:
        for entry in self._table:
            if isinstance(entry, _KeyValuePair):
                yield (entry.key, entry.value)

    def clear(self) -> None:
        self._table = [None] * self._capacity
        self._size = 0

    def __rich_repr__(self) -> Result:
        yield from self.items()

    @property
    def load_factor(self) -> float:
        return self._size / self._capacity

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def probing_strategy(self) -> ProbingStrategy:
        return self._probing_strategy
