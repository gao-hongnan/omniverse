from __future__ import annotations

from typing import Generic

from ...core.errors import InvalidConfiguration
from ...core.types import KeyT, ValueT


class _LFUNode(Generic[KeyT, ValueT]):  # noqa: UP046
    __slots__ = ("key", "value", "freq", "prev", "next")

    def __init__(self, key: KeyT, value: ValueT, freq: int) -> None:
        self.key: KeyT = key
        self.value: ValueT = value
        self.freq: int = freq
        self.prev: _LFUNode[KeyT, ValueT] | None = None
        self.next: _LFUNode[KeyT, ValueT] | None = None


class _FreqBucket(Generic[KeyT, ValueT]):  # noqa: UP046
    __slots__ = ("head", "tail", "size")

    def __init__(self) -> None:
        self.head: _LFUNode[KeyT, ValueT] = _LFUNode.__new__(_LFUNode)
        self.tail: _LFUNode[KeyT, ValueT] = _LFUNode.__new__(_LFUNode)
        self.head.prev = None
        self.head.next = self.tail
        self.tail.prev = self.head
        self.tail.next = None
        self.size: int = 0

    def push_front(self, node: _LFUNode[KeyT, ValueT]) -> None:
        first = self.head.next
        assert first is not None
        node.prev = self.head
        node.next = first
        self.head.next = node
        first.prev = node
        self.size += 1

    def unlink(self, node: _LFUNode[KeyT, ValueT]) -> None:
        prev_node = node.prev
        next_node = node.next
        assert prev_node is not None and next_node is not None
        prev_node.next = next_node
        next_node.prev = prev_node
        self.size -= 1

    def pop_back(self) -> _LFUNode[KeyT, ValueT]:
        victim = self.tail.prev
        assert victim is not None and victim is not self.head
        self.unlink(victim)
        return victim

    def is_empty(self) -> bool:
        return self.size == 0


class LFUCache(Generic[KeyT, ValueT]):  # noqa: UP046
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise InvalidConfiguration(f"capacity must be positive, got {capacity}")
        self._capacity: int = capacity
        self._index: dict[KeyT, _LFUNode[KeyT, ValueT]] = {}
        self._buckets: dict[int, _FreqBucket[KeyT, ValueT]] = {}
        self._min_freq: int = 0

    def _bucket(self, freq: int) -> _FreqBucket[KeyT, ValueT]:
        bucket = self._buckets.get(freq)
        if bucket is None:
            bucket = _FreqBucket()
            self._buckets[freq] = bucket
        return bucket

    def _promote(self, node: _LFUNode[KeyT, ValueT]) -> None:
        current = self._buckets[node.freq]
        current.unlink(node)
        if current.is_empty() and node.freq == self._min_freq:
            self._min_freq += 1
        node.freq += 1
        self._bucket(node.freq).push_front(node)

    def get(self, key: KeyT) -> ValueT | None:
        node = self._index.get(key)
        if node is None:
            return None
        self._promote(node)
        return node.value

    def put(self, key: KeyT, value: ValueT) -> None:
        existing = self._index.get(key)
        if existing is not None:
            existing.value = value
            self._promote(existing)
            return
        if len(self._index) >= self._capacity:
            victim_bucket = self._buckets[self._min_freq]
            victim = victim_bucket.pop_back()
            del self._index[victim.key]
        node: _LFUNode[KeyT, ValueT] = _LFUNode(key, value, freq=1)
        self._bucket(1).push_front(node)
        self._index[key] = node
        self._min_freq = 1

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        return key in self._index
