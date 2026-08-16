from __future__ import annotations

from typing import Generic

from ...core.errors import InvalidConfiguration
from ...core.types import KeyT, ValueT


class _LRUNode(Generic[KeyT, ValueT]):  # noqa: UP046
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: KeyT, value: ValueT) -> None:
        self.key: KeyT = key
        self.value: ValueT = value
        self.prev: _LRUNode[KeyT, ValueT] | None = None
        self.next: _LRUNode[KeyT, ValueT] | None = None


class LRUCache(Generic[KeyT, ValueT]):  # noqa: UP046
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise InvalidConfiguration(f"capacity must be positive, got {capacity}")
        self._capacity: int = capacity
        self._index: dict[KeyT, _LRUNode[KeyT, ValueT]] = {}
        self._head: _LRUNode[KeyT, ValueT] = _LRUNode.__new__(_LRUNode)
        self._tail: _LRUNode[KeyT, ValueT] = _LRUNode.__new__(_LRUNode)
        self._head.prev = None
        self._head.next = self._tail
        self._tail.prev = self._head
        self._tail.next = None

    def _unlink(self, node: _LRUNode[KeyT, ValueT]) -> None:
        prev_node = node.prev
        next_node = node.next
        assert prev_node is not None and next_node is not None
        prev_node.next = next_node
        next_node.prev = prev_node

    def _push_front(self, node: _LRUNode[KeyT, ValueT]) -> None:
        first = self._head.next
        assert first is not None
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node

    def _evict_lru(self) -> None:
        victim = self._tail.prev
        assert victim is not None and victim is not self._head
        self._unlink(victim)
        del self._index[victim.key]

    def get(self, key: KeyT) -> ValueT | None:
        node = self._index.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key: KeyT, value: ValueT) -> None:
        existing = self._index.get(key)
        if existing is not None:
            existing.value = value
            self._unlink(existing)
            self._push_front(existing)
            return
        if len(self._index) >= self._capacity:
            self._evict_lru()
        node: _LRUNode[KeyT, ValueT] = _LRUNode(key, value)
        self._push_front(node)
        self._index[key] = node

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        return key in self._index
