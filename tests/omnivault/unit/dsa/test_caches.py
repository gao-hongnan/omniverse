from __future__ import annotations

import pytest

from omnivault.dsa.containers.associative.lfu_cache import LFUCache
from omnivault.dsa.containers.associative.lru_cache import LRUCache
from omnivault.dsa.core.errors import InvalidConfiguration


@pytest.mark.unit
class TestLRUCache:
    def test_lc146_official_trace(self) -> None:
        cache: LRUCache[int, int] = LRUCache(capacity=2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1
        cache.put(3, 3)
        assert cache.get(2) is None
        cache.put(4, 4)
        assert cache.get(1) is None
        assert cache.get(3) == 3
        assert cache.get(4) == 4

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            LRUCache[int, int](capacity=0)
        with pytest.raises(InvalidConfiguration):
            LRUCache[int, int](capacity=-3)

    def test_update_existing_does_not_evict(self) -> None:
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 100)
        cache.put("c", 3)
        assert cache.get("a") == 100
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_contains_and_len(self) -> None:
        cache: LRUCache[str, int] = LRUCache(capacity=3)
        cache.put("x", 1)
        cache.put("y", 2)
        assert "x" in cache
        assert "z" not in cache
        assert len(cache) == 2


@pytest.mark.unit
class TestLFUCache:
    def test_lc460_official_trace(self) -> None:
        cache: LFUCache[int, int] = LFUCache(capacity=2)
        cache.put(1, 1)
        cache.put(2, 2)
        assert cache.get(1) == 1
        cache.put(3, 3)
        assert cache.get(2) is None
        assert cache.get(3) == 3
        cache.put(4, 4)
        assert cache.get(1) is None
        assert cache.get(3) == 3
        assert cache.get(4) == 4

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            LFUCache[int, int](capacity=0)

    def test_tie_break_by_recency(self) -> None:
        cache: LFUCache[str, int] = LFUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
