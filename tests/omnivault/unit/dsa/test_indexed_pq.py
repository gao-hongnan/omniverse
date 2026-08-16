from __future__ import annotations

import pytest

from omnivault.dsa.containers.priority.indexed_pq import IndexedPriorityQueue
from omnivault.dsa.core.errors import EmptyContainer, InvalidConfiguration, KeyNotFound


class TestIndexedPriorityQueue:
    @pytest.fixture
    def populated(self) -> IndexedPriorityQueue[str, int]:
        pq: IndexedPriorityQueue[str, int] = IndexedPriorityQueue()
        pq.insert("a", 5)
        pq.insert("b", 3)
        pq.insert("c", 7)
        return pq

    @pytest.mark.unit
    def test_pop_min_returns_smallest(self, populated: IndexedPriorityQueue[str, int]) -> None:
        assert populated.pop_min() == ("b", 3)

    @pytest.mark.unit
    def test_decrease_key_then_pop(self, populated: IndexedPriorityQueue[str, int]) -> None:
        populated.decrease_key("c", 1)
        assert populated.pop_min() == ("c", 1)
        assert populated.pop_min() == ("b", 3)
        assert populated.pop_min() == ("a", 5)

    @pytest.mark.unit
    def test_peek_min_does_not_remove(self, populated: IndexedPriorityQueue[str, int]) -> None:
        assert populated.peek_min() == ("b", 3)
        assert len(populated) == 3

    @pytest.mark.unit
    def test_contains_and_len(self, populated: IndexedPriorityQueue[str, int]) -> None:
        assert "a" in populated
        assert "z" not in populated
        assert len(populated) == 3

    @pytest.mark.unit
    def test_decrease_key_unknown_raises(self, populated: IndexedPriorityQueue[str, int]) -> None:
        with pytest.raises(KeyNotFound):
            populated.decrease_key("z", 1)

    @pytest.mark.unit
    def test_decrease_key_larger_priority_raises(self, populated: IndexedPriorityQueue[str, int]) -> None:
        with pytest.raises(InvalidConfiguration):
            populated.decrease_key("a", 99)

    @pytest.mark.unit
    def test_pop_min_empty_raises(self) -> None:
        pq: IndexedPriorityQueue[str, int] = IndexedPriorityQueue()
        with pytest.raises(EmptyContainer):
            pq.pop_min()

    @pytest.mark.unit
    def test_insert_duplicate_key_raises(self, populated: IndexedPriorityQueue[str, int]) -> None:
        with pytest.raises(InvalidConfiguration):
            populated.insert("a", 10)

    @pytest.mark.unit
    def test_is_empty(self) -> None:
        pq: IndexedPriorityQueue[str, int] = IndexedPriorityQueue()
        assert pq.is_empty()
        pq.insert("a", 1)
        assert not pq.is_empty()
