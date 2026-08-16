from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from omnivault.dsa.containers.priority.priority_queue import BinaryHeapPriorityQueue, PriorityQueueItem
from omnivault.dsa.containers.priority.priority_queue.concrete import HeapType

if TYPE_CHECKING:
    pass


class TestPriorityQueueItem:
    @pytest.mark.unit
    def test_priority_queue_item_ordering(self) -> None:
        item1 = PriorityQueueItem(value="low", priority=1, insertion_order=0)
        item2 = PriorityQueueItem(value="medium", priority=2, insertion_order=1)
        item3 = PriorityQueueItem(value="high", priority=3, insertion_order=2)

        assert item1 < item2 < item3
        assert item3 > item2 > item1
        assert item1 <= item2 <= item3
        assert item3 >= item2 >= item1

    @pytest.mark.unit
    def test_priority_queue_item_same_priority_insertion_order(self) -> None:
        item1 = PriorityQueueItem(value="first", priority=1, insertion_order=0)
        item2 = PriorityQueueItem(value="second", priority=1, insertion_order=1)

        assert item1 < item2
        assert item2 > item1
        assert item1 != item2

    @pytest.mark.unit
    def test_priority_queue_item_equality(self) -> None:
        item1 = PriorityQueueItem(value="test", priority=1, insertion_order=0)
        item2 = PriorityQueueItem(value="different", priority=1, insertion_order=0)

        assert item1 == item2


class TestBinaryHeapPriorityQueue:
    @pytest.fixture(params=[HeapType.MIN, HeapType.MAX])
    def heap_type(self, request: pytest.FixtureRequest) -> HeapType:
        return cast("HeapType", request.param)

    @pytest.fixture
    def empty_pq(self, heap_type: HeapType) -> BinaryHeapPriorityQueue[str, int]:
        return BinaryHeapPriorityQueue(heap_type=heap_type)

    @pytest.fixture
    def filled_pq(self, heap_type: HeapType) -> BinaryHeapPriorityQueue[str, int]:
        pq = BinaryHeapPriorityQueue[str, int](heap_type=heap_type)
        priorities = [3, 1, 4, 1, 5, 9, 2, 6]
        values = ["c", "a1", "d", "a2", "e", "i", "b", "f"]

        for value, priority in zip(values, priorities, strict=False):
            pq.enqueue(value, priority)

        return pq

    @pytest.mark.unit
    def test_empty_priority_queue_properties(self, empty_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        assert empty_pq.is_empty()
        assert len(empty_pq) == 0
        assert not empty_pq
        assert list(empty_pq) == []
        assert empty_pq.to_list() == []

    @pytest.mark.unit
    def test_enqueue_single_item(self, empty_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        empty_pq.enqueue("test", 5)
        assert not empty_pq.is_empty()
        assert len(empty_pq) == 1
        assert empty_pq.peek() == "test"
        assert empty_pq.peek_priority() == 5
        assert "test" in empty_pq

    @pytest.mark.unit
    def test_min_heap_ordering(self) -> None:
        pq = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MIN)

        priorities = [3, 1, 4, 2, 5]
        values = ["c", "a", "d", "b", "e"]

        for value, priority in zip(values, priorities, strict=False):
            pq.enqueue(value, priority)

        expected_order = ["a", "b", "c", "d", "e"]
        dequeued = []

        while not pq.is_empty():
            dequeued.append(pq.dequeue())

        assert dequeued == expected_order

    @pytest.mark.unit
    def test_max_heap_ordering(self) -> None:
        pq = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MAX)

        priorities = [3, 1, 4, 2, 5]
        values = ["c", "a", "d", "b", "e"]

        for value, priority in zip(values, priorities, strict=False):
            pq.enqueue(value, priority)

        expected_order = ["e", "d", "c", "b", "a"]
        dequeued = []

        while not pq.is_empty():
            dequeued.append(pq.dequeue())

        assert dequeued == expected_order

    @pytest.mark.unit
    def test_stable_ordering_same_priority(self) -> None:
        pq = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MIN)

        pq.enqueue("first", 1)
        pq.enqueue("second", 1)
        pq.enqueue("third", 1)

        assert pq.dequeue() == "first"
        assert pq.dequeue() == "second"
        assert pq.dequeue() == "third"

    @pytest.mark.unit
    def test_peek_operations_dont_modify(self, filled_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        initial_len = len(filled_pq)

        peek_item = filled_pq.peek()
        peek_priority = filled_pq.peek_priority()

        assert len(filled_pq) == initial_len
        assert filled_pq.peek() == peek_item
        assert filled_pq.peek_priority() == peek_priority

    @pytest.mark.unit
    def test_dequeue_from_empty_raises_error(self, empty_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        with pytest.raises(IndexError, match="empty priority queue"):
            empty_pq.dequeue()

    @pytest.mark.unit
    def test_peek_from_empty_raises_error(self, empty_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        with pytest.raises(IndexError, match="empty priority queue"):
            empty_pq.peek()

        with pytest.raises(IndexError, match="empty priority queue"):
            empty_pq.peek_priority()

    @pytest.mark.unit
    def test_clear_priority_queue(self, filled_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        filled_pq.clear()
        assert filled_pq.is_empty()
        assert len(filled_pq) == 0

    @pytest.mark.unit
    def test_iterator_behavior(self, filled_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        items = list(filled_pq)
        assert len(items) == len(filled_pq)

        expected_values = ["c", "a1", "d", "a2", "e", "i", "b", "f"]
        for item in items:
            assert item in expected_values

    @pytest.mark.unit
    def test_contains_operation(self, filled_pq: BinaryHeapPriorityQueue[str, int]) -> None:
        assert "a1" in filled_pq
        assert "c" in filled_pq
        assert "nonexistent" not in filled_pq

    @pytest.mark.unit
    def test_repr_output(self) -> None:
        pq = BinaryHeapPriorityQueue[str, int]()
        pq.enqueue("test", 1)
        repr_str = repr(pq)
        assert "BinaryHeapPriorityQueue" in repr_str

    @pytest.mark.unit
    def test_change_priority_existing_item(self) -> None:
        pq = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MIN)

        pq.enqueue("low", 3)
        pq.enqueue("high", 1)

        assert pq.peek() == "high"

        success = pq.change_priority("low", 0)
        assert success
        assert pq.peek() == "low"

    @pytest.mark.unit
    def test_change_priority_nonexistent_item(self) -> None:
        pq = BinaryHeapPriorityQueue[str, int]()
        pq.enqueue("test", 1)

        success = pq.change_priority("nonexistent", 5)
        assert not success

    @pytest.mark.unit
    def test_merge_same_heap_type(self) -> None:
        pq1 = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MIN)
        pq2 = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MIN)

        pq1.enqueue("a", 1)
        pq1.enqueue("c", 3)

        pq2.enqueue("b", 2)
        pq2.enqueue("d", 4)

        pq1.merge(pq2)

        assert len(pq1) == 4

        dequeued = []
        while not pq1.is_empty():
            dequeued.append(pq1.dequeue())

        assert dequeued == ["a", "b", "c", "d"]

    @pytest.mark.unit
    def test_merge_different_heap_types_raises_error(self) -> None:
        pq1 = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MIN)
        pq2 = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MAX)

        with pytest.raises(ValueError, match="different heap types"):
            pq1.merge(pq2)

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_large_priority_queue_operations(self, heap_type: HeapType, size: int) -> None:
        pq = BinaryHeapPriorityQueue[int, int](heap_type=heap_type)

        for i in range(size):
            pq.enqueue(i, i)

        assert len(pq) == size

        if heap_type == HeapType.MIN:
            for i in range(size):
                assert pq.dequeue() == i
        else:
            for i in range(size - 1, -1, -1):
                assert pq.dequeue() == i

        assert pq.is_empty()

    @pytest.mark.edge_case
    def test_duplicate_values_different_priorities(self) -> None:
        pq = BinaryHeapPriorityQueue[str, int](heap_type=HeapType.MIN)

        pq.enqueue("same", 3)
        pq.enqueue("same", 1)
        pq.enqueue("same", 2)

        assert pq.dequeue() == "same"
        assert pq.dequeue() == "same"
        assert pq.dequeue() == "same"

    @pytest.mark.benchmark
    def test_performance_with_random_data(self, random_ints: list[int]) -> None:
        pq = BinaryHeapPriorityQueue[int, int]()

        for value in random_ints:
            pq.enqueue(value, value)

        sorted_values = []
        while not pq.is_empty():
            sorted_values.append(pq.dequeue())

        assert sorted_values == sorted(random_ints)
