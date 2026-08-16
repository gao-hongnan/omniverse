from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from omnivault.dsa.containers.linear.queue import AbstractQueue, ArrayQueue, LinkedListQueue

if TYPE_CHECKING:
    from tests.omnivault.unit.dsa.conftest import TestData


class TestQueueImplementations:
    @pytest.fixture(params=[ArrayQueue, LinkedListQueue])
    def queue_class(self, request: pytest.FixtureRequest) -> type[AbstractQueue[Any]]:
        return cast("type[AbstractQueue[Any]]", request.param)

    @pytest.fixture
    def empty_queue(self, queue_class: type[AbstractQueue[Any]]) -> AbstractQueue[Any]:
        return queue_class()

    @pytest.fixture
    def filled_queue(self, queue_class: type[AbstractQueue[Any]], test_data: TestData) -> AbstractQueue[int]:
        queue: AbstractQueue[int] = queue_class()
        for item in test_data.integers[:5]:
            queue.enqueue(item)
        return queue

    @pytest.mark.unit
    def test_empty_queue_properties(self, empty_queue: AbstractQueue[Any]) -> None:
        assert empty_queue.is_empty()
        assert len(empty_queue) == 0
        assert not empty_queue
        assert list(empty_queue) == []
        assert empty_queue.to_list() == []

    @pytest.mark.unit
    def test_enqueue_single_item(self, empty_queue: AbstractQueue[int]) -> None:
        empty_queue.enqueue(42)
        assert not empty_queue.is_empty()
        assert len(empty_queue) == 1
        assert empty_queue.peek() == 42
        assert 42 in empty_queue

    @pytest.mark.unit
    def test_enqueue_dequeue_fifo_order(self, empty_queue: AbstractQueue[int], test_data: TestData) -> None:
        items = test_data.integers[:5]

        for item in items:
            empty_queue.enqueue(item)

        dequeued_items = []
        while not empty_queue.is_empty():
            dequeued_items.append(empty_queue.dequeue())

        assert dequeued_items == items

    @pytest.mark.unit
    def test_peek_does_not_modify_queue(self, filled_queue: AbstractQueue[int]) -> None:
        initial_len = len(filled_queue)
        first_item = filled_queue.peek()

        assert len(filled_queue) == initial_len
        assert filled_queue.peek() == first_item
        assert filled_queue.dequeue() == first_item

    @pytest.mark.unit
    def test_dequeue_from_empty_raises_error(self, empty_queue: AbstractQueue[Any]) -> None:
        with pytest.raises(IndexError, match="empty queue"):
            empty_queue.dequeue()

    @pytest.mark.unit
    def test_peek_from_empty_raises_error(self, empty_queue: AbstractQueue[Any]) -> None:
        with pytest.raises(IndexError, match="empty queue"):
            empty_queue.peek()

    @pytest.mark.unit
    def test_clear_queue(self, filled_queue: AbstractQueue[int]) -> None:
        filled_queue.clear()
        assert filled_queue.is_empty()
        assert len(filled_queue) == 0

    @pytest.mark.unit
    def test_extend_multiple_items(self, empty_queue: AbstractQueue[int], test_data: TestData) -> None:
        items = test_data.integers[:5]
        empty_queue.extend(items)

        assert len(empty_queue) == len(items)
        assert empty_queue.to_list() == items

    @pytest.mark.unit
    def test_iterator_behavior(self, filled_queue: AbstractQueue[int], test_data: TestData) -> None:
        expected = test_data.integers[:5]
        assert list(filled_queue) == expected

        for i, item in enumerate(filled_queue):
            assert item == expected[i]

    @pytest.mark.unit
    def test_contains_operation(self, filled_queue: AbstractQueue[int], test_data: TestData) -> None:
        for item in test_data.integers[:5]:
            assert item in filled_queue

        assert 999 not in filled_queue

    @pytest.mark.unit
    def test_repr_output(self, queue_class: type[AbstractQueue[Any]]) -> None:
        queue: AbstractQueue[int] = queue_class([1, 2, 3])
        repr_str = repr(queue)
        assert queue_class.__name__ in repr_str
        assert "1" in repr_str
        assert "2" in repr_str
        assert "3" in repr_str

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_large_queue_operations(self, queue_class: type[AbstractQueue[Any]], size: int) -> None:
        queue: AbstractQueue[int] = queue_class()

        for i in range(size):
            queue.enqueue(i)

        assert len(queue) == size

        for i in range(size):
            assert queue.dequeue() == i

        assert queue.is_empty()

    @pytest.mark.edge_case
    def test_alternating_enqueue_dequeue(self, empty_queue: AbstractQueue[int]) -> None:
        empty_queue.enqueue(0)

        for i in range(1, 10):
            empty_queue.enqueue(i)
            assert empty_queue.dequeue() == i - 1

        assert len(empty_queue) == 1
        assert empty_queue.dequeue() == 9


class TestArrayQueueSpecific:
    @pytest.mark.unit
    def test_array_resizing_optimization(self) -> None:
        queue = ArrayQueue[int]()

        for i in range(1000):
            queue.enqueue(i)

        for _ in range(501):
            queue.dequeue()

        assert len(queue) == 499
        assert queue._front == 0

    @pytest.mark.unit
    def test_initialization_with_items(self, test_data: TestData) -> None:
        items = test_data.integers[:5]
        queue = ArrayQueue(items)

        assert len(queue) == len(items)
        assert queue.to_list() == items

        items.append(999)
        assert 999 not in queue


class TestLinkedListQueueSpecific:
    @pytest.mark.unit
    def test_initialization_with_items(self, test_data: TestData) -> None:
        items = test_data.integers[:5]
        queue = LinkedListQueue(items)

        assert len(queue) == len(items)
        assert queue.to_list() == items

    @pytest.mark.unit
    def test_front_rear_pointers(self) -> None:
        queue = LinkedListQueue[int]()

        assert (queue._front, queue._rear) == (None, None)

        queue.enqueue(1)
        front = queue._front
        assert front is queue._rear
        assert front is not None
        assert front.value == 1

        queue.enqueue(2)
        front, rear = queue._front, queue._rear
        assert front is not rear
        assert front is not None
        assert front.value == 1
        assert rear is not None
        assert rear.value == 2

        queue.dequeue()
        front = queue._front
        assert front is queue._rear
        assert front is not None
        assert front.value == 2

        queue.dequeue()
        assert (queue._front, queue._rear) == (None, None)
