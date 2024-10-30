from typing import cast, List, TypeVar
import pytest

from omnivault.dsa.queue.concrete import QueueList, DeQueueList

from omnivault._types._generic import T


class TestQueueList:
    @pytest.fixture
    def empty_queue(self) -> QueueList[int]:
        return QueueList[int]()

    @pytest.fixture
    def populated_queue(self) -> QueueList[int]:
        queue: QueueList[int] = QueueList()
        for i in range(1, 4):  # Add 1, 2, 3
            queue.enqueue(i)
        return queue

    def test_queue_initialization(self, empty_queue: QueueList[int]) -> None:
        assert empty_queue.size == 0
        assert empty_queue.is_empty() is True
        assert empty_queue.queue_items == []

    def test_enqueue(self, empty_queue: QueueList[int]) -> None:
        empty_queue.enqueue(1)
        assert empty_queue.size == 1
        assert empty_queue.peek() == 1
        assert empty_queue.queue_items == [1]

        empty_queue.enqueue(2)
        assert empty_queue.size == 2
        assert empty_queue.peek() == 1
        assert empty_queue.queue_items == [2, 1]

    def test_dequeue(self, populated_queue: QueueList[int]) -> None:
        assert populated_queue.dequeue() == 1
        assert populated_queue.size == 2
        assert populated_queue.peek() == 2

        assert populated_queue.dequeue() == 2
        assert populated_queue.size == 1
        assert populated_queue.peek() == 3

        assert populated_queue.dequeue() == 3
        assert populated_queue.is_empty() is True

    def test_peek(self, populated_queue: QueueList[int]) -> None:
        assert populated_queue.peek() == 1
        assert populated_queue.size == 3  # Ensure peek doesn't remove item

    def test_empty_queue_operations(self, empty_queue: QueueList[int]) -> None:
        with pytest.raises(Exception, match="Queue is empty"):
            empty_queue.peek()

        with pytest.raises(Exception, match="Queue is empty"):
            empty_queue.dequeue()

    def test_iteration(self, populated_queue: QueueList[int]) -> None:
        expected: List[int] = [1, 2, 3]
        result: List[int] = []

        for item in populated_queue:
            result.append(item)

        assert result == expected
        assert populated_queue.is_empty() is True  # Iterator should consume queue

    def test_generic_type_support(self) -> None:
        string_queue: QueueList[str] = QueueList()
        string_queue.enqueue("hello")
        string_queue.enqueue("world")

        assert string_queue.dequeue() == "hello"
        assert string_queue.peek() == "world"


class TestDeQueueList:
    @pytest.fixture
    def empty_deque(self) -> DeQueueList[int]:
        return DeQueueList[int]()

    @pytest.fixture
    def populated_deque(self) -> DeQueueList[int]:
        deque: DeQueueList[int] = DeQueueList()
        for i in range(1, 4):  # Add 1, 2, 3
            deque.add_rear(i)
        return deque

    def test_deque_initialization(self, empty_deque: DeQueueList[int]) -> None:
        assert empty_deque.size == 0
        assert empty_deque.is_empty() is True
        assert empty_deque.queue_items == []

    def test_add_front(self, empty_deque: DeQueueList[int]) -> None:
        empty_deque.add_front(1)
        assert empty_deque.size == 1
        assert empty_deque.peek_front() == 1
        assert empty_deque.peek_rear() == 1

        empty_deque.add_front(2)
        assert empty_deque.size == 2
        assert empty_deque.peek_front() == 2
        assert empty_deque.peek_rear() == 1

    def test_add_rear(self, empty_deque: DeQueueList[int]) -> None:
        empty_deque.add_rear(1)
        assert empty_deque.size == 1
        assert empty_deque.peek_front() == 1
        assert empty_deque.peek_rear() == 1

        empty_deque.add_rear(2)
        assert empty_deque.size == 2
        assert empty_deque.peek_front() == 1
        assert empty_deque.peek_rear() == 2

    def test_remove_front(self, populated_deque: DeQueueList[int]) -> None:
        assert populated_deque.remove_front() == 1
        assert populated_deque.size == 2
        assert populated_deque.peek_front() == 2

    def test_remove_rear(self, populated_deque: DeQueueList[int]) -> None:
        assert populated_deque.remove_rear() == 3
        assert populated_deque.size == 2
        assert populated_deque.peek_rear() == 2

    def test_peek_operations(self, populated_deque: DeQueueList[int]) -> None:
        assert populated_deque.peek_front() == 1
        assert populated_deque.peek_rear() == 3
        assert populated_deque.size == 3  # Ensure peeks don't modify deque

    def test_empty_deque_operations(self, empty_deque: DeQueueList[int]) -> None:
        with pytest.raises(Exception, match="Queue is empty"):
            empty_deque.peek_front()

        with pytest.raises(Exception, match="Queue is empty"):
            empty_deque.peek_rear()

        with pytest.raises(Exception, match="Queue is empty"):
            empty_deque.remove_front()

        with pytest.raises(Exception, match="Queue is empty"):
            empty_deque.remove_rear()

    def test_generic_type_support(self) -> None:
        string_deque: DeQueueList[str] = DeQueueList()
        string_deque.add_front("hello")
        string_deque.add_rear("world")

        assert string_deque.peek_front() == "hello"
        assert string_deque.peek_rear() == "world"
        assert string_deque.remove_front() == "hello"
        assert string_deque.remove_rear() == "world"
