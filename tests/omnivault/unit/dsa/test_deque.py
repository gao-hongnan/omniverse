from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from omnivault.dsa.containers.linear.deque import AbstractDeque, ArrayDeque, LinkedListDeque

if TYPE_CHECKING:
    from tests.omnivault.unit.dsa.conftest import TestData


class TestDequeImplementations:
    @pytest.fixture(params=[ArrayDeque, LinkedListDeque])
    def deque_class(self, request: pytest.FixtureRequest) -> type[AbstractDeque[Any]]:
        return cast("type[AbstractDeque[Any]]", request.param)

    @pytest.fixture
    def empty_deque(self, deque_class: type[AbstractDeque[Any]]) -> AbstractDeque[Any]:
        return deque_class()

    @pytest.fixture
    def filled_deque(self, deque_class: type[AbstractDeque[Any]], test_data: TestData) -> AbstractDeque[int]:
        deque: AbstractDeque[int] = deque_class()
        for item in test_data.integers[:5]:
            deque.add_rear(item)
        return deque

    @pytest.mark.unit
    def test_empty_deque_properties(self, empty_deque: AbstractDeque[Any]) -> None:
        assert empty_deque.is_empty()
        assert len(empty_deque) == 0
        assert not empty_deque
        assert list(empty_deque) == []
        assert empty_deque.to_list() == []

    @pytest.mark.unit
    def test_add_front_single_item(self, empty_deque: AbstractDeque[int]) -> None:
        empty_deque.add_front(42)
        assert not empty_deque.is_empty()
        assert len(empty_deque) == 1
        assert empty_deque.peek_front() == 42
        assert empty_deque.peek_rear() == 42
        assert 42 in empty_deque

    @pytest.mark.unit
    def test_add_rear_single_item(self, empty_deque: AbstractDeque[int]) -> None:
        empty_deque.add_rear(42)
        assert not empty_deque.is_empty()
        assert len(empty_deque) == 1
        assert empty_deque.peek_front() == 42
        assert empty_deque.peek_rear() == 42
        assert 42 in empty_deque

    @pytest.mark.unit
    def test_add_both_ends(self, empty_deque: AbstractDeque[int]) -> None:
        empty_deque.add_rear(2)
        empty_deque.add_front(1)
        empty_deque.add_rear(3)
        empty_deque.add_front(0)

        assert len(empty_deque) == 4
        assert empty_deque.to_list() == [0, 1, 2, 3]

    @pytest.mark.unit
    def test_remove_front(self, filled_deque: AbstractDeque[int], test_data: TestData) -> None:
        expected_first = test_data.integers[0]
        removed = filled_deque.remove_front()

        assert removed == expected_first
        assert len(filled_deque) == 4
        assert filled_deque.peek_front() == test_data.integers[1]

    @pytest.mark.unit
    def test_remove_rear(self, filled_deque: AbstractDeque[int], test_data: TestData) -> None:
        expected_last = test_data.integers[4]
        removed = filled_deque.remove_rear()

        assert removed == expected_last
        assert len(filled_deque) == 4
        assert filled_deque.peek_rear() == test_data.integers[3]

    @pytest.mark.unit
    def test_peek_operations_dont_modify(self, filled_deque: AbstractDeque[int], test_data: TestData) -> None:
        initial_len = len(filled_deque)

        front = filled_deque.peek_front()
        rear = filled_deque.peek_rear()

        assert len(filled_deque) == initial_len
        assert filled_deque.peek_front() == front == test_data.integers[0]
        assert filled_deque.peek_rear() == rear == test_data.integers[4]

    @pytest.mark.unit
    def test_remove_from_empty_raises_error(self, empty_deque: AbstractDeque[Any]) -> None:
        with pytest.raises(IndexError, match="empty deque"):
            empty_deque.remove_front()

        with pytest.raises(IndexError, match="empty deque"):
            empty_deque.remove_rear()

    @pytest.mark.unit
    def test_peek_from_empty_raises_error(self, empty_deque: AbstractDeque[Any]) -> None:
        with pytest.raises(IndexError, match="empty deque"):
            empty_deque.peek_front()

        with pytest.raises(IndexError, match="empty deque"):
            empty_deque.peek_rear()

    @pytest.mark.unit
    def test_clear_deque(self, filled_deque: AbstractDeque[int]) -> None:
        filled_deque.clear()
        assert filled_deque.is_empty()
        assert len(filled_deque) == 0

    @pytest.mark.unit
    def test_extend_operations(self, empty_deque: AbstractDeque[int], test_data: TestData) -> None:
        items = test_data.integers[:3]

        empty_deque.extend_rear(items)
        assert len(empty_deque) == 3

        empty_deque.extend_front([0, -1])
        assert empty_deque.to_list() == [-1, 0, 1, 2, 3]

    @pytest.mark.unit
    def test_convenience_methods(self, empty_deque: AbstractDeque[int]) -> None:
        empty_deque.append(1)
        empty_deque.appendleft(0)
        empty_deque.append(2)

        assert empty_deque.to_list() == [0, 1, 2]

        assert empty_deque.pop() == 2
        assert empty_deque.popleft() == 0
        assert empty_deque.to_list() == [1]

    @pytest.mark.unit
    def test_iterator_behavior(self, filled_deque: AbstractDeque[int], test_data: TestData) -> None:
        expected = test_data.integers[:5]
        assert list(filled_deque) == expected

        for i, item in enumerate(filled_deque):
            assert item == expected[i]

    @pytest.mark.unit
    def test_contains_operation(self, filled_deque: AbstractDeque[int], test_data: TestData) -> None:
        for item in test_data.integers[:5]:
            assert item in filled_deque

        assert 999 not in filled_deque

    @pytest.mark.unit
    def test_repr_output(self, deque_class: type[AbstractDeque[Any]]) -> None:
        deque: AbstractDeque[int] = deque_class([1, 2, 3])
        repr_str = repr(deque)
        assert deque_class.__name__ in repr_str
        assert "1" in repr_str
        assert "2" in repr_str
        assert "3" in repr_str

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_large_deque_operations(self, deque_class: type[AbstractDeque[Any]], size: int) -> None:
        deque: AbstractDeque[int] = deque_class()

        for i in range(size // 2):
            deque.add_rear(i)
            deque.add_front(-i - 1)

        assert len(deque) == size

        for i in range(size // 2):
            assert deque.remove_front() == -size // 2 + i
            assert deque.remove_rear() == size // 2 - i - 1

        assert deque.is_empty()

    @pytest.mark.edge_case
    def test_alternating_operations(self, empty_deque: AbstractDeque[int]) -> None:
        empty_deque.add_rear(0)
        empty_deque.add_front(-1)

        assert empty_deque.remove_front() == -1

        for i in range(1, 5):
            empty_deque.add_rear(i)
            empty_deque.add_front(-i - 1)

        assert len(empty_deque) == 9
        assert empty_deque.peek_front() == -5
        assert empty_deque.peek_rear() == 4


class TestArrayDequeSpecific:
    @pytest.mark.unit
    def test_initialization_with_items(self, test_data: TestData) -> None:
        items = test_data.integers[:5]
        deque = ArrayDeque(items)

        assert len(deque) == len(items)
        assert deque.to_list() == items

        items.append(999)
        assert 999 not in deque

    @pytest.mark.unit
    def test_maxlen_constraint(self) -> None:
        deque = ArrayDeque[int](maxlen=3)

        for i in range(5):
            deque.add_rear(i)

        assert len(deque) == 3
        assert deque.to_list() == [2, 3, 4]

    @pytest.mark.unit
    def test_maxlen_front_operations(self) -> None:
        deque = ArrayDeque([1, 2, 3], maxlen=3)

        deque.add_front(0)
        assert deque.to_list() == [0, 1, 2]

        deque.add_front(-1)
        assert deque.to_list() == [-1, 0, 1]

    @pytest.mark.unit
    def test_reverse_operation(self) -> None:
        deque = ArrayDeque([1, 2, 3, 4, 5])
        deque.reverse()
        assert deque.to_list() == [5, 4, 3, 2, 1]

    @pytest.mark.unit
    def test_rotate_operation(self) -> None:
        deque = ArrayDeque([1, 2, 3, 4, 5])

        deque.rotate(2)
        assert deque.to_list() == [4, 5, 1, 2, 3]

        deque.rotate(-2)
        assert deque.to_list() == [1, 2, 3, 4, 5]

    @pytest.mark.unit
    def test_rotate_empty_deque(self) -> None:
        deque = ArrayDeque[int]()
        deque.rotate(5)
        assert deque.is_empty()


class TestLinkedListDequeSpecific:
    @pytest.mark.unit
    def test_initialization_with_items(self, test_data: TestData) -> None:
        items = test_data.integers[:5]
        deque = LinkedListDeque(items)

        assert len(deque) == len(items)
        assert deque.to_list() == items

    @pytest.mark.unit
    def test_maxlen_constraint(self) -> None:
        deque = LinkedListDeque[int](maxlen=3)

        for i in range(5):
            deque.add_rear(i)

        assert len(deque) == 3
        assert deque.to_list() == [2, 3, 4]

    @pytest.mark.unit
    def test_node_pointers(self) -> None:
        deque = LinkedListDeque[int]()

        assert (deque._front, deque._rear) == (None, None)

        deque.add_rear(1)
        front = deque._front
        assert front is deque._rear
        assert front is not None
        assert front.value == 1

        deque.add_rear(2)
        front, rear = deque._front, deque._rear
        assert front is not rear
        assert front is not None
        assert front.value == 1
        assert rear is not None
        assert rear.value == 2

        deque.add_front(0)
        front, rear = deque._front, deque._rear
        assert front is not None
        assert rear is not None
        assert front.value == 0
        assert rear.value == 2
        assert front.next is not None
        assert front.next.value == 1

    @pytest.mark.unit
    def test_reverse_operation(self) -> None:
        deque = LinkedListDeque([1, 2, 3, 4, 5])
        deque.reverse()
        assert deque.to_list() == [5, 4, 3, 2, 1]

    @pytest.mark.unit
    def test_rotate_operation(self) -> None:
        deque = LinkedListDeque([1, 2, 3, 4, 5])

        deque.rotate(2)
        assert deque.to_list() == [4, 5, 1, 2, 3]

        deque.rotate(-1)
        assert deque.to_list() == [5, 1, 2, 3, 4]
