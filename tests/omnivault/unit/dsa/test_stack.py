from typing import Any, List, TypeVar

import pytest

from omnivault.dsa.stack.concrete import StackList

# Define a generic type variable for testing
T = TypeVar("T")


@pytest.fixture
def empty_stack() -> StackList[Any]:
    """Fixture to create an empty StackList."""
    return StackList()


@pytest.fixture
def populated_stack() -> StackList[int]:
    """Fixture to create a StackList populated with integers."""
    stack = StackList[int]()
    for item in [1, 2, 3, 4, 5]:
        stack.push(item)
    return stack


class TestStackList:
    """Test suite for the StackList class."""

    @pytest.mark.parametrize(
        "initial_items",
        [
            [],
            [1],
            [1, 2, 3],
            ["a", "b", "c"],
            [1.1, 2.2, 3.3],
        ],
    )
    def test_initialization(self, initial_items: List[T]) -> None:
        """Test initializing the stack with different initial items."""
        stack = StackList[T]()
        for item in initial_items:
            stack.push(item)
        assert len(stack) == len(initial_items)
        assert stack.stack_items == initial_items

    def test_is_empty_on_new_stack(self, empty_stack: StackList[Any]) -> None:
        """Test that a new stack is empty."""
        assert empty_stack.is_empty() is True
        assert len(empty_stack) == 0

    def test_push(self, empty_stack: StackList[int]) -> None:
        """Test pushing items onto the stack."""
        empty_stack.push(10)
        assert not empty_stack.is_empty()
        assert len(empty_stack) == 1
        assert empty_stack.peek() == 10

        empty_stack.push(20)
        assert len(empty_stack) == 2
        assert empty_stack.peek() == 20

    def test_pop(self, populated_stack: StackList[int]) -> None:
        """Test popping items from the stack."""
        assert len(populated_stack) == 5
        top = populated_stack.pop()
        assert top == 5
        assert len(populated_stack) == 4
        assert populated_stack.peek() == 4

        top = populated_stack.pop()
        assert top == 4
        assert len(populated_stack) == 3
        assert populated_stack.peek() == 3

    def test_peek(self, populated_stack: StackList[int]) -> None:
        """Test peeking the top item of the stack."""
        top = populated_stack.peek()
        assert top == 5
        assert len(populated_stack) == 5  # Ensure size is unchanged

    def test_pop_empty_stack(self, empty_stack: StackList[Any]) -> None:
        """Test popping from an empty stack raises an exception."""
        with pytest.raises(Exception) as exc_info:
            empty_stack.pop()
        assert str(exc_info.value) == "Stack is empty"

    def test_peek_empty_stack(self, empty_stack: StackList[Any]) -> None:
        """Test peeking an empty stack raises an exception."""
        with pytest.raises(IndexError):
            empty_stack.peek()  # This will raise IndexError as per list behavior

    def test_iteration(self, populated_stack: StackList[int]) -> None:
        """Test iterating over the stack."""
        items: List[int] = list(populated_stack)
        assert items == [5, 4, 3, 2, 1]
        assert populated_stack.is_empty()
        assert len(populated_stack) == 0

    @pytest.mark.parametrize(
        "items",
        [
            [1],
            [1, 2],
            [1, 2, 3],
            ["x", "y", "z"],
            [1.0, 2.0, 3.0],
        ],
    )
    def test_stack_with_various_types(self, items: List[T]) -> None:
        """Test stack operations with various data types."""
        stack = StackList[T]()
        for item in items:
            stack.push(item)
        assert len(stack) == len(items)
        assert stack.peek() == items[-1]

        for expected in reversed(items):
            popped = stack.pop()
            assert popped == expected
        assert stack.is_empty()

    def test_size_property(self, populated_stack: StackList[int]) -> None:
        """Test the size property of the stack."""
        assert populated_stack.size == 5
        populated_stack.pop()
        assert populated_stack.size == 4
        populated_stack.push(6)
        assert populated_stack.size == 5

    def test_len_dunder_method(self, populated_stack: StackList[int]) -> None:
        """Test the __len__ dunder method."""
        assert len(populated_stack) == 5
        populated_stack.pop()
        assert len(populated_stack) == 4

    def test_str_representation(self, populated_stack: StackList[int]) -> None:
        """Optionally, test the string representation if __str__ or __repr__ is defined."""
        # Assuming StackList has a __repr__ method
        expected_repr = "StackList(stack_items=[1, 2, 3, 4, 5])"
        assert repr(populated_stack) == expected_repr

    def test_exception_message_on_pop_empty(self, empty_stack: StackList[Any]) -> None:
        """Test the exception message when popping from an empty stack."""
        with pytest.raises(Exception) as exc_info:
            empty_stack.pop()
        assert "Stack is empty" in str(exc_info.value)
