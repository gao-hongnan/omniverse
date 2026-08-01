import pytest

from omnivault.dsa.stack.concrete import StackList


@pytest.fixture
def empty_stack() -> StackList[int]:
    """Fixture to create an empty StackList."""
    return StackList[int]()


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
            pytest.param([], id="empty"),
            pytest.param([1], id="single-int"),
            pytest.param([1, 2, 3], id="ints"),
            pytest.param(["a", "b", "c"], id="strs"),
            pytest.param([1.1, 2.2, 3.3], id="floats"),
        ],
    )
    def test_initialization[ItemT](self, initial_items: list[ItemT]) -> None:
        """Test initializing the stack with different initial items."""
        stack = StackList[ItemT]()

        for item in initial_items:
            stack.push(item)

        assert len(stack) == len(initial_items)
        assert stack.stack_items == initial_items

    def test_is_empty_on_new_stack(self, empty_stack: StackList[int]) -> None:
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

    def test_pop_empty_stack(self, empty_stack: StackList[int]) -> None:
        """Test popping from an empty stack raises with the exact message."""
        # The source raises a bare `Exception("Stack is empty")`, so Exception
        # is the narrowest type available; the match pins the exact message.
        with pytest.raises(Exception, match="Stack is empty"):
            empty_stack.pop()

    def test_peek_empty_stack(self, empty_stack: StackList[int]) -> None:
        """Test peeking an empty stack raises IndexError from the backing list."""
        with pytest.raises(IndexError, match="list index out of range"):
            empty_stack.peek()

    def test_iteration(self, populated_stack: StackList[int]) -> None:
        """Test iterating over the stack."""
        items: list[int] = list(populated_stack)

        assert items == [5, 4, 3, 2, 1]
        assert populated_stack.is_empty()
        assert len(populated_stack) == 0

    @pytest.mark.parametrize(
        "items",
        [
            pytest.param([1], id="single-int"),
            pytest.param([1, 2], id="two-ints"),
            pytest.param([1, 2, 3], id="three-ints"),
            pytest.param(["x", "y", "z"], id="strs"),
            pytest.param([1.0, 2.0, 3.0], id="floats"),
        ],
    )
    def test_stack_with_various_types[ItemT](self, items: list[ItemT]) -> None:
        """Test stack operations with various data types."""
        stack = StackList[ItemT]()
        for item in items:
            stack.push(item)

        assert len(stack) == len(items)
        assert stack.peek() == items[-1]

        popped_items = [stack.pop() for _ in items]
        assert popped_items == list(reversed(items))
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
        """Test the official string representation."""
        expected_repr = "StackList(stack_items=[1, 2, 3, 4, 5])"

        assert repr(populated_stack) == expected_repr
