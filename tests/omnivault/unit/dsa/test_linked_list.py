from typing import Any

import pytest

from omnivault.dsa.linked_list.base import DoublyNode, SinglyNode


class TestSinglyNode:
    def test_init_with_data(self) -> None:
        """Test initialization with only value."""
        node = SinglyNode[int](value=1)
        assert node.value == 1
        assert node.next is None

    def test_init_with_next(self) -> None:
        """Test initialization with both value and next node."""
        next_node = SinglyNode[int](value=2)
        node = SinglyNode[int](value=1, next=next_node)
        assert node.value == 1
        assert node.next is next_node
        assert node.next.value == 2

    def test_link_nodes(self) -> None:
        """Test linking nodes after creation."""
        node1 = SinglyNode[str](value="first")
        node2 = SinglyNode[str](value="second")
        node1.next = node2
        assert node1.next is node2
        assert node1.next.value == "second"

    @pytest.mark.parametrize(
        "value",
        [
            42,  # int
            "hello",  # str
            3.14,  # float
            True,  # bool
            [1, 2, 3],  # list
            {"key": "value"},  # dict
        ],
    )
    def test_generic_type_support(self, value: Any) -> None:
        """Test that node supports various value types."""
        node = SinglyNode[Any](value=value)
        assert node.value == value


class TestDoublyNode:
    def test_init_with_data(self) -> None:
        """Test initialization with only value."""
        node = DoublyNode[int](value=1)
        assert node.value == 1
        assert node.next is None
        assert node.prev is None

    def test_init_with_next_and_prev(self) -> None:
        """Test initialization with value, next, and prev nodes."""
        node1 = DoublyNode[int](value=1)
        node2 = DoublyNode[int](value=2)
        node3 = DoublyNode[int](value=3)

        # Link nodes
        node2.prev = node1
        node2.next = node3
        node1.next = node2
        node3.prev = node2

        # Test node2's connections
        assert node2.value == 2
        assert node2.prev is node1
        assert node2.next is node3

        # Test bidirectional linking
        assert node2.prev.value == 1
        assert node2.next.value == 3
        assert node2.prev.next is node2
        assert node2.next.prev is node2

    def test_link_nodes(self) -> None:
        """Test linking nodes after creation."""
        node1 = DoublyNode[str](value="first")
        node2 = DoublyNode[str](value="second")
        node3 = DoublyNode[str](value="third")

        # Create chain: node1 <-> node2 <-> node3
        node1.next = node2
        node2.prev = node1
        node2.next = node3
        node3.prev = node2

        # Test forward traversal
        assert node1.next is node2
        assert node2.next is node3

        # Test backward traversal
        assert node3.prev is node2
        assert node2.prev is node1

    @pytest.mark.parametrize(
        "value",
        [
            42,  # int
            "hello",  # str
            3.14,  # float
            True,  # bool
            [1, 2, 3],  # list
            {"key": "value"},  # dict
        ],
    )
    def test_generic_type_support(self, value: Any) -> None:
        """Test that node supports various value types."""
        node = DoublyNode[Any](value=value)
        assert node.value == value

    def test_circular_reference(self) -> None:
        """Test creating a circular reference with doubly linked nodes."""
        node1 = DoublyNode[int](value=1)
        node2 = DoublyNode[int](value=2)

        # Create circular reference
        node1.next = node2
        node2.prev = node1
        node2.next = node1
        node1.prev = node2

        # Test circular nature
        assert node1.next is node2
        assert node2.next is node1
        assert node1.prev is node2
        assert node2.prev is node1
