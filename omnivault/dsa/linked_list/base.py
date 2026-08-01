from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, override, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class LinkedNode[ValueT](Protocol):
    """
    Protocol defining the minimum interface for a linked list node.
    Using Protocol allows for structural typing and better flexibility.
    """

    value: ValueT


class SinglyNode[ValueT](BaseModel):
    """Concrete implementation of a singly-linked node.

    A node implementation for singly linked lists that contains value and a reference
    to the next node in the sequence.

    Parameters
    ----------
    value : ValueT
        The current value to be stored in the node.
    next : SinglyNode[ValueT] | None, optional
        Reference to the next node in the sequence, by default None.

    Examples
    --------
    >>> node = SinglyNode(value=1)
    >>> print(node.value)
    1
    >>> print(node.next)
    None
    >>> node.next = SinglyNode(value=2)
    >>> print(node.next.value)
    2
    """

    value: ValueT
    next: SinglyNode[ValueT] | None = None


class DoublyNode[ValueT](BaseModel):
    """Concrete implementation of a doubly-linked node.

    A node implementation for doubly linked lists that contains value and references
    to both the next and previous nodes in the sequence.

    Parameters
    ----------
    value : ValueT
        The value to be stored in the node.
    next : DoublyNode[ValueT] | None, optional
        Reference to the next node in the sequence, by default None.
    prev : DoublyNode[ValueT] | None, optional
        Reference to the previous node in the sequence, by default None.

    Examples
    --------
    >>> node = DoublyNode(value=1)
    >>> print(node.value)
    1
    >>> node.next = DoublyNode(value=2)
    >>> node.next.prev = node
    >>> print(node.next.value)
    2
    >>> print(node.next.prev.value)
    1
    """

    value: ValueT
    next: DoublyNode[ValueT] | None = None
    prev: DoublyNode[ValueT] | None = None


@runtime_checkable
class LinkedListProtocol[ValueT](Protocol):
    """
    Protocol defining the public interface for linked lists.
    This ensures structural subtyping for any linked list implementation.
    """

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[ValueT]: ...

    def append(self, value: ValueT) -> None: ...

    def prepend(self, value: ValueT) -> None: ...

    def remove(self, value: ValueT) -> bool: ...

    def clear(self) -> None: ...


class AbstractLinkedList[ValueT, NodeT: LinkedNode[Any]](ABC):
    """
    Abstract base class for linked list implementations.

    Generic Parameters
    ------------------
    ValueT
        Type of value stored in the list
    NodeT
        Type of node used (must conform to LinkedNode protocol)

    Attributes
    ----------
    head : SinglyNode[ValueT] | None
        The head node of the linked list.
    size : int
        The number of nodes in the linked list.
    """

    def __init__(self, values: Sequence[ValueT] | None = None) -> None:
        self._head: NodeT | None = None
        self._size: int = 0

        self._values = values

        if values is not None:
            self.add_multiple_nodes()

    @abstractmethod
    def append(self, value: ValueT) -> None:
        """Add an element to the end of the list."""

    @abstractmethod
    def prepend(self, value: ValueT) -> None:
        """Add an element to the beginning of the list."""

    @abstractmethod
    def remove(self, value: ValueT) -> bool:
        """Remove the first occurrence of value from the list."""

    @abstractmethod
    def traverse(self) -> str:
        """Traverse through each node in the linked list and return a string."""

    def add_multiple_nodes(self) -> None:  # noqa: B027  # optional override hook; no-op default is intentional.
        """Add multiple nodes to the linked list.
        Useful when initializing a linked list with multiple values."""

    def clear(self) -> None:
        """Remove all elements from the list."""
        self._head = None
        self._size = 0

    def is_empty(self) -> bool:
        """Check if the linked list is empty.

        A linked list is empty if the head is None.

        Returns:
            bool: True if the linked list is empty, False otherwise.
        """
        return self.head is None

    @property
    def size(self) -> int:
        """Get the size of the linked list. The increment must be done in the
        `append`, `prepend` methods and also in the `remove` method since we
        need to decrement the size when a node is removed."""
        return self._size

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return bool(self._size)

    @override
    def __repr__(self) -> str:
        return self.traverse()

    @abstractmethod
    def __iter__(self) -> Iterator[ValueT]: ...

    @property
    def head(self) -> NodeT | None:
        """Get the head node of the list."""
        return self._head
