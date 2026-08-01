from typing import Protocol


class QueueProtocol[ItemT](Protocol):
    """Protocol defining the interface for a basic queue.

    This protocol defines the minimum interface that any queue implementation
    must satisfy. It specifies a FIFO (First-In-First-Out) queue where items
    are added to the rear and removed from the front.

    Type Parameters
    --------------
    ItemT
        The type of items stored in the queue, covariant since items are only
        retrieved, not stored through the base protocol methods
    """

    @property
    def size(self) -> int:
        """Get the number of items in the queue.

        Returns
        -------
        int
            Current number of items in the queue
        """
        ...

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns
        -------
        bool
            True if queue contains no items, False otherwise
        """
        ...

    def peek(self) -> ItemT:
        """View the next item to be dequeued without removing it.

        Returns
        -------
        ItemT
            The item at the front of the queue

        Raises
        ------
        Exception
            If the queue is empty
        """
        ...


class MutableQueueProtocol[ItemT](QueueProtocol[ItemT], Protocol):
    """Protocol defining a mutable queue interface.

    Extends the basic QueueProtocol to include methods for modifying the queue.
    This protocol defines a standard FIFO queue interface with enqueue and
    dequeue operations.

    Type Parameters
    --------------
    ItemT
        The type of items stored in the queue
    """

    def enqueue(self, item: ItemT) -> None:
        """Add an item to the rear of the queue.

        Parameters
        ----------
        item : ItemT
            The item to add
        """
        ...

    def dequeue(self) -> ItemT:
        """Remove and return the item at the front of the queue.

        Returns
        -------
        ItemT
            The item at the front of the queue

        Raises
        ------
        Exception
            If the queue is empty
        """
        ...


class DequeProtocol[ItemT](Protocol):
    """Protocol defining the interface for a basic double-ended queue.

    This protocol defines the minimum read-only interface that any deque
    implementation must satisfy. It allows inspection of both ends of the queue.

    Type Parameters
    --------------
    ItemT
        The type of items stored in the deque, covariant since items are only
        retrieved, not stored through the base protocol methods
    """

    @property
    def size(self) -> int:
        """Get the number of items in the deque.

        Returns
        -------
        int
            Current number of items in the deque
        """
        ...

    def is_empty(self) -> bool:
        """Check if the deque is empty.

        Returns
        -------
        bool
            True if deque contains no items, False otherwise
        """
        ...

    def peek_front(self) -> ItemT:
        """View the item at the front without removing it.

        Returns
        -------
        ItemT
            The item at the front of the deque

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...

    def peek_rear(self) -> ItemT:
        """View the item at the rear without removing it.

        Returns
        -------
        ItemT
            The item at the rear of the deque

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...


class MutableDequeProtocol[ItemT](DequeProtocol[ItemT], Protocol):
    """Protocol defining a mutable double-ended queue interface.

    Extends the basic DequeProtocol to include methods for modifying the deque.
    This protocol defines a standard double-ended queue interface with operations
    at both ends.

    Type Parameters
    --------------
    ItemT
        The type of items stored in the deque
    """

    def add_front(self, item: ItemT) -> None:
        """Add an item to the front of the deque.

        Parameters
        ----------
        item : ItemT
            The item to add to the front
        """
        ...

    def add_rear(self, item: ItemT) -> None:
        """Add an item to the rear of the deque.

        Parameters
        ----------
        item : ItemT
            The item to add to the rear
        """
        ...

    def remove_front(self) -> ItemT:
        """Remove and return the item from the front.

        Returns
        -------
        ItemT
            The item from the front

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...

    def remove_rear(self) -> ItemT:
        """Remove and return the item from the rear.

        Returns
        -------
        ItemT
            The item from the rear

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...
