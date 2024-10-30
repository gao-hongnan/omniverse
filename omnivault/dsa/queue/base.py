from typing import Protocol

from omnivault._types._generic import T, T_co

# T_co = TypeVar("T_co", covariant=True)  # Covariant type for read-only operations


class QueueProtocol(Protocol[T_co]):
    """Protocol defining the interface for a basic queue.

    This protocol defines the minimum interface that any queue implementation
    must satisfy. It specifies a FIFO (First-In-First-Out) queue where items
    are added to the rear and removed from the front.

    Type Parameters
    --------------
    T_co
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

    def peek(self) -> T_co:
        """View the next item to be dequeued without removing it.

        Returns
        -------
        T_co
            The item at the front of the queue

        Raises
        ------
        Exception
            If the queue is empty
        """
        ...


class MutableQueueProtocol(QueueProtocol[T], Protocol[T]):
    """Protocol defining a mutable queue interface.

    Extends the basic QueueProtocol to include methods for modifying the queue.
    This protocol defines a standard FIFO queue interface with enqueue and
    dequeue operations.

    Type Parameters
    --------------
    T
        The type of items stored in the queue
    """

    def enqueue(self, item: T) -> None:
        """Add an item to the rear of the queue.

        Parameters
        ----------
        item : T
            The item to add
        """
        ...

    def dequeue(self) -> T:
        """Remove and return the item at the front of the queue.

        Returns
        -------
        T
            The item at the front of the queue

        Raises
        ------
        Exception
            If the queue is empty
        """
        ...


class DequeProtocol(Protocol[T_co]):
    """Protocol defining the interface for a basic double-ended queue.

    This protocol defines the minimum read-only interface that any deque
    implementation must satisfy. It allows inspection of both ends of the queue.

    Type Parameters
    --------------
    T_co
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

    def peek_front(self) -> T_co:
        """View the item at the front without removing it.

        Returns
        -------
        T_co
            The item at the front of the deque

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...

    def peek_rear(self) -> T_co:
        """View the item at the rear without removing it.

        Returns
        -------
        T_co
            The item at the rear of the deque

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...


class MutableDequeProtocol(DequeProtocol[T], Protocol[T]):
    """Protocol defining a mutable double-ended queue interface.

    Extends the basic DequeProtocol to include methods for modifying the deque.
    This protocol defines a standard double-ended queue interface with operations
    at both ends.

    Type Parameters
    --------------
    T
        The type of items stored in the deque
    """

    def add_front(self, item: T) -> None:
        """Add an item to the front of the deque.

        Parameters
        ----------
        item : T
            The item to add to the front
        """
        ...

    def add_rear(self, item: T) -> None:
        """Add an item to the rear of the deque.

        Parameters
        ----------
        item : T
            The item to add to the rear
        """
        ...

    def remove_front(self) -> T:
        """Remove and return the item from the front.

        Returns
        -------
        T
            The item from the front

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...

    def remove_rear(self) -> T:
        """Remove and return the item from the rear.

        Returns
        -------
        T
            The item from the rear

        Raises
        ------
        Exception
            If the deque is empty
        """
        ...
