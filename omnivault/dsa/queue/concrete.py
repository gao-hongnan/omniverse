from __future__ import annotations

from typing import Generic, List

from omnivault._types._generic import T
from omnivault.dsa.queue.base import MutableDequeProtocol, MutableQueueProtocol


class QueueList(Generic[T], MutableQueueProtocol[T]):
    """A queue implementation using Python's built-in list.

    This implementation treats the end of the list as the start of the queue and
    the start of the list as the end of the queue.

    Parameters
    ----------
    None

    Attributes
    ----------
    _queue_items : List[T]
        The list storing the queue items

    Example
    -------
    >>> queue: QueueList[int] = QueueList()
    >>> queue.enqueue(1)
    >>> queue.enqueue(2)
    >>> queue.enqueue(3)

    >>> print(f"Queue size: {queue.size}")
    3

    >>> for item in queue:
    ...     print(item)
    3
    2
    1
    """

    _queue_items: List[T]

    def __init__(self) -> None:
        self._queue_items = []

    def __len__(self) -> int:
        """Return the number of items in the queue.

        Returns
        -------
        int
            The size of the queue
        """
        return len(self._queue_items)

    def __iter__(self) -> QueueList[T]:
        """Make the queue iterable.

        Returns
        -------
        QueueList[T]
            The queue iterator
        """
        return self

    def __next__(self) -> T:
        """Get the next item during iteration.

        Returns
        -------
        T
            The next item in the queue

        Raises
        ------
        StopIteration
            When the queue is empty
        """
        if self.is_empty():
            raise StopIteration
        return self.dequeue()

    @property
    def queue_items(self) -> List[T]:
        """Get the queue items.

        Returns
        -------
        List[T]
            The list of queue items
        """
        return self._queue_items

    @property
    def size(self) -> int:
        """Get the size of the queue.

        Returns
        -------
        int
            The number of items in the queue
        """
        return len(self)

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns
        -------
        bool
            True if queue is empty, False otherwise
        """
        return self.size == 0

    def peek(self) -> T:
        """Get the item at the start of the queue without removing it.

        Returns
        -------
        T
            The item at the start of the queue
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self._queue_items[-1]

    def enqueue(self, item: T) -> None:
        """Add an item to the end of the queue.

        Parameters
        ----------
        item : T
            The item to add to the queue
        """
        self._queue_items.insert(0, item)

    def dequeue(self) -> T:
        """Remove and return the item at the start of the queue.

        Returns
        -------
        T
            The item at the start of the queue

        Raises
        ------
        Exception
            If the queue is empty
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self._queue_items.pop()


class DeQueueList(Generic[T], MutableDequeProtocol[T]):
    """A double-ended queue implementation using Python's built-in list.

    This implementation allows adding and removing items from both ends of the queue.

    Parameters
    ----------
    None

    Attributes
    ----------
    _queue_items : List[T]
        The list storing the queue items
    """

    _queue_items: List[T]

    def __init__(self) -> None:
        self._queue_items = []

    def __len__(self) -> int:
        """Return the number of items in the dequeue.

        Returns
        -------
        int
            The size of the dequeue
        """
        return len(self._queue_items)

    @property
    def queue_items(self) -> List[T]:
        """Get the queue items.

        Returns
        -------
        List[T]
            The list of queue items
        """
        return self._queue_items

    @property
    def size(self) -> int:
        """Get the size of the queue.

        Returns
        -------
        int
            The number of items in the queue
        """
        return len(self)

    def is_empty(self) -> bool:
        """Check if the queue is empty.

        Returns
        -------
        bool
            True if queue is empty, False otherwise
        """
        return self.size == 0

    def peek_front(self) -> T:
        """Get the item at the front of the queue without removing it.

        Returns
        -------
        T
            The item at the front of the queue

        Raises
        ------
        Exception
            If the queue is empty
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self._queue_items[-1]

    def peek_rear(self) -> T:
        """Get the item at the rear of the queue without removing it.

        Returns
        -------
        T
            The item at the rear of the queue
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self._queue_items[0]

    def add_front(self, item: T) -> None:
        """Add an item to the front of the queue.

        Parameters
        ----------
        item : T
            The item to add to the front
        """
        self._queue_items.append(item)

    def add_rear(self, item: T) -> None:
        """Add an item to the rear of the queue.

        Parameters
        ----------
        item : T
            The item to add to the rear
        """
        self._queue_items.insert(0, item)

    def remove_front(self) -> T:
        """Remove and return the item from the front of the queue.

        Returns
        -------
        T
            The item from the front

        Raises
        ------
        Exception
            If the queue is empty
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self._queue_items.pop()

    def remove_rear(self) -> T:
        """Remove and return the item from the rear of the queue.

        Returns
        -------
        T
            The item from the rear

        Raises
        ------
        Exception
            If the queue is empty
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self._queue_items.pop(0)
