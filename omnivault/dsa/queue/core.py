from __future__ import annotations

from collections import deque
from typing import Generic, List, TypeVar

T = TypeVar("T")


class QueueList(Generic[T]):
    """Creates a queue that uses python's default list as the underlying
    data structure.

    Attributes:
        queue_items (List[T]): The list that stores the items in the queue.
            We treat the end of the list as the start of the queue and
            the start of the list as the end of the queue.
    """

    _queue_items: List[T]

    def __init__(self) -> None:
        self._queue_items = []

    def __len__(self) -> int:
        """Return the size of the queue."""
        return len(self.queue_items)

    def __iter__(self) -> QueueList[T]:
        """Iterate over the queue items."""
        return self

    def __next__(self) -> T:
        """Return the next item in the queue."""
        if self.is_empty():
            raise StopIteration
        return self.dequeue()

    @property
    def queue_items(self) -> List[T]:
        """Read only property for the queue items."""
        return self._queue_items

    @property
    def size(self) -> int:
        """Return the size of the queue.

        Returns:
            (int): The size of the queue.
        """
        return len(self)

    def is_empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            (bool): True if queue is empty, False otherwise.
        """
        return self.size == 0

    def enqueue(self, item: T) -> None:
        """Insert an item at the end of the queue.

        In this implementation, the item is inserted at the start of the list.

        Args:
            item (T): The current item to be queued.
        """
        self.queue_items.insert(0, item)

    def dequeue(self) -> T:
        """Pop an item from the start of the queue.

        In this implementation, the item at the end of the list is returned and removed.
        We are using the list's pop method to do this.

        Raises:
            (Exception): If queue is empty.

        Returns:
            (T): The item at the start of the queue.
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self.queue_items.pop()


class DeQueueList(Generic[T]):
    """Creates a double-ended queue that uses python's default list as the underlying
    data structure.

    Attributes:
        queue_items (List[T]): The list that stores the items in the queue.
            We treat the end of the list as the start of the queue and
            the start of the list as the end of the queue.
    """

    _queue_items: List[T]

    def __init__(self) -> None:
        self._queue_items = []

    def __len__(self) -> int:
        """Return the size of the dequeue."""
        return len(self.queue_items)

    @property
    def queue_items(self) -> List[T]:
        """Read only property for the queue items."""
        return self._queue_items

    @property
    def size(self) -> int:
        """Return the size of the queue.

        Returns:
            (int): The size of the queue.
        """
        return len(self)

    def is_empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            (bool): True if queue is empty, False otherwise.
        """
        return self.size == 0

    def add_front(self, item: T) -> None:
        """Insert an item at the front of the queue.

        Args:
            item (T): The current item to be added.
        """
        self.queue_items.append(item)

    def add_rear(self, item: T) -> None:
        """Insert an item at the end of the queue.

        In this implementation, the item is inserted at the start of the list.

        Args:
            item (T): The current item to be queued.
        """
        self.queue_items.insert(0, item)

    def remove_front(self) -> T:
        """Pop an item from the start of the queue.

        In this implementation, the item at the end of the list is returned and removed.
        We are using the list's pop method to do this.

        Raises:
            (Exception): If queue is empty.

        Returns:
            (T): The item at the start of the queue.
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self.queue_items.pop()

    def remove_rear(self) -> T:
        """Pop an item from the end of the queue.

        Raises:
            (Exception): If queue is empty.

        Returns:
            (T): The item at the end of the queue.
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        return self.queue_items.pop(0)


if __name__ == "__main__":
    queue = QueueList()
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)

    print(f"Queue size: {queue.size}")

    for item in queue:
        print(item)
