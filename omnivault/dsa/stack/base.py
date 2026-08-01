from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import overload


class EmptyStackError(IndexError):
    """Raised when an operation requires a non-empty stack."""


class Stack[ItemT](ABC):
    """
    This interface defines the contract for a stack data structure.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, iterable: Iterable[ItemT]) -> None: ...

    def __init__(self, iterable: Iterable[ItemT] | None = None) -> None:
        """Construct a new stack object.

        Parameters
        ----------
        iterable : Iterable[ItemT] | None
            An iterable to initialize the stack with, by default None
        """
        self._stack_items: list[ItemT] = []
        if iterable is not None:
            for item in iterable:
                self.push(item)

    @abstractmethod
    def push(self, item: ItemT) -> None:
        """Push an item on top of the stack."""
        raise NotImplementedError

    @abstractmethod
    def pop(self) -> ItemT:
        """Pop an item from the top of the stack."""
        raise NotImplementedError

    @abstractmethod
    def peek(self) -> ItemT:
        """Return the top most item in the stack without modifying the stack."""
        raise NotImplementedError

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if stack is empty."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the stack."""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[ItemT]:
        """Return an iterator for the stack."""
        raise NotImplementedError
