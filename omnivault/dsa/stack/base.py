from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Generator

from omnivault.dsa.typings.generics import T


class Stack(ABC, Generic[T]):
    """
    This interface defines the contract for a stack data structure.
    """

    @abstractmethod
    def push(self, item: T) -> None:
        """Push an item on top of the stack."""
        raise NotImplementedError

    @abstractmethod
    def pop(self) -> T:
        """Pop an item from the top of the stack."""
        raise NotImplementedError

    @abstractmethod
    def peek(self) -> T:
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
    def __iter__(self) -> Generator[T, None, None]:
        """Return an iterator for the stack."""
        raise NotImplementedError
