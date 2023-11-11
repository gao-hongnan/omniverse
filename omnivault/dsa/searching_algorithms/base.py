from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Iterable

T = TypeVar("T", int, float)  # T should be of type int, float or str


class BinarySearchStrategy(ABC):
    """Interface for Binary Search Strategies (Strategy Design Pattern)."""

    @abstractmethod
    def search(self, nums: Iterable[T], target: int) -> int:
        """Search for a target from a sorted array nums."""

    @abstractmethod
    def mid_strategy(self, left: int, right: int) -> int:
        """Strategy for calculating the middle index."""


class BinarySearch:
    def __init__(self, strategy: BinarySearchStrategy) -> None:
        """
        Usually, the Context (here is BinarySearch executor)
        accepts a strategy through the constructor, but also
        provides a setter to change it at runtime.
        """
        self._strategy = strategy

    @property
    def strategy(self) -> BinarySearchStrategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BinarySearchStrategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def find_target(self, nums: Iterable[T], target: int) -> int:
        """Find the target."""
        return self._strategy.search(nums, target)
