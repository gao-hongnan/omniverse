"""
The base module for the searching algorithms in the omnivault.dsa package. It
provides foundational classes and functionalities that are shared across various
search strategies and the context implementation.

This module typically includes abstract base classes or common utility functions
that are utilized by the concrete strategy classes in strategies.py and the
context class in context.py.
"""

# pylint:disable=all
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence

from omnivault.dsa.typings.generics import Real


class Search(ABC):
    @abstractmethod
    def search(self, container: Sequence[Real], target: Real) -> int:
        ...


class BinarySearch(Search):
    """Interface for Binary Search Strategies (Strategy Design Pattern)."""

    @abstractmethod
    def mid_strategy(self, left: int, right: int) -> int:
        """Strategy for calculating the middle index."""
