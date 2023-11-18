"""
The base module for the searching algorithms in the omnivault.dsa package. This
is the interface for the Strategy Design Pattern that will be implemented by the
concrete search strategies in strategies.py.

This module typically includes abstract base classes or common utility functions
that are utilized by the concrete strategy classes in strategies.py and the
context class in context.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

from omnivault.dsa.typings.generics import Real
from omnivault.dsa.typings.newtype import NonNegativeInt


class Search(ABC):
    """Base class for Search Strategies (Strategy Design Pattern)."""

    @abstractmethod
    def search(
        self, container: Sequence[Real], target: Real
    ) -> NonNegativeInt | Literal[-1]:
        """Searches for the target in the container and returns the index of the
        target if found, otherwise returns -1."""
