"""
The context module in the omnivault.dsa.searching_algorithms package. It defines
the SearchContext class, which acts as the primary interface for executing
search strategies in the Strategy Design Pattern.

The SearchContext class holds a reference to a search strategy and delegates the
searching task to it. This allows for dynamic swapping of search algorithms at
runtime, adhering to the principles of the Strategy Pattern.
"""
from typing import Sequence

from omnivault._types._generic import Real
from omnivault.dsa.searching_algorithms.base import Search


class SearchContext:
    """The Context defines the interface of interest to clients."""

    def __init__(self, strategy: Search) -> None:
        """The context defines the interface of interest to clients and accepts
        a strategy through the constructor."""
        self._strategy = strategy

    @property
    def strategy(self) -> Search:
        """The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Search) -> None:
        """Usually, the Context allows replacing a Strategy object at runtime."""
        self._strategy = strategy

    def execute_search(self, container: Sequence[Real], target: Real) -> int:
        """Here the context delegates some work to the strategy object instead
        of implementing multiple versions of the algorithm on its own."""
        return self.strategy.search(container, target)
