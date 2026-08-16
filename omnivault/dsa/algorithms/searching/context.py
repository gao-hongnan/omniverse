from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.types import Real, SearchResult
from .base import Search

if TYPE_CHECKING:
    from collections.abc import Sequence


class SearchContext:
    def __init__(self, strategy: Search) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> Search:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Search) -> None:
        self._strategy = strategy

    def execute_search(self, container: Sequence[Real], target: Real) -> SearchResult:
        return self.strategy.search(container, target)
