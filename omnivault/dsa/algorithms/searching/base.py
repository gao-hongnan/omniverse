from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ...core.types import Real, SearchResult

if TYPE_CHECKING:
    from collections.abc import Sequence


# ABC retained: shared interface contract with no concrete defaults; subclasses inherit the abstract search slot per rules/python-typings.md exception clause
class Search(ABC):
    @abstractmethod
    def search(self, container: Sequence[Real], target: Real) -> SearchResult: ...
