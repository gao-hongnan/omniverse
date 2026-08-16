from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...core.types import Real, SearchResult
from .base import Search

if TYPE_CHECKING:
    from collections.abc import Sequence


class LinearSearchForLoop(Search):
    def search(self, container: Sequence[Real], target: Real) -> SearchResult:
        for index, item in enumerate(container):
            if item == target:
                return index
        return -1


class LinearSearchWhileLoop(Search):
    def search(self, container: Sequence[Real], target: Real) -> SearchResult:
        index = 0
        length = len(container)
        while index < length:
            if container[index] == target:
                return index
            index += 1
        return -1


class LinearSearchRecursive(Search):
    def search(self, container: Sequence[Real], target: Real) -> SearchResult:
        def recursive(container: Sequence[Real], target: Real, index: int = 0) -> SearchResult:
            if not container:
                return -1
            if container[0] == target:
                return index
            return recursive(container[1:], target, index + 1)

        return recursive(container, target)


class LinearSearchTailRecursive(Search):
    def search(self, container: Sequence[Real], target: Real) -> SearchResult:
        def recursive(container: Sequence[Real], target: Real, index: int = 0) -> SearchResult:
            if not container:
                return -1
            if container[0] == target:
                return index
            return recursive(container[1:], target, index + 1)

        return recursive(container, target)


class IterativeBinarySearchExactMatch(Search):
    def search(self, container: Sequence[Real], target: Real) -> SearchResult:
        left_index = 0
        right_index = len(container) - 1

        while left_index <= right_index:
            mid_index = self._calculate_mid_index(left=left_index, right=right_index)

            if container[mid_index] == target:
                return mid_index
            elif container[mid_index] < target:
                left_index = mid_index + 1
            else:
                right_index = mid_index - 1

        return -1

    def _calculate_mid_index(self, left: int, right: int) -> int:
        return left + math.floor((right - left) / 2)


class RecursiveBinarySearchExactMatch(Search):
    def search(self, container: Sequence[Real], target: Real) -> SearchResult:
        def recursive(left: int, right: int) -> SearchResult:
            if left > right:
                return -1

            mid_index = self._calculate_mid_index(left, right)

            if container[mid_index] < target:
                return recursive(mid_index + 1, right)
            elif container[mid_index] > target:
                return recursive(left, mid_index - 1)
            else:
                return mid_index

        return recursive(0, len(container) - 1)

    def _calculate_mid_index(self, left: int, right: int) -> int:
        return left + math.floor((right - left) / 2)
