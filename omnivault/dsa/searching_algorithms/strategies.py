"""
This module, part of the omnivault.dsa.searching_algorithms package, implements
various concrete search strategies as per the Strategy Design Pattern. Each
class in this module defines a specific search algorithm, like linear search or
binary search.

These strategies are designed to be used interchangeably by the SearchContext
class in context.py, demonstrating the flexibility and decoupling offered by the
Strategy Pattern.
"""
from __future__ import annotations

import math
from typing import Sequence

from omnivault.dsa.searching_algorithms.base import BinarySearch, Search
from omnivault.dsa.typings.generics import Real


class LinearSearchForLoop(Search):
    def search(self, container: Sequence[Real], target: Real) -> int:
        for index, item in enumerate(container):
            if item == target:
                return index
        return -1


class LinearSearchWhileLoop(Search):
    def search(self, container: Sequence[Real], target: Real) -> int:
        index = 0
        length = len(container)
        while index < length:
            if container[index] == target:
                return index
            index += 1
        return -1


class LinearSearchRecursive(Search):
    def search(self, container: Sequence[Real], target: Real) -> int:
        def recursive(container: Sequence[Real], target: Real, index: int = 0) -> int:
            if not container:
                return -1
            if container[0] == target:
                return index
            return recursive(container[1:], target, index + 1)

        return recursive(container, target)


class LinearSearchTailRecursive(Search):
    def search(self, container: Sequence[Real], target: Real) -> int:
        def recursive(container: Sequence[Real], target: Real, index: int = 0) -> int:
            if not container:
                return -1
            if container[0] == target:
                return index
            return recursive(container[1:], target, index + 1)

        return recursive(container, target)


class IterativeBinarySearchExactMatch(BinarySearch):
    """Leetcode calls this template 1:
    https://leetcode.com/explore/learn/card/binary-search/125/template-i/
    """

    def search(self, container: Sequence[Real], target: Real) -> int:
        """Search for a target from a sorted array container."""

        left_index = 0
        right_index = len(container) - 1

        while left_index <= right_index:
            mid_index = self.mid_strategy(left=left_index, right=right_index)
            # Check if target is present at mid
            if container[mid_index] == target:
                return mid_index

            # If target is greater, we discard left half, so we update left_index
            elif container[mid_index] < target:
                left_index = mid_index + 1

            # If target is smaller, we discard right half, so we update right_index
            else:  # container[mid_index] > target
                right_index = mid_index - 1

        # Search has ended and target is not present in the container, so we return -1
        return -1

    def mid_strategy(self, left: int, right: int) -> int:
        # (left_index + right_index) // 2 will cause overflow.
        mid_index = left + math.floor((right - left) / 2)
        return mid_index


class RecursiveBinarySearchExactMatch(BinarySearch):
    """Template 1 but recursive."""

    def search(self, container: Sequence[Real], target: Real) -> int:
        """Search for a target from a sorted array container."""

        def recursive(l: int, r: int) -> int:
            if l > r:
                return -1

            mid_index = self.mid_strategy(l, r)

            if container[mid_index] < target:
                return recursive(l=mid_index + 1, r=r)
            elif container[mid_index] > target:
                return recursive(l=l, r=mid_index - 1)
            else:
                return mid_index

        l, r = 0, len(container) - 1
        return recursive(l, r)

    def mid_strategy(self, left: int, right: int) -> int:
        mid_index = left + math.floor((right - left) / 2)
        return mid_index
