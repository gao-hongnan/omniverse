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
from typing import Literal, Sequence, Union

from omnivault._types._alias import NonNegativeInt
from omnivault._types._generic import Real
from omnivault.dsa.searching_algorithms.base import Search


class LinearSearchForLoop(Search):
    """
    Implements a linear search algorithm using a for loop.

    This class provides a concrete implementation of the `Search` abstract base
    class. It performs a linear search on a sequence to find a target element.
    The search is conducted iteratively using a for loop, examining each element
    sequentially until the target is found or the end of the sequence is reached.
    """

    def search(self, container: Sequence[Real], target: Real) -> Union[NonNegativeInt, Literal[-1]]:
        for index, item in enumerate(container):
            if item == target:
                return index
        return -1


class LinearSearchWhileLoop(Search):
    """
    Implements a linear search algorithm using a while loop.

    This class provides a concrete implementation of the `Search` abstract base
    class. It iteratively searches through the given sequence using a while loop,
    comparing each element with the target until the target is found or the end
    of the sequence is reached.
    """

    def search(self, container: Sequence[Real], target: Real) -> Union[NonNegativeInt, Literal[-1]]:
        index = 0
        length = len(container)
        while index < length:
            if container[index] == target:
                return index
            index += 1
        return -1


class LinearSearchRecursive(Search):
    """
    Implements a linear search algorithm using recursion.

    This class provides a recursive implementation of the `Search` abstract base
    class. It searches for a target element in a sequence by recursively
    examining each element. The recursion starts from the first element and
    proceeds until the target is found or the sequence is fully traversed.

    Note
    ----
    Since Python does not support tail-call optimization, this recursive
    approach may not be efficient for very large sequences.
    """

    def search(self, container: Sequence[Real], target: Real) -> Union[NonNegativeInt, Literal[-1]]:
        def recursive(container: Sequence[Real], target: Real, index: int = 0) -> int:
            if not container:
                return -1
            if container[0] == target:
                return index
            return recursive(container[1:], target, index + 1)

        return recursive(container, target)


class LinearSearchTailRecursive(Search):
    """
    Implements a linear search algorithm using tail recursion.

    This class attempts a tail-recursive approach to implement linear search,
    similar to `LinearSearchRecursive`. However, due to the lack of tail-call
    optimization in Python, this implementation does not provide the usual
    benefits of tail recursion and behaves similarly to regular recursion.
    """

    def search(self, container: Sequence[Real], target: Real) -> Union[NonNegativeInt, Literal[-1]]:
        def recursive(container: Sequence[Real], target: Real, index: int = 0) -> Union[NonNegativeInt, Literal[-1]]:
            if not container:
                return -1
            if container[0] == target:
                return index
            return recursive(container[1:], target, index + 1)

        return recursive(container, target)


class IterativeBinarySearchExactMatch(Search):
    """Leetcode calls this template 1:
    https://leetcode.com/explore/learn/card/binary-search/125/template-i/

    Implements an iterative binary search algorithm for exact matches.

    This class provides an iterative implementation of binary search following
    the template often used in problems such as those found on LeetCode.
    It is designed to work on sorted sequences, dividing the search space in half
    with each step, until the target element is found or the search space is exhausted.
    """

    def search(self, container: Sequence[Real], target: Real) -> Union[NonNegativeInt, Literal[-1]]:
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

    def mid_strategy(self, left: NonNegativeInt, right: NonNegativeInt) -> NonNegativeInt:
        """Strategy for calculating the middle index."""

        # (left_index + right_index) // 2 will cause overflow.
        mid_index = left + math.floor((right - left) / 2)
        return mid_index


class RecursiveBinarySearchExactMatch(Search):
    """Template 1 but recursive."""

    def search(self, container: Sequence[Real], target: Real) -> int:
        """Search for a target from a sorted array container."""

        def recursive(l: NonNegativeInt, r: NonNegativeInt) -> Union[NonNegativeInt, Literal[-1]]:
            if l > r:  # base case
                return -1

            mid_index = self.mid_strategy(l, r)

            if container[mid_index] < target:
                return recursive(l=mid_index + 1, r=r)
            elif container[mid_index] > target:
                return recursive(l=l, r=mid_index - 1)
            else:  # base case
                return mid_index

        l, r = 0, len(container) - 1
        return recursive(l, r)

    def mid_strategy(self, left: NonNegativeInt, right: NonNegativeInt) -> NonNegativeInt:
        """Strategy for calculating the middle index."""

        mid_index = left + math.floor((right - left) / 2)
        return mid_index
