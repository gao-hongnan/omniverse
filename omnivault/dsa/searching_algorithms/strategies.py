from __future__ import annotations

import math
from typing import Iterable

from rich.pretty import pprint

from dsa.searching_algorithms.base import BinarySearch, BinarySearchStrategy, T


class IterativeBinarySearchExactMatch(BinarySearchStrategy):
    """Leetcode calls this template 1:
    https://leetcode.com/explore/learn/card/binary-search/125/template-i/
    """

    def search(self, nums: Iterable[T], target: int) -> int:
        """Search for a target from a sorted array nums."""

        left_index = 0
        right_index = len(nums) - 1

        while left_index <= right_index:
            mid_index = self.mid_strategy(left=left_index, right=right_index)
            # Check if target is present at mid
            if nums[mid_index] == target:
                return mid_index

            # If target is greater, we discard left half, so we update left_index
            elif nums[mid_index] < target:
                left_index = mid_index + 1

            # If target is smaller, we discard right half, so we update right_index
            else:  # nums[mid_index] > target
                right_index = mid_index - 1

        # Search has ended and target is not present in the nums, so we return -1
        return -1

    def mid_strategy(self, left: int, right: int) -> int:
        # (left_index + right_index) // 2 will cause overflow.
        mid_index = left + math.floor((right - left) / 2)
        return mid_index


class RecursiveBinarySearchExactMatch(BinarySearchStrategy):
    """Template 1 but recursive."""

    def search(self, nums: Iterable[T], target: int) -> int:
        """Search for a target from a sorted array nums."""

        def recursive(l: int, r: int) -> int:
            if l > r:
                return -1

            mid_index = self.mid_strategy(l, r)

            if nums[mid_index] < target:
                return recursive(l=mid_index + 1, r=r)
            elif nums[mid_index] > target:
                return recursive(l=l, r=mid_index - 1)
            else:
                return mid_index

        l, r = 0, len(nums) - 1
        return recursive(l, r)

    def mid_strategy(self, left: int, right: int) -> int:
        mid_index = left + math.floor((right - left) / 2)
        return mid_index


if __name__ == "__main__":
    binary_search = BinarySearch(strategy=IterativeBinarySearchExactMatch())
    result = binary_search.find_target([2, 5, 8, 12, 16, 23, 38, 56, 72, 91], 23)
    pprint(result)

    # Changing the strategy to RecursiveBinarySearchExactMatch
    binary_search.strategy = RecursiveBinarySearchExactMatch()
    result = binary_search.find_target([2, 5, 8, 12, 16, 23, 38, 56, 72, 91], 23)
    pprint(result)
