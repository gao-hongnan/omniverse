from __future__ import annotations

import math
from typing import Iterable, Tuple, TypeVar, Callable

T = TypeVar("T", str, int, float)  # T should be of type int, float or str


def unordered_sequential_search_iterative(
    container: Iterable[T], target: T
) -> Tuple[bool, int]:
    """If the target element is found in the container, returns True and its index,
    else, return False and -1 to indicate the not found index."""
    is_found = False  # a flag to indicate so your return is more meaningful
    index = 0
    for item in container:
        if item == target:
            is_found = True
            return is_found, index
        index += 1
    return is_found, -1


def unordered_sequential_search_recursive(
    container: Iterable[T], target: T, index: int = 0
) -> int:
    """Recursive implementation of unordered Sequential Search."""
    if len(container) == 0:  # if not container is also fine
        return -1  # not found

    if container[0] == target:  # this is base case
        return index  # found

    # notice we increment index by 1 to mean index += 1 in the iterative case
    return unordered_sequential_search_recursive(
        container[1:], target, index + 1
    )  # recursive case


def ordered_sequential_search(container: Iterable[T], target: T) -> Tuple[bool, int]:
    """Sequential search for ordered container."""
    is_found = False  # a flag to indicate so your return is more meaningful
    index = 0
    for item in container:
        if item == target:
            is_found = True
            return is_found, index
        index += 1
        if item > target:
            return is_found, -1
    # do not forget this if not if target > largest element in container, this case is not covered
    return is_found, -1


def search(nums: Iterable[T], target: int) -> int:
    def recursive(l: int, r: int, nums: Iterable[T], target: int) -> int:
        if l > r:
            return -1

        mid_strategy: Callable = lambda l, r: (l + r) // 2
        mid_index: int = mid_strategy(l, r)

        if nums[mid_index] < target:
            return recursive(l=mid_index + 1, r=r, nums=nums, target=target)
        elif nums[mid_index] > target:
            return recursive(l=l, r=mid_index - 1, nums=nums, target=target)
        else:
            return mid_index

    l, r = 0, len(nums) - 1
    return recursive(l, r, nums, target)


if __name__ == "__main__":
    ordered_list = [0, 1, 2, 8, 13, 17, 19, 32, 42]
    left_index = 0
    right_index = len(ordered_list) - 1
    print(search(nums=ordered_list, target=42))
    # print(binary_search_iterative(ordered_list, -1, left_index, right_index))
    # print(binary_search_iterative(ordered_list, 3.5, left_index, right_index))
    # print(binary_search_iterative(ordered_list, 42, left_index, right_index))
    # print(binary_search_recursive(ordered_list, -1, left_index, right_index))
    # print(binary_search_recursive(ordered_list, 3.5, left_index, right_index))
    # print(binary_search_recursive(ordered_list, 42, left_index, right_index))
