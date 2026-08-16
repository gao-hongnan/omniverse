from __future__ import annotations

import random
from collections.abc import MutableSequence
from typing import Any, Callable, cast, overload

from ..core.types import Comparable, ComparableT, ItemT


class SortKey(Comparable):
    def __init__(self, obj: Any, key: Callable[[Any], Any]) -> None:
        self.obj = obj
        self.key_value = key(obj)

    def __lt__(self, other: Any, /) -> bool:
        if isinstance(other, SortKey):
            return bool(self.key_value < other.key_value)
        return NotImplemented

    def __le__(self, other: Any, /) -> bool:
        if isinstance(other, SortKey):
            return bool(self.key_value <= other.key_value)
        return NotImplemented

    def __gt__(self, other: Any, /) -> bool:
        if isinstance(other, SortKey):
            return bool(self.key_value > other.key_value)
        return NotImplemented

    def __ge__(self, other: Any, /) -> bool:
        if isinstance(other, SortKey):
            return bool(self.key_value >= other.key_value)
        return NotImplemented


@overload
def quick_sort(arr: MutableSequence[ComparableT], key: None = None) -> None: ...


@overload
def quick_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any]) -> None: ...


def quick_sort(arr: MutableSequence[Any], key: Callable[[Any], Any] | None = None) -> None:
    if key is None:
        _quick_sort_impl(arr, 0, len(arr) - 1)
    else:
        _quick_sort_with_key(arr, 0, len(arr) - 1, key)


def _quick_sort_impl(arr: MutableSequence[ComparableT], low: int, high: int) -> None:
    if low < high:
        pi = _partition(arr, low, high)
        _quick_sort_impl(arr, low, pi - 1)
        _quick_sort_impl(arr, pi + 1, high)


def _quick_sort_with_key(arr: MutableSequence[ItemT], low: int, high: int, key: Callable[[ItemT], Any]) -> None:
    if low < high:
        pi = _partition_with_key(arr, low, high, key)
        _quick_sort_with_key(arr, low, pi - 1, key)
        _quick_sort_with_key(arr, pi + 1, high, key)


def _partition(arr: MutableSequence[ComparableT], low: int, high: int) -> int:
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def _partition_with_key(arr: MutableSequence[ItemT], low: int, high: int, key: Callable[[ItemT], Any]) -> int:
    pivot_key = SortKey(arr[high], key)
    i = low - 1

    for j in range(low, high):
        if SortKey(arr[j], key) <= pivot_key:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


@overload
def randomized_quick_sort(arr: MutableSequence[ComparableT], key: None = None) -> None: ...


@overload
def randomized_quick_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any]) -> None: ...


def randomized_quick_sort(arr: MutableSequence[Any], key: Callable[[Any], Any] | None = None) -> None:
    if key is None:
        _randomized_quick_sort_impl(arr, 0, len(arr) - 1)
    else:
        _randomized_quick_sort_with_key(arr, 0, len(arr) - 1, key)


def _randomized_quick_sort_impl(arr: MutableSequence[ComparableT], low: int, high: int) -> None:
    if low < high:
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]
        pi = _partition(arr, low, high)
        _randomized_quick_sort_impl(arr, low, pi - 1)
        _randomized_quick_sort_impl(arr, pi + 1, high)


def _randomized_quick_sort_with_key(
    arr: MutableSequence[ItemT], low: int, high: int, key: Callable[[ItemT], Any]
) -> None:
    if low < high:
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]
        pi = _partition_with_key(arr, low, high, key)
        _randomized_quick_sort_with_key(arr, low, pi - 1, key)
        _randomized_quick_sort_with_key(arr, pi + 1, high, key)


def merge_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any] | None = None) -> None:
    if len(arr) <= 1:
        return

    temp: list[ItemT | None] = [None] * len(arr)
    if key is None and hasattr(arr[0] if arr else None, "__le__"):
        _merge_sort_impl(cast("MutableSequence[Any]", arr), cast("list[Any]", temp), 0, len(arr) - 1)
    else:
        _merge_sort_with_key(arr, temp, 0, len(arr) - 1, key or (lambda x: x))


def _merge_sort_impl(arr: MutableSequence[ComparableT], temp: list[ComparableT | None], left: int, right: int) -> None:
    if left < right:
        mid = (left + right) // 2
        _merge_sort_impl(arr, temp, left, mid)
        _merge_sort_impl(arr, temp, mid + 1, right)
        _merge(arr, temp, left, mid, right)


def _merge_sort_with_key(
    arr: MutableSequence[ItemT], temp: list[ItemT | None], left: int, right: int, key: Callable[[ItemT], Any]
) -> None:
    if left < right:
        mid = (left + right) // 2
        _merge_sort_with_key(arr, temp, left, mid, key)
        _merge_sort_with_key(arr, temp, mid + 1, right, key)
        _merge_with_key(arr, temp, left, mid, right, key)


def _merge(arr: MutableSequence[ComparableT], temp: list[ComparableT | None], left: int, mid: int, right: int) -> None:
    for i in range(left, right + 1):
        temp[i] = arr[i]

    i, j, k = left, mid + 1, left

    while i <= mid and j <= right:
        left_val = temp[i]
        right_val = temp[j]
        assert left_val is not None and right_val is not None
        if left_val <= right_val:
            arr[k] = left_val
            i += 1
        else:
            arr[k] = right_val
            j += 1
        k += 1

    while i <= mid:
        val = temp[i]
        assert val is not None
        arr[k] = val
        i += 1
        k += 1

    while j <= right:
        val = temp[j]
        assert val is not None
        arr[k] = val
        j += 1
        k += 1


def _merge_with_key(
    arr: MutableSequence[ItemT], temp: list[ItemT | None], left: int, mid: int, right: int, key: Callable[[ItemT], Any]
) -> None:
    for i in range(left, right + 1):
        temp[i] = arr[i]

    i, j, k = left, mid + 1, left

    while i <= mid and j <= right:
        left_val = temp[i]
        right_val = temp[j]
        assert left_val is not None and right_val is not None
        if SortKey(left_val, key) <= SortKey(right_val, key):
            arr[k] = left_val
            i += 1
        else:
            arr[k] = right_val
            j += 1
        k += 1

    while i <= mid:
        val = temp[i]
        assert val is not None
        arr[k] = val
        i += 1
        k += 1

    while j <= right:
        val = temp[j]
        assert val is not None
        arr[k] = val
        j += 1
        k += 1


@overload
def heap_sort(arr: MutableSequence[ComparableT], key: None = None) -> None: ...


@overload
def heap_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any]) -> None: ...


def heap_sort(arr: MutableSequence[Any], key: Callable[[Any], Any] | None = None) -> None:
    n = len(arr)

    if key is None:
        for i in range(n // 2 - 1, -1, -1):
            _heapify(arr, n, i)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            _heapify(arr, i, 0)
    else:
        for i in range(n // 2 - 1, -1, -1):
            _heapify_with_key(arr, n, i, key)

        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            _heapify_with_key(arr, i, 0, key)


def _heapify(arr: MutableSequence[ComparableT], n: int, i: int) -> None:
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)


def _heapify_with_key(arr: MutableSequence[ItemT], n: int, i: int, key: Callable[[ItemT], Any]) -> None:
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and SortKey(arr[left], key) > SortKey(arr[largest], key):
        largest = left

    if right < n and SortKey(arr[right], key) > SortKey(arr[largest], key):
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify_with_key(arr, n, largest, key)


def radix_sort(arr: MutableSequence[int]) -> None:
    if not arr:
        return

    max_num = max(arr)
    exp = 1

    while max_num // exp > 0:
        _counting_sort_by_digit(arr, exp)
        exp *= 10


def _counting_sort_by_digit(arr: MutableSequence[int], exp: int) -> None:
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    for i in range(n):
        arr[i] = output[i]


def counting_sort(arr: MutableSequence[int], max_val: int | None = None) -> None:
    if not arr:
        return

    if max_val is None:
        max_val = max(arr)

    min_val = min(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    output = [0] * len(arr)

    for num in arr:
        count[num - min_val] += 1

    for i in range(1, range_val):
        count[i] += count[i - 1]

    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    for i in range(len(arr)):
        arr[i] = output[i]


def bucket_sort(arr: MutableSequence[float], bucket_count: int | None = None) -> None:
    if not arr:
        return

    if bucket_count is None:
        bucket_count = len(arr)

    min_val, max_val = min(arr), max(arr)
    bucket_range = (max_val - min_val) / bucket_count

    if bucket_range == 0:
        return

    buckets: list[list[float]] = [[] for _ in range(bucket_count)]

    for num in arr:
        bucket_index = min(int((num - min_val) / bucket_range), bucket_count - 1)
        buckets[bucket_index].append(num)

    for bucket in buckets:
        bucket.sort()

    index = 0
    for bucket in buckets:
        for num in bucket:
            arr[index] = num
            index += 1


def intro_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any] | None = None) -> None:
    if len(arr) <= 1:
        return

    max_depth = 2 * (len(arr).bit_length() - 1)
    _intro_sort_impl(arr, 0, len(arr) - 1, max_depth, key)


def _intro_sort_impl(
    arr: MutableSequence[ItemT], low: int, high: int, depth: int, key: Callable[[ItemT], Any] | None
) -> None:
    size = high - low + 1

    if size <= 16:
        _insertion_sort_range(arr, low, high, key)
    elif depth == 0:
        _heap_sort_range(arr, low, high, key)
    else:
        if key is None and hasattr(arr[low] if low < len(arr) else None, "__le__"):
            pi = _partition(cast("MutableSequence[Any]", arr), low, high)
        else:
            pi = _partition_with_key(arr, low, high, key or (lambda x: x))
        _intro_sort_impl(arr, low, pi - 1, depth - 1, key)
        _intro_sort_impl(arr, pi + 1, high, depth - 1, key)


def _insertion_sort_range(arr: MutableSequence[ItemT], low: int, high: int, key: Callable[[ItemT], Any] | None) -> None:
    for i in range(low + 1, high + 1):
        current = arr[i]
        j = i - 1

        if key is None and hasattr(current, "__le__"):
            current_any: Any = current
            arr_any: MutableSequence[Any] = arr
            while j >= low and arr_any[j] > current_any:
                arr[j + 1] = arr[j]
                j -= 1
        elif key is not None:
            current_key = SortKey(current, key)
            while j >= low and SortKey(arr[j], key) > current_key:
                arr[j + 1] = arr[j]
                j -= 1

        arr[j + 1] = current


def _heap_sort_range(arr: MutableSequence[ItemT], low: int, high: int, key: Callable[[ItemT], Any] | None) -> None:
    size = high - low + 1

    if key is None and hasattr(arr[low] if low < len(arr) else None, "__le__"):
        arr_any = cast("MutableSequence[Any]", arr)
        for i in range(size // 2 - 1, -1, -1):
            _heapify_range(arr_any, low, size, i)

        for i in range(size - 1, 0, -1):
            arr[low], arr[low + i] = arr[low + i], arr[low]
            _heapify_range(arr_any, low, i, 0)
    else:
        key_func: Callable[[ItemT], Any] = key if key is not None else (lambda x: x)
        for i in range(size // 2 - 1, -1, -1):
            _heapify_range_with_key(arr, low, size, i, key_func)

        for i in range(size - 1, 0, -1):
            arr[low], arr[low + i] = arr[low + i], arr[low]
            _heapify_range_with_key(arr, low, i, 0, key_func)


def _heapify_range(arr: MutableSequence[ComparableT], offset: int, n: int, i: int) -> None:
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[offset + left] > arr[offset + largest]:
        largest = left

    if right < n and arr[offset + right] > arr[offset + largest]:
        largest = right

    if largest != i:
        arr[offset + i], arr[offset + largest] = arr[offset + largest], arr[offset + i]
        _heapify_range(arr, offset, n, largest)


def _heapify_range_with_key(
    arr: MutableSequence[ItemT], offset: int, n: int, i: int, key: Callable[[ItemT], Any]
) -> None:
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and SortKey(arr[offset + left], key) > SortKey(arr[offset + largest], key):
        largest = left

    if right < n and SortKey(arr[offset + right], key) > SortKey(arr[offset + largest], key):
        largest = right

    if largest != i:
        arr[offset + i], arr[offset + largest] = arr[offset + largest], arr[offset + i]
        _heapify_range_with_key(arr, offset, n, largest, key)


def tim_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any] | None = None) -> None:
    min_merge = 32
    n = len(arr)

    for i in range(0, n, min_merge):
        _insertion_sort_range(arr, i, min(i + min_merge - 1, n - 1), key)

    size = min_merge
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size - 1
            end = min(start + size * 2 - 1, n - 1)

            if mid < end:
                if key is None and hasattr(arr[start] if start < len(arr) else None, "__le__"):
                    _merge_tim(cast("MutableSequence[Any]", arr), start, mid, end)
                else:
                    _merge_tim_with_key(arr, start, mid, end, key or (lambda x: x))

        size *= 2


def _merge_tim(arr: MutableSequence[ComparableT], left: int, mid: int, right: int) -> None:
    left_part = list(arr[left : mid + 1])
    right_part = list(arr[mid + 1 : right + 1])

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1

    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1

    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1


def _merge_tim_with_key(
    arr: MutableSequence[ItemT], left: int, mid: int, right: int, key: Callable[[ItemT], Any]
) -> None:
    left_part = list(arr[left : mid + 1])
    right_part = list(arr[mid + 1 : right + 1])

    i = j = 0
    k = left

    while i < len(left_part) and j < len(right_part):
        if SortKey(left_part[i], key) <= SortKey(right_part[j], key):
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1

    while i < len(left_part):
        arr[k] = left_part[i]
        i += 1
        k += 1

    while j < len(right_part):
        arr[k] = right_part[j]
        j += 1
        k += 1


def insertion_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any] | None = None) -> None:
    for i in range(1, len(arr)):
        current = arr[i]
        j = i - 1

        if key is None and hasattr(current, "__le__"):
            current_any: Any = current
            arr_any: MutableSequence[Any] = arr
            while j >= 0 and arr_any[j] > current_any:
                arr[j + 1] = arr[j]
                j -= 1
        elif key is not None:
            current_key = SortKey(current, key)
            while j >= 0 and SortKey(arr[j], key) > current_key:
                arr[j + 1] = arr[j]
                j -= 1

        arr[j + 1] = current


def selection_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any] | None = None) -> None:
    n = len(arr)

    for i in range(n):
        min_idx = i

        if key is None and n > 0 and hasattr(arr[0], "__lt__"):
            arr_any: MutableSequence[Any] = arr
            for j in range(i + 1, n):
                if arr_any[j] < arr_any[min_idx]:
                    min_idx = j
        elif key is not None:
            min_key = SortKey(arr[min_idx], key)
            for j in range(i + 1, n):
                if SortKey(arr[j], key) < min_key:
                    min_idx = j
                    min_key = SortKey(arr[j], key)

        arr[i], arr[min_idx] = arr[min_idx], arr[i]


def bubble_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any] | None = None) -> None:
    n = len(arr)

    for i in range(n):
        swapped = False

        if key is None and n > 0 and hasattr(arr[0], "__gt__"):
            arr_any: MutableSequence[Any] = arr
            for j in range(0, n - i - 1):
                if arr_any[j] > arr_any[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
        elif key is not None:
            for j in range(0, n - i - 1):
                if SortKey(arr[j], key) > SortKey(arr[j + 1], key):
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True

        if not swapped:
            break


def cocktail_sort(arr: MutableSequence[ItemT], key: Callable[[ItemT], Any] | None = None) -> None:
    n = len(arr)
    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False

        if key is None and n > 0 and hasattr(arr[0], "__gt__"):
            arr_any: MutableSequence[Any] = arr
            for i in range(start, end):
                if arr_any[i] > arr_any[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True

            if not swapped:
                break

            end -= 1
            swapped = False

            for i in range(end - 1, start - 1, -1):
                if arr_any[i] > arr_any[i + 1]:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True
        elif key is not None:
            for i in range(start, end):
                if SortKey(arr[i], key) > SortKey(arr[i + 1], key):
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True

            if not swapped:
                break

            end -= 1
            swapped = False

            for i in range(end - 1, start - 1, -1):
                if SortKey(arr[i], key) > SortKey(arr[i + 1], key):
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    swapped = True

        start += 1
