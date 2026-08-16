from __future__ import annotations

from typing import Literal

import pytest

from omnivault.dsa.containers.linear.monotonic import MonotonicDeque, MonotonicStack
from omnivault.dsa.core.errors import EmptyContainer, InvalidConfiguration


@pytest.mark.unit
class TestMonotonicStack:
    @pytest.mark.parametrize(
        ("values", "mode", "expected_final_stack"),
        [
            ([2, 1, 5, 6, 2, 3], "decreasing", [6, 3]),
            ([3, 1, 4, 1, 5, 9, 2, 6], "increasing", [1, 1, 2, 6]),
        ],
    )
    def test_invariant_preserved_after_each_push(
        self,
        values: list[int],
        mode: Literal["increasing", "decreasing"],
        expected_final_stack: list[int],
    ) -> None:
        stack: MonotonicStack[int] = MonotonicStack(mode=mode)
        for value in values:
            stack.push(value)
        drained: list[int] = []
        while not stack.is_empty():
            drained.append(stack.pop())
        drained.reverse()
        assert drained == expected_final_stack

    def test_next_greater_element_lc496(self) -> None:
        nums = [2, 1, 2, 4, 3]
        result = [-1] * len(nums)
        stack: MonotonicStack[int] = MonotonicStack(mode="decreasing")
        for i, value in enumerate(nums):
            while not stack.is_empty() and nums[stack.peek_index()] < value:
                result[stack.pop_index()] = value
            stack.push_index(i, value)
        assert result == [4, 2, 4, -1, -1]

    def test_peek_empty_raises(self) -> None:
        stack: MonotonicStack[int] = MonotonicStack(mode="decreasing")
        with pytest.raises(EmptyContainer):
            stack.peek()

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            MonotonicStack[int](mode="bogus")  # type: ignore[arg-type]


@pytest.mark.unit
class TestMonotonicDeque:
    @pytest.mark.parametrize(
        ("nums", "k", "expected"),
        [
            ([1, 3, -1, -3, 5, 3, 6, 7], 3, [3, 3, 5, 5, 6, 7]),
            ([1], 1, [1]),
            ([9, 11], 2, [11]),
            ([4, -2], 2, [4]),
        ],
    )
    def test_sliding_window_max_lc239(self, nums: list[int], k: int, expected: list[int]) -> None:
        deque: MonotonicDeque[int] = MonotonicDeque(mode="decreasing")
        result: list[int] = []
        for i, value in enumerate(nums):
            deque.push(i, value)
            while not deque.is_empty() and deque.front_index() <= i - k:
                deque.pop_front()
            if i >= k - 1:
                result.append(deque.front_value())
        assert result == expected
