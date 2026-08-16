from __future__ import annotations

import pytest

from omnivault.dsa.containers.linear.min_max_stack import MinMaxStack
from omnivault.dsa.core.errors import EmptyContainer


@pytest.mark.unit
class TestMinMaxStack:
    def test_lc155_official_trace(self) -> None:
        stack: MinMaxStack[int] = MinMaxStack()
        stack.push(-2)
        stack.push(0)
        stack.push(-3)
        assert stack.min() == -3
        stack.pop()
        assert stack.peek() == 0
        assert stack.min() == -2

    @pytest.mark.parametrize(
        ("values", "expected_min", "expected_max"),
        [
            ([5, 3, 7, 1, 9, 2], 1, 9),
            ([10], 10, 10),
            ([4, 4, 4], 4, 4),
        ],
    )
    def test_min_max_after_pushes(self, values: list[int], expected_min: int, expected_max: int) -> None:
        stack: MinMaxStack[int] = MinMaxStack()
        for value in values:
            stack.push(value)
        assert stack.min() == expected_min
        assert stack.max() == expected_max

    def test_min_after_pop_restores_previous(self) -> None:
        stack: MinMaxStack[int] = MinMaxStack()
        for value in [3, 1, 5, 4, 2]:
            stack.push(value)
        assert stack.min() == 1
        stack.pop()
        stack.pop()
        assert stack.min() == 1
        stack.pop()
        stack.pop()
        assert stack.min() == 3

    def test_min_on_empty_raises(self) -> None:
        stack: MinMaxStack[int] = MinMaxStack()
        with pytest.raises(EmptyContainer):
            stack.min()
