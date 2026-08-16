from __future__ import annotations

import pytest

from omnivault.dsa.containers.linear.stack.base import AbstractStack
from omnivault.dsa.containers.linear.stack.concrete import ArrayStack, LinkedListStack
from omnivault.dsa.core.errors import EmptyContainer


def test_abstract_stack_is_generic_with_itemt() -> None:
    params = getattr(AbstractStack, "__type_params__", ())
    names = [p.__name__ for p in params]
    assert names == ["ItemT"], f"expected [ItemT], got {names}"


@pytest.mark.parametrize("cls", [ArrayStack, LinkedListStack])
def test_stack_pop_raises_empty_container_on_empty(cls: type[AbstractStack[int]]) -> None:
    stack: AbstractStack[int] = cls()
    with pytest.raises(IndexError, match="pop from an empty stack"):
        stack.pop()
    with pytest.raises(EmptyContainer, match="pop from an empty stack"):
        cls().pop()


@pytest.mark.parametrize("cls", [ArrayStack, LinkedListStack])
def test_stack_peek_raises_empty_container_on_empty(cls: type[AbstractStack[int]]) -> None:
    with pytest.raises(EmptyContainer, match="peek from an empty stack"):
        cls().peek()


@pytest.mark.parametrize("cls", [ArrayStack, LinkedListStack])
def test_stack_push_pop_roundtrip(cls: type[AbstractStack[int]]) -> None:
    stack: AbstractStack[int] = cls()
    for value in (1, 2, 3):
        stack.push(value)
    assert stack.pop() == 3
    assert stack.pop() == 2
    assert stack.pop() == 1
