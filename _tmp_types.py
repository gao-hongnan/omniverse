from omnivault.dsa.stack.concrete import StackList
from omnivault._types._generic import T
from typing import (
    Union,
    Any,
    List,
    Tuple,
    Dict,
    Callable,
    TypeVar,
    Generic,
    Optional,
    Type,
    cast,
)
from rich.pretty import pprint


def append_any_to_stack(stack: StackList[Any], item: Any) -> StackList[Any]:
    stack.push(item)
    return stack


stack: StackList[Any] = StackList()
stack = append_any_to_stack(stack, 1)
stack = append_any_to_stack(stack, "a")
stack = append_any_to_stack(stack, 2)


def append_generic_to_stack(stack: StackList[T], item: T) -> StackList[T]:
    stack.push(item)
    return stack


stack = StackList[Union[int, str]]()
stack = append_generic_to_stack(stack, 1)
stack = append_generic_to_stack(stack, "a")
stack = append_generic_to_stack(stack, 2)
