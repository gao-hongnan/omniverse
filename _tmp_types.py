from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from rich.pretty import pprint

from omnivault._types._generic import T
from omnivault.dsa.stack.concrete import StackList


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


import numpy as np
from numpy.typing import NDArray


def _compute_argmin_assignment(
    x: NDArray[np.float64], centroids: NDArray[np.float64]
) -> Tuple[int, float]:
    # Function implementation...
    return (0, 0.0)
