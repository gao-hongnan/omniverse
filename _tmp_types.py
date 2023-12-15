from typing import Any, Tuple, Union

from omnivault._types._generic import T
from omnivault.dsa.stack.concrete import StackList

import subprocess

# Insecure use of subprocess with shell=True
subprocess.call("echo 'Hello, world!'", shell=True)


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


def append_int_to_stack(stack: StackList[int], item: int) -> StackList[int]:
    stack.push(item)
    return stack


stack = StackList[int]()
stack = append_int_to_stack(stack, 1)
stack = append_int_to_stack(
    stack, "a"
)  # Error: Argument 2 to "append_int_to_stack" has incompatible type "str"; expected "int"


import numpy as np
from numpy.typing import NDArray


def _compute_argmin_assignment(x: NDArray[np.float64], centroids: NDArray[np.float64]) -> Tuple[int, float]:
    # Function implementation...
    return (0, 0.0)
