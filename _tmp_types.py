from omnivault.dsa.stack.concrete import StackList
from omnivault._types._generic import T

def append_int_to_stack(stack: StackList[int], value: int) -> StackList[int]:
    stack.push(value)
    return stack


stack_int = StackList[int]()
stack_int = append_int_to_stack(stack_int, 1)
stack_int = append_int_to_stack(stack_int, 2)
stack_int = append_int_to_stack(stack_int, "3")
print(stack_int.stack_items)

def append_str_to_stack(stack: StackList[str], value: str) -> StackList[str]:
    stack.push(value)
    return stack

stack_str = StackList[str]()
stack_str = append_str_to_stack(stack_str, "1")
stack_str = append_str_to_stack(stack_str, "2")
stack_str = append_str_to_stack(stack_str, 3)
print(stack_str.stack_items)

def append_to_stack(stack: StackList[T], value: T) -> StackList[T]:
    stack.push(value)
    return stack

stack = StackList[T]()
stack = append_to_stack(stack, 1)
stack = append_to_stack(stack, "2")
stack = append_to_stack(stack, 3)
print(stack.stack_items)