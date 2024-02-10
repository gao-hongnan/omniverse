from typing import List, TypeVar, Any

T = TypeVar('T')

def append_and_return_list(list_: List[T], element: T) -> List[T]:
    list_.append(element)
    return list_

list_of_ints: List[int] = [1, 2, 3, 4, 5]

# This will cause a mypy error
new_list_of_ints: List[int] = append_and_return_list(list_=list_of_ints, element="abcedf")
print(new_list_of_ints)


