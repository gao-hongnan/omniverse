---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Concept

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Stack-orange)
![Tag](https://img.shields.io/badge/Tag-Array-orange)
![Tag](https://img.shields.io/badge/Tag-Linked_List-orange)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from IPython.display import display
from typing import Generator, List, Union, Any
from rich.pretty import pprint

import sys
from pathlib import Path

def find_root_dir(current_path: Path | None = None, marker: str = '.git') -> Path | None:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path | None
        The starting path to search from. If None, the current working directory
        `Path.cwd()` is used.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path | None
        The path to the root directory. Returns None if the marker is not found.
    """
    if not current_path:
        current_path = Path.cwd()
    current_path = current_path.resolve()
    for parent in [current_path, *current_path.parents]:
        if (parent / marker).exists():
            return parent
    return None

root_dir = find_root_dir(marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.dsa.stack.base import Stack
    from omnivault._types._generic import T
else:
    raise ImportError("Root directory not found.")
```

## Learning Objectives

1. Understand the fundamental concept of a **Stack** as an abstract data type in
   computer science, characterized by its **_Last In, First Out (LIFO)_**
   principle.
2. Grasp the intuition behind a stack as a collection with restricted access,
   analogous to real-world examples such as a stack of books or a **_stack of
   plates_** in a restaurant.
3. Learn the core operations of a stack, including `push`, `pop`, `peek`,
   `is_empty`, and `size`, along with their descriptions and significance.
4. Comprehend the implementation details of the `StackList` class in Python,
   particularly focusing on its usage of a dynamic array (Python list) for stack
   operations and its implementation as a **Generic** class for type
   flexibility.
5. Be able to implement and use the `StackList` class, including pushing items
   onto the stack, checking its size, peeking at the top item, popping items
   off, and iterating over the stack.
6. Understand the **time complexity** of the stack operations, particularly the
   average and amortized worst-case time complexities, along with an explanation
   of **amortized analysis** and **exponential growth strategy**.
7. Recognize the **space complexity** of the `StackList` class and how it varies
   based on the types and sizes of the elements stored in the stack.

## Introduction

The [**Stack**](<https://en.wikipedia.org/wiki/Stack_(abstract_data_type)>) data
structure is a fundamental concept in computer science, acting as an abstract
model for storing and managing data in a specific order. This
[**_Last In, First Out (LIFO)_**](https://www.geeksforgeeks.org/lifo-last-in-first-out-approach-in-programming/)
principle underpins its functionality, where the last element added to the stack
is the first one to be removed. Understanding stacks is crucial for various
computing tasks, from algorithm design to system operation.

### Intuition

At its core, a stack represents a collection with restricted access, akin to a
real-world stack of items. The primary intuition behind a stack is its
sequential access pattern - you can only add or remove items from the top. This
structure is analogous to a stack of books; you can only remove the top book
without disturbing the others. In computer science, stacks are used to store
data in a way that provides fast access to the last item added, but does not
allow for random access to other elements.

### Analogy: The Stack of Plates

An excellent analogy for understanding stacks is the **_stack of plates_** in a
restaurant, particularly in a sushi restaurant. Imagine this scenario:

-   The **plate loader** in a sushi restaurant, where plates are neatly stacked
    one over the other, serves as a vivid illustration of a stack.
-   When you finish eating from a plate, you place it on top of the stack on the
    **plate loader**. This action is akin to the `push` operation in a stack,
    where an element is added to the top.
-   Now, consider the stack transitioning into a coding environment. We initiate
    an empty stack `s` represented as `s = []`. In this representation, **_the
    end of the list is treated as the top of the stack_**.
-   As you add more plates (e.g., `p1`, `p2`), you `push` them onto the stack:
    `s.push(p1)`, leading to `s = [p1]`, and then `s.push(p2)`, resulting in
    `s = [p1, p2]`.
-   When a waiter clears the topmost plate, this is similar to the `pop`
    operation, which **_returns and removes_** the top item of the stack. Thus,
    executing `s.pop()` would return `p2`, modifying the stack to `s = [p1]`.

Here, we have went through the two fundamental operations (amongst others) on
stack: `push` and `pop`.

-   `push` operation pushes something on the top of the stack (appending to a
    list);
-   `pop` operation returns and removes the top most item from the stack
    (popping from the list).

```{figure} ./assets/stack-1.svg
---
name: stack-1
---
Stack Diagram Flow.
```

````{admonition} Tikz Code
:class: dropdown
```latex
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric}

\begin{document}
\begin{tikzpicture}[
    box/.style={draw, rectangle, minimum height=1cm, minimum width=1cm, thick},
    stack/.style={draw, rectangle, minimum width=1.2cm, minimum height=4cm, thick, align=center},
    arrow/.style={thick, -Stealth, bend left=45}
]

% Empty stack
\node[stack, label=below:empty stack] (empty) at (0,0) {};

% First push
\node[stack, label=below:push] (push1) at (3,0) {};
\node[box, fill=blue!30] (element1) at ([yshift=0.5cm]push1.south) {1};
\draw[arrow] ([yshift=-0.5cm]push1.north) -- (element1.north);
\node[above=0.1cm of push1.north] {push(1)};

% Second push
\node[stack, label=below:push] (push2) at (6,0) {};
\node[box, fill=blue!30] (element2a) at ([yshift=0.5cm]push2.south) {1};
\node[box, fill=blue!30] (element2b) at ([yshift=1.5cm]push2.south) {2};
\draw[arrow] ([yshift=-0.5cm]push2.north) -- (element2b.north);
\node[above=0.1cm of push2.north] {push(2)};

% Third push
\node[stack, label=below:push] (push3) at (9,0) {};
\node[box, fill=blue!30] (element3a) at ([yshift=0.5cm]push3.south) {1};
\node[box, fill=blue!30] (element3b) at ([yshift=1.5cm]push3.south) {2};
\node[box, fill=blue!30] (element3c) at ([yshift=2.5cm]push3.south) {3};
\draw[arrow] ([yshift=-0.5cm]push3.north) -- (element3c.north);
\node[above=0.1cm of push3.north] {push(3)};

% Pop
\node[stack, label=below:pop] (pop1) at (12,0) {};
\node[box, fill=blue!30] (element4a) at ([yshift=0.5cm]pop1.south) {1};
\node[box, fill=blue!30] (element4b) at ([yshift=1.5cm]pop1.south) {2};
\draw[arrow] (element4b.north) -- ([yshift=1.5cm]element4b.north);

\node[above=0.1cm of pop1.north] {pop(3)};

\end{tikzpicture}
\end{document}
```
````

### Real-World Use Case

In the real world, stacks are ubiquitous in scenarios requiring reverse order
processing. For instance, in web browsers, the **Back** button functionality is
a classic example. Each visited webpage is "pushed" onto a stack, and clicking
**Back** "pops" the pages in reverse visitation order, demonstrating stack’s
utility in history navigation[^stack-applications].

## Stack with List as Underlying Data Structure

In computer science, a stack is an abstract data type that serves as a
collection of elements with two principal operations: push, which adds an
element to the collection, and pop, which removes the most recently added
element. In many programming languages, the built-in list or array structures
are used to implement the stack due to their efficiency and simplicity. In
Python, the `list` is particularly well-suited for this purpose because it
provides built-in methods for stack operations, such as `append()` for push and
`pop()` for pop operations, which both run in $\mathcal{O}(1)$ average time
complexity.

### Why Use a List for Stack Implementation?

Python lists are dynamic arrays behind the scenes. This makes them ideal for
implementing stacks because:

-   They provide **amortized constant time complexity** ($\mathcal{O}(1)$) for
    adding and removing items from the end[^list-pop-time-complexity].
-   Lists are **dynamic**, so they can grow and shrink on demand, which suits
    the variable size nature of stacks.
-   They come with built-in methods that directly correspond to stack operations
    (`append` for push and `pop` without an index for pop).

Using Python lists sidesteps the need for managing the capacity of the
underlying data structure, which is an advantage over using a fixed-size array.

### Operations

Before implementing the stack, let's define the operations that will be
supported by the `StackList` class:

```{list-table} Stack List Operations
:header-rows: 1
:name: stack-list-operations

* - Operation
  - Description
* - `push`
  - Appends an item to the end of the `_stack_items` list, effectively pushing
    it onto the stack.
* - `pop`
  - Removes and returns the last item from `_stack_items`, adhering to the Last
    In, First Out (LIFO) principle. Raises an exception if the stack is empty.
* - `peek`
  - Returns the last item without removing it, providing a view of the top item
    of the stack. Raises an exception if the stack is empty.
* - `is_empty`
  - Returns `True` if the stack is empty, otherwise `False`.
* - `size`
  - A property that returns the number of items in the stack, utilizing the
    `__len__` method to return the length of `_stack_items`.
```

### Implementation

Our class inherits `Generic[T]` to make it a generic class, meaning it can store
items of any type. This flexibility is provided by the use of the generic type
variable `T` in `TypeVar("T")`. Furthermore, since `StackList` is a container,
inheritance from `Generic[T]` is necessary to ensure type safety and clarity.
For example, we can instantiate a `StackList` of integers by specifying the type
`StackList[int]`, and the type hints will be enforced by the class (by a static
type checker such as `mypy`) so that that particular stack instance can only
store integers.

Stack being a container, we will also implement some dunder methods:

```{list-table} Stack List Dunder Methods
:header-rows: 1
:name: stack-list-dunder-methods

* - Dunder Method
  - Description
* - `__len__`
  - Returns the length of the stack, which is the length of `_stack_items`.
* - `__iter__`
  - Returns an iterator over the stack, which is implemented using a generator
    that yields items from the stack using `self.pop()`. This means each
    iteration over the `StackList` instance will modify the stack by removing
    its elements.
```

```{code-cell} ipython3
class StackList(Stack[T]):
    def __len__(self) -> int:
        return len(self.stack_items)

    def __iter__(self) -> Generator[T, None, None]:
        while not self.is_empty():
            yield self.pop()

    @property
    def stack_items(self) -> List[T]:
        return self._stack_items

    @property
    def size(self) -> int:
        return len(self)

    def is_empty(self) -> bool:
        return not self.stack_items

    def peek(self) -> T:
        return self.stack_items[-1]

    def pop(self) -> T:
        if self.is_empty():
            raise Exception("Stack is empty")
        return self.stack_items.pop()

    def push(self, item: T) -> None:
        self.stack_items.append(item)
```

```{prf:remark} Some Remarks
:label: stack-list-remarks

The implementation of `__iter__` makes the `StackList` a single-use iterable, as
iterating over it empties the stack. This is an important detail to be aware of,
as it differs from the typical behavior of iterables in Python.
```

### Example

The provided example demonstrates pushing integers onto the stack and then
performing various operations:

1. **Pushing Items**: Integers `1` to `6` are pushed onto the stack.
2. **Checking Size**: The size of the stack is checked, which should be `6`
   after pushing six items.
3. **Peeking**: The `peek` method is used to view the top item of the stack
   without removing it. In this case, it should be `6`.
4. **Popping**: The `pop` method is used to remove and return the top item of
   the stack, which would be `6`.
5. **Iterating with next**: Using `next(iter(stack))` pops and returns the next
   item (now the top of the stack), demonstrating the destructive nature of the
   iteration implemented in `__iter__`.
6. **Iterating Over the Stack**: The final loop iterates over the remaining
   items in the stack, popping and printing each one. This will empty the stack.

```{code-cell} ipython3
stack = StackList[int]() # means stack should only store integers
items = [1, 2, 3, 4, 5, 6]

for item in items:
    print(f"pushing {item} onto the stack")
    stack.push(item)

length = len(stack)
print(f"stack size = {length}")

top_item = stack.peek()
print(f"peek = {top_item}")

removed_item = stack.pop()
print(f"pop and return the top of the stack = {removed_item}")

print(
    f"call next(iter) on stack will pop and return the top "
    f"of the stack = {next(iter(stack))}"
)

print()

for item in stack:
    print(item)

print()
print(f"stack size = {len(stack)}")
print(f"stack is empty? {stack.is_empty()}")
```

### The Importance of Generic Types

The `StackList` class is a generic class, meaning it can store items of any
type. This flexibility is provided by the use of the generic type variable `T`
in `TypeVar("T")`. Furthermore, since `StackList` is a container, inheritance
from `Generic[T]` is necessary to ensure type safety and clarity. For example,
we can instantiate a `StackList` of integers by specifying the type
`StackList[int]`, and the type hints will be enforced by the class (by a static
type checker such as `mypy`) so that that particular stack instance can only
store integers.

Consider the following example, where we have a function named
`append_int_to_stack` that takes a `StackList[int]` and an integer value, and
returns a `StackList[int]` with the value appended to it. This function will
only accept a `StackList` that stores integers, and will return a `StackList`
that stores integers.

```{code-cell} ipython3
def append_int_to_stack(stack: StackList[int], value: int) -> StackList[int]:
    stack.push(value)
    return stack

stack_int = StackList[int]()
stack_int = append_int_to_stack(stack_int, 1)
stack_int = append_int_to_stack(stack_int, 2)
stack_int = append_int_to_stack(stack_int, "3")
pprint(stack_int.stack_items)
```

The last line of the code above will raise a `mypy` error:

```python
error: Argument 2 to "append_int_to_stack" has incompatible type "str"; expected "int"  [arg-type]
```

because we are trying to push a string onto a stack that only accepts integers.

Similarly, we can define a function that takes a `StackList[str]` and returns a
`StackList[str]`:

```{code-cell} ipython3
def append_str_to_stack(stack: StackList[str], value: str) -> StackList[str]:
    stack.push(value)
    return stack

stack_str = StackList[str]()
stack_str = append_str_to_stack(stack_str, "1")
stack_str = append_str_to_stack(stack_str, "2")
stack_str = append_str_to_stack(stack_str, 3)
print(stack_str.stack_items)
```

and this will also raise a `mypy` error:

```python
error: Argument 2 to "append_str_to_stack" has incompatible type "int"; expected "str"  [arg-type]
```

because we are trying to push an integer onto a stack that only accepts strings.

To push both integers and strings onto a stack, we can define a function that
takes a `StackList[Union[str, int]]` and returns a `StackList[Union[str, int]]`:

```{code-cell} ipython3
def append_to_stack(stack: StackList[T], value: T) -> StackList[T]:
    stack.push(value)
    return stack

stack = StackList[Union[str, int]]()
stack = append_to_stack(stack, 1)
stack = append_to_stack(stack, "2")
stack = append_to_stack(stack, 3)
print(stack.stack_items)
```

Now if you run `mypy` on the code above, it will not raise any errors. However,
we type hint the function `append_to_stack` with `StackList[T]` instead of
`StackList[Union[str, int]]`, and this is because `T` is a generic type variable
that can be bound to any type. In this case, `T` is bound to `Union[str, int]`
because we specified `StackList[Union[str, int]]` when we defined `stack`.

Note that you cannot define `stack = StackList[T]` without specifying the type
`T` because `T` is a generic type variable, and `mypy` will raise an error:

```python
error: Type variable "omnivault._types._generic.T" is unbound  [valid-type]
```

When you create your own generic class like `StackList`, the type variable `T`
must be bound to a specific type at the point of instantiation. This is a
requirement for user-defined generics to ensure type safety and consistency.
This is consistent with the behavior of built-in generic classes such as
`List[T]`, which also require the type variable `T` to be bound to a specific
type at the point of instantiation. So you cannot also define something like
`array = list[T]([1, 2, 3])` without specifying the type `T`.

This is the power of generic types - they allow us to write code that is
type-safe and clear, and they enable us to write functions that are generic
enough to work with different types of stacks.

### Time Complexity

Given a stack of size $N$ implemented using a Python list, the average and
amortized worst-case time complexities of the fundamental operations are as
follows:

-   **`push` (append)**: Appending an item to the end of a Python list has an
    average time complexity of $\mathcal{O}(1)$. This constant time complexity
    results from Python lists being implemented as dynamic arrays. Although they
    occasionally need to be resized—an operation that takes $\mathcal{O}(N)$
    time—the allocation strategy of Python lists employs an exponential growth
    factor. This means that resizes happen less frequently as the list grows.
    Thus, while the worst-case complexity of a single `append` operation can be
    $\mathcal{O}(N)$ (when resizing is required), the cost of resizing is spread
    over a large number of `append` operations, leading to an amortized time
    complexity of $\mathcal{O}(1)$. This behavior is well-documented in the
    [Python Time Complexity page](https://wiki.python.org/moin/TimeComplexity),
    which lists both the average and amortized worst-case complexities for
    `list.append` as $\mathcal{O}(1)$.

-   **`pop` (without an index)**: The `pop` method in Python, when used without
    an index, removes the last item of the list. This operation has an average
    and amortized worst-case time complexity of $\mathcal{O}(1)$, as it directly
    accesses and removes the element at the end of the dynamic array without
    needing to shift any elements. This behavior is consistent with the Python
    documentation referenced above.

-   **`peek`**: Retrieving the item at the top of the stack without removing it,
    achieved by a direct access to the last element of the list (i.e.,
    `list[-1]`), is an operation with a time complexity of $\mathcal{O}(1)$.
    This constant time complexity is due to the array-based nature of Python
    lists, which allows for direct indexing.

-   **`is_empty`**: The `is_empty` method checks whether the list is empty,
    equivalent to verifying if the length of the list is zero. This is a
    constant-time operation ($\mathcal{O}(1)$) in Python because the list object
    maintains and updates its count of elements.

-   **`size` (len)**: Obtaining the number of elements in the list, as done by
    the `size` property using the `__len__` method, is a $\mathcal{O}(1)$
    operation. Python lists automatically keep track of their size, enabling
    quick access to this information.

Some more remarks on **_amortized worst-case time complexity_**:

```{prf:remark} Amortized Worst-Case Time Complexity
:label: stack-list-amortized-worst-case-time-complexity

1. **Amortized Analysis**: In amortized analysis, we average the time complexity
   over a sequence of operations, not just a single operation. For Python lists,
   when a resizing occurs (which is an $\mathcal{O}(N)$ operation), it doesn't
   happen with every append. Python lists grow in such a way that the resizes
   happen exponentially less often as the size of the list grows. This strategy
   ensures that, averaged over a large number of appends, the time per operation
   is still constant, or $\mathcal{O}(1)$.

2. **Exponential Growth Strategy**: When a Python list needs to resize, it
   doesn't just increase its size by one element. Instead, it typically
   increases its size by a larger amount (approximately doubling, although the
   exact growth factor may vary). This means that, although the individual
   operation of resizing and copying the list is $\mathcal{O}(N)$, such
   operations happen so infrequently that their cost is "amortized" over the
   many $\mathcal{O}(1)$ append operations, resulting in an overall
   $\mathcal{O}(1)$ amortized time complexity.

3. **Worst-Case vs. Amortized Worst-Case**: The worst-case scenario for a single
   operation of `list.append()` can indeed be $\mathcal{O}(N)$ (when a resize
   occurs), but when considering the worst-case in an amortized sense (across
   multiple operations), it averages out to $\mathcal{O}(1)$.
```

```{list-table} Stack List Time Complexity
:header-rows: 1
:name: stack-list-time-complexity

- - Operations
  - Average Time Complexity
  - Amortized Worst-Case Time Complexity
- - `push`
  - $\mathcal{O}(1)$
  - $\mathcal{O}(1)$ [occasional resizing]
- - `pop`
  - $\mathcal{O}(1)$
  - $\mathcal{O}(1)$
- - `peek`
  - $\mathcal{O}(1)$
  - $\mathcal{O}(1)$
- - `is_empty`
  - $\mathcal{O}(1)$
  - $\mathcal{O}(1)$
- - `size`
  - $\mathcal{O}(1)$
  - $\mathcal{O}(1)$
```

If you treat the list's start as top of the stack, then you might need to use
`insert(0)` and `pop(0)`, and these are $\mathcal{O}(N)$ operations.

### Space Complexity

The space complexity of the `StackList` class is $\mathcal{O}(N)$, where $N$ is
the number of elements in the stack. This linear relationship arises because the
primary storage is the list `_stack_items`, whose size grows directly with the
number of elements added to the stack. If the stack's elements are themselves
containers, such as lists or sets, the overall space complexity will depend on
the sizes of these containers. In the case where each element has a similar size
`M`, the space complexity can be approximated as $\mathcal{O}(N \times M)$. For
variable-sized containers, the complexity becomes the sum of the sizes of all
elements, i.e., $\mathcal{O}\left(\sum_{i=1}^{N} M_i\right)$, where `M_i` is the
size of the `i`-th element.

## References and Further Readings

-   [Tech Interview Handbook - Stack Algorithms](https://www.techinterviewhandbook.org/algorithms/stack/)
-   [GeeksforGeeks - Introduction to Stack Data Structure](https://www.geeksforgeeks.org/introduction-to-stack-data-structure-and-algorithm-tutorials/)
-   [Runestone Academy - The Stack Abstract Data Type](https://runestone.academy/ns/books/published/pythonds3/BasicDS/TheStackAbstractDataType.html)

[^stack-applications]:
    [Stack Applications](https://www.geeksforgeeks.org/real-time-application-of-data-structures/#application-of-stack)

[^list-pop-time-complexity]:
    [List pop() time complexity](https://stackoverflow.com/questions/195625/what-is-the-time-complexity-of-popping-elements-from-list-in-python)
