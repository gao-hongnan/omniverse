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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gao-hongnan/gaohn-dsa/blob/main/content/stack/concept.ipynb)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from IPython.display import display
from typing import Iterable, TypeVar, Optional


import sys
from pathlib import Path

def find_root_dir(current_path: Path = Path.cwd(), marker: str = '.git') -> Optional[Path]:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path
        The starting path to search from.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path or None
        The path to the root directory. Returns None if the marker is not found.
    """
    current_path = current_path.resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    return None

root_dir = find_root_dir(marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
else:
    raise ImportError("Root directory not found.")
```

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

- The **plate loader** in a sushi restaurant, where plates are neatly stacked
  one over the other, serves as a vivid illustration of a stack.
- When you finish eating from a plate, you place it on top of the stack on the
  **plate loader**. This action is akin to the `push` operation in a stack,
  where an element is added to the top.
- Now, consider the stack transitioning into a coding environment. We initiate
  an empty stack `s` represented as `s = []`. In this representation, **_the end
  of the list is treated as the top of the stack_**.
- As you add more plates (e.g., `p1`, `p2`), you `push` them onto the stack:
  `s.push(p1)`, leading to `s = [p1]`, and then `s.push(p2)`, resulting in
  `s = [p1, p2]`.
- When a waiter clears the topmost plate, this is similar to the `pop`
  operation, which **_returns and removes_** the top item of the stack. Thus,
  executing `s.pop()` would return `p2`, modifying the stack to `s = [p1]`.

Here, we have went through the two fundamental operations (amongst others) on
stack: `push` and `pop`.

- `push` operation pushes something on the top of the stack (appending to a
  list);
- `pop` operation returns and removes the top most item from the stack (popping
  from the list).

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
`pop()` for pop operations, which both run in $O(1)$ average time complexity.

### Why Use a List for Stack Implementation?

Python lists are dynamic arrays behind the scenes. This makes them ideal for
implementing stacks because:

- They provide **amortized constant time complexity** ($O(1)$) for adding and
  removing items from the end[^list-pop-time-complexity].
- Lists are **dynamic**, so they can grow and shrink on demand, which suits the
  variable size nature of stacks.
- They come with built-in methods that directly correspond to stack operations
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
from __future__ import annotations

from typing import List, Generator, TypeVar, Generic

T = TypeVar("T", covariant=False, contravariant=False)

class StackList(Generic[T]):
    def __init__(self) -> None:
        self._stack_items: List[T] = []

    def __len__(self) -> int:
        return len(self._stack_items)

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
        return not self._stack_items

    def peek(self) -> T:
        if self.is_empty():
            raise Exception("Stack is empty")
        return self._stack_items[-1]

    def pop(self) -> T:
        if self.is_empty():
            raise Exception("Stack is empty")
        return self._stack_items.pop()

    def push(self, item: T) -> None:
        self._stack_items.append(item)
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
stack = StackList[int]()
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

### Time Complexity

Consider a stack of $N$ items. The time complexity of the `push` and `pop`
operations is $O(1)$, as they both involve accessing the last item of the list
`_stack_items`. The `peek` operation also runs in $O(1)$ time, as it only
accesses the last item of the list without removing it. The `is_empty` method
runs in $O(1)$ time, as it only checks if the list `_stack_items` is empty.
Finally, the `size` property runs in $O(1)$ time, as it simply returns the
length of the list `_stack_items`.

```{list-table} Stack List Time Complexity
:header-rows: 1
:name: stack-list-time-complexity

* - Operations
  - Time Complexity
* - `push`
  - $\mathcal{O}(1)$
* - `pop`
  - $\mathcal{O}(1)$
```

The time complexity for both `push` and `pop` are $\mathcal{O}(1)$, an obvious
consequence because the native python `list`'s operations `append` and `pop` are
also $\mathcal{O}(1)$, so the result follows.

If you treat the list's start as top of the stack, then you might need to use
`insert(0)` and `pop(0)`, and these are $\mathcal{O}(N)$ operations.

### Space Complexity

Space complexity: $\mathcal{O}(N)$. The space required depends on the number of
items stored in the list `stack_items`, so if `stack_items` stores up to $N$
items, then space complexity is $\mathcal{O}(N)$.

## Implementing Stack Using Linked List

We need to think a bit little different from list where you easily visualize a
list's first and last element as the bottom and top of the stack respectively.

For Linked List, you think of a reversed list. That is to say, the `head` node
of the Linked List is the **top** of the stack and the last node (not the `None`
node) will be the beginning of the stack.

Ref:
[https://www.geeksforgeeks.org/stack-data-structure-introduction-program/?ref=lbp](https://www.geeksforgeeks.org/stack-data-structure-introduction-program/?ref=lbp)

```{code-cell} ipython3
from typing import Optional, Any

class LinkedListNode:
    """
    The LinkedListNode object is initialized with a value and can be linked to the next node by setting the next_node attribute to a LinkedListNode object.
    This node is Singular associated with Singly Linked List.

    Attributes:
        curr_node_value (Any): The value associated with the created node.
        next_node (LinkedListNode): The next node in the linked list. Note the distinction between curr_node_value and next_node, the former is the value of the node, the latter is the pointer to the next node.

    Examples:
        >>> node = Node(1)
        >>> print(node.curr_node_value)
        1
        >>> print(node.next_node)
        None
        >>> node.next_node = Node(2)
        >>> print(node.next_node.curr_node_value)
        2
        >>> print(node.next_node.next_node)
        None
    """

    curr_node_value: Any
    next_node: Optional["LinkedListNode"]

    def __init__(self, curr_node_value: Any = None) -> None:
        self.curr_node_value = curr_node_value
        self.next_node = None
```

```{code-cell} ipython3
class StackLinkedList:
    def __init__(self) -> None:
        self.head = None  # top of the stack

    def is_empty(self) -> bool:
        """Check if the stack is empty.

        The stack is empty if the head is None.

        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        return self.head is None

    def push(self, curr_node_value: Any) -> None:
        """Push a new node on top of the stack.

        # if push a value say 10 inside,, then the new node will be the head of the stack.
        # if push another value say 20 inside, then the 20 will be the head of the stack.
        # everytime you push a value it must be the pushed node become head.
        # so if you push 10, 20, 30, then it must be 30 -> 20 -> 10 -> None.
        # so think of base case if push 10 what happens?
        # as usual the logic is:
            - Start with the base case self.head to be None first, this will keep incrementing as we push more values.
            - Create a new node with the value of curr_node_value whenever a new value is pushed.
            - If we push in a 10, the newly_pushed_node holds the value of 10.
            - We set newly_pushed_node.next_node to become self.head so now newly_pushed_node becomes 10 -> None.
            - Now set self.head to be the newly_pushed_node so next time we push another value, it will be new_value -> 10 -> None.
            - If we push in a 20, the newly_pushed_node variables holds 20.
            - We set newly_pushed_node.next_node to become self.head so now newly_pushed_node becomes 20 -> (10 -> None).
            - The logic continues.

        Args:
            curr_node_value (Any): The current item (node) pushed into the stack.
        """

        newly_pushed_node = LinkedListNode(curr_node_value)
        newly_pushed_node.next_node = self.head
        self.head = newly_pushed_node
        print(f"Pushed {curr_node_value} onto the stack")

    def pop(self) -> Any:
        """Pop an item from the top of the stack.

        In this implementation, the item at the head of the Linked List is returned and removed.

        # logic is pop the head and it can always work since whenever you access self.head, the current value it holds is the first value and also the top of the stack.
        # - popped_node: set to self.head.
        # - self.head: set to self.head.next_node which is akin to removing the head and now the next value is the new head.
        # - popped_value: this is the current node value of popped_node.

        Raises:
            Exception: If stack is empty.

        Returns:
            Any: The top most item in the stack.
        """

        if self.is_empty():
            raise Exception("Stack is empty")

        popped_node = self.head
        self.head = self.head.next_node
        popped_value = popped_node.curr_node_value
        print(f"Popped {popped_value} from the stack")

        return popped_value

    def peek(self) -> Any:
        """Peek at the top of the stack.

        In this implementation, the item at the head of the Linked List is returned.

        Raises:
            Exception: If stack is empty.

        Returns:
            Any: The top most item in the stack.
        """

        if self.is_empty():
            raise Exception("Stack is empty")

        return self.head.curr_node_value

# Driver code
stack = StackLinkedList()
stack.push(10)
stack.push(20)
stack.push(30)
_ = stack.pop()
```

### Time Complexity

Time complexity: $\mathcal{O}(1)$ for both `push` and `pop` as no **traversal**
is involved.

### Space Complexity

Space complexity: $\mathcal{O}(N)$. The space required depends on the number of
items stored in the list `stack_items`, so if `stack_items` stores up to $N$
items, then space complexity is $\mathcal{O}(N)$.

## References and Further Readings

- [Tech Interview Handbook - Stack Algorithms](https://www.techinterviewhandbook.org/algorithms/stack/)
- [GeeksforGeeks - Introduction to Stack Data Structure](https://www.geeksforgeeks.org/introduction-to-stack-data-structure-and-algorithm-tutorials/)
- [Runestone Academy - The Stack Abstract Data Type](https://runestone.academy/ns/books/published/pythonds3/BasicDS/TheStackAbstractDataType.html)

[^stack-applications]:
    [Stack Applications](https://www.geeksforgeeks.org/real-time-application-of-data-structures/#application-of-stack)

[^list-pop-time-complexity]:
    [List pop() time complexity](https://stackoverflow.com/questions/195625/what-is-the-time-complexity-of-popping-elements-from-list-in-python)
