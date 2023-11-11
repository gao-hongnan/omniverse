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

## Introduction

[**Linear search**](https://en.wikipedia.org/wiki/Linear_search), also known as
sequential search, is a fundamental algorithm in the field of computer science,
serving as a bedrock for understanding more complex algorithms. It represents
the most _intuitive_ method of searching for an element within a list. The
simplicity of linear search lies in its straightforward approach: examining each
element in the list **_sequentially_** until the desired element is found or the
end of the list is reached.

This algorithm, in its essence, begins with the assumption of an **unordered**
list. Without any prior knowledge of the arrangement of elements, linear search
treats each item in the list with **equal importance**, methodically checking
each until a match is found. This aspect of linear search, while simple, is
crucial in understanding how more complex searching algorithms evolve and
optimize under different conditions.

### Extensions

As we progress from the basic unordered scenario, the concept of an ordered list
introduces a subtle yet significant shift in the algorithm’s operation. In an
ordered list, the elements are arranged in a known sequence, either ascending or
descending. This arrangement allows for certain optimizations in the search
process. For instance, the search can be terminated early if the current element
in the sequence is greater (or lesser, depending on the order) than the element
being searched for, as it confirms the absence of the target element in the
remaining part of the list.

Furthermore, exploring the concept of a probabilistic array adds another
dimension to our understanding of linear search. In a probabilistic setting,
each element's likelihood of being the target is different and known. This
knowledge can be leveraged to modify the traditional linear search into a more
efficient version, where the order of search is determined based on the
probability of finding the target element, thus potentially reducing the average
number of comparisons needed.

### Intuition

The **linear search** algorithm operates on a fundamental principle of
_sequential access_ in data structures. This intuition is best understood by
considering the data structure it commonly operates on, such as a
[list](<https://en.wikipedia.org/wiki/List_(abstract_data_type)>). In computer
science, the representation of a list in memory is typically implemented as a
linear data structure. This means that the elements of the list are stored in
[**contiguous**](<https://en.wikipedia.org/wiki/Fragmentation_(computing)>)
memory locations. Each element in the list is stored at a memory address that
sequentially follows the previous element, adhering to a defined order.

To elucidate the mechanism of linear search, let's consider the task of locating
a specific target element $\tau$ within a list $\mathcal{A}$. The process
unfolds in a _sequential manner_: beginning with the first element, the
algorithm examines each subsequent element in turn. This _sequential traversal_
is the hallmark of linear search, distinguishing it from more complex search
algorithms that may jump or divide the list in their search process.

```{figure} ./assets/linear_search_geeksforgeeks.png
---
name: linear_search_diagram
---
Linear Search Algorithm. Image credit to [GeeksforGeeks](https://www.geeksforgeeks.org/linear-search/).
```

As the algorithm progresses through list $\mathcal{A}$, searching for the target
element $\tau$, there are two primary outcomes to consider:

1. **Element Discovery**: If the element $\tau$ is encountered at any point
   during the traversal, the search is immediately successful. The algorithm can
   then return a positive result, such as `True`, or, more informatively, the
   index $i$ at which $\tau$ was found in $\mathcal{A}$. Returning the index $i$
   provides not only confirmation of the element's presence but also its exact
   _location_ within the list. This outcome signifies that
   $\tau = \mathcal{A}[i]$, where $\mathcal{A}[i]$ denotes the element at the
   $i^{th}$ position in list $\mathcal{A}$.

2. **Exhaustive Search without Discovery**: Conversely, if the traversal reaches
   the end of the list without encountering $\tau$, it implies that the element
   is not present in $\mathcal{A}$. In this case, the algorithm concludes with a
   negative result, typically returning `False`. This denotes that for all
   indices $i$ in $\mathcal{A}$, $\tau \neq \mathcal{A}[i]$.

The _simplicity_ of this approach is its most defining characteristic (read:
brute force). Unlike algorithms that rely on specific data structure properties
(such as [binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm),
which requires a **sorted** array), linear search _makes no such assumptions_.
It treats each element in $\mathcal{A}$ with equal priority and does not benefit
from any particular arrangement of the data. We will see later on some
extensions of linear search that leverage the ordering of elements to optimize
the search process or if the list is probabilistic.

## Contiguous

- **Contiguous Memory Allocation**: In most programming languages, when a list
  (or an array, which is a fixed-size list) is created, a contiguous block of
  memory is allocated. This contiguous allocation ensures that each element in
  the list can be accessed by computing its memory address based on the starting
  memory address of the list and the element's position (or index) in the list.

- **Index-Based Access**: The position of each element in the list is determined
  by its index, a numerical representation starting from 0 (in zero-indexed
  languages like Python, C, and Java) or 1 (in one-indexed languages like Lua
  and MATLAB). The memory address of an element at index $n$ can be calculated
  using the formula: `Base Address + (n * Size of Element)`. This formula
  leverages the fact that elements are stored at evenly spaced intervals in
  memory.

- **Relative Positioning**: The sequential storage implies that the memory
  location of an element is relative to its immediate neighbors. For example, in
  a list of integers (assuming each integer occupies 4 bytes and the list starts
  at memory address 100), the element at index 1 would be at memory address 104,
  index 2 at 108, and so on. This relative positioning enables efficient access
  and traversal operations, as moving to the next or previous element involves a
  consistent step in memory.

Understanding this memory model is crucial for appreciating how basic operations
on lists, like indexing and iterating, are performed and why they have the time
complexities they do. For instance, accessing any element in a list via its
index is an $\mathcal{O}(1)$ operation because it involves a direct calculation
to find the element's memory address, regardless of the list's size.

## Unordered Sequential Search

We start with the simplest scenario: searching for an element in an unordered
list $\mathcal{A}$.

We make some explicit assumptions:

1. **All Integers**: The set/array $\mathcal{A}$ is a set/list of integers.
2. **Unsorted**: The set/array $\mathcal{A}$ need not be ordered/sorted.
3. **Uniform Probability of Each Element Being the Target**: This assumption
   implies that before the search begins, every element in the set/array
   $\mathcal{A}$ is equally likely to be the target element $\tau$. In other
   words, there is no prior information or distribution pattern that indicates
   some elements are more likely to be $\tau$ than others. This assumption is
   crucial for the linear search algorithm, as it underpins the decision to
   search each element in $\mathcal{A}$ sequentially without prioritizing any
   specific elements.

### Algorithm (Iterative)

#### Pseudocode

````{prf:algorithm} Unordered Linear Search Pseudocode (Iterative)
:label: unordered-linear-search-pseudocode-iterative

```
Algorithm : unordered_linear_search_iterative(A, t)

    Input : A = [a_0, a_1, ..., a_{N-1}] (list of elements),
            t (target value)
    Output: Index of t in A or -1 if not found

    1: Initialize n = 0

    2: while n < N do
        3:    if A[n] == t then
        4:        return n
        5:    end if
        6:    n = n + 1
       7: end while

    8: return -1
```
````

#### Mathematical Representation

```{prf:algorithm} Unordered Linear Search Mathematical Representation (Iterative)
:label: unordered-linear-search-mathematical-representation-iterative

Given a list $\mathcal{A}$ of $N$ elements with values or records
$\mathcal{A}_0, \mathcal{A}_1, \ldots, \mathcal{A}_{N-1}$, and a target value
$\tau$, the goal is to find the index of the target $\tau$ in $\mathcal{A}$
using an iterative linear search procedure.

The search space $\mathcal{S}$ for the linear search is defined as the set of
indices in $\mathcal{A}$:

$$
\mathcal{S} = \{ n \in \mathbb{N} \,|\, 0 \leq n < N \}
$$

The algorithm performs the following steps:

1. **Initialization**:

   1. Set the initial index $n = 0$.

2. **Iterative Procedure**: While $n < N$:

   1. Check the current element: If $\mathcal{A}_n = \tau$, the search
      terminates successfully. Return $n$ as the index where $\tau$ was found in
      $\mathcal{A}$.
   2. If $\mathcal{A}_n \neq \tau$, increment the index: $n = n + 1$.

3. **Termination**: The algorithm terminates when $n \geq N$, indicating the end
   of the list has been reached without finding $\tau$. In this case, return
   $-1$ to indicate that $\tau$ is not found in $\mathcal{A}$.
```

#### Correctness

After detailing the iterative approach to the **Unordered Sequential Search**,
it is imperative to delve into the **correctness** of the algorithm. This aspect
is crucial in
[algorithmic design](https://en.wikipedia.org/wiki/Algorithm_design) as it
assures that the method not only works in theory but also functions correctly in
practical scenarios. The correctness of an algorithm can typically be
established by proving two key properties: its **invariance** and
**termination**.

- **Invariance** ensures that the algorithm always maintains certain conditions
  that lead to a correct solution,
- **Termination** guarantees that the algorithm will eventually complete its
  execution.

```{prf:proof}
Here's the proof:

1. **Invariance**: Throughout the execution of the linear search algorithm, the
   invariant we maintain is that the target element $\tau$ exists in the list
   $\mathcal{A}$ in the unexplored part of the search space $\mathcal{S}$.
   Initially, $\mathcal{S}$ encompasses the entire list, i.e.,
   $\mathcal{S} = \{ n \in \mathbb{N} \,|\, 0 \leq n < N \}$. With each
   iteration, if $\mathcal{A}_n \neq \tau$, the current index $n$ is eliminated
   from the search space, effectively shrinking $\mathcal{S}$ by one element.
   This process ensures that if $\tau$ is present in $\mathcal{A}$, it will be
   in the remaining $\mathcal{S}$ until it is found.

2. **Termination**: The termination of the algorithm is defined by two distinct
   conditions:
   - **Discovery of $\tau$**: If at any iteration $\mathcal{A}_n = \tau$, the
     algorithm terminates and returns the index $n$. This indicates a successful
     search.
   - **Exhaustion of $\mathcal{S}$**: If the algorithm iterates through the
     entire list without finding $\tau$ (i.e., $n \geq N$), it concludes that
     $\tau$ is not present in $\mathcal{A}$. It then terminates and returns
     \(-1\), indicating an unsuccessful search.

The proof focuses on the maintenance of the invariant and the termination
conditions, ensuring that the algorithm is both correct (it will find $\tau$ if
it is present) and finite (it will terminate after a finite number of steps
regardless of the outcome).
```

### Implementation

```{code-cell} ipython3
from __future__ import annotations

from typing import TypeVar, Tuple, Iterable, Sequence

T = TypeVar("T")


def unordered_linear_search_iterative(
    container: Sequence[T], target: T
) -> Tuple[bool, int]:
    """If the target element is found in the container, returns True and its index,
    else, return False and -1 to indicate the not found index."""
    is_found = False  # a flag to indicate so your return is more meaningful
    index = 0
    for item in container:
        if item == target:
            is_found = True
            return is_found, index
        index += 1
    return is_found, -1

# use while loop

def unordered_linear_search_iterative_while_loop(
    container: Sequence[T], target: T
) -> Tuple[bool, int]:
    """If the target element is found in the container, returns True and its index,
    else, return False and -1 to indicate the not found index."""

    index = 0
    length = len(container)

    while index < length:
        if container[index] == target:
            return True, index
        index += 1
    return False, -1

unordered_list = [1, 2, 32, 8, 17, 19, 42, 13, 0]

print(unordered_linear_search_iterative(unordered_list, -1)) # smaller than smallest element
print(unordered_linear_search_iterative(unordered_list, 45)) # larger than largest element
print(unordered_linear_search_iterative(unordered_list, 13)) # in the middle
```

#### Time Complexity

We need to split the time complexity into a few cases, this is because the time
complexity **_heavily_** depends on the position of the target element we are
searching for.

If the element we are searching for is at the beginning of the list, then the
time complexity is $\mathcal{O}(1)$, because we only need to check the first
element.

If the element is at the end of the list, then the time complexity is
$\mathcal{O}(n)$, because we need to check every element in the list.

On average, the time complexity is $\mathcal{O}(\frac{n}{2})$. This average
means that for a list with $n$ elements, there is an equal chance that the
element we are searching for is at the beginning, middle, or end of the list. In
short, it is a uniform distribution. And therefore the expected time complexity
is $\mathcal{O}(\frac{n}{2})$.

However, so far we assumed that the element we are searching for is in the list.
If the element is not in the list, then the time complexity is $\mathcal{O}(n)$
for all cases, because we need to check every element in the list.

```{list-table} Time Complexity of Sequential Search
:header-rows: 1
:name: sequential_search_time_complexity

* - Case
  - Worst Case
  - Average Case
  - Best Case
* - Element is in the list
  - $\mathcal{O}(n)$
  - $\mathcal{O}(\frac{n}{2})$
  - $\mathcal{O}(1)$
* - Element is not in the list
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
```

#### Space Complexity

Space complexity: $\mathcal{O}(1)$ because we are keeping track of one
boolean/index variable in the loop. However, if we count the space of the list,
then the space complexity is $\mathcal{O}(n)$ since the list is of size $n$.

However, the consensus is that, if the list given is a constant list, and not
part of the algorithm, we will not count the size of the list, and thus the
space complexity is $\mathcal{O}(1)$.

### Implementation (Recursive)

```{code-cell} ipython3
def unordered_sequential_search_recursive(
    container: Iterable[T], target: T, index: int = 0
) -> int:
    """Recursive implementation of unordered Sequential Search."""
    if len(container) == 0:  # if not container is also fine
        return -1  # not found

    if container[0] == target:  # this is base case
        return index  # found

    # notice we increment index by 1 to mean index += 1 in the iterative case
    return unordered_sequential_search_recursive(
        container[1:], target, index + 1
    )  # recursive case

unordered_list = [1, 2, 32, 8, 17, 19, 42, 13, 0]

print(unordered_sequential_search_recursive(unordered_list, -1)) # smaller than smallest element
print(unordered_sequential_search_recursive(unordered_list, 45)) # larger than largest element
print(unordered_sequential_search_recursive(unordered_list, 13)) # in the middle
```

Let's see if our implementation obeys the 3 Laws of Recursion
({prf:ref}`axiom_three_laws_of_recursion`).

We need to shrink our `container` list from $n$ all the way down, and at the
same time, keep track of our `index` to point to the correct index of the
`container`.

1. We have two base cases: - in `lines 5-6`, we first check if the list is
   empty, if it is, means we reached till the end of the list and have not found
   the `target` element, and hence return `-1`. - in `lines 8-9`, if the list's
   first element is the `target`, then return the `index` since we found it.
2. Has our recursive algorithm change its state and move towards our base case?
   Yes, because after each function call at `lines 12-14`, we slice our list by
   `[1:]`, which means we drop the first element, and move on to check if the
   "next" element is our `target`. Here, we also need to increment `index` by 1
   since we need to recover the index if we found the `target`.
3. This is a recursive algorithm because the function calls itself at
   `lines 12-14`.

```{admonition} Tip
:class: tip

Time to revisit this recursion for revision, especially understand how
recursion is stacking function calls and popping it later.

I also think converting from an iterative solution to recursive is easier
than just thinking of recursion straight. You just need to observe
what variables are changing in **states** in iterative,
and try to do the same to its recursive counterpart.
```

Using Python Tutor to visualize recursive calls
[here](https://pythontutor.com/render.html#code=def%20f%28container,%20target,%20index%3D0%29%3A%0A%20%20%20%20if%20len%28container%29%20%3D%3D%200%3A%20%20%23%20if%20not%20container%20is%20also%20fine%0A%20%20%20%20%20%20%20%20return%20-1%20%20%23%20not%20found%0A%0A%20%20%20%20if%20container%5B0%5D%20%3D%3D%20target%3A%20%20%23%20this%20is%20base%20case%0A%20%20%20%20%20%20%20%20return%20index%20%20%23%20found%0A%0A%20%20%20%20%23%20notice%20we%20increment%20index%20by%201%20to%20mean%20index%20%2B%3D%201%20in%20the%20iterative%20case%0A%20%20%20%20return%20f%28container%5B1%3A%5D,%20target,%20index%20%2B%201%29%20%20%23%20recursive%20case%0A%20%20%20%20%0Aunordered_list%20%3D%20%5B1,%202,%2032,%208,%2017,%2019,%2042,%2013,%200%5D%0Aprint%28f%28unordered_list,%2013%29%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false).

<iframe src="https://pythontutor.com/render.html#code=def%20f%28container,%20target,%20index%3D0%29%3A%0A%20%20%20%20if%20len%28container%29%20%3D%3D%200%3A%20%20%23%20if%20not%20container%20is%20also%20fine%0A%20%20%20%20%20%20%20%20return%20-1%20%20%23%20not%20found%0A%0A%20%20%20%20if%20container%5B0%5D%20%3D%3D%20target%3A%20%20%23%20this%20is%20base%20case%0A%20%20%20%20%20%20%20%20return%20index%20%20%23%20found%0A%0A%20%20%20%20%23%20notice%20we%20increment%20index%20by%201%20to%20mean%20index%20%2B%3D%201%20in%20the%20iterative%20case%0A%20%20%20%20return%20f%28container%5B1%3A%5D,%20target,%20index%20%2B%201%29%20%20%23%20recursive%20case%0A%20%20%20%20%0Aunordered_list%20%3D%20%5B1,%202,%2032,%208,%2017,%2019,%2042,%2013,%200%5D%0Aprint%28f%28unordered_list,%2013%29%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false" width="800" height="600"></iframe>

## Ordered Sequential Search

Previously, we showed how to perform sequential search on a list, which does not
assumes order.

We noticed that when the item is not in the list, the time complexity is
$\mathcal{O}(n)$, because we need to check every element in the list. This can
be alleviated if we assume that the list is ordered, and we can stop searching
when we reach an element that is greater than the element we are searching for.

For now, we will assume the list contains a list of integers, but this can be
generalized to other data types through mapping. For example, we can map the
alphabet to a list of integers, and then perform ordered sequential search on
the list of integers.

### Algorithm (Iterative)

```{prf:algorithm} Basic Ordered Linear Search Algorithm (Iterative)
:label: basic_ordered_linear_search_iterative

Given an ordered list $L$ of $n$ elements with values or records $L_0, L_1, ..., L_{n-1}$
such that $L_0 \leq L_1 \leq ... \leq L_{n-1}$, and target value $T$, the following subroutine uses ordered linear search to find the index of the target $T$ in $L$.

1. Set $i$ to 0.
2. If $L_i = T$, the search terminates successfully; return $i$. Else, go to step 3.
3. If $L_i > T$, the search terminates unsuccessfully; return $-1$.
```

### Implementation (Iterative)

```{code-cell} ipython3
def ordered_sequential_search(container: Iterable[T], target: T) -> Tuple[bool, int]:
    """Sequential search for ordered container."""
    is_found = False  # a flag to indicate so your return is more meaningful
    index = 0
    for item in container:
        if item == target:
            is_found = True
            return is_found, index
        index += 1
        if item > target:
            return is_found, -1
    # do not forget this if not if target > largest element in container, this case is not covered
    return is_found, -1
```

The reason for not using `enumerate` to get the index of a number in a list when
iterating is to minimize the usage of in-built functions.

```{code-cell} ipython3
ordered_list = [0, 1, 2, 8, 13, 17, 19, 32, 42]
print(ordered_sequential_search(ordered_list, -1)) # smaller than smallest element
print(ordered_sequential_search(ordered_list, 45)) # larger than largest element
print(ordered_sequential_search(ordered_list, 13)) # in the middle
```

#### Time Complexity

Note that for ordered sequential search, the time complexity does not change for
the case when the item is in the list.

However, for the case when the item is not in the list, we have our best case
scenario to be $\mathcal{O}(1)$, because upon checking our first element, and if
the first element is already greater than the element we are searching for, then
we can stop searching and return `False`.

For the worst case scenario, it is still $\mathcal{O}(n)$ since we have to check
every element in the list.

But, for the average case, it is now $\mathcal{O}(\frac{n}{2})$, because we can
stop searching when we reach an element that is greater than the element we are
searching for.

```{list-table} Time Complexity of Ordered Sequential Search
:header-rows: 1
:name: ordered_sequential_search_time_complexity

* - Case
  - Worst Case
  - Average Case
  - Best Case
* - Element is in the list
  - $\mathcal{O}(n)$
  - $\mathcal{O}(\frac{n}{2})$
  - $\mathcal{O}(1)$
* - Element is not in the list
  - $\mathcal{O}(n)$
  - $\mathcal{O}(\frac{n}{2})$
  - $\mathcal{O}(1)$
```

#### Space Complexity

Similarly, the space complexity is still $\mathcal{O}(1)$.

## Further Readings

- [GeeksforGeeks Linear Search](https://www.geeksforgeeks.org/linear-search/)
- [Runestone Academy Sequential Search](https://runestone.academy/ns/books/published/pythonds/SortSearch/TheSequentialSearch.html)
- [Wikipedia Linear Search](https://en.wikipedia.org/wiki/Linear_search)
- [Stack Overflow Recursive Linear Search](https://stackoverflow.com/questions/4295608/recursive-linear-search-returns-list-index)
- [Ozaner Sequential Search](https://ozaner.github.io/sequential-search/)
