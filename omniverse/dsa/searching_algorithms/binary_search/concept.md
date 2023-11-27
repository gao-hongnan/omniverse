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
    from omnivault.utils.visualization.tabbed_svg_viewer import create_tabbed_svg_viewer
    from omnivault.dsa.searching_algorithms.context import SearchContext
    from omnivault.dsa.searching_algorithms.strategies import IterativeBinarySearchExactMatch, RecursiveBinarySearchExactMatch
else:
    raise ImportError("Root directory not found.")

T = TypeVar("T", str, int, float)  # T should be of type int, float or str
```

## Introduction

**[Binary Search](https://en.wikipedia.org/wiki/Binary_search_algorithm)** is an
exceptionally efficient
**[search algorithm](https://en.wikipedia.org/wiki/Search_algorithm)** used
ubiquitously in computer science. At its heart, binary search is a
**[divide-and-conquer](https://en.wikipedia.org/wiki/Divide-and-conquer_algorithm)**
strategy that halves the search space with each iteration, providing a search
efficiency that is
**[logarithmic](https://en.wikipedia.org/wiki/Logarithmic_time_complexity)** in
the size of the dataset.

In binary search, the dataset must be in a
**[sorted](https://en.wikipedia.org/wiki/Sorting_algorithm)** format, as the
algorithm operates by repeatedly dividing the search interval in half. It starts
with an interval covering the whole array, and if the value of the search key is
less than the middle element of the interval, it narrows the interval to the
lower half. Otherwise, it narrows it to the upper half. Repeatedly, this process
continues until the value is found or the interval is empty.

Binary search can be used not only to find whether a specific value exists in a
sorted array but also to find the position of an insert for a new value to keep
the array sorted. Because of this and its impressive efficiency, it underpins
many fundamental computer science algorithms and data structures, such as
**[binary search trees](https://en.wikipedia.org/wiki/Binary_search_tree)** and
**[B-trees](https://en.wikipedia.org/wiki/B-tree)**.

Applications of binary search in computer science are many and varied:

-   **Search Algorithms**: Binary search is one of the fundamental search
    algorithms, applied when dealing with sorted lists or arrays to find the
    existence and position of a particular value.

-   **Database and File-System Indexing**: Many
    **[database indexing](https://en.wikipedia.org/wiki/Database_index)**
    techniques use structures like B-trees and binary search trees, which at
    their core use a binary search approach for efficiently locating records.

-   **Numerical Computation**: Binary search is used in
    **[numeric computations](https://en.wikipedia.org/wiki/Numerical_analysis)**
    for tasks like
    **[root-finding](https://en.wikipedia.org/wiki/Root-finding_algorithm)** or
    for implementing
    **[numerical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization)**
    algorithms.

-   **Version Control Systems**: In
    **[software version control systems](https://en.wikipedia.org/wiki/Version_control)**,
    binary search is used in "bisecting", which is a method to find which commit
    introduced a particular bug.

-   **Memory Management**: In
    **[memory management](https://en.wikipedia.org/wiki/Memory_management)**
    tasks, binary search is often used to search in address spaces and page
    tables in a fast and efficient way.

## Intuition

Assume we have a sorted list of numbers, denoted $\mathcal{A}$, of size $N$ and
we want to find a particular number in the list, called the target $\tau$.

Then, we know from **[Linear Search](../linear_search/concept.md)** that we
still have to traverse each element until we find the target $\tau$, or reach
the end of the list. This yields a time complexity of $\mathcal{O}(N)$, with $N$
being the size of the list.

However, if the list is **sorted**, we can leverage a strategy known as
**[divide and conquer](https://en.wikipedia.org/wiki/Divide-and-conquer_algorithm)**
to improve our search speed significantly. This is where
**[Binary Search](https://en.wikipedia.org/wiki/Binary_search_algorithm)** comes
into play.

The principle behind binary search is that we continually **halve** our search
space, which we denote as $\mathcal{S}$ (to be conceptualized later). We begin
with the whole list, then determine whether our target value is in the first or
the second half by comparing it with the middle element. If our target is less
than the middle element, it must lie in the first half of the list, so we can
immediately discard the second half. Conversely, if our target is larger than
the middle element, it must reside in the second half, and we discard the first
half.

This process continues until we either find our target or our search space
$\mathcal{S}$ is exhausted, i.e., there's no element left to check. Each
iteration effectively **halves** the problem size, resulting in a logarithmic
time complexity of $\mathcal{O}(\log N)$. This makes binary search markedly more
efficient than linear search, particularly in large, sorted datasets.

A neat connection exists between binary search and the concept of a
**[Binary Search Tree (BST)](https://en.wikipedia.org/wiki/Binary_search_tree)**.
In a BST, each node has a value larger than all the values in its left subtree
and smaller than all the values in its right subtree. This is akin to how binary
search operates, dividing the list into two halves, with one half always less
than and the other always greater than the middle. Indeed, if you perform an
inorder traversal of a BST (left, root, right), you retrieve the elements in
sorted order, just as you would have them arranged for a binary search. So the
middle number of the sorted list is the "root" of the BST, and the left and
right halves of the list are the left and right subtrees of the root. This is a
great way to visualize the process of binary search and how it operates.

In what follows, we conceptualize the intuition via examples and visualizations,
and lastly through defining the algorithm rigorously.

## Example

Consider the following sorted list of numbers $\mathcal{A}$:

$$
\mathcal{A} = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
$$

and we seek to find if the number $23$ exists in the list.

### Visualization

We create a tabbed SVG viewer to visualize the binary search algorithm in the
explanation below.

```{code-cell} ipython3
# List of SVG image paths
svg_images = [
    "./assets/binary-search-0.svg",
    "./assets/binary-search-1.svg",
    "./assets/binary-search-2.svg",
    "./assets/binary-search-3.svg",
]

tab_titles = ["Step 1", "Step 2", "Step 3", "Step 4"]

tab = create_tabbed_svg_viewer(svg_images, tab_titles)
display(tab)
```

### Explanation

1. We start with the entire list $\mathcal{A}$, which is our search space
   $\mathcal{S}$.

    In particular, the initial search space can be solely determined by two
    pointers (indexes) $\ell$ and $r$, where $\ell$ is the leftmost index and
    $r$ is the rightmost index of the array. This means that we are searching
    for the target value $\tau$ in the range $[\ell, r]$.

    - $\ell = 0$ since indexing in python starts from 0,
    - $r = N - 1 = 10 - 1 = 9$

2. We compute the middle index $m$ of the search space $\mathcal{S}$, which is
   the index of the middle element of the array.

    - In this case, $m = 4$ because our mid strategy uses the
      [**floor function**](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions):

        $$
        \begin{aligned}
        m &= \left \lfloor \frac{0 + 9}{2} \right \rfloor \\
          &= \left \lfloor \frac{9}{2} \right \rfloor \\
          &= 4.
        \end{aligned}
        $$

    - Now one may ask why do we use the floor function and not the ceiling
      function? It really depends on how you update the boundary indexes $\ell$
      and $r$. See a
      [discussion here to understand in details](https://stackoverflow.com/questions/27655955/why-does-binary-search-algorithm-use-floor-and-not-ceiling-not-in-an-half-open).

3. At this junction we are at step 2 in the visualization above. The middle
   index is our pivot point that **_halves_** the search space $\mathcal{S}$
   into two subspaces, namely the left and right subspaces.

    We now compare the middle element $\mathcal{A}_m$ with the target value
    $\tau$. We will have 3 cases:

    - $\mathcal{A}_m = \tau$: We found the target value $\tau$ at index $m$ in
      the array $\mathcal{A}$, so we return $m$.
    - $\mathcal{A}_m < \tau$: The target value $\tau$ is greater than the middle
      element $\mathcal{A}_m$, so we discard the left half of the array
      $\mathcal{A}$ and search the right half.
    - $\mathcal{A}_m > \tau$: The target value $\tau$ is less than the middle
      element $\mathcal{A}_m$, so we discard the right half of the array
      $\mathcal{A}$ and search the left half.

    - And in our case, our value at index $m=4$ is $\mathcal{A}_m=16$ and
      $\mathcal{A}_m < \tau = 23$, so we discard the left half of the array
      $\mathcal{A}$ and search the right half. When we discard the left half, we
      want to also discard the middle element $\mathcal{A}_m$ because we know
      that it is not the target value $\tau$.

    - So we update the left index $\ell$ to be

        $$\ell \leftarrow m + 1 = 4 + 1 = 5.$$

    - There is no need to update the right index $r$ because we know that the
      target value $\tau$ is in the right half of the array $\mathcal{A}$, if it
      exists.

4. By just shifting the boundary index(es), we have also reduced our search
   space $\mathcal{S}$.

    - In this case, our search space $\mathcal{S}$ is now the right half of the
      array $\mathcal{A}$, which is $[\ell, r] = [5, 9]$.

    - We now compute the middle index $m$ of the search space $\mathcal{S}$,
      which is the index of the middle element of the array.

    - In this case, $m = 7$ because our mid strategy uses the
      [**floor function**](https://en.wikipedia.org/wiki/Floor_and_ceiling_functions):

        $$
        \begin{aligned}
        m &= \left \lfloor \frac{5 + 9}{2} \right \rfloor \\
          &= \left \lfloor \frac{14}{2} \right \rfloor \\
          &= 7.
        \end{aligned}
        $$

    - And in our case, our value at index $m=7$ is $\mathcal{A}_m=56$ and
      $\mathcal{A}_m > \tau = 23$, so we discard the right half of the array
      $\mathcal{A}$ and search the left half. When we discard the right half, we
      want to also discard the middle element $\mathcal{A}_m$ because we know
      that it is not the target value $\tau$.

    - So we update the right index $r$ to be

        $$r \leftarrow m - 1 = 7 - 1 = 6.$$

    - There is no need to update the left index $\ell$ because we know that the
      target value $\tau$ is in the left half of the array $\mathcal{A}$, if it
      exists.

5. We continue searching until we find the target value $\tau$ or the search
   space $\mathcal{S}$ is exhausted, i.e., there's no element left to check.

    - In this case, our search space $\mathcal{S}$ is now the left half of the
      array $\mathcal{A}$, which is $[\ell, r] = [5, 6]$.

    - We now compute the middle index $m$ of the search space $\mathcal{S}$,
      which is the index of the middle element of the array.

    - In this case, $m = 5$:

        $$
        \begin{aligned}
        m &= \left \lfloor \frac{5 + 6}{2} \right \rfloor \\
          &= \left \lfloor \frac{11}{2} \right \rfloor \\
          &= 5.
        \end{aligned}
        $$

    - And in our case, our value at index $m=5$ is $\mathcal{A}_m=23$ and
      $\mathcal{A}_m = \tau = 23$, so we return $m=5$.

    - We found the target value $\tau$ at index $m$ in the array $\mathcal{A}$,
      so we return $m$.

Now, let's visualize this. Here's a table that shows the state of the variables
at each step:

```{list-table} Binary Search State Table
:header-rows: 1
:name: binary-search-state-table-1

* - Step
  - $\ell$
  - $r$
  - $m$
  - $\mathcal{A}_m$
  - Target $\tau$
  - Action
* - 1
  - 0
  - 9
  - 4
  - 16
  - 23
  - $\mathcal{A}_m < \tau$, so $\ell = m + 1$
* - 2
  - 5
  - 9
  - 7
  - 56
  - 23
  - $\mathcal{A}_m > \tau$, so $r = m - 1$
* - 3
  - 5
  - 6
  - 5
  - 23
  - 23
  - $\mathcal{A}_m = \tau$, so return $m$
```

At the start, we're looking at the full list and the search space $\mathcal{S}$
is defined by the initial boundary indexes $\ell = 0$ and $r=9$. At each step,
we adjust $\ell$ or $r$ to reduce the search space $\mathcal{S}$ and close in on
the target number based on whether it's greater than or less than the middle
element. Finally, when we find the target number, we return the index.

````{prf:remark} What if we want to find 39?
:label: binary-search-remark-what-if-we-want-to-find-39

Something we did not touch on is the terminating condition. If we were to find
a non-existing target $\tau = 39$, we would eventually reach a point where
$\ell > r$. Let's continue from the previous example.

Recall we found our index at $\ell = 5$ and $r = 6$ to host the target $\tau=23$.
Now assume our target is $\tau = 39$. We would have the following:

```{list-table} Binary Search State Table
:header-rows: 1
:name: binary-search-state-table-2

* - Step
  - $\ell$
  - $r$
  - $m$
  - $\mathcal{A}_m$
  - Target $\tau$
  - Action
  - Search Space $\mathcal{S}$
* - 1
  - 0
  - 9
  - 4
  - 16
  - 39
  - $\mathcal{A}_m < \tau$, so $\ell = m + 1$
  - $\mathcal{S} = [0, 9]$
* - 2
  - 5
  - 9
  - 7
  - 56
  - 39
  - $\mathcal{A}_m > \tau$, so $r = m - 1$
  - $\mathcal{S} = [5, 9]$
* - 3
  - 5
  - 6
  - 5
  - 23
  - 39
  - $\mathcal{A}_m < \tau$, so $\ell = m + 1$
  - $\mathcal{S} = [5, 6]$
* - 4
  - 6
  - 6
  - 6
  - 38
  - 39
  - $\mathcal{A}_m < \tau$, so $\ell = m + 1$
  - $\mathcal{S} = [6, 6]$
* - 5
  - 7
  - 6
  - ?
  - ?
  - 39
  - Terminating condition since $\ell > r$
  - ?
```

We have seen that the seach space $\mathcal{S}$ is reduced to $[6, 6]$ and we
cannot reduce it further. At step 4, where the search space $\mathcal{S}$ is
reduce to $[6, 6]$, we have how many elements in the search space? Only 1!
So if that element is not the target $\tau$, then we know that the target $\tau$
does not exist in the array $\mathcal{A}$. This is the terminating condition.
````

## Assumptions and Constraints

### Assumptions

1. **Sorted Array**: We assume the input array is already sorted in ascending
   order, and binary search is not suitable for unsorted arrays.
2. **Unique Elements**: As mentioned, the array consists of unique elements.
   This implies that the return index for any valid target is unique.
3. **Deterministic**: The input array does not change during the course of the
   algorithm, and there are no external factors affecting its content.

### Constraints

The constaints/assumptions are made below, we follow the same set of assumptions
from [LeetCode's Binary Search](https://leetcode.com/problems/binary-search/).

-   The array must be sorted in ascending order.
-   The array consists of **unique** elements of type `int` with base 10.
-   **Array Length**: The length of the array `nums` will always be in the range
    `1 <= len(nums) <= 10^4`.
-   **Array Content**: Every element in the array, as well as the target value,
    is guaranteed to be in the range `-10^4 <= nums[i], target <= 10^4`.
-   **Return Value**: The function should return the index of the `target` if it
    exists in `nums` and `-1` if it doesn't.

### Why Left + Right // 2 May Cause Overflow?

In programming, when two integers are added, and the result goes beyond the
maximum value that can be stored in that integer type, it's referred to as an
integer overflow.

Consider a 32-bit signed integer in most languages. The largest value it can
hold is $2^{31} - 1$, because one bit is reserved for the sign. When you add two
large numbers, their sum might exceed this value, causing an overflow.

Now, let's explore the mid calculation:

$$m = \left\lfloor \frac{\ell + r}{2} \right\rfloor$$

If $\ell$ and $r$ are very large (close to the maximum value for the integer
type), their sum can overflow, even though their average (mid value) might still
be within the valid range.

Example: Let's take a hypothetical situation where the maximum value an integer
can store is $15$ (in reality, this is much larger, but we're simplifying for
the sake of illustration).

Suppose $\ell = 7$ and $r = 14$. When you add these together, you get
$\ell + r =
21$, which exceeds our hypothetical maximum value of $15$. Hence, an
overflow occurs.

However, if we use the safer approach:

$$m = l + \left\lfloor \frac{r - l}{2} \right\rfloor$$

With our values, we'd get:

$$m = 7 + \left\lfloor \frac{14 - 7}{2} \right\rfloor = 7 + 3 = 10$$

This approach avoids overflow because $\ell$ and
$\left\lfloor \frac{r - l}{2}
\right\rfloor$ will always be valid integers and
their sum will be too.

In real-world applications, this is relevant for very large arrays when using
32-bit integers. Using 64-bit integers (like `long long` in C++ or `int64` in
some other languages) delays the point where overflow might occur, but the
principle remains. The alternate method for calculating mid is a safer approach
that avoids the potential overflow issue altogether.

## Algorithm (Iterative + Exact Match)

The algorithm discussed above is recognized as the most generic and precisely
fitting version according to LeetCode. For more details, you can refer to their
explanation under the "Template I" section of binary search, available at
[this link](https://leetcode.com/explore/learn/card/binary-search/125/template-i).

We plan to explore additional variants of this algorithm in subsequent
discussions.

In what follows we formalize the algorithm in pseudocode and mathematical
notation.

### Pseudocode

````{prf:algorithm} Pseudocode
:label: binary-search-pseudocode-iterative

```
Algorithm: binary_search_iterative(A, t)

    Input:  A = [a_0, a_1, ..., a_{N-1}] (sorted list of elements),
            t (target value)
    Output: Index of t in A or -1 if not found

    1: Set l = 0 and r = N - 1    // (Initialization)

    2: while l <= r (Iterative Procedure)
        3:    m = l + (r - l) // 2    // (Calculate mid-point)
        4:    if A[m] == t then
        5:        return m        //(Target found)
        6:    elif A[m] < t then
        7:        l = m + 1       // (Discard left half)
        8:    else
        9:        r = m - 1       //(Discard right half)
       10: end while

    11: return -1 (Target not found)
```
````

```{prf:algorithm} Mathematical Representation (Iterative)
:label: binary-search-mathematical-representation-iterative

Given a sorted list $\mathcal{A}$ of $N$ elements $\mathcal{A}_0,
\mathcal{A}_1, \ldots, \mathcal{A}_{N-1}$, and a target value $\tau$, the
objective is to locate $\tau$'s index in $\mathcal{A}$ using an _iterative
binary search_.

The search space $\mathcal{S}$, initially the entire array $\mathcal{A}$, is
dynamically reduced in each iteration. It can be defined as $\mathcal{S} =
\mathcal{A}[l:r+1]$ at any given iteration, where $\ell$ and $r$ are the current
lower and upper bounds of the search space.

Binary search proceeds as follows:

1. **Initialization**:
   - Set $\ell = 0$ and $r = N - 1$.

2. **Iterative Procedure**:
   - While $\ell \leq r$, calculate the mid-point $m = l + (r - l) // 2$.
   - If $\mathcal{A}_m = \tau$, return $m$.
   - If $\mathcal{A}_m < \tau$, update $\ell = m + 1$ (discard left half).
   - Otherwise, update $r = m - 1$ (discard right half).

3. **Termination**:
   - If $\ell > r$, $\tau$ is not in $\mathcal{A}$. Return $-1$.
```

```{prf:algorithm} Pseudocode
:label: binary-search-pseudocode1

Algorithm: `binarySearch(nums, target)`

Input: `nums` (sorted array of integers), `target` (integer to search for)

Output: `index` (index of the target in the array, or -1 if not found)

Binary search operates on a sorted array. It starts by comparing the target
value with the middle element of the array. If the target equals this middle
element, the position of this element in the array is returned. If the target is
less than the middle element, the search continues in the lower half of the
array. If the target is greater than the middle element, the search continues in
the upper half of the array. By doing this, the algorithm eliminates the half of
the array in which the target cannot lie in each iteration.

Given an array $\mathcal{A}$ of $N$ elements with values or records
$A_0, A_1, \dots, \mathcal{A}_{n-1}$, sorted such that
$A_0 \leq A_1 \leq \dots \leq \mathcal{A}_{n-1}$, and target value $T$, the following
subroutine uses binary search to find the index of $T$ in $\mathcal{A}$.

1. Set $L$ to 0 and $R$ to $n-1$ (the indices of the leftmost and rightmost
   elements of the array).
2. If $L > R$, the search terminates as unsuccessful, return -1. In other words,
   here is while $L \leq R$.
3. Set $m$ to the floor of $(L + R) / 2$ (the index of the middle element of the
   array). In other words, $m = \lfloor \frac{L + R}{2} \rfloor$. In the event
   where the length of the array $\mathcal{A}$ is odd, then $m$ will be the index
   of the
   middle element (exactly middle). In the event where the length of the array
   $\mathcal{A}$ is even, then $m$ will be the index of the leftmost element of
   the right
   half. In practice, to avoid overflow, use
   $m = L + \lfloor \frac{R - L}{2} \rfloor$.
4. If $\mathcal{A}_m < T$, set $L$ to $m + 1$ and go to step 2.
5. If $\mathcal{A}_m > T$, set $R$ to $m - 1$ and go to step 2.
6. If $\mathcal{A}_m = T$, the search is done; return $m$.
```

Note that we can use the ceiling function in step 3. This may change the result
if the target value appears more than once in the array.

### Mathematical Representation (Recursive)

The binary search algorithm is a divide-and-conquer algorithm that halves the
search space at each step. It can be formally described using the notation for
sequences, with $a$ representing the sequence (i.e., the array), $N$
representing the length of the sequence, $i$ and $j$ representing the low and
high indices of the search space, and $m$ representing the midpoint.

Here's a rigorous mathematical version for binary search:

```{prf:algorithm} Mathematical Representation
:label: binary-search-mathematical-representation

Define the function $f: (\mathcal{A}, \ell, r, T) \rightarrow k$ as follows:

$$
f(\mathcal{A}, \ell, r, T) =
    \begin{cases}
      -1 & \text{if } r < \ell \\
      f(A, \ell, m - 1, T) & \text{if } \mathcal{A}_m > T \\
      f(A, m + 1, r, T) & \text{if } \mathcal{A}_m < T \\
      m & \text{if } \mathcal{A}_m = T
    \end{cases}
$$

where

-   $\mathcal{A}$ is a sequence of sorted elements (the array)
-   $\ell$ and $r$ are the leftmost and rightmost indices of the search space,
    respectively
-   $m = \ell + \lfloor \frac{r - \ell}{2} \rfloor$ is the index of the middle
    element of the search space
    - $m$ is a function of $\ell$ and $r$.
-   $\mathcal{A}_m$ is the element at index $m$ in the sequence $\mathcal{A}$
-   $T$ is the target value

This function operates on the half of the search space where the target value
may exist (as determined by comparing the target value $T$ to the middle value
of the search space). It divides the size of the problem in half at each step,
which is what makes it a divide-and-conquer algorithm.
```

This representation of the binary search algorithm emphasizes that it works by
reducing the size of the problem at each step. If the target value is less than
the middle value of the array, then the algorithm searches the left half of the
array, and if the target value is greater than the middle value, then it
searches the right half. If the middle value is equal to the target, then the
search is successful, and the index of the middle value is returned. If the size
of the array is 0, then the search is unsuccessful, and -1 is returned.

### Explanation

1. The search space is initially defined as the entire array, from index `L = 0`
   to `R = len(array) - 1`.

2. The middle point `M` of the search space is defined as `M = (L + R) // 2`.

3. We compare the value at the midpoint `M` with the target value. There are
   three possible outcomes:

    - If `array[M] == target`, then we have found the target at index `M` and
      the search is complete.

    - If `array[M] > target`, then the target must, if it exists, be in the left
      half of the array, specifically in the range from `L` to `M - 1`, because
      all values at and after `M` are greater than the target. We update
      `R = M - 1`.

    - If `array[M] < target`, then the target must, if it exists, be in the
      right half of the array, specifically in the range from `M + 1` to `R`,
      because all values before `M` are less than the target. We update
      `L = M + 1`.

We repeat the process until we find the target or the search space is empty
(`L > R`).

### Correctness

For the proof of correctness, assume that the binary search algorithm does not
correctly find the target. This means that either the algorithm did not return
the target when it was present in the array or returned an incorrect value when
the target was not in the array.

However, each step of the binary search algorithm precisely follows the sorted
property of the array. If the target is less than the value at the midpoint, we
know that the target, if it exists, must be in the left half. Similarly, if the
target is greater than the value at the midpoint, it must be in the right half
if it exists. Therefore, the algorithm correctly narrows down the search space
at each step based on the sorted property of the array.

This contradicts our assumption that the binary search algorithm does not
correctly find the target, and thus proves that the binary search algorithm is
correct.

More formally:

I'm still assuming the list indices start from 1 for the sake of simplicity.
However, in most programming languages (including Python), they start from 0.

```{prf:proof}
We want to prove that the binary search algorithm correctly finds a target value
in a sorted list or correctly reports that the target is not in the list.

Let's denote the proposition $P(n)$: "For all sorted lists of length $N$, binary
search either finds the target or correctly reports it is not in the list."

**Base Case:**

For $n=1$, binary search correctly reports whether the single element is equal
to the target. So, $P(1)$ is true.

**Inductive Step:**

We assume that $P(k)$ holds for all $k$ such that $1 \leq k < n$ (Inductive
Hypothesis). We want to show $P(n)$ is true.

We can write down the steps of binary search for a sorted list of length $N$ as
follows:

1. Compute $m = \lfloor \frac{n}{2} \rfloor$ (mid-point).

2. If the target equals the $m^{th}$ element of the list, then we have found the
   target.

3. If the target is less than the $m^{th}$ element, then we recursively search
   the left subarray of length $m-1$.

4. If the target is greater than the $m^{th}$ element, then we recursively
   search the right subarray of length $n-m$.

Both the left subarray in step 3 and the right subarray in step 4 have length
less than $N$, so by our inductive hypothesis, our algorithm works correctly in
these cases. Therefore, we conclude that $P(n)$ is true.

This completes our induction and the proof that the binary search algorithm is
correct.

**Q.E.D.**
```

## Test Cases

1. **Standard Test Case**

    ```python
    container = [1, 3, 5, 7, 9]
    target = 5
    # Expected Output: 2 (since container[2] = 5)
    ```

2. **Target at Start**

    ```python
    container = [1, 3, 5, 7, 9]
    target = 1
    # Expected Output: 0 (since container[0] = 1)
    ```

3. **Target at End**

    ```python
    container = [1, 3, 5, 7, 9]
    target = 9
    # Expected Output: 4 (since container[4] = 9)
    ```

4. **Target Not in List**

    ```python
    container = [1, 3, 5, 7, 9]
    target = 4
    # Expected Output: -1 (since 4 is not in the container)
    ```

5. **Large Numbers in Array**

    ```python
    container = [10000, 20000, 30000, 40000]
    target = 30000
    # Expected Output: 2 (since container[2] = 30000)
    ```

## Edge Cases

1. **Empty List**

    Although the problem constraints state the minimum size of the array is 1,
    it's always a good idea to consider what would happen with an empty array.

    ```python
    container = []
    target = 1
    # Expected Output: -1 (since the array is empty)
    ```

2. **Single Element Array**

    ```python
    container = [3]
    target = 3
    # Expected Output: 0 (since container[0] = 3)
    ```

3. **Single Element Array, Target Not Present**

    ```python
    container = [3]
    target = 4
    # Expected Output: -1 (since 4 is not in the container)
    ```

4. **Target Value is Minimum Possible Value**

    ```python
    container = [-10000, -9999, -9998]
    target = -10000
    # Expected Output: 0 (since container[0] = -10000)
    ```

5. **Target Value is Maximum Possible Value**

    ```python
    container = [9998, 9999, 10000]
    target = 10000
    # Expected Output: 2 (since container[2] = 10000)
    ```

## Solution (Iterative - Find the Exact Value)

### Intuition

See the intuition above.

### Visualization

See the visualization above.

### Algorithm

See the algorithm above.

### Claim

...

### Proof

...

### Implementation

See my dsa repo.

Think of the following:

-   Terminating condition: the search space is empty.

    -   We don't use `len(nums) == 0` because this question usually want us to
        have auxiliary space complexity of $\mathcal{O}(1)$. And thus `nums` may
        not be mutated directly here (retrospectively).
    -   We generally use left, right pointers and if `left > right` then the
        search space is empty. Why?

        -   The `left` pointer moves right (`left = mid + 1`), and the `right`
            pointer moves left (`right = mid - 1`), so if they cross, it means
            we've checked all possible elements.
        -   The `left <= right` condition ensures that when `left` and `right`
            are pointing to the same element (i.e., the search space has only
            one item left), we still check this last element.
        -   When `left > right`, there are no elements left to check in the
            search space, and the algorithm can terminate.

        This condition works for the binary search paradigm where we exclude the
        middle element at each step after checking it. There are other paradigms
        where the `left` and `right` pointers do not exclude the middle element
        after checking it, and the terminating condition for those may be
        `left < right`. However, for the classic binary search, the
        `left > right` condition is used to indicate an empty search space.

-   Distinguishing Syntax:
    -   Initial Condition: `left = 0`, `right = length-1`
    -   Termination: `left > right`
    -   Searching Left: `right = mid-1`
    -   Searching Right: `left = mid+1`

### Tests

```{code-cell} ipython3
strategy = IterativeBinarySearchExactMatch()
context = SearchContext(strategy)
result = context.execute_search([2, 5, 8, 12, 16, 23, 38, 56, 72, 91], 23)
assert result == 5
```

### Time Complexity

Let's consider a sorted list `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` and we want to
find the target value `9`.

We see that for an array of $10$ elements, I purposely choose `9` as the target
value, which is also the last element. In a typical sequential search, it will
take $10$ iterations to find the target value `9`, and therefore the time
complexity is $O(n)$.

However, we took a total of $4$ iterations to find the target value `9` in the
array for a binary search. How did we get to this result?

To analyze the binary search algorithm, we need to recall that each comparison
eliminates about half of the remaining items from consideration. What is the
maximum number of comparisons this algorithm will require to check the entire
list {cite}`pythonds3`?

Let's say we have an array of $N$ elements.

-   The first comparison eliminates about half of the remaining items from
    consideration. Thus, after the first comparison, we have about $\frac{n}{2}$
    elements left.
-   The second comparison eliminates about half of the remaining items from
    consideration. Thus, after the second comparison, we have about
    $\frac{n}{2^2}$ elements left.
-   The third comparison eliminates about half of the remaining items from
    consideration. Thus, after the third comparison, we have about
    $\frac{n}{2^3}$ elements left.
-   The $k$-th comparison eliminates about half of the remaining items from
    consideration. Thus, after the $k$-th comparison, we have about
    $\frac{n}{2^k}$ elements left.

Note that we say approximately/about because the number of elements left after
the $i$-th comparison is not always "half". Using back the same example
previously, if we have an array of $10$ elements, and we want to find $9$, then
after the first comparison, we discard the first half of the array,
`[0, 1, 2, 3, 4]`, and we have `[5, 6, 7, 8, 9]` left. This is indeed
$\frac{n}{2} = \frac{10}{2} = 5$ elements left. However, after the second
comparison, we discard the first half of the array, `[5, 6]`, and we have
`[7, 8, 9]` left. This is now $3$ elements left, which is not exactly half of
the remaining items from consideration since `[5, 6, 7, 8, 9]` is an array of
odd length.

```{list-table} Number of items left after $k$-th comparison
:header-rows: 1
:name: items_left_binary_search

* - Comparisons
  - Approximate number of items left
* - $i = 1$
  - $\frac{n}{2}$
* - $i = 2$
  - $\frac{n}{2^2}$
* - $i = 3$
  - $\frac{n}{2^3}$
* - $\ldots$
  - $\ldots$
* - $i = k$
  - $\frac{n}{2^k}$
```

If we split the container/list enough times, eventually we will have only one
item left {cite}`pythonds3`. The last item is either the target value or it is
not.

So our stopping condition is when the number of items left is $1$. Consequently,
we solve for $k$ in the equation $\frac{n}{2^k} = 1$:

$$
\begin{align*}
\frac{n}{2^k} &\iff 1 \\
2^k &\iff n \\
\log_2 2^k &\iff \log_2 n \\
k &\iff \log_2 n
\end{align*}
$$

This means that the maximum number of comparisons is $\log_2 n$. In other words,
after approximately $\log_2 n$ comparisons, we can reduce the size of the list
to $1$ and since we will not be able to divide the list any further, we can
conclude that the target value is either the last element or it is not in the
list, ending the search.

In terms of big-O, we say that the binary search algorithm takes
$\mathcal{O}(\log_2 n)$ time to search for an item in a list of $N$ items, which
means the maximum number of comparisons is in a logarithmic relationship to the
number of items in the list {cite}`pythonds3`.

The time complexity table is listed below, the best case is $\mathcal{O}(1)$ for
the same reason as the sequential search algorithm, where the `target` element
is in the middle, and we just need to make one comparison. For the worst case,
the element is either in the first or last index, or it is not in the list at
all. In this case, we need to make $\log_2 n$ comparisons.

```{list-table} Best, Worst, and Average Case Analysis of Binary Search
:header-rows: 1
:name: binary_search_time_complexity_iterative

* - Case
  - Worst Case
  - Average Case
  - Best Case
* - Element is in the list
  - $\mathcal{O}(\log_2 n)$
  - $\mathcal{O}(\log_2 n)$[^average_case]
  - $\mathcal{O}(1)$
* - Element is not in the list
  - $\mathcal{O}(\log_2 n)$
  - $\mathcal{O}(\log_2 n)$
  - $\mathcal{O}(\log_2 n)$
```

### Space Complexity

We break down the space complexity of binary search into three parts: input,
auxiliary and total space complexity.

#### Input Space Complexity

Input space complexity is the space used to store the input to the problem. For
a binary search, the input is the array or list we are searching through, and a
target value. We do not usually consider the space taken by the inputs when
analyzing the space complexity of an algorithm, unless the algorithm modifies
the input in place.

But if we do want to consider the input space, that is the space taken by the
inputs, which is $\mathcal{O}(n) + \mathcal{O}(1) = \mathcal{O}(n)$, where $N$
is the length of the array and $\mathcal{O}(1)$ is the space taken by the target
value.

#### Auxiliary Space Complexity

Auxiliary space complexity is the extra space or the temporary space used by an
algorithm. In the case of binary search, we only need three variables to hold
the left, right and middle indices (`L`, `R`, `m`). These variables occupy
constant space, so the auxiliary space complexity is $\mathcal{O}(1)$.

For the iterative approach, only one stack frame is used, contributing to the
auxiliary space complexity of O(1). There is no extra allocation in each
iteration of the while loop in terms of variables or object instances.

#### Total Space Complexity

Total space complexity is the sum of input and auxiliary space complexities. For
binary search, if the input space complexity is not considered and the auxiliary
space complexity is $\mathcal{O}(1)$, the total space complexity of binary
search is $\mathcal{O}(1)$, else it is $\mathcal{O}(n)$.

To summarize, the binary search algorithm is very space-efficient as it only
requires constant auxiliary space to perform the search, and it does not modify
the input array or list.

## Solution (Recursive - Find the Exact Value)

### Intuition

...

### Visualization

...

### Algorithm

#### Pseudocode

...

#### Mathematical Representation

The binary search algorithm is a divide-and-conquer algorithm that halves the
search space at each step. It can be formally described using the notation for
sequences, with $a$ representing the sequence (i.e., the array), $N$
representing the length of the sequence, $i$ and $j$ representing the low and
high indices of the search space, and $m$ representing the midpoint.

Here's a rigorous mathematical version for binary search:

```{prf:algorithm} Mathematical Representation
:label: binary-search-mathematical-representation-algorithm-exact-match

Define the function $f: (\mathcal{A}, \ell, r, T) \rightarrow k$ as follows:

$$
f(\mathcal{A}, \ell, r, T) =
    \begin{cases}
      -1 & \text{if } r < \ell \\
      f(A, \ell, m - 1, T) & \text{if } \mathcal{A}_m > T \\
      f(A, m + 1, r, T) & \text{if } \mathcal{A}_m < T \\
      m & \text{if } \mathcal{A}_m = T
    \end{cases}
$$

where

-   $\mathcal{A}$ is a sequence of sorted elements (the array)
-   $\ell$ and $r$ are the leftmost and rightmost indices of the search space,
    respectively
-   $m = \ell + \lfloor \frac{r - \ell}{2} \rfloor$ is the index of the middle
    element of the search space
    - $m$ is a function of $\ell$ and $r$.
-   $\mathcal{A}_m$ is the element at index $m$ in the sequence $\mathcal{A}$
-   $T$ is the target value

This function operates on the half of the search space where the target value
may exist (as determined by comparing the target value $T$ to the middle value
of the search space). It divides the size of the problem in half at each step,
which is what makes it a divide-and-conquer algorithm.
```

This representation of the binary search algorithm emphasizes that it works by
reducing the size of the problem at each step. If the target value is less than
the middle value of the array, then the algorithm searches the left half of the
array, and if the target value is greater than the middle value, then it
searches the right half. If the middle value is equal to the target, then the
search is successful, and the index of the middle value is returned. If the size
of the array is 0, then the search is unsuccessful, and -1 is returned.

### Claim

...

### Proof

...

### Implementation

See dsa folder.

Using Python Tutor to visualize recursive calls
[here](https://pythontutor.com/render.html#code=import%20math%0Afrom%20typing%20import%20Iterable,%20TypeVar,%20Tuple%0A%0AT%20%3D%20TypeVar%28%22T%22,%20str,%20int,%20float%29%20%20%23%20T%20should%20be%20of%20type%20int,%20float%20or%20str%0A%0Adef%20binary_search_recursive%28%0A%20%20%20%20container%3A%20Iterable%5BT%5D,%20target%3A%20T,%20left_index%3A%20int,%20right_index%3A%20int%0A%29%20-%3E%20int%3A%0A%20%20%20%20%22%22%22Binary%20search%20recursive%20implementation.%22%22%22%0A%20%20%20%20mid_index%20%3D%20left_index%20%2B%20math.floor%28%28right_index%20-%20left_index%29%20/%202%29%0A%20%20%20%20if%20left_index%20%3C%3D%20right_index%3A%0A%20%20%20%20%20%20%20%20if%20container%5Bmid_index%5D%20%3D%3D%20target%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20return%20mid_index%20%20%23%20base%20case%201%0A%20%20%20%20%20%20%20%20elif%20container%5Bmid_index%5D%20%3C%20target%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20return%20binary_search_recursive%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20container,%20target,%20mid_index%20%2B%201,%20right_index%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%0A%20%20%20%20%20%20%20%20else%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20return%20binary_search_recursive%28container,%20target,%20left_index,%20mid_index%20-%201%29%0A%20%20%20%20else%3A%0A%20%20%20%20%20%20%20%20return%20-1%20%20%23%20base%20case%202%0A%20%20%20%20%0Aordered_list%20%3D%20%5B0,%201,%202,%208,%2013,%2017,%2019,%2032,%2042%5D%0Aleft_index%20%3D%200%0Aright_index%20%3D%20len%28ordered_list%29%20-%201%0Aprint%28binary_search_recursive%28ordered_list,%2042,%20left_index,%20right_index%29%29&cumulative=false&curInstr=17&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false).

<iframe width="800" height="500" frameborder="0" src="https://pythontutor.com/iframe-embed.html#code=import%20math%0Afrom%20typing%20import%20Iterable,%20TypeVar,%20Tuple%0A%0AT%20%3D%20TypeVar%28%22T%22,%20str,%20int,%20float%29%20%20%23%20T%20should%20be%20of%20type%20int,%20float%20or%20str%0A%0Adef%20binary_search_recursive%28%0A%20%20%20%20container%3A%20Iterable%5BT%5D,%20target%3A%20T,%20left_index%3A%20int,%20right_index%3A%20int%0A%29%20-%3E%20int%3A%0A%20%20%20%20%22%22%22Binary%20search%20recursive%20implementation.%22%22%22%0A%20%20%20%20mid_index%20%3D%20left_index%20%2B%20math.floor%28%28right_index%20-%20left_index%29%20/%202%29%0A%20%20%20%20if%20left_index%20%3C%3D%20right_index%3A%0A%20%20%20%20%20%20%20%20if%20container%5Bmid_index%5D%20%3D%3D%20target%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20return%20mid_index%20%20%23%20base%20case%201%0A%20%20%20%20%20%20%20%20elif%20container%5Bmid_index%5D%20%3C%20target%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20return%20binary_search_recursive%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20container,%20target,%20mid_index%20%2B%201,%20right_index%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%0A%20%20%20%20%20%20%20%20else%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20return%20binary_search_recursive%28container,%20target,%20left_index,%20mid_index%20-%201%29%0A%20%20%20%20else%3A%0A%20%20%20%20%20%20%20%20return%20-1%20%20%23%20base%20case%202%0A%20%20%20%20%0Aordered_list%20%3D%20%5B0,%201,%202,%208,%2013,%2017,%2019,%2032,%2042%5D%0Aleft_index%20%3D%200%0Aright_index%20%3D%20len%28ordered_list%29%20-%201%0Aprint%28binary_search_recursive%28ordered_list,%2042,%20left_index,%20right_index%29%29&codeDivHeight=400&codeDivWidth=350&cumulative=false&curInstr=0&heapPrimitives=nevernest&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false"> </iframe>

### Tests

```{code-cell} ipython3
# Changing the strategy to RecursiveBinarySearchExactMatch
context.strategy = RecursiveBinarySearchExactMatch()
result = context.execute_search([2, 5, 8, 12, 16, 23, 38, 56, 72, 91], 23)
assert result == 5
```

### Time Complexity

#### Master Theorem

We have a recurrence relation of the form:

$$T(n) = a \cdot T\left(\frac{n}{b}\right) + f(n)$$

Where:

-   $a$ is the number of subproblems in the recursion.
-   $\frac{n}{b}$ is the size of each subproblem. (All subproblems are assumed
    to have the same size.)
-   $f(n)$ represents the cost of the work done outside the recursive calls,
    which includes the cost of dividing the problem and the cost of merging the
    solutions.

For our binary search, we have:

$$T(n) = T\left(\frac{n}{2}\right) + \mathcal{O}(1)$$

This translates into $a = 1$, $b = 2$, and $f(n) = \mathcal{O}(1)$, which means
$d = 0$ since $\mathcal{O}(n^0) = \mathcal{O}(1)$.

Now let's proceed with the Master Theorem. The Master Theorem states that the
solution to the recurrence relation:

$$T(n) = a \cdot T\left(\frac{n}{b}\right) + f(n)$$

is given as follows:

1. If $f(n) = \mathcal{O}(n^c)$, where $c < \log_b{a}$, then
   $T(n) = \Theta(n^{\log_b{a}})$.
2. If $f(n) = \mathcal{O}(n^c)$, where $c = \log_b{a}$, then
   $T(n) = \Theta(n^c \log n)$.
3. If $f(n) = \mathcal{O}(n^c)$, where $c > \log_b{a}$, then
   $T(n) = \Theta(f(n))$.

Comparing $d$ with $\log_b a$, we see that $d = \log_b a = \log_2 1 = 0$.

So, we're in the second case of the Master Theorem. According to the second case
of the Master Theorem, if $f(n) = \Theta(n^d)$, where $d = \log_b a$, then
$T(n) = \Theta(n^d \log n)$.

Substitute $d = 0$ into $T(n) = \Theta(n^d \log n)$, we get
$T(n) = \Theta(\log n)$, which means that the time complexity of binary search
is $\mathcal{O}(\log n)$.

#### Repeated Substitution

Let's denote the time complexity of our function as $\mathcal{T}(n)$, where $N$
is the number of elements being considered during a given recursive call.
Initially, $N$ is the size of the entire list, but with each recursive call, it
gets halved.

On every recursive call to the `recursive` function, the list is divided into
two halves, and only one half is considered for further search. Hence, the size
of the problem is reduced to half its previous size.

To find the recurrence relation, let's break down the operations:

1. We calculate the middle index, which takes constant time $\mathcal{O}(1)$.
2. We make a decision based on the middle element, again taking $\mathcal{O}(1)$
   time.
3. We make a recursive call, but only on half of the current list.

Putting this into a recurrence relation:

$$\mathcal{T}(n) = \mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(1)$$

This is a standard divide-and-conquer recurrence relation. We can solve it using
the Master Theorem or repeated substitution.

Using repeated substitution:

$$
\begin{align*}
\mathcal{T}(n) &= \mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(1) \\
\mathcal{T}(n) &= \left[\mathcal{T}\left(\frac{n}{4}\right) + \mathcal{O}(1)\right] + \mathcal{O}(1) \\
\mathcal{T}(n) &= \mathcal{T}\left(\frac{n}{4}\right) + 2\mathcal{O}(1) \\
\mathcal{T}(n) &= \mathcal{T}\left(\frac{n}{8}\right) + 3\mathcal{O}(1) \\
&\vdots \\
\mathcal{T}(n) &= \mathcal{T}\left(\frac{n}{2^k}\right) + k\mathcal{O}(1)
\end{align*}
$$

Now, $\mathcal{T}\left(\frac{n}{2^k}\right)$ will be $\mathcal{T}(1)$ (i.e., a
constant) when $\frac{n}{2^k} = 1$ or $k = \log n$.

So, the expansion becomes:

$$\mathcal{T}(n) = \mathcal{T}(1) + \log n \times \mathcal{O}(1)$$

Given that $\mathcal{T}(1)$ is a constant time, the dominating factor here is
$\log n$. Therefore, the time complexity is:

$$\mathcal{T}(n) = \mathcal{O}(\log n)$$

### Space Complexity

For recursive algorithms, the space complexity is often determined by the
maximum depth of the recursion. In other words, it's based on the maximum number
of recursive calls that are in progress at the same time.

So the intuition is simple, we already established there can be a maximum of
$\log n$ splits, thus it follows that the recursion depth is bounded by
$\log n$.

The space complexity of a recursive binary search gets divided into three parts:
input, auxiliary, and total space complexity.

#### Input Space Complexity

Input space complexity is the space used to store the input to the problem. For
binary search, the input is the array or list we are searching through, and a
target value.

In most cases, we don't consider the space taken by the inputs when analyzing
the space complexity of an algorithm, unless the algorithm modifies the input in
place. But if we do want to consider the input space, that's the space taken by
the inputs, which is $\mathcal{O}(n) + \mathcal{O}(1) = \mathcal{O}(n)$, where
$N$ is the length of the array, and $\mathcal{O}(1)$ is the space taken by the
target value.

#### Auxiliary Space Complexity

Auxiliary space complexity is the extra space or temporary space used by an
algorithm. In the case of recursive binary search, we still only need three
variables to hold the left, right, and middle indices (`l`, `r`, `mid_index`).
These variables occupy constant space, so the auxiliary space complexity is
$\mathcal{O}(1)$ per recursive call.

However, because this is a recursive function, we also have to consider the
space taken up by the recursion stack. In the worst-case scenario, binary search
makes $\mathcal{O}(\log n)$ recursive calls, and each call adds a level to the
stack. So, the auxiliary space complexity in this case is $\mathcal{O}(\log n)$.

Note that an iterative implementation of binary search would have a space
complexity of $\mathcal{O}(1)$, as it doesn't need additional space that grows
with the input size. It only uses a constant amount of space to store the
pointers and the target element.

For the recursive binary search:

-   Each time we make a recursive call, we essentially halve the input size.
-   At most, we would need to make $\log n$ recursive calls (since we are
    dividing by 2 each time) before we either find our target or exhaust the
    list.
-   Each of these calls gets pushed onto the call stack.

Therefore, the maximum height of the call stack, and hence the space complexity
due to the recursive calls, is $O(\log n)$.

#### Total Space Complexity

Total space complexity is the sum of input and auxiliary space complexities. For
binary search, if the input space complexity is not considered and the uses
$\mathcal{O}(\log n)$ auxiliary space, the total space complexity of recursive
binary search is $\mathcal{O}(\log n)$; otherwise, it is $\mathcal{O}(n)$.

To summarize, the recursive binary search algorithm is still very
space-efficient as it only requires logarithmic auxiliary space to perform the
search, and it does not modify the input array or list.

## When to use Binary Search?

If we can discover some kind of **monotonicity**, for example, if `condition(k)`
is `True` then `condition(k + 1)` is `True`, then we can consider binary search.

More formally, we have:

The essential precondition to apply binary search is the presence of a
**monotonic property**. This is a property that allows us to decide which half
of the search space should be eliminated based on the comparison between the
target value and the value at the current index.

```{prf:definition} Monotonicity
:label: monotonicity

In more formal terms, a function or sequence is said to have the property of
monotonicity if it is either entirely non-increasing or non-decreasing. A
function that increases monotonically does not necessarily increase constantly,
but it does not decrease at any point. Similarly, a function that decreases
monotonically does not necessarily decrease constantly, but it does not increase
at any point.

1. A sequence or function $f$ is said to be **monotone increasing** (or
   non-decreasing) on an interval $I$ if for all $x, y \in I$, if $x \leq y$,
   then $f(x) \leq f(y)$. In simple terms, as we move along the interval, the
   function value does not decrease; it either increases or stays the same.

2. Similarly, a sequence or function $f$ is said to be **monotone decreasing**
   (or non-increasing) on an interval $I$ if for all $x, y \in I$, if
   $x \leq y$, then $f(x) \geq f(y)$. That is, as we move along the interval,
   the function value does not increase; it either decreases or stays the same.
```

In the context of binary search, when the `condition` function has a monotonic
property (either always `True` to `False`, or always `False` to `True`), it
means that there is a clear threshold or tipping point in the sorted array that
divides the array into two halves - the first half where the `condition`
function is `True` and the second half where the `condition` function is `False`
(or vice versa).

That's where binary search comes into play: it allows us to effectively locate
that threshold by iteratively narrowing down the search space. If we find that
the `condition` is `True` for a given middle element (let's call it `mid`), we
know that all elements on the right of `mid` will also satisfy `condition`
(because of the monotonic property), so we can safely ignore the right half.
Conversely, if `condition(mid)` is `False`, we can ignore the left half.

If we can't establish such a monotonic property, it's difficult (or even
impossible) to decide which half of the array to eliminate, rendering binary
search ineffective or incorrect. Therefore, confirming the existence of this
monotonicity is crucial before deciding to use binary search.

## Solution (Minimize $k$, $s.t.$ condition($k$) is True)

Before going into details, we see the below:

```{prf:remark} Finding Target and Finding First True
:label: finding-target-and-finding-first-true

Finding a target in a sorted array and finding the "first True" in a sorted
Boolean array are conceptually similar because both rely on a monotonic
condition. In the first case, the condition is "Is the element at the current
index greater or equal to the target?" In the second case, it's "Is the element
at the current index True?"

To bridge the gap:

1. Consider the feasible function $f(x)$ that maps each element in the sorted
   array to either True or False based on whether the element is greater or
   equal to the target. This makes the problem equivalent to finding the "first
   True" in a sorted Boolean array derived from $f(x)$.
2. In both problems, once you identify an element that satisfies the condition
   (either being the target or being True), you can be sure that no elements
   satisfying the condition exist in the half of the array that is 'less' than
   the current element.

In the context of finding a specific target element $x$ in a sorted array, the
feasible function $f(i)$ would map to True for all elements greater than or
equal to $x$ and False for all elements less than $x$. So, if the array is
$[1, 3, 5, 7]$ and the target is $5$, the mapped Boolean array based on $f(i)$
would be $\text{FFFFTTT}$, making it a sorted Boolean array. In this setup,
"finding the first True" indeed corresponds to "finding the target element."
```

The problem of finding a target number in a sorted array is a "minimize k s.t.
condition(k) is True" problem because you're essentially looking for the
smallest (left-most) index `k` where the condition "array value at `k` is
greater than or equal to the target" is True.

In other words, you're trying to find the minimum `k` such that
`nums[k] >= target`. This can either be the first occurrence of the target in
the array (if the target exists in the array) or the position where the target
could be inserted to maintain the sorted order of the array (if the target does
not exist in the array).

This fits the structure of "minimize k s.t. condition(k) is True" because you
are minimizing the index `k` subject to a condition (i.e., `nums[k] >= target`).

In the binary search template, this is implemented as the `condition(mid)`
function. The binary search algorithm keeps adjusting the search boundaries
(i.e., `left` and `right`) based on whether the condition is met at the
mid-point, and keeps narrowing down to the smallest `k` (left-most position)
where the condition is True. This is why this problem fits into the "minimize k
s.t. condition(k) is True" structure.

```{code-cell} ipython3
def binary_search(nums: List[int], target: int) -> int:
    def condition(k: int, nums: List[int]) -> bool:
        return nums[k] >= target

    left, right = 0, len(array)
    while left < right:
        mid = left + (right - left) // 2
        if condition(k=mid, nums=nums):
            right = mid
        else:
            left = mid + 1
    return left

# Example 1
array = [1, 2, 3, 4, 5]
target = 3
result = binary_search(nums=array, target=target)
assert result == 2
```

So you essentially combined the two steps of finding the target and finding the
left-most occurrence of the target into one step.

## References and Further Readings

1. [LeetCode: Binary Search](https://leetcode.com/explore/learn/card/binary-search/125/template-i/)
2. [LeetCode: Binary Search Editorial](https://leetcode.com/problems/binary-search/editorial/)
3. [GeeksforGeeks: Binary Search](https://www.geeksforgeeks.org/binary-search/)
4. [Runestone Academy: The Binary Search](https://runestone.academy/ns/books/published/pythonds3/SortSearch/TheBinarySearch.html)
5. [Wikipedia: Binary Search Algorithm](https://en.wikipedia.org/wiki/Binary_search_algorithm)
6. [Algomonster: Binary Search](https://algo.monster/problems/binary_search_intro)
7. [Generic Template from Leetcode User](https://leetcode.com/problems/koko-eating-bananas/solutions/769702/python-clear-explanation-powerful-ultimate-binary-search-template-solved-many-problems/)
8. [Strong Induction - Binary Search Correctness](http://www.cs.cornell.edu/courses/cs211/2006sp/Lectures/L06-Induction/binary_search.html)

[^average_case]:
    The average case is about the same as the worst case, but if you want to be
    pedantic, there is a mathematical derivation in
    [Wikipedia](https://en.wikipedia.org/wiki/Binary_search_algorithm#Derivation_of_average_case).

```

```

```

```