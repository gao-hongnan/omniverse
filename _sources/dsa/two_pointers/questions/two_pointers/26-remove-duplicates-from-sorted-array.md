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

# Remove Duplicates from Sorted Array

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
[![LeetCode Problem](https://img.shields.io/badge/LeetCode-26-FFA116?style=social&logo=leetcode)](https://leetcode.com/problems/remove-duplicates-from-sorted-array)
![Difficulty](https://img.shields.io/badge/Difficulty-Easy-green)
![Tag](https://img.shields.io/badge/Tag-Binary_Search-orange)
![Tag](https://img.shields.io/badge/Tag-Two_Pointers-orange)
![Tag](https://img.shields.io/badge/Tag-Array-orange)

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
    from omnivault.dsa.utils import compare_test_case
    from omnivault._types._generic import T
else:
    raise ImportError("Root directory not found.")
```

## Problem

Given an integer array `nums` sorted in non-decreasing order, _remove the
duplicates [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm)_
such that each unique element appears only once. The relative order of the
elements should be kept the same. Then return the number of unique elements in
`nums`.

Consider the number of unique elements of `nums` to be `k`, to get accepted, you
need to do the following things:

-   Change the array `nums` such that the first `k` elements of `nums` contain
    the unique elements in the order they were present in `nums` initially. The
    remaining elements of `nums` are not important as well as the size of
    `nums`.
-   Return `k`.

## Intuition

If we are allowed to use extra memory, we can use a hash table to store the
unique elements in the array. Then, we can return the length of the hash table
as the length of the array without duplicates. But since we are not allowed to
use extra memory, we need to find a way to do this in-place. Usually this kind
of pattern is solved using the two-pointer technique.

The two-pointer algorithm in question can be intuitively understood as a "runner
and walker" or "fast and slow" strategy.

Let's say you're tasked with removing duplicates from a sorted array of numbers.
Imagine these numbers are placed on a straight pathway, sorted from low to high,
and duplicates are grouped together. You have two markers or pointers – the
"fast" one and the "slow" one.

You start both pointers at the beginning of the pathway (the first element of
the array). The fast pointer races through the pathway, visiting every number,
while the slow one walks, taking its time. Whenever the fast pointer finds a new
number (which is not a duplicate, since duplicates are grouped together), it
signals the slow pointer to move one step forward and copy this new number.

So the slow pointer is building a collection of unique numbers in-place (i.e.,
using the same array), because it only moves and copies when the fast pointer
signals that it has found a new unique number. The fast pointer, on the other
hand, is doing the job of scanning through each element in the array.

By the time the fast pointer reaches the end of the array, the slow pointer
would have gone through all unique elements, replacing duplicates with new
unique elements in the original array. The position of the slow pointer (plus
one, since array indexing typically starts from zero) will give the length of
the array without duplicates.

This is the intuition behind the algorithm. It uses two pointers at different
speeds to efficiently scan and modify the array in-place. The proof shows that
this approach is sound and will give us the expected outcome.

## Solution (Two-Pointers Moving In The Same Direction)

Consider an array $\mathcal{A}$ sorted in non-decreasing order. We seek to
remove the duplicates in-place and return the new length of the array.

### Intuition

TODO.

### Visualisation

See [visualization here](https://algo.monster/problems/remove_duplicates).

### Algorithm

We use the two-pointer technique where we initiate two pointers, `slow` and
`fast`. The `slow` pointer represents the location at which we want to place the
next unique element we find as we scan through the array. The `fast` pointer
scans through the array.

```{prf:algorithm} Two-Pointers
:label: remove-duplicates-from-sorted-array-two-pointers-algorithm-1

Let's define the following variables:

-   $i$: as the slow pointer;
-   $j$: as the fast pointer.

The algorithm is as follows:

1. Start with two pointers at the beginning of the array (index $i$) and one at
   the same location (index $j$).
2. Compare the elements at indices $i$ and $j$ in array $\mathcal{A}$, i.e.,
   check if $\mathcal{A}[i] = \mathcal{A}[j]$.
3. If the elements are not equal ($\mathcal{A}[i] \neq \mathcal{A}[j]$),
   increment $i$ (i.e., $i = i + 1$), and replace the element at index $i$ with
   the element at $j$ (i.e., $\mathcal{A}[i] = \mathcal{A}[j]$).
4. If the elements are equal ($\mathcal{A}[i] = \mathcal{A}[j]$), leave $i$ as
   it is and only move $j$ to the next position (i.e., $j = j + 1$).
5. Repeat steps 2-4 until $j$ has scanned the entire array, i.e., until $j = n$,
   where $n$ is the length of the array.
6. At this point, $i + 1$ gives the length of the array with all unique
   elements, and the elements from $\mathcal{A}[0]$ to $\mathcal{A}[i]$ are all
   unique elements.
```

In code like syntax, it would look like this:

```{prf:algorithm} Two-Pointers (Code)
:label: remove-duplicates-from-sorted-array-two-pointers-algorithm-2

1. Start with two pointers, one at the beginning of the array (index `slow`) and
   one at the same location (index `fast`).
2. Compare the elements at indices `slow` and `fast`.
3. If the elements are not equal, increment `slow`, replace the element at index
   `slow` with the element at `fast`.
4. If the elements are equal, leave `slow` as it is and only move `fast` to the
   next position.
5. Repeat steps 2-4 until `fast` has scanned the entire array.
```

### Claim

We first make a claim and proceed to prove it.

```{prf:theorem} Claim
:label: remove-duplicates-from-sorted-array-two-pointers-claim

**Claim:** Given an array $\mathcal{A}$ sorted in non-decreasing order, the
two-pointer algorithm correctly removes all duplicates from $\mathcal{A}$
in-place, such that each element appears only once, and returns the length of
the new array.
```

### Proof

We make a formal mathematical proof using the property of sorted arrays. The
algorithm mentioned is often referred to as the "two-pointer" method, and it
works efficiently on sorted arrays due to their inherent order.

```{prf:proof}
Let's denote the array at the beginning as $\mathcal{A}$ of length $n$. Let $j$
be the fast pointer and $i$ be the slow pointer. Both are indices of
$\mathcal{A}$, such that $0 \leq i, j < n$.

Our goal is to prove that for all $k$ where $0 \leq k \leq i$, the elements
in the subarray $\mathcal{A}[0,...,k]$ are unique and sorted.

We will use induction on $j$.

**Base Case:** When $j = 0$, $i = 0$ and the array consists of only one element.
It is sorted and contains no duplicate elements. Therefore, the claim holds for
the base case.

**Inductive Step:** Let's assume that the claim is true for some $j = k$. That
is, the subarray $\mathcal{A}[0,...,i]$ contains unique and sorted elements for
some $i \leq k$.

Now, let's consider $j = k + 1$. There are two possibilities:

1. $\mathcal{A}[j] = \mathcal{A}[i]$: The element at index $j$ is a duplicate of
   the element at index $i$. The algorithm doesn't increment $i$, and the
   invariant still holds because the unique and sorted subarray hasn't changed.

2. $\mathcal{A}[j] \neq \mathcal{A}[i]$: The element at index $j$ is a new
   unique element. The algorithm increments $i$ and replaces $\mathcal{A}[i]$
   with $\mathcal{A}[j]$. This maintains the invariant, as the subarray up to
   index $i$ remains sorted and unique.

By induction, we can conclude that the algorithm maintains the invariant at
every step, and hence the final array up to index $i$ consists of unique and
sorted elements.

The space complexity is $\mathcal{O}(1)$, since we didn't use any extra space in
the process, which confirms the in-place nature of the algorithm.

Thus, our claim is proven.
```

The induction only involves $j$ to avoid complexity as considering both $i$ and
$j$ would make the proof more complex. However, the proof is still valid.

We give an informal proof of the claim below, which may be more intuitive.

In the sorted array, all duplicate elements are contiguous. When the `fast`
pointer encounters a new unique element (i.e., `nums[fast] != nums[slow]`), we
increment `slow` and replace `nums[slow]` with `nums[fast]`. Thus, we always
keep the first occurrence of each element, and duplicates are overwritten by new
unique elements.

By the end of the algorithm, `slow` is at the last unique element's index. So
the length of the array with all unique elements is `slow + 1`.

The proof follows from the correctness of the algorithm, which is guaranteed by
the [**loop invariant**](https://en.wikipedia.org/wiki/Loop_invariant): at the
start of each iteration of the loop, the array up to index `slow` consists of
sorted unique elements of the array, and `fast` is the index of the element to
compare with `nums[slow]`. This invariant is clearly true at the start, and each
iteration of the loop maintains it, so when the loop terminates, the array up to
index `slow` contains all unique elements.

The space complexity is $\mathcal{O}(1)$ since no additional space is used.
Thus, the algorithm meets the in-place requirement. Therefore, the algorithm is
correct.

### Implementation

```{code-cell} ipython3
from typing import List

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        slow: int = 0

        for fast in range(len(nums)):
            if nums[fast] != nums[slow]:
                slow += 1
                nums[slow] = nums[fast]
        return slow + 1
```

### Tests

```{code-cell} ipython3
nums: List[int] = [1, 1, 1]
compare_test_case(
    actual=Solution().removeDuplicates(nums=nums),
    expected=1,
    description="test all duplicates",
)
```

## References and Further Readings

-   [Leetcode: Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/editorial/)
-   [Algomonster: Remove Duplicates from Sorted Array](https://algo.monster/problems/remove_duplicates)
