---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Two Sum II - Input Array Is Sorted

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
[![LeetCode Problem](https://img.shields.io/badge/LeetCode-167-FFA116?style=social&logo=leetcode)](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted)
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)
![Tag](https://img.shields.io/badge/Tag-Binary_Search-orange)
![Tag](https://img.shields.io/badge/Tag-Two_Pointers-orange)
![Tag](https://img.shields.io/badge/Tag-Array-orange)

```{contents}
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

Given a **1-indexed** array of integers `numbers` that is already **_sorted in
non-decreasing order_**, find two numbers such that they add up to a specific
`target` number. The two numbers are `numbers[index1]` and `numbers[index2]`
where `1 <= index1 < index2 <= numbers.length`.

_Return the indices of the two numbers_, `index1` and `index2`, **added by one**
as an integer array `[index1, index2]` of _length 2_.

The tests are generated such that there is **exactly one solution**. You **may
not** use the same element twice.

Your solution must use only constant extra space.

## Intuition

Since the given input array is sorted in non-decreasing order, we can make use
of a
[**two-pointer technique**](https://leetcode.com/articles/two-pointer-technique/).
We start with one pointer at the beginning of the array and another at the end.
We calculate the sum of the elements at the two pointer positions. If the sum
equals the target, we have found our solution. If the sum is less than the
target, we move the left pointer to the right (increasing the sum). If the sum
is more than the target, we move the right pointer to the left (decreasing the
sum). We continue this process until we find a solution or the pointers meet.

## Assumptions

-   Each given input will have exactly **one solution**.
-   The **same element** cannot be used twice in the pair.
-   The `numbers` array is **1-indexed** and sorted in **non-decreasing** order.
-   All the elements in the `numbers` array are **integers**.
-   The `target` is also an **integer**.
-   The `numbers` array will at least have **two elements** (since we are
    looking for a pair).
-   Your solution must use only **constant extra space**.

## Constraints

-   It's assumed that the `numbers` array size is in the range of:

    $$
    2 \leq \text{numbers.length} \leq 3 \times 10^4.
    $$

-   Each element in `numbers` array is an integer in the range of:

    $$
    -10^3 \leq \text{numbers[i]} \leq 10^3.
    $$

-   The `target` is an integer in the range of:

    $$
    -10^3 \leq \text{target} \leq 10^3.
    $$

-   `numbers` is sorted in non-decreasing order.
-   There is only **one valid answer**.

### What are Constraints for?

The constraints for this problem are as follows:

1. The length of the `numbers` array is in the range of
   $2 \leq
   \text{numbers.length} \leq 3 \times 10^4$. This means that there
   will be at least two numbers and at most 30,000 numbers in the array. This
   helps us understand the potential size of the input and informs the
   complexity of our solution. For example, a solution with a time complexity of
   $\mathcal{O}(n^2)$ might not be feasible, while a solution with a time
   complexity of $\mathcal{O}(n)$ or $\mathcal{O}(n \log n)$ would be
   acceptable.

2. Each number in the `numbers` array is an integer in the range of
   $-1000 \leq
   \text{numbers[i]} \leq 1000$. This means that each number in
   the array can be as low as -1000 and as high as 1000. Understanding the range
   of numbers in the array can be important for avoiding potential issues with
   integer overflow and for considering possible edge cases.

3. The `target` is an integer in the range of
   $-1000 \leq \text{target} \leq
   1000$. This informs us about the potential
   values that the target sum can take. It's important to consider the range of
   the target when thinking about possible edge cases.

4. The `numbers` array is sorted in non-decreasing order. This means that each
   number in the array is less than or equal to the next number. This
   information is crucial because it allows us to use certain techniques (like
   the two-pointer technique) that wouldn't work on an unsorted array.

5. The tests are generated such that there is exactly one solution. This means
   that there will always be exactly two numbers in the array that add up to the
   target. This simplifies the problem because we don't have to consider cases
   where there might be multiple pairs of numbers that sum to the target, or
   cases where there is no pair that sums to the target.

Overall, these constraints give us important information about the size and
nature of the input data, which can help us determine feasible approaches to
solving the problem and anticipate potential edge cases.

## Test Cases

-   Test Case 1: `numbers = [2, 7, 11, 15], target = 9`, Return: `[1, 2]`
-   Test Case 2: `numbers = [2, 3, 4], target = 6`, Return: `[1, 3]`
-   Test Case 3: `numbers = [2, 7, 11, 15], target = 18`, Return: `[2, 3]`

## Edge Cases

-   Edge Case 1: `numbers = [1, 2], target = 3`, Return: `[1, 2]` (Minimum size
    array)
-   Edge Case 2: `numbers = [-1, 0, 1, 2], target = 1`, Return: `[2, 3]`
    (Negative numbers and zero)
-   Edge Case 3: `numbers = [0, 0, 3, 4], target = 0`, Return: `[1, 2]` (Target
    is zero)

## Walkthrough / Whiteboarding

Consider `numbers = [2, 7, 11, 15], target = 9`.

1. Initialize `left` pointer at index 1 and `right` pointer at index 4.
2. Calculate the sum `numbers[left] + numbers[right] = 2 + 15 = 17`.
3. Since the sum is greater than the target, move the `right` pointer one step
   to the left.
4. Calculate the new sum `numbers[left] + numbers[right] = 2 + 11 = 13`.
5. The sum is still greater than the target, move the `right` pointer one step
   to the left.
6. Calculate the new sum `numbers[left] + numbers[right] = 2 + 7 = 9`.
7. The sum equals the target, return `[left, right] = [1, 2]`.

## Theoretical Best Time Complexity

The theoretical best time complexity for this problem is $\mathcal{O}(n)$, where
$n$ is the length of the `numbers` array. This is because the most efficient
solution to this problem involves traversing the array once, which requires
$\mathcal{O}(n)$ time.

## Theoretical Best Space Complexity

The theoretical best (auxiliary) space complexity for this problem is
$\mathcal{O}(1)$. This is because we only use a constant amount of space to
store the `left` and `right` pointers. We do not use any additional data
structures that scale with the input size.

The given problem requires the solution to use only constant extra space. This
constraint means that the space usage of our solution should not increase with
the size of the input array. Our solution meets this requirement as it only uses
two variables to keep track of the `left` and `right` pointers, irrespective of
the size of the input array.

## Space-Time Tradeoff

There is no space-time tradeoff for this problem. This is because we cannot our
space complexity is already at constant space, and our time complexity is
already at linear time. We cannot improve either of these metrics without
changing the problem itself.

If the problem is solved using hashmap, then as compared to the brute force
method, there indeed is a trade-off between space and time complexity (i.e.
sacrificing space for time).

## Solution (Two-Pointers Moving In Different Directions)

Consider an array $\mathcal{A}$ sorted in ascending order and a target sum $t$.
We seek to find a pair of elements $\mathcal{A}[i]$ and $\mathcal{A}[j]$ (where
$i \neq j$) such that $\mathcal{A}[i] + \mathcal{A}[j] = t$.

### Intuition

TODO.

### Visualisation

See [visualization here](https://algo.monster/problems/two_sum_sorted).

### Algorithm

We can use a two-pointer technique to solve this problem. The two pointers left
and right start at the beginning and end of the array respectively. We compare
the sum of the two numbers at indices left and right with the target sum target
$t$. If the sum is equal to $t$, we have found a solution and return the
indices. If the sum is less than $t$, we increment left by one. If the sum is
greater than $t$, we decrement right by one. We repeat this process until left
and right are equal, at which point we have checked every pair of numbers.

```{prf:algorithm} Two-Pointers
:label: two-sum-167-two-sum-ii-input-array-is-sorted-two-pointers-algorithm

The two-pointer algorithm operates as follows:

1. Start with two pointers, one at the beginning of the array (index $i$) and
   one at the end (index $j$).
2. Calculate the sum $\mathcal{A}[i] + \mathcal{A}[j]$.
3. If the sum is equal to the target $t$, we have found a solution.
4. If the sum is less than $t$, increment $i$.
5. If the sum is greater than $t$, decrement $j$.
6. Repeat steps 2-5 until $i = j$, at which point every pair has been checked.
```

### Claim

We make a more formal mathematical proof using the property of sorted arrays.
The algorithm mentioned is often referred to as the "two-pointer" or "two-sum"
method, and it works very efficiently on sorted arrays due to their inherent
order.

Firstly, let's clarify what we're aiming to prove:

```{prf:theorem} Claim
:label: two-sum-167-two-sum-ii-input-array-is-sorted-two-pointers-claim

**Claim:** Given an array $\mathcal{A}$ sorted in ascending order and a target
sum $t$, there exists a pair of elements $\mathcal{A}[i]$ and $\mathcal{A}[j]$
(where $i \neq j$) such that $\mathcal{A}[i] + \mathcal{A}[j] = t$ if and only
if the two-pointer algorithm finds such a pair.
```

### Proof

```{prf:proof}
Assume that there is a solution to the problem, i.e., there exist indices $i^*$
and $j^*$ (with $i^* < j^*$) such that
$\mathcal{A}[i^*] + \mathcal{A}[j^*] = t$. This is also part of the assumption
in the question.

By contradiction, assume that the algorithm doesn't find this solution. Then, it
must be the case that either $i$ was incremented past $i^*$ when $j$ was still
greater than $j^*$, or $j$ was decremented past $j^*$ when $i$ was still less
than $i^*$.

In the first case, this means that for some $j' > j^*$, we had
$\mathcal{A}[i^*] + \mathcal{A}[j'] < t$, and therefore $i$ was incremented.
However, this contradicts the fact that $\mathcal{A}$ is sorted in ascending
order, as it implies that $\mathcal{A}[j'] < \mathcal{A}[j^*]$, which is not
possible.

In the second case, this means that for some $i' < i^*$, we had
$\mathcal{A}[i'] + \mathcal{A}[j^*] > t$, and therefore $j$ was decremented.
However, this contradicts the fact that $\mathcal{A}$ is sorted in ascending
order, as it implies that $\mathcal{A}[i'] > \mathcal{A}[i^*]$, which is not
possible.

Therefore, the algorithm must find a solution if it exists. This completes the
proof.
```

In other words, the intuition behind the proof is as follows:

Let $\mathcal{A} = [a_1, a_2, \ldots, a_n]$ be a sorted array (ascending), and
let $t$ be the target sum. Assume without loss of generality that there exists
an unqiue solution pair $a_i + a_j = t$ where $i < j$. Then, the two-pointer
algorithm will find this solution pair. Why so?

Consider the following cases:

1. If for a given index pair $(k, l)$, we have $a_k + a_l < t$, then we know
   that $a_k + a_m < t$ for all $m > l$. This is because the array is sorted in
   ascending order, so $a_m > a_l$ for all $m > l$. Therefore, we can increment
   $k$ to $k + 1$ and continue the search. If you were to instead decrement $l$
   to $l - 1$, you will never find a solution pair, as the sum will only
   decrease further.
2. If for a given index pair $(k, l)$, we have $a_k + a_l > t$, then we know
   that $a_m + a_l > t$ for all $m < k$. This is because the array is sorted in
   ascending order, so $a_m < a_k$ for all $m < k$. Therefore, we can decrement
   $l$ to $l - 1$ and continue the search. Similarly, if you were to instead
   increment $k$ to $k + 1$, you will never find a solution pair, as the sum
   will only increase further.

Without loss of generality, if our left pointer were to reach to index $i$
first, and the right pointer is at index $l$ that is greater than $j$, then we
know that then at this point in time, $a_i + a_l > t$ and based on the above
argument, we just need to decrement $l$ to $l - 1$ to continue the search.

### Implementation

```{code-cell} ipython3
from typing import List

class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1

        while left < right:
            two_sum = numbers[left] + numbers[right]

            # this means two pointers have not met each other, if met, stop
            if two_sum > target:
                right -= 1
                continue

            # this means two pointers have not met each other, if met, stop
            if two_sum < target:
                left += 1
                continue

            if two_sum == target:
                return [left + 1, right + 1]
```

### Tests

```{code-cell} ipython3
numbers = [2, 7, 11, 15]
target = 9
assert Solution().twoSum(numbers, target) == [1, 2]
```

### Time Complexity

Let's denote $\mathcal{T}(n)$ as the time complexity of the two-pointer
function. We can break down the time complexity of each major step.

The time complexity of the algorithm is $\mathcal{O}(n)$. This is because we
traverse the list of length $n$ a single time. Every operation inside the loop,
including the comparisons and pointer adjustments, has a time complexity of
$\mathcal{O}(1)$.

Consider the while loop inside the function. Each line inside this loop has a
time complexity of $\mathcal{O}(1)$, meaning it takes a constant amount of time
to execute, regardless of the size of the input. So, a typical iteration inside
the loop would require approximately $\mathcal{O}(6)$ time, given that there are
six operations within the loop.

However, since the loop runs $n$ times (where $n$ is the length of the input
list), the overall time required would be $n$ times the time required for a
single iteration. This gives us:

$$
\begin{aligned}
\mathcal{T}(n) &= n \cdot \mathcal{O}(3) \\
&\approx \mathcal{O}(3n).
\end{aligned}
$$

In big-O notation, we typically omit constant multipliers since we're primarily
interested in how the runtime scales with the size of the input. Hence, we
simplify $\mathcal{O}(3n)$ to $\mathcal{O}(n)$. This signifies that the runtime
of the algorithm grows linearly with the size of the input list.

The time complexity of the two-pointer approach for the Two Sum Sorted problem
varies depending on the case:

-   **Best Case**: The two numbers that sum to the target are at the beginning
    and end of the list. In this case, the function finds the solution in the
    first iteration, so the time complexity is $\mathcal{O}(1)$.

-   **Average Case**: On average, the two numbers that sum to the target are
    evenly distributed in the list. Therefore, on average, the function has to
    inspect $\frac{n}{2}$ elements in `numbers` before finding the pair that
    sums to `target`. So technically, we can think of the average time
    complexity to be $\mathcal{O}\left(\frac{n}{2}\right)$. However, as we
    simplify this to its highest order term and ignore the coefficients in Big O
    notation, the average case complexity becomes $\mathcal{O}(n)$.

-   **Worst Case**: The two numbers that sum to the target are at the middle of
    the list. In this case, the function has to iterate through the entire list,
    so the time complexity is $\mathcal{O}(n)$.

```{list-table} Time Complexity of Two-Sum Sorted Function Using Two-Pointer Approach
:header-rows: 1
:name: two-sum-167-two-sum-ii-input-array-is-sorted-two-pointers

* - Case
  - Complexity
  - Description
* - Best Case
  - $\mathcal{O}(1)$
  - The two numbers that sum to the target are at the beginning and end of the list.
* - Average Case
  - $\mathcal{O}(\frac{n}{2})$
  - On average, the two numbers that sum to the target are evenly distributed in
    the list.
* - Worst Case
  - $\mathcal{O}(n)$
  - The two numbers that sum to the target are at the middle of the list.
```

### Space Complexity

If we denote $\mathcal{S}(n)$ as the space complexity of the two-pointer
function, we can break down the space complexity of each major step.

The auxiliary space complexity of the two-pointer approach for the Two Sum
Sorted problem is $\mathcal{O}(1)$. This is because we only use a constant
amount of space to store the input list and the two pointers. Unlike the hash
table approach in the Two Sum problem, we don't need to create any additional
data structures that grow with the size of the input.

```{list-table} Space Complexity of Two-Sum Sorted Function Using Two-Pointer Approach
:header-rows: 1
:name: two-sum-sorted-space-complexity-two-pointer

* - Type
  - Complexity
  - Description
* - Input Space
  - $\mathcal{O}(n)$
  - The input space depends on the size of the input list `numbers`.
* - Auxiliary Space
  - $\mathcal{O}(1)$
  - The auxiliary space is constant as it only uses two integer variables to
    store the pointers.
* - Total Space
  - $\mathcal{O}(n)$
  - The total space is the sum of the input and auxiliary space.
```

#### Input Space

The input space for this function is the space needed to store the input data,
which in this case is the list `numbers`. Since `numbers` is a list of `n`
integers, the space required to store this list is proportional to `n`.
Therefore, the input space complexity is $\mathcal{O}(n)$.

#### Auxiliary Space

The auxiliary space is the extra space or the temporary space used by the
algorithm. In the two-pointer approach, we use two integer variables `left` and
`right` to keep track of the pointers. These variables take a constant amount of
space. Therefore, the auxiliary space complexity is $\mathcal{O}(1)$.

#### Total Space

The total space required by an algorithm is the sum of the input space and the
auxiliary space. In this case, the total space is
$\mathcal{O}(n) + \mathcal{O}(1)$, which simplifies to $\mathcal{O}(n)$.
Therefore, the total space complexity of the two-pointer approach is
$\mathcal{O}(n)$.

To summarize, the two-pointer approach for the Two Sum Sorted problem has a time
complexity of $\mathcal{O}(n)$ and a space complexity of $\mathcal{O}(1)$.

## References and Further Readings

-   [Algomonster: Two Sum II - Input array is sorted](https://algo.monster/problems/two_sum_sorted)
-   [Leetcode: Two Sum II - Input array is sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/editorial/)
