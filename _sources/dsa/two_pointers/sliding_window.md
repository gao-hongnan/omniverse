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

# Sliding Window

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Sliding_Window-orange)

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
    from omnivault.dsa.utils import compare_test_cases
    from omnivault._types._generic import T
else:
    raise ImportError("Root directory not found.")
```

## Fix Sized Window

### Find maximum (or minimum) sum of a subarray of size k

Given an array (list) `nums` consisted of only non-negative integers, find the
largest sum among all subarrays of length `k` in `nums`.

For example, if the input is `nums = [1, 2, 3, 7, 4, 1]`, `k = 3`, then the
output would be `14` as the largest length `3` subarray sum is given by
`[3, 7, 4]` which sums to `14`.

#### Solution 1 (Not Optimal)

```{code-cell} ipython3
def subarray_sum_fixed(nums: List[int], k: int) -> int:
    if len(nums) < k:
        return 0

    maximum = float("-inf")
    total_num = len(nums)

    for index in range(total_num):
        if index == total_num - k:  # no need + 1 since start from 0
            break
        subarray = nums[index:index + k]
        sum_ = sum(subarray)
        if sum_ >= maximum:
            maximum = sum_
    return maximum
```

##### Test

```{code-cell} ipython3
nums = [1, 2, 3, 7, 4, 1]
k = 3

compare_test_cases(
    actual_list = [subarray_sum_fixed(nums, k)],
    expected_list = [14],
    description_list = ["subarray_sum_fixed"],
)
```

##### Time Complexity

At first glance, it seems like the time complexity of the above algorithm will
be $\mathcal{O}(n)$ where $n$ is the total number of elements in the input since
we are iterating through each element of the input array up till $n-k+1$.

But there is a hidden factor in the above algorithm: the call to `sum` for each
subarray of size $k$. Remember, the `sum` function takes $\mathcal{O}(k)$ time
for each subarray of size $k$. Therefore, the overall time complexity of our
algorithm will become $\mathcal{O}(n*k)$.

And let's consider where $k << n$, then the time complexity will still be
$\mathcal{O}(n)$ since $k$ is small.

If $k$ is large or approaching $n$, then the time complexity will be
$\mathcal{O}(n^2)$? Not really, since the maximum number of subarrays of size
$k$ in an array of size $n$ is $n-k+1$. Therefore, the time complexity of our
algorithm will be $\mathcal{O}((n-k+1)*k)$, which is asymptotically equivalent
to $\mathcal{O}(k)$ which is same as $\mathcal{O}(n)$.

Consider the case where $k=n$:

If $k = n$, where $n$ is the length of the array, then there would only be one
subarray of length $n$ (the entire array itself). Therefore, the algorithm would
only need to compute the sum of this one subarray, which would result in a time
complexity of $\mathcal{O}(n)$, not $\mathcal{O}(n^2)$.

However, in the general case where $k < n$, the time complexity of the algorithm
would be $\mathcal{O}(n*k)$, because the algorithm needs to compute the sum of
each possible subarray of length $k$ in the array.

So while the worst-case time complexity of the algorithm is $\mathcal{O}(n*k)$,
if $k = n$, the time complexity would actually be $\mathcal{O}(n)$.

Furthermore, the above solution may not be faithful to the "sliding window"
approach.

##### Space Complexity

The space (auxiliary) complexity of the given `subarray_sum_fixed` function is
$\mathcal{O}(k)$, where $k$ is the size of the subarray being considered.

Here's a breakdown of the space usage in the function:

-   `maximum` variable: This variable is used to track the maximum sum
    encountered so far. Its space requirement is constant, as it only stores a
    single integer value.

-   `total_num` variable: This variable stores the total number of elements in
    the `nums` list. Its space requirement is constant, as it only stores a
    single integer value.

-   `subarray` variable: This variable is used to store a subarray of size k. In
    each iteration of the loop, a new subarray is created and stored in this
    variable. The space requirement for the `subarray` variable is
    $\mathcal{O}(k)$ since it stores k elements.

-   `sum_` variable: This variable stores the sum of the elements in the
    subarray. Its space requirement is constant, as it only stores a single
    integer value.

Overall, the dominant factor affecting the space complexity is the size of the
subarray, which is $k$. Therefore, the space (auxiliary) complexity of the
function is $\mathcal{O}(k)$.

But this is a bit tricky since strictly speaking, the space complexity **_does
not_** depend on the input size. It depends on the size of the subarray being
considered, which is $k$.

#### Solution 2 (Optimal)

The approach described is called a "sliding window" approach because we're
essentially sliding a window of a fixed size (k) along the array, and
calculating the sum of the elements within that window.

Here's a step-by-step explanation:

1. We start by defining a window of size `k` at the leftmost part of the array.
   The sum of the elements in this window is calculated and stored in a variable
   called `window_sum`.

2. The maximum sum found so far, `largest`, is initially set to the
   `window_sum`.

3. We then "slide" this window one position to the right. To calculate the new
   `window_sum`, we add the value of the new element on the right and subtract
   the value of the element that was previously on the leftmost position of the
   window.

4. If the new `window_sum` is larger than the `largest` sum found so far, we
   update `largest` to be the new `window_sum`.

5. Repeat steps 3 and 4 until we've slid the window to the end of the array. The
   `largest` sum at the end of this process is the largest sum of `k`
   consecutive elements in the array.

Here's an example:

Consider an array [1, 3, 2, 6, -1, 4, 1, 8, 2] and `k = 5`.

In the beginning, our window of size 5 includes the elements [1, 3, 2, 6, -1].
The `window_sum` is 11 (1+3+2+6-1). So initially, `largest = window_sum = 11`.

The window slides to the right to include the next element 4 and exclude the
leftmost element 1. Now, the window includes the elements [3, 2, 6, -1, 4]. The
new `window_sum` is 14 (11 - 1 + 4), which is larger than `largest`, so we
update `largest` to 14.

We continue this process until we reach the end of the array. The final
`largest` value will be the maximum sum of `k` consecutive elements.

Here's a diagram to illustrate this process:

```
Array: [1, 3, 2, 6, -1, 4, 1, 8, 2]
k = 5

Steps:

# Window position              window_sum       largest
{1, 3, 2, 6, -1}, 4, 1, 8, 2     11               11
1, {3, 2, 6, -1, 4}, 1, 8, 2     14               14
1, 3, {2, 6, -1, 4, 1}, 8, 2     12               14
1, 3, 2, {6, -1, 4, 1, 8}, 2     18               18
1, 3, 2, 6, {-1, 4, 1, 8, 2}     14               18
```

At the end, the maximum sum of 5 consecutive elements in the array is 18.

It is worth noting that that the total number of subarrays of size $k$ in an
array of size $n$ is $n-k+1$. Why? Because the first subarray of size $k$ will
start at index 0 and end at index $k-1$. The second subarray of size $k$ will
start at index 1 and end at index $k$. The third subarray of size $k$ will start
at index 2 and end at index $k+1$. And so on. The last subarray of size $k$ will
start at index $n-k$ and end at index $n-1$.

Therefore, the total number of subarrays of size $k$ in an array of size $n$ is
$n - k + 1$. This is because for each starting index of the subarray, we "shift"
it one position to the right, until we reach the point where the end of the
subarray is at the last index of the array.

Here's a visualization for better understanding:

Consider an array of size $n = 7$ and we want subarrays of size $k = 3$.

Array: [a, b, c, d, e, f, g]

Subarrays of size 3:

1. [a, b, c], d, e, f, g (starts at index 0, ends at index 2)
2. a, [b, c, d], e, f, g (starts at index 1, ends at index 3)
3. a, b, [c, d, e], f, g (starts at index 2, ends at index 4)
4. a, b, c, [d, e, f], g (starts at index 3, ends at index 5)
5. a, b, c, d, [e, f, g] (starts at index 4, ends at index 6)

As you can see, the last subarray starts at index $n - k = 7 - 3 = 4$ and ends
at index $n - 1 = 6$.

So, we have 5 subarrays of size 3 in an array of size 7, which matches with the
formula $n - k + 1 = 7 - 3 + 1 = 5$.

##### Implementation

```{code-cell} ipython3
from typing import List

def subarray_sum_fixed(nums, k):
    window_sum = 0
    for i in range(k):
        window_sum += nums[i]
    largest = window_sum
    for right in range(k, len(nums)):
        left = right - k
        window_sum -= nums[left]
        window_sum += nums[right]
        largest = max(largest, window_sum)
    return largest
```

In a mathematical text, the sliding window algorithm can be represented using
summations and indexing of the input list. Here's how we can explain it:

Consider a sequence of numbers $a = [a_1, a_2, ..., a_n]$ where $n$ is the total
number of elements in the sequence, and a window size $k$.

The sum of elements in the initial window (first $k$ elements) is given by:

$$
S = \sum_{i=1}^{k} a_i
$$

This sum $S$ can also be considered as the maximum sum of subarray of length $k$
found so far.

As we slide the window by one position to the right, the sum of the new window
can be calculated by subtracting the first element of the previous window
($a_i$) and adding the first element not included in the previous window
($a_{i+k}$):

$$
S = S - a_i + a_{i+k}
$$

This operation is repeated until the window reaches the end of the list, i.e.,
$i = n - k + 1$.

At each step, we update the maximum sum found so far:

$$
\text{{max_sum}} = \max(\text{{max_sum}}, S)
$$

This will eventually give us the maximum sum of any subarray of length $k$ in
the sequence $a$. The key advantage of this approach is that each update
operation is done in constant time, leading to a linear time complexity ($O(n)$)
for the entire algorithm.

## The Sliding Window Technique

The sliding window technique, a specific application of the two pointers
technique, is an algorithmic strategy used to solve a variety of problems that
involve searching for subsequences in an array or list that satisfy a particular
condition. The so-called "window" is defined by two pointers or indices into the
array.

### Generalized Sliding Window Technique

Consider a sequence $S = [s_1, s_2, \ldots, s_n]$ of $n$ elements where
$s*i \in \mathbb{R}$ for all $1 \leq i \leq n$. We define two pointers, $i$ and
$j$ (with $1 \leq i \leq j \leq n$), which demarcate the boundaries of the
"window". This window represents a contiguous subsequence
$s_i, s*{i+1}, \ldots, s_j$ of the sequence, or more commonly denoted as:

$$
S[i:j] = [s_i, s*{i+1}, \ldots, s_j]
$$

The objective is often to find a window that satisfies a specific condition
$C(S, i, j)$. This condition is a function of the sequence and the current
window, and varies depending on the problem. At the start, we set $i = j = 1$,
which corresponds to a window that only includes the first element of the
sequence.

We then iteratively adjust the positions of $i$ and $j$ based on the condition
$C$ and the specific problem requirements. This process continues until we meet
a suitable stopping condition. This is often when the right pointer $j$ has
traversed the entire sequence.

### Fixed Window Size Sliding Window Technique

In the fixed-size sliding window technique, we start by determining the window
size $k$. The window is defined by the pointers $i$ and $j$ such that
$j - i + 1 = k$, and it remains constant throughout the algorithm.

The algorithm is as follows:

1. Decide the window size $k$.
2. Initialize $i = 1$ and $j = k$.
3. Repeat the following until a stopping condition is met (usually $j > n$):
    - Evaluate $C(S, i, j)$ and process the window as needed.
    - Increment both $i$ and $j$ synchronously to slide the window.

In this variant, the window slides through the sequence as $i$ and $j$ are
incremented together, maintaining the same size $k$.

### Variable Window Size Sliding Window Technique

In the variable-size sliding window technique, the size of the window can change
dynamically during the execution of the algorithm. When the condition $C$ is
satisfied, we expand the window by incrementing $j$. When the condition $C$ is
not satisfied, we shrink the window from the left by incrementing $i$.

The algorithm is as follows:

1. Initialize $i = j = 1$.
2. Repeat the following until a stopping condition is met (usually $j > n$):
    - If $C(S, i, j)$ is satisfied, increment $j$ to expand the window.
    - Else, increment $i$ to shrink the window.

This variant allows for more flexibility and is often useful in problems where
the optimal window size is not known beforehand and depends on the specific
condition $C$.
