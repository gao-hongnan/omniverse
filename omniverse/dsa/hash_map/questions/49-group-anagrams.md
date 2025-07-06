---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Group Anagrams

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
[![LeetCode Problem](https://img.shields.io/badge/LeetCode-49-FFA116?style=social&logo=leetcode)](https://leetcode.com/problems/group-anagrams)
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)
![Tag](https://img.shields.io/badge/Tag-String-orange)
![Tag](https://img.shields.io/badge/Tag-Array-orange)
![Tag](https://img.shields.io/badge/Tag-HashMap-orange)
![Tag](https://img.shields.io/badge/Tag-Sorting-orange)

```{contents}
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from typing import Optional, List, Union, Any

from IPython.display import display, HTML
import rich

import sys
from pathlib import Path
```

## Problem

**Problem Statement:** Given an array of strings `strs`, group the anagrams
together. You can return the answer in any order.

## Examples

**Example 1:**

```text
Input: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
Output: [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
Explanation:
- There is no string in `strs` that can be rearranged to form "bat".
- The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
- The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
```

**Example 2:**

```text
Input: strs = [""]
Output: [[""]]
```

**Example 3:**

```text
Input: strs = ["a"]
Output: [["a"]]
```

## Constraints

-   `1 \leq \text{strs.length} \leq 10^4`
-   `0 \leq \text{strs}[i].\text{length} \leq 100`
-   `strs[i]` consists of lowercase English letters.

## Intuition

-   Two strings are anagrams if and only if their sorted strings are equal.
-   Two strings are anagrams if and only if their character counts (respective
    number of occurrences of each character) are the same.

We can then implement two solutions later.

## Solution (Categorize by Sorted String)

### Intuition

Two strings are anagrams if and only if their sorted strings are equal.

### Implementation

Solution without using `defaultdict`:

```{code-cell} ipython3
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        seen: Dict[str, List[str]] = {}
        results: List[List[str]]

        # no need check for 0 length as constraint said so
        if len(strs) == 1:
            return [strs] # if only one string, return it

        for str_ in strs:
            # assumes sorted will sort empty string
            sorted_str = ''.join(sorted(str_))
            if sorted_str not in seen:
                seen[sorted_str] = [str_]
            else:
                seen[sorted_str].append(str_)

        for key, value in seen.items():
            results.append(value)
        return results
```

Solution using `defaultdict`:

```{code-cell} ipython3
from collections import defaultdict
from typing import List, Dict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        seen = defaultdict(list)

        for str_ in strs:
            sorted_str = ''.join(sorted(str_))
            seen[sorted_str].append(str_)

        return list(seen.values())
```

### Tests

```{code-cell} ipython3
print(Solution().groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
```

### Time Complexity

We will use the solution without using `defaultdict` for the time complexity
analysis.

1. Iterating through each string in `strs`: $\mathcal{O}(N)$, where $N$ is the
   number of strings.

2. Sorting each string: Since sorting takes $\mathcal{O}(K \log K)$ time, where
   $K$ is the maximum length of a string, this step contributes a complexity of
   $\mathcal{O}(N \cdot K \log K)$.

    Sorting a sequence of $K$ elements using comparison-based sorting
    algorithms, such as merge sort or quick sort, has a time complexity of
    $\mathcal{O}(K \log K)$. This is because the sorting algorithm divides the
    sequence into two halves, sorts each half recursively, and then merges the
    sorted halves. This leads to a time complexity of $\log K$ levels of
    recursion, and at each level, all $K$ elements are processed, resulting in
    the overall complexity of $\mathcal{O}(K \log K)$.

3. As for the `join` operation, it concatenates a list of strings into a single
   string. In Python, the `join` method is implemented efficiently and takes
   linear time with respect to the total length of the concatenated string. In
   the given code snippet, the `join` operation is applied to a sorted list of
   characters from the original string, so its complexity is $\mathcal{O}(K)$,
   where $K$ is the length of the string.

4. Iterating through the dictionary `seen`: This takes $\mathcal{O}(N)$ time as
   there will be at most $N$ keys.

So, the overall time complexity of this code is:

$$
\begin{aligned}
\mathcal{T}(N, K) &= \mathcal{O}(N) + \mathcal{O}(K \log K) + \mathcal{O}(K) + \mathcal{O}(N) \\
&= \mathcal{O}(N \cdot K \log K)
\end{aligned}
$$

### Space Complexity

We will use the solution without using `defaultdict` for the time complexity
analysis.

#### Input Space Complexity

The input space complexity refers to the space required to store the input
itself. In this case, it's the array of strings `strs`. Since there are $N$
strings and the maximum length of a string is $K$, the input space complexity
is:

$$
\mathcal{O}(N \cdot K)
$$

#### Auxiliary Space Complexity

1. **Space for the `seen` dictionary**: This will store at most $N$ sorted keys
   and values, each having a complexity of $\mathcal{O}(K)$, resulting in a
   complexity of $\mathcal{O}(N \cdot K)$.
2. **Space for the `results` list**: This will store all the values from the
   `seen` dictionary, which again leads to a space complexity of
   $\mathcal{O}(N \cdot K)$.

Since the `results` list in the original code is simply a reorganization of the
`seen` dictionary values, the space complexity for these two structures combined
is:

$$
\mathcal{O}(N \cdot K) + \mathcal{O}(N \cdot K) = \mathcal{O}(N \cdot K)
$$

The `join` operation does not contribute to the space complexity, as it's
working on existing strings and does not create additional space that scales
with the input size. The keys and values in the `seen` dictionary are the
primary factors contributing to the space complexity in this code.

#### Total Space Complexity

Summing the input space complexity (which remains $\mathcal{O}(N \cdot K)$) and
the revised auxiliary space complexity, we get the total space complexity:

$$
\mathcal{O}(N \cdot K) + \mathcal{O}(N \cdot K) = \mathcal{O}(N \cdot K)
$$

Even though we are storing both the `seen` dictionary and the `results` list,
the overall space complexity remains proportional to the product of $N$ and $K$
because they are not storing additional information beyond what's in the input.
Therefore, the space complexity remains $\mathcal{O}(N \cdot K)$.

## Solution (Categorize by Count)

### Intuition

Two strings are anagrams if and only if their character counts (respective
number of occurrences of each character) are the same.

For instance, given the below strings:

```python
strs = ["aab", "aba", "baa", "abcdef"]
```

We can represent them using `ord` as follows:

```python
anagrams = {
    (2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0): ["aab", "aba", "baa"],
    (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ..., 0): ["abcdef"],
}
```

and then return the values of the dictionary as the final result.

This works because the character counts are the same for all anagrams.

### Implementation

```{code-cell} ipython3
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = defaultdict(list)

        for str_ in strs:
            counts = [0] * 26
            for char in str_:
                counts[ord(char) - ord('a')] += 1
            anagrams[tuple(counts)].append(str_)

        return list(anagrams.values())
```

### Tests

```{code-cell} ipython3
print(Solution().groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
```

### Time Complexity

1. **Iterating through Input Strings**: There are $N$ strings, so this takes
   $\mathcal{O}(N)$ time.
2. **Initializing Character Counts**: Initializing the counts takes
   $\mathcal{O}(26)$ time, equivalent to $\mathcal{O}(1)$.
3. **Counting Characters**: Iterating through each character in a string takes
   $\mathcal{O}(K)$ time, where $K$ is the maximum length of a string.
4. **Updating the Anagrams Dictionary**: Updating the dictionary takes constant
   time $\mathcal{O}(1)$ for each string.

Combining these parts, the total time complexity is:

$$
\mathcal{O}(N) + \mathcal{O}(N \cdot K) = \mathcal{O}(N \cdot K)
$$

### Space Complexity

#### Input Space Complexity

The space required to store the input, which consists of $N$ strings, each of
length at most $K$. This gives an input space complexity of:

$$
\mathcal{O}(N \cdot K)
$$

#### Auxiliary Space Complexity

The space required for additional data structures used in the algorithm, not
including the input and output. In this case, it includes the `anagrams`
dictionary and the `counts` array. Since the dictionary stores all $N$ strings
and their corresponding character counts, and the `counts` array takes constant
space, the auxiliary space complexity is:

$$
\mathcal{O}(N \cdot K) + \mathcal{O}(1) = \mathcal{O}(N \cdot K)
$$

#### Total Space Complexity

The sum of the input space and auxiliary space complexities gives the total
space complexity of the algorithm:

$$
\mathcal{O}(N \cdot K) + \mathcal{O}(N \cdot K) = \mathcal{O}(N \cdot K)
$$

In this case, the input space and auxiliary space complexities are the same, so
the total space complexity remains proportional to the product of $N$ and $K$.

## References and Further Readings

-   [Leetcode: Group Anagrams](https://leetcode.com/problems/group-anagrams/)
