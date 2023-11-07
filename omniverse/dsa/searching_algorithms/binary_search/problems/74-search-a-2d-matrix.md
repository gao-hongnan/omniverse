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

# Search a 2D Matrix

<a href="https://leetcode.com/problems/search-a-2d-matrix/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-74-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow) ![Tag](https://img.shields.io/badge/Tag-BinarySearch-orange)
![Tag](https://img.shields.io/badge/Tag-Array-orange) ![Tag](https://img.shields.io/badge/Tag-Matrix-orange)

```{contents}
:local:
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
parent_dir = str(Path().resolve().parents[3])
sys.path.append(parent_dir)
```

## Problem

A clear and concise statement of the problem.

## Example

Small example(s) to illustrate the problem.

## Intuition

Brief explanation of the initial thought process for solving the problem.

## Assumptions

List of any assumptions made in the problem solving process.

## Constraints

### What are the Constraints for?

Explanation of the constraints and their impact on the problem and solution.

## Test Cases

Set of test cases for validating the solution.

## Edge Cases

Discussion of any potential edge cases in the problem.

## Walkthrough / Whiteboarding

Detailed walkthrough of the problem-solving process.

## Theoretical Best Time Complexity

Discussion of the theoretical best time complexity for this problem.

## Theoretical Best Space Complexity

Discussion of the theoretical best space complexity for this problem.

## Space-Time Tradeoff

Analysis of the tradeoff between space and time complexity for the problem.

## Solution (Potentially Multiple)

### Intuition

This is a really simple problem if you have solved the basic 1D binary search
problem. The only difference is that you have to perform binary search on each
row of the matrix.

Of course, brute force works just as easily, but since the time complexity
constraint is on the $\log$ scale, we can't afford to do that since looping over
a $m \times n$ matrix takes $\mathcal{O}(mn)$ time. But if we just do binary
search on each row, we can get $\mathcal{O}(m \log n)$ time, which is much
better.

The time complexity of searching in a 2D matrix can be either $m \log n$ or
$\log(mn)$ depending on how the binary search is applied.

If you are applying binary search on each row of the matrix individually, the
time complexity would be $m \log n$. This is because for each of the $m$ rows,
you perform a binary search operation that has a time complexity of $\log n$,
assuming n is the number of columns.

However, if you treat the 2D matrix as a flattened 1D array, the time complexity
of the binary search becomes $\log(mn)$. This is because you're now performing a
single binary search operation over $mn$ elements. To convert between a 2D index
and a 1D index in this case, you can use the formula $index = row * n + column$.

The advantage of this latter approach is that it takes advantage of the fact
that the entire 2D matrix is sorted, not just individual rows or columns, so
you're using more of the information available to you to make the search faster.
This is why you often see $\log(mn)$ used as the time complexity for searching a
sorted 2D matrix.

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

#### Pseudocode

Detailed description of the algorithm used to solve the problem.

#### Mathematical Representation

Math formulation

#### Correctness

...

### Claim

Statement claiming the correctness of the algorithm.

### Proof

Proof showing the correctness of the algorithm.

### Implementation

```{code-cell} ipython3
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            # perform binary search on each row
            l, r = 0, len(row) - 1

            while l <= r:
                m = (l + r) // 2
                if row[m] == target:
                    return True
                elif row[m] < target:
                    l = m + 1
                else:
                    r = m - 1
        return False
```

### Tests

Set of tests for validating the algorithm.

### Time Complexity

Analysis of the time complexity of the solution.

### Space Complexity

#### Input Space Complexity

Analysis of the space complexity of the input.

#### Auxiliary Space Complexity

Analysis of the space complexity excluding the input and output space.

#### Total Space Complexity

Analysis of the total space complexity of the solution.

## References and Further Readings

Any useful references or resources for further reading.
