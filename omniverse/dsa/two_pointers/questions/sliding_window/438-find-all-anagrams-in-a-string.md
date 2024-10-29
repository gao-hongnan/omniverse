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

# Find All Anagrams in a String

<a href="https://leetcode.com/problems/find-all-anagrams-in-a-string/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-438-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)
![Tag](https://img.shields.io/badge/Tag-Sliding_Window-orange)
![Tag](https://img.shields.io/badge/Tag-Hash_Table-orange)
![Tag](https://img.shields.io/badge/Tag-String-orange)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from typing import Optional, List, Union, Any

import rich

import sys
from pathlib import Path
parent_dir = str(Path().resolve().parents[2])
sys.path.append(parent_dir)

from common_utils.tests.core import compare_test_case_dsa, compare_test_cases_dsa
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

Brief explanation of the core ideas or insights that form the basis for the solution.

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

Detailed description of the algorithm used to solve the problem.

### Claim

Statement claiming the correctness of the solution.

### Proof

Proof showing the correctness of the solution.

### Implementation

SEE `tmpp.py`

### Tests

Set of tests for validating the solution.

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
