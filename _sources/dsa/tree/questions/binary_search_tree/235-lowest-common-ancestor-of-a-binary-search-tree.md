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

# Lowest Common Ancestor of a Binary Search Tree

<a href="https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-235-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow) ![Tag](https://img.shields.io/badge/Tag-BinarySearchTree-orange)
![Tag](https://img.shields.io/badge/Tag-DFS-orange) ![Tag](https://img.shields.io/badge/Tag-Recursion-orange)
![Tag](https://img.shields.io/badge/Tag-Iterative-orange)

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

from omnivault.dsa.trees.binary import BinaryTreeNode
from omnivault.dsa.trees.utils import build_binary_tree_from_list_preorder
from omnivault.dsa.trees.utils import print_binary_tree
from omnivault.dsa.utils import compare_test_case, compare_test_cases
```

## Problem

A clear and concise statement of the problem.

## Example

```{code-cell} ipython3
tree_values = [4, 2, 1, None, None, 3, None, None, 7, None, None]
root = build_binary_tree_from_list_preorder(tree_values)
print_binary_tree(root, node_info=lambda n: (str(n.value), n.left, n.right), is_top=True)
```

## Intuition

Here are key observations that can be made about the problem:

1. If $p \leq r \leq q$ or $q \leq r \leq p$, then $r$ is the lowest common
   ancestor of $p$ and $q$.
    1. What this means is that if the very root node of the tree is between $p$
       and $q$, then the root node is the lowest common ancestor.
    2. This observation stems from the fact that if $p$ and $q$ are on opposite
       sides of $r$, then $r$ is the lowest common ancestor. You can prove this
       by contradiction.
2. Otherwise, if $p < r$ and $q < r$, then the lowest common ancestor of $p$ and
   $q$ is in the left subtree of $r$.
    1. In fact you do not need to check both, just one is sufficient because of
       the first observation.
3. Otherwise, if $p > r$ and $q > r$, then the lowest common ancestor of $p$ and
   $q$ is in the right subtree of $r$.
    1. In fact you do not need to check both, just one is sufficient because of
       the first observation.

if not, p and q must either exist in left or right subtree of root. Now since
the node can be its own ancestor it just needs us to find out the first
encounter of p or q in the left or right subtree, once found that node is the
LCA. Why? Because once you contain the problem to only left or right, assume
without loss of generality that it is the left subtree, then the first encounter
say it is $p$, then we know that $q$ must be somewhere below $p$, so $p$ is the
lowest common ancestor of $p$ and $q$.

This is hinged upon the assumption $p$ and $q$ are in the tree (exists). See
constraints.

Another part is

```python
        if root == p or root == q:
            return root
```

How I arrived is base case, so initially the drill is `if not root: return` but
here it is not case cause assumption says number of nodes is at least 2.

So I came up with this. And it turns out it also fulfills my logic on if we
first encounter p or q in the left or right subtree, then that node is the LCA.

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

Brief explanation of the core ideas or insights that form the basis for the
solution.

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

#### Pseudocode

#### Mathematical Representation

#### Correctness Proof

### Implementation

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

-   <https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/editorial>
