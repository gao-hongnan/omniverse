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

# Validate Binary Search Tree

<a href="https://leetcode.com/problems/validate-binary-search-tree/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-98-blue"/></a>
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

We will do the easiest way that minimizes the changes to the binary search tree.

An important observation is for any binary search tree, and consider that it has
$m$ leaf nodes $l_1, l_2, \ldots, l_m$, then for the target node $t$ that is
ready to be inserted, we can insert $t$ into any of the leaf nodes $l_1, l_2,
\ldots, l_m$ and the resulting binary search tree will be the same.

Without loss of generality, we will insert the target node $t$ into the leaf
node $l_1$. Then either $g(t) < g(l_1)$ or $g(t) > g(l_1)$. So if $g(t) < g(l_1)$,
then we will insert $t$ into the left child of $l_1$, and if $g(t) > g(l_1)$,
then we will insert $t$ into the right child of $l_1$.

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


### Claim

Statement claiming the correctness of the algorithm.

### Proof

Proof showing the correctness of the algorithm.

### Implementation

```{code-cell} ipython3
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        nodes = []

        def dfs_inorder(root, nodes):
            if not root:
                return None

            dfs_inorder(root.left, nodes)
            nodes.append(root.val)
            dfs_inorder(root.right, nodes)
            return nodes

        nodes = dfs_inorder(root, nodes)

        min_value = float("-inf")
        for node in nodes:
            if node > min_value:
                min_value = node
            else:
                return False
        return True
```

```{code-cell} ipython3
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def validate(node, lower_limit=float('-inf'), upper_limit=float('inf')):
            if not node:
                return True

            if node.val <= lower_limit or node.val >= upper_limit:
                return False

            left_subtree_is_valid = validate(node.left, lower_limit, node.val)
            right_subtree_is_valid = validate(node.right, node.val, upper_limit)

            return left_subtree_is_valid and right_subtree_is_valid

        return validate(root)
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

- https://leetcode.com/problems/validate-binary-search-tree/editorial/