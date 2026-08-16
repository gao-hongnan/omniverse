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

# Search in a Binary Search Tree

<a href="https://leetcode.com/problems/search-in-a-binary-search-tree/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-700-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Easy-green) ![Tag](https://img.shields.io/badge/Tag-BinarySearchTree-orange)
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

## Solution (Recursive)

### Intuition

Brief explanation of the core ideas or insights that form the basis for the
solution.

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

#### Pseudocode

#### Mathematical Representation

```{prf:algorithm} Mathematical Representation
:label: 700-search-in-a-binary-search-tree-mathematical-representation

If we were to represent the recursive calls to `searchBST` as a mathematical
function $f(v, x)$, where $f(v, x)$ represents the node in the subtree
rooted at node $v$ with value equal to $x$, then the recursion for the given
binary search tree would look like this:

$$
f(v, x) =
\begin{cases}
      \emptyset & \text{if } v = \emptyset \\
      f\Big(\ell(v), x\Big) & \text{if } g(v) > x \\
      f\Big(r(v), x\Big) & \text{if } g(v) < x \\
      v & \text{if } g(v) = x
\end{cases}
$$

where

-   $\emptyset$ is the empty tree, or `None` in Python
-   $v$ is a node in the tree
-   $\ell(v)$ is the left child of $v$
-   $r(v)$ is the right child of $v$
-   $g(v)$ is the value of node $v$, and $x$ is the target value to search for.

In this formulation, the function $f$ takes two arguments: a node $v$ and a
target value $x$. The function $\ell(v)$ returns the left child of the node $v$
and $r(v)$ returns the right child. Depending on the comparison of the value of
node $v$ and the target $x$, the function $f$ recursively calls itself on either
the left child or the right child of $v$. If $v$ is empty, it returns
$\emptyset$. If the value of node $v$ equals the target $x$, it returns $v$
itself. This function could be seen as a search algorithm on a binary tree.
```

This means the following:

-   Base case is when the node $v$ is empty. This means that if you've reached
    the end of the tree and haven't found a node with value $x$, then return
    `None`.
    -   This is slightly different than recursion with backtracking, where once
        you reach a base case, you backtrack and try a different path.
    -   Here, once you reach a base case, you just return `None` and the
        recursion stops because if you've reached a base case in a BST, then you
        know that the node with value $x$ doesn't exist.
-   If the value of the node $v$ (represented by $g(v)$) is greater than $x$,
    then you know that the node with value $x$ can only exist in the left
    subtree of $v$.
-   If the value of the node $v$ (represented by $g(v)$) is less than $x$, then
    you know that the node with value $x$ can only exist in the right subtree of
    $v$.
-   Take note: if $f(v, x) = v$ is reached at any point, then you know that the
    node with value $x$ exists in the tree. The program will terminate, and node
    $v$ will be returned.

#### Backtracking

Recursion in the context of the `searchBST` function is a form of Depth-First
Search (DFS), but without backtracking. It continually goes deeper into the tree
following a specific path (left or right) until it either finds the target value
or hits a `None` node (the base case), in which case it returns `None`. This is
because once it starts down a path, it doesn't need to backtrack or explore
other paths, because the Binary Search Tree (BST) property ensures that if a
node isn't down a certain path, it won't be found by backtracking and going down
a different path.

In contrast, in algorithms where backtracking is used, the algorithm might need
to "undo" some of its previous choices and try different paths when it realizes
the current path won't lead to a solution. This is common in problems where
there are many potential paths to explore and the algorithm needs to find one or
all paths that meet certain conditions (e.g., Sudoku solvers, N-queens problem,
etc). In these problems, reaching a dead end doesn't mean the solution doesn't
exist in the problem space; it just means the current path doesn't lead to a
solution.

### Claim

Statement claiming the correctness of the algorithm.

### Proof

Proof showing the correctness of the algorithm.

### Implementation

```{code-cell} ipython3
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None

        if root.val > val:
            left_subtree = self.searchBST(root.left, val)
            return left_subtree
        elif root.val < val:
            right_subtree = self.searchBST(root.right, val)
            return right_subtree

        return root
```

#### Mistake

Mistake: Did not return in `if` and `elif`.

In this context, the `return` keyword is used to send back the result of the
recursive calls to `self.searchBST`. Here's how the recursion works:

-   When you call `self.searchBST(root.left, val)` or
    `self.searchBST(root.right, val)`, you're making a recursive call. This
    means you're calling the same function (`searchBST`), but with a different
    argument (either `root.left` or `root.right` instead of `root`).
-   This recursive call will return a value - either a `TreeNode` (if a node
    with the desired value is found) or `None` (if no such node is found).
-   The `return` keyword in the `if` and `elif` branches is necessary to send
    this value back up the call stack. Without the `return` keyword, the
    function would discard the result of the recursive call and continue
    executing, ultimately returning `None` (which is the default return value of
    a Python function if no other return statement is encountered).

In other words, the `return` keyword is necessary to "pass along" the result of
the recursive call. Without it, your function wouldn't be able to correctly
report whether or not it found a node with the desired value.

To give a concrete example, let's say you're searching for the value `5` in a
binary search tree, and the root node has the value `3`. Here's what would
happen:

-   The root node's value is less than `5`, so you make a recursive call with
    `root.right` (the right child of the root).
-   This recursive call returns a `TreeNode` with the value `5`.
-   You need to return this `TreeNode` from the original call to `searchBST`, so
    you use the `return` keyword to send it back up the call stack.

If you omitted the `return` keyword, the `TreeNode` found by the recursive call
would be discarded, and the original call to `searchBST` would return `None`,
incorrectly indicating that no node with the value `5` was found.

```{prf:remark} Mistake in Mathematical Representation
:label: 700-search-in-a-binary-search-tree-mistake-in-mathematical-representation

The function `searchBST` is designed to find a node in a binary search tree
(BST) that has a specified value. It does this by recursively calling itself on
the left or right child of the current node, depending on whether the current
node's value is greater than or less than the desired value.

If you do not return the results of the recursive calls (i.e., `left_subtree`
and `right_subtree`), then the function will not be able to "pass up" the node
it found back to the original call $f(v=root, x)$. This means the function will
always return `None`, even if it found the node you were looking for.

If you were to remove the return statements in the recursive calls,
the functionality of the code would no longer match your mathematical
representation. Currently, the math representation is:

$$
f(v, x) =
\begin{cases}
      \emptyset & \text{if } v = \emptyset \\
      f\Big(\ell(v), x\Big) & \text{if } g(v) > x \\
      f\Big(r(v), x\Big) & \text{if } g(v) < x \\
      v & \text{if } g(v) = x
\end{cases}
$$

If you remove the return statements in the recursive calls, it would change to
something like:

$$
f(v, x) =
\begin{cases}
      \emptyset & \text{if } v = \emptyset \\
      \emptyset & \text{if } g(v) \neq x
\end{cases}
$$

In this case, $f(v, x)$ only returns a node if the node's value is exactly $x$.
Otherwise, it always returns $\emptyset$, regardless of whether $v$ exists
elsewhere in the tree. This is because the return statement only exists in the
base case and at the last line. This would fail to correctly search for a
value in the BST because it doesn't explore the tree beyond the current node
unless the current node's value is exactly $x$.
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

-   <https://leetcode.com/explore/learn/card/introduction-to-data-structure-binary-search-tree/141/basic-operations-in-a-bst/1019/>
- https://leetcode.com/problems/search-in-a-binary-search-tree/editorial/