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

# Binary Tree Paths

<a href="https://leetcode.com/problems/binary-tree-paths/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-257-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Easy-green) ![Tag](https://img.shields.io/badge/Tag-BinaryTree-orange)
![Tag](https://img.shields.io/badge/Tag-DFS-orange) ![Tag](https://img.shields.io/badge/Tag-Recursion-orange)
![Tag](https://img.shields.io/badge/Tag-Iterative-orange) ![Tag](https://img.shields.io/badge/Tag-String-orange)
![Tag](https://img.shields.io/badge/Tag-Backtracking-orange)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Iterator,
    Optional,
    TypeVar,
    List,
    Union,
    TypeVar,
    Optional,
    Tuple,
)
import matplotlib.pyplot as plt
import networkx as nx
import rich
from rich.jupyter import print

import sys
from pathlib import Path

parent_dir = str(Path().resolve().parents[3])
sys.path.append(parent_dir)

from omnivault.dsa.trees.binary import BinaryTreeNode
from omnivault.dsa.trees.utils import build_binary_tree_from_list_preorder
from omnivault.dsa.trees.utils import print_binary_tree
```

## Problem

A clear and concise statement of the problem.

## Example

```{code-cell} ipython3
tree_values = [1, 2, 4, None, None, 5, None, None, 3, None, None]
root = build_binary_tree_from_list_preorder(tree_values)
print_binary_tree(root, node_info=lambda n: (str(n.value), n.left, n.right), is_top=True)
```

Then the paths should be:

```python
["1->2->4", "1->2->5", "1->3"]
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

## Wrong Solution 1

This is wrong because:

-   Mutation of `local_path` is not undone when backtracking.
-   Returning `GLOBAL_PATHS` is awkward.

To correct it, simply replace `local_path` with `local_path.copy()` in each
recursive call.

```{code-cell} ipython3
class Solution:
    def binaryTreePaths(self, root: Optional[BinaryTreeNode]) -> List[str]:

        GLOBAL_PATHS: List[str] = []

        def dfs(
            root: Optional[BinaryTreeNode], local_path: List[str]
        ) -> Union[Literal[None], Literal[not None]]:
            if not root:
                return []

            local_path.append(str(root.val))

            left: Union[Literal[None], Literal[not None]] = dfs(root.left, local_path)
            right: Union[Literal[None], Literal[not None]] = dfs(root.right, local_path)

            if left == [] and right == []:
                joined: str = "->".join(local_path)
                GLOBAL_PATHS.append(joined)

            return GLOBAL_PATHS

        return dfs(root, local_path=[])
```

## Wrong Solution 2

```{code-cell} ipython3
class Solution:
    def binaryTreePaths(self, root: Optional[BinaryTreeNode]) -> List[str]:

        GLOBAL_PATHS: List[str] = []

        def dfs(
            root: Optional[BinaryTreeNode], local_path: List[str]
        ) -> Union[Literal[None], Literal[not None]]:
            if not root:
                return None

            local_path.append(str(root.val))

            left: Union[Literal[None], Literal[not None]] = dfs(
                root.left, local_path.copy()
            )
            right: Union[Literal[None], Literal[not None]] = dfs(
                root.right, local_path.copy()
            )

            if left is None and right is None:
                joined: str = "->".join(local_path)
                GLOBAL_PATHS.append(joined)
            return None

        dfs(root, local_path=[])
        return GLOBAL_PATHS
```

is wrong while

```{code-cell} ipython3
class Solution:
    def binaryTreePaths(self, root: Optional[BinaryTreeNode]) -> List[str]:

        GLOBAL_PATHS: List[str] = []

        def dfs(
            root: Optional[BinaryTreeNode], local_path: List[str]
        ) -> Union[Literal[None], Literal[not None]]:
            if not root:
                return []

            local_path.append(str(root.val))

            left: Union[Literal[None], Literal[not None]] = dfs(
                root.left, local_path.copy()
            )
            right: Union[Literal[None], Literal[not None]] = dfs(
                root.right, local_path.copy()
            )

            if left == [] and right == []:
                joined: str = "->".join(local_path)
                GLOBAL_PATHS.append(joined)
            return None

        dfs(root, local_path=[])
        return GLOBAL_PATHS
```

works. We just changed return `None` to return `[]` in the base case. How come?

When you return `[]` instead of `None`, and you check for
`if left == [] and right == []`, the logic behaves differently.

In the code where you use `[]`, the condition `if left == [] and right == []` is
true only when both the left and right children of the node are `None`. This is
because the function returns `[]` only when it reaches a `None` node, so both
left and right have to be `None` for the condition to be true.

When you replace the `[]` with `None`, the condition
`if left is None and right is None` is true not only when both the left and
right children are `None`, but also when the left or right child is an actual
leaf node itself (with no children). The `None` return value no longer uniquely
identifies a `None` child; it also identifies a leaf node's left and right
children.

> In other words, you returned `None` when you reached a leaf node, `line 24` is
> where you reached a leaf node, and because of this, consider the tree in
> preorder `1-2-None-5-None-None-3-None-None`. When you reach the leaf node `5`,
> indeed you return both `left` and `right` as `None`, but you also return
> `None` for the parent node `2` as well. This is because `left` and `right` are
> both `None` (because they are leaf nodes), so the condition
> `if left is None and right is None` is true, and you return `None` for the
> parent node `2`. This is why you get `1->2` as a path in the output. This will
> cascade up the tree, and even your root node `1` will return `None` as well.

## Solution (Top-down Recursive Preorder Traversal)

### Intuition

The idea is use preorder traversal to traverse the tree, and keep track of
whether the current node is a leaf node or not. If it is a leaf node, then its
left and right children are `None`, and we can add the current path to the
global list of paths.

Consider our earlier example:

```text
    1
   / \
  2   3
 / \
4   5
```

Now just by eyeballing, if we traverse till the leaf node `4`, then we can see
that both its left and right children are `None`, so the call stack could look
like this when it encounter the left node of `4`:

```python
1 -> 2 -> 4 -> None
```

and when you call `dfs` on the node `None`, you hit the base case, and return
`None`. This `None` is returned to the parent node `4`'s `left` call. Now
consider the right node of `4`, the call stack could look like this:

```python
1 -> 2 -> 4 -> None
```

and similarly, when you call `dfs` on the node `None`, you hit the base case,
and return `None`. This `None` is returned to the parent node `4`'s `right`
call.

So since both the left and right children of the node `4` are `None`, the
condition `if left is None and right is None` is true, and so we can add the
current path to the global list of paths. The current path is `1->2->4`.

But the problem here is the call for node `4` ended, so we return something, and
like the previous wrong solution, if you just return `None` as the final return,
then once `4` is popped from the stack, the parent node `2`'s left (`4`) will be
assigned `None`. This is not what we want. We want to return something that is
not `None` because for node `2`, we know that there is already a left child so
it should never fulfill the condition `if left is None and right is None`.

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

1. **Decision Space** $\mathcal{X}$: In the context of this code, $\mathcal{X}$
   represents the set of all possible paths from the root to any node in the
   binary tree. Each node's value is a decision or a choice that is made as the
   DFS progresses.

2. **Constraints** $\mathcal{C}$: The constraints in this problem are implicit
   and involve following the binary tree's structure. You only move from a node
   to its left or right child, and you stop and record the path when you reach a
   leaf node (a node with no left or right child).

3. **Solution Space** $\mathcal{S}$: $\mathcal{S}$ contains all feasible
   solutions that meet the constraints. In this problem, $\mathcal{S}$ is the
   set of all paths from the root to the leaf nodes.

4. **Partial Solution**: A partial sequence of choices, which, in this case, is
   represented by the `local_path` list. It stores the path from the root to the
   current node being visited.

5. **Complete Solution**: A sequence of choices that completely covers the
   solution space, which is the path from the root to a leaf node.

6. **Backtracking**: This is demonstrated by the line `local_path.pop()`, which
   removes the last element from the `local_path` list when a leaf node is
   reached or when backtracking from a subtree. By doing this, the algorithm
   effectively reverts to the previous state and is ready to explore a different
   path.

#### Pseudocode

Detailed description of the algorithm used to solve the problem.

#### Mathematical Representation

Math formulation

#### Correctness

Prove the correctness of the algorithm

### Claim

Statement claiming the correctness of the algorithm.

### Proof

Proof showing the correctness of the algorithm.

### Implementation

We first show a raw, not so optimized implementation of the algorithm.

```{code-cell} ipython3
class Solution:
    def binaryTreePaths(self, root: Optional[BinaryTreeNode]) -> List[str]:

        GLOBAL_PATHS: List[str] = []

        def dfs(
            root: Optional[BinaryTreeNode], local_path: List[str]
        ) -> Union[Literal[None], Literal[not None]]:
            if not root:
                return None

            local_path.append(str(root.val))

            left: Union[Literal[None], Literal[not None]] = dfs(
                root.left, local_path.copy()
            )
            right: Union[Literal[None], Literal[not None]] = dfs(
                root.right, local_path.copy()
            )

            if left is None and right is None:
                joined: str = "->".join(local_path)
                GLOBAL_PATHS.append(joined)

            return not None

        dfs(root, local_path=[])
        return GLOBAL_PATHS
```

1. **Check for Completion**: The code checks for a leaf node (a complete
   solution) by the condition `if left is None and right is None:` and then
   processes the path by joining and appending it to `GLOBAL_PATHS`.

2. **Extend the Partial Solution**: For each node, the code appends the value to
   the `local_path` and recursively calls the `dfs` function on the left and
   right children. This effectively explores all possible choices in the
   decision space.

3. **Prune**: There is no explicit pruning in this code since the binary tree's
   structure naturally guides the search, but one could argue that returning
   early when `not root` acts as a form of pruning by avoiding unnecessary
   exploration.

4. **Backtrack**: As mentioned, the line `local_path.pop()` is where the
   backtracking occurs, resetting the state to explore other paths.

By mapping the code to the formal definitions, readers can see how the abstract
concepts are practically implemented in this specific problem. The systematic
exploration of the binary tree using DFS, combined with backtracking to revert
the state and explore different paths, aligns with the general principles of
backtracking algorithms.

### Tests

Set of tests for validating the algorithm.

### Time Complexity

Analysis of the time complexity of the solution.

### Space Complexity

The `local_path.copy()` method is used to create a copy of the path list when
making the recursive calls. This ensure that the modifications to the path list
in one branch of the recursion do not affect the path list in other branches.

Since the implementation create a new list for each node in the tree, the space
complexity is $\mathcal{O}(n \times d)$, where $n$ is the number of nodes in the
tree, and $d$ is the maximum depth of the tree. In a balanced binary tree, this
would be $\mathcal{O}(n \log n)$, and in a skewed tree (where every parent has
only one child), it would be $\mathcal{O}(n^2)$.

#### Input Space Complexity

Analysis of the space complexity of the input.

#### Auxiliary Space Complexity

Analysis of the space complexity excluding the input and output space.

#### Total Space Complexity

Analysis of the total space complexity of the solution.

## Solution (Top-down Recursive Preorder Traversal with Backtracking)

### Intuition

As we have seen in the previous solution, the `local_path.copy()` method is used
to create a copy of the path list when making the recursive calls. This ensure
that the modifications to the path list in one branch of the recursion do not
affect the path list in other branches.

But this adds to the space complexity of the solution. We can avoid this by
using backtracking. We can remove the last element from the path list when we
are done processing the current node, and this will ensure that the path list
remains the same when we make the recursive calls.

In a way, we can visualize as it follows:

```python
1->2->4
```

is a valid path, and once we finish traversing the left subtree, we remove the
last element from the path list, and we get:

```python
1->2
```

and when we traverse the right subtree, we get:

```python
1->2->5
```

and once we finish traversing the right subtree, we remove the last element from
the path list, and we get:

```python
1->2
```

and since `2` does not fulfill the condition
`if left is None and right is None`, we do not add it to the global list of
paths. But we still need to pop the last element from the path list, and we get:

```python
1
```

and when we traverse the right subtree, we get:

```python
1->3
```

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

#### Pseudocode

Detailed description of the algorithm used to solve the problem.

#### Mathematical Representation

Math formulation

#### Correctness

Prove the correctness of the algorithm

### Claim

Statement claiming the correctness of the algorithm.

### Proof

Proof showing the correctness of the algorithm.

### Implementation

```{code-cell} ipython3
class Solution:
    def binaryTreePaths(self, root: Optional[BinaryTreeNode]) -> List[str]:

        GLOBAL_PATHS: List[str] = []

        def dfs(
            root: Optional[BinaryTreeNode], local_path: List[str]
        ) -> Union[Literal[None], Literal[not None]]:
            if not root:
                return None

            local_path.append(str(root.val))

            left: Union[Literal[None], Literal[not None]] = dfs(root.left, local_path)
            right: Union[Literal[None], Literal[not None]] = dfs(root.right, local_path)

            if left is None and right is None:
                joined: str = "->".join(local_path)
                GLOBAL_PATHS.append(joined)
            local_path.pop()

            return not None

        dfs(root, local_path=[])
        return GLOBAL_PATHS
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
