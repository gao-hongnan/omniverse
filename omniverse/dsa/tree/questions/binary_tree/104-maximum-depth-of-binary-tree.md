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

# Maximum Depth of Binary Tree

<a href="https://leetcode.com/problems/maximum-depth-of-binary-tree">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-104-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Easy-green) ![Tag](https://img.shields.io/badge/Tag-BinaryTree-orange)
![Tag](https://img.shields.io/badge/Tag-DFS-orange) ![Tag](https://img.shields.io/badge/Tag-Recursion-orange)
![Tag](https://img.shields.io/badge/Tag-Iterative-orange)

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

Given the `root` of a binary tree, return its _maximum_ depth.

A binary tree's **maximum depth** is the number of nodes along the longest path
from the root node down to the farthest leaf node.

## Example

```{code-cell} ipython3
tree_values = [1, 2, 4, None, None, 5, None, None, 3, None, None]
root = build_binary_tree_from_list_preorder(tree_values)
print_binary_tree(root, node_info=lambda n: (str(n.value), n.left, n.right), is_top=True)
```

Then the maximum depth of the tree is $3$ because the longest path from the root
node down to the farthest leaf node is $3$. One possible path is
$1 \rightarrow
2 \rightarrow 4$.

## Intuition

Think in the perspective of a node in a tree, what do you want to know from your
children? You want to know the maximum depth of your children's subtrees. What
do you want to tell your parents? You want to tell your parents the maximum
depth of your subtree. So the recursion is pretty straightforward:

1. **Parents ask children**: Tell me the maximum depth of your subtree! I do not
   care how you get it, just tell me the maximum depth of your subtree. This
   means finding the longest path from you (my child) to any leaf in your
   subtree (child's children).
2. **Children answer parents**: the longest path from me to any leaf in my
   children is the maximum of the maximum depth of my left subtree and the
   maximum depth of my right subtree plus one.


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

## Solution (Bottom-up Recursive Postorder Traversal)

### Intuition

The provided solution is a **bottom-up** approach, often referred to as
**postorder** because you compute the depth for the children first (i.e., the
left and right subtrees), and then compute the depth for the current node based
on its children.

The recursive call `self.maxDepth(root.left)` is made first, computing the depth
of the entire left subtree before moving on to the right subtree with
`self.maxDepth(root.right)`. Finally, the maximum depth of the two is determined
and incremented by one to account for the current node (`root`), which makes
this a bottom-up computation. Think of our postorder traversal, indeed we visit
the two children of a node before visiting the node itself. In here, we are
computing the depth of the two children before computing the depth of the node
itself.

Top-down approaches, on the other hand, often involve passing down values from
the root to the leaves. For this particular problem (computing tree depth), both
approaches will work.

Let's discuss the base case, the return value, and the state of the recursive
function.

#### Base Case

```{prf:remark} The base case
:label: 104-maximum-depth-of-binary-tree-dfs-bottom-up-base-case

The base case is when the node is `None`, in which case the depth is 0. This
makes sense because the depth of a tree with no nodes is 0.
```

#### Return Value (Passing Values Up From Child To Parent)

```{prf:remark} The return statement's role
:label: 104-maximum-depth-of-binary-tree-dfs-bottom-up-return

Firstly, be very clear the `return` statement's role in the recursive function.
When the `return` statement is returned at a **_particular node_**, it is
returning the maximum depth of the subtree rooted at that node.

For instance, in the running example above, consider the node with value `2`,
when it finishes executing the left subtree of `2`, it returns `1` which is the
maximum depth of the left subtree rooted at `2` (tree `4`). Similarly, when it
finishes executing the right subtree (tree `5`) of `2`, it returns `1` which is
the maximum depth of the right subtree rooted at `2`. Finally, when it finishes
executing the subtree rooted at `2`, it returns `2` which is the maximum depth
of the subtree rooted at `1`.
```

#### States (Passing Values Down From Parent To Child)

```{prf:remark} The state of the recursive function
:label: 104-maximum-depth-of-binary-tree-dfs-bottom-up-state

There isn't a state that's explicitly passed down from parent to child. This is
because the function does not need any additional information from the parent to
perform its computation at the child.

However, the top-down approach does require passing down the current depth from
the parent to the child, which we will see later.
```

### Visualization

The below visual is courtesy of
[the answer here](https://stackoverflow.com/questions/76685730/understanding-the-internal-stack-frames-in-a-recursive-function-call).

Each execution context of a function creates a stackframe. You could imagine the
stack frames as boxes, where each new box is placed on top of a previous box
(stacking them). Names like `root` have a scope and lifetime that is limited to
the "box" they are defined in.

In the following visualisation, execution goes from top to bottom, and smaller
boxes are placed on top of larger boxes, so it is like looking at the stack from
"above":

```none
┌─maxDepth─────────────────────────────────┐
│ root is TreeNode(1)                      │
│ ┌─dfs_postorder(root)──────────────────┐ │
│ │ node is TreeNode(1)                  │ │
│ │ ┌─dfs_postorder(node.left)─────────┐ │ │
│ │ │ node is TreeNode(2)              │ │ │
│ │ │ ┌─dfs_postorder(node.left)─────┐ │ │ │
│ │ │ │ node is TreeNode(4)          │ │ │ │
│ │ │ │ ┌─dfs_postorder(node.left)─┐ │ │ │ │
│ │ │ │ │ node is None             │ │ │ │ │
│ │ │ │ │ return 0                 │ │ │ │ │
│ │ │ │ └──────────────────────────┘ │ │ │ │
│ │ │ │ ┌─dfs_postorder(node.right)┐ │ │ │ │
│ │ │ │ │ node is None             │ │ │ │ │
│ │ │ │ │ return 0                 │ │ │ │ │
│ │ │ │ └──────────────────────────┘ │ │ │ │
│ │ │ │ return max(0, 0) + 1  # 1    │ │ │ │
│ │ │ └──────────────────────────────┘ │ │ │
│ │ │ ┌─dfs_postorder(node.right)────┐ │ │ │
│ │ │ │ node is TreeNode(5)          │ │ │ │
│ │ │ │ ┌─dfs_postorder(node.left)─┐ │ │ │ │
│ │ │ │ │ node is None             │ │ │ │ │
│ │ │ │ │ return 0                 │ │ │ │ │
│ │ │ │ └──────────────────────────┘ │ │ │ │
│ │ │ │ ┌─dfs_postorder(node.right)┐ │ │ │ │
│ │ │ │ │ node is None             │ │ │ │ │
│ │ │ │ │ return 0                 │ │ │ │ │
│ │ │ │ └──────────────────────────┘ │ │ │ │
│ │ │ │ return max(0, 0) + 1  # 1    │ │ │ │
│ │ │ └──────────────────────────────┘ │ │ │
│ │ │ return max(1, 1) + 1  # 2        │ │ │
│ │ └──────────────────────────────────┘ │ │
│ │ ┌─dfs_postorder(node.right)────────┐ │ │
│ │ │ node is TreeNode(3)              │ │ │
│ │ │ ┌─dfs_postorder(node.left)─────┐ │ │ │
│ │ │ │ node is None                 │ │ │ │
│ │ │ │ return 0                     │ │ │ │
│ │ │ └──────────────────────────────┘ │ │ │
│ │ │ ┌─dfs_postorder(node.right)────┐ │ │ │
│ │ │ │ node is None                 │ │ │ │
│ │ │ │ return 0                     │ │ │ │
│ │ │ └──────────────────────────────┘ │ │ │
│ │ │ return max(0, 0) + 1  # 1        │ │ │
│ │ └──────────────────────────────────┘ │ │
│ │ return max(2, 1) + 1  # 3            │ │
│ └──────────────────────────────────────┘ │
│ return 3                                 │
└──────────────────────────────────────────┘
```

What is not shown here is the return address, which is also part of a stack
frame: when an inner function call returns, the calling function context
(preserved in a stack frame) has a trace of where to resume its execution.

When this recursion is "emulated" with an explicit stack and no recursive calls,
it is not uncommon to replace this idea of a "return address" with the "next
task to do", which is visiting the right child. So that is where the idea comes
from to first push the right child (as a deferred task) and then the left child
(as the immediate task).

### Algorithm

#### Pseudocode

````{prf:algorithm} Pseudocode
:label: 104-maximum-depth-of-binary-tree-dfs-bottom-up-pseudocode

Algorithm: `maxDepth(node)`

Input: `node` (root of the binary tree)

Output: `max_depth` (maximum depth of the binary tree)

```
BEGIN maxDepth(node)
    IF node is None THEN
        RETURN 0
    ELSE
        left_depth ← maxDepth(node.left)
        right_depth ← maxDepth(node.right)
        max_depth ← 1 + max(left_depth, right_depth)
    RETURN max_depth
END
CALL maxDepth with the root node as input.
```
````

#### Mathematical Representation

```{prf:algorithm} Mathematical Representation
:label: 104-maximum-depth-of-binary-tree-dfs-bottom-up-mathematical-representation

Define the depth function $f$ for a node $v$ in the binary tree as follows:

$$
f(v) =
    \begin{cases}
      0 & \text{if } v = \emptyset \\
      1 + \max\bigg\{f\Big(\ell(v)\Big), f\Big(r(v)\Big)\bigg\} & \text{otherwise}
    \end{cases}
$$

where

- $\emptyset$ is the empty tree, or `None` in Python
- $v$ is a node in the tree
- $\ell(v)$ is the left child of $v$
- $r(v)$ is the right child of $v$
```

```{admonition} Notation Abuse
:class: dropdown

Note that originally I abused notation:

$$
f(v) =
    \begin{cases}
      0 & \text{if } v = \emptyset \\
      1 + \max\bigg\{f(v_{\ell}), f(v_{r})\bigg\} & \text{otherwise}
    \end{cases}
$$

where

- $\emptyset$ is the empty tree, or `None` in Python
- $v$ is a node in the tree
- $v_{\ell}$ is the left child of $v$
- $v_r$ is the right child of $v$

Compared to our earlier definition, our earlier one is better because
it makes explicit that the functions $f(\ell(v))$ and $f(r(v))$ operate on the
left and right children of $v$, respectively. The use of $\ell(v)$ and $r(v)$ as
functions, rather than subscripts, clarifies that these are operations that take
$v$ as an argument and return its left and right child, respectively.
```

This means that $f(v)$ is 0 if $v$ is $\emptyset$ (the base case), and otherwise
it's 1 plus the maximum of $f$ applied to the left and right children of $v$. This
mirrors the recursive logic of the pseudocode `maxDepth`.

For any binary tree rooted at $r$ with left subtree $l$ and right subtree $r$,
the maximum depth of the tree can be found by computing $f(r)$.

Note that the depth of any tree with a single node (a leaf) is 1, as

$$
f(v) = 1 + \max\left(f(\emptyset), f(\emptyset)\right) = 1.
$$

So the maximum depth of a tree can be found by applying this recursive function
to the root node, which will in turn apply it to all of its descendants. The
depth of the entire tree is then given by the return value of the function when
applied to the root node.

#### Example

If we were to represent the recursive calls to `maxDepth` as a mathematical
function $f$, where $f(v)$ represents the maximum depth of the subtree rooted at
node $v$, then the recursion for the given tree would look like this:

$$
\begin{aligned}
f(1)    &= 1 + \max(f(2), f(3)) \\
        &= 1 + \max(1 + \max(f(4), f(5)), f(3)) \\
        &= 1 + \max(1 + \max(1, 1), f(3)) \\
        &= 1 + \max(2, f(3)) \\
        &= 1 + \max(2, 1) \\
        &= 1 + 2 \\
        &= 3
\end{aligned}
$$

For node 1 (root):

$$
f(1) = 1 + \max(f(2), f(3))
$$

For node 2:

$$
f(2) = 1 + \max(f(4), f(5))
$$

The leaves, nodes 4 and 5, have no children, so their depth is 1:

$$
f(4) = f(5) = 1 + \max(f(\emptyset), f(\emptyset)) = 1
$$

because by definition, $f(\emptyset) = 0$ as specified by the base case.

For node 3:

$$
f(3) = 1
$$

So you can see that $f(v)$ depends on the maximum of $f$ applied to its
children, plus 1 for the current node. This matches the Python code: `maxDepth`
of a node is 1 (for the node itself) plus the maximum of the `maxDepth` of its
children.

The final depth of the tree will be calculated as the value of $f(1)$, which
depends on $f(2)$ and $f(3)$, and so on.

> So the recursive calls for each node are basically asking the left and right
> children for their maximum depth, and then returning the maximum of the two
> plus one for the current node.

### Claim

Claim the algorithm is correct.

### Proof

Prove the correctness of the algorithm.

### Implementation

```{code-cell} ipython3
class Solution:
    def maxDepth(self, root: BinaryTreeNode) -> Union[int, Literal[0]]:
        if not root:
            return 0

        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)
        return max(left_depth, right_depth) + 1

    def maxDepth_no_max(self, root: BinaryTreeNode) -> Union[int, Literal[0]]:
        if not root:
            return 0

        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        if left_depth > right_depth:
            max_depth = left_depth + 1
        else:
            max_depth = right_depth + 1
        return max_depth

    def maxDepth_helper(self, root: BinaryTreeNode) -> Union[int, Literal[0]]:
        def dfs_postorder(root: BinaryTreeNode) -> Union[int, Literal[0]]:
            if not root:
                return 0

            left_depth = dfs_postorder(root.left)
            right_depth = dfs_postorder(root.right)
            return max(left_depth, right_depth) + 1
        return dfs_postorder(root)
```

Now the third solution involves adding a helper function `dfs_postorder` to
perform the recursive computation. This is a common pattern in recursive
solutions, where you have a recursive function that takes in some arguments and
a helper function that does the actual recursion. In this context, the helper
function makes clear to me the following:

1. The helper is a postorder depth-first search traversal of the tree.
2. When you call `maxDepth_helper`, you know that the return is `dfs_postorder`
   of the root node, which is `max(left_depth, right_depth) + 1`. It helps me
   understand the code better.

### Tests

```{code-cell} ipython3
assert Solution().maxDepth(root) == 3
assert Solution().maxDepth_no_max(root) == 3
assert Solution().maxDepth_helper(root) == 3
```

### Time Complexity

TODO.

### Space Complexity

TODO.

## From Recursive to Iterative to Recursive

To visualize the stack calls, we can think of it this way:

```text
maxDepth(1)
    maxDepth(2)
        maxDepth(4)
            maxDepth(None)
            maxDepth(None)
        maxDepth(5)
            maxDepth(None)
            maxDepth(None)
    maxDepth(3)
        maxDepth(None)
        maxDepth(None)
```

so translating this to stack calls, we have:

```text
[1]
[1, 2]
[1, 2, 4]
[1, 2, 4, None]
[1, 2, 4, None, None]
[1, 2, 4, None]
[1, 2, 5]
[1, 2, 5, None]
[1, 2, 5, None, None]
[1, 2, 5, None]
[1, 2]
[1, 2, None]
[1, 2, None, None]
[1, 2, None]
[1, 2]
[1]
[1, 3]
[1, 3, None]
[1, 3, None, None]
[1, 3, None]
[1]
[]
```

```{code-cell} ipython3
from enum import Enum

class State(Enum):
    PROCESS_LEFT = 1
    PROCESS_RIGHT = 2
    UPDATE_MAX_DEPTH = 3

def maxDepth(root):
    if not root:
        return 0

    stack = [(root, State.PROCESS_LEFT, 1)]
    max_depth = 0

    while stack:
        node, state, depth = stack.pop()

        if node is None:
            continue

        if state == State.PROCESS_LEFT:
            stack.append((node, State.PROCESS_RIGHT, depth))
            if node.left:
                stack.append((node.left, State.PROCESS_LEFT, depth + 1))
        elif state == State.PROCESS_RIGHT:
            stack.append((node, State.UPDATE_MAX_DEPTH, depth))
            if node.right:
                stack.append((node.right, State.PROCESS_LEFT, depth + 1))
        else:
            max_depth = max(max_depth, depth)

    return max_depth
```

```{code-cell} ipython3
assert maxDepth(root) == 3
```

Let's go through the process using the same binary tree we've been using for
examples:

```text
    1
   / \
  2   3
 / \
4   5
```

Let's trace the function and call `A` our `State.PROCESS_LEFT`, `B` our
`State.PROCESS_RIGHT`, and `C` our `State.UPDATE_MAX_DEPTH`.

**Step 0:**

-   stack = `[(root, 'A', 1)]` (root is node 1)
-   max_depth = 0

**Step 1:**

-   Pop (node 1, 'A', 1) from stack.
-   node is not None and marker is 'A', so add (node 1, 'B', 1) and (node 2,
    'A', 2) to stack.
-   stack = `[(node 1, 'B', 1), (node 2, 'A', 2)]`
-   max_depth remains 0.

**Step 2:**

-   Pop (node 2, 'A', 2) from stack.
-   node is not None and marker is 'A', so add (node 2, 'B', 2) and (node 4,
    'A', 3) to stack.
-   stack = `[(node 1, 'B', 1), (node 2, 'B', 2), (node 4, 'A', 3)]`
-   max_depth remains 0.

**Step 3:**

-   Pop (node 4, 'A', 3) from stack.
-   node is not None and marker is 'A', so add (node 4, 'B', 3) to stack. (node
    4 doesn't have children, so no additional entries)
-   stack = `[(node 1, 'B', 1), (node 2, 'B', 2), (node 4, 'B', 3)]`
-   max_depth remains 0.

**Step 4:**

-   Pop (node 4, 'B', 3) from stack.
-   node is not None and marker is 'B', so add (node 4, 'C', 3) to stack. (node
    4 doesn't have a right child, so no additional entries)
-   stack = `[(node 1, 'B', 1), (node 2, 'B', 2), (node 4, 'C', 3)]`
-   max_depth remains 0.

**Step 5:**

-   Pop (node 4, 'C', 3) from stack.
-   node is not None and marker is 'C', so max_depth is updated to
    max(max_depth, depth) = max(0, 3) = 3.
-   stack = `[(node 1, 'B', 1), (node 2, 'B', 2)]`
-   max_depth is now 3.

**Step 6:**

-   Pop (node 2, 'B', 2) from stack.
-   node is not None and marker is 'B', so add (node 2, 'C', 2) and (node 5,
    'A', 3) to stack.
-   stack = `[(node 1, 'B', 1), (node 2, 'C', 2), (node 5, 'A', 3)]`
-   max_depth remains 3.

**Step 7:**

-   Pop (node 5, 'A', 3) from stack.
-   node is not None and marker is 'A', so add (node 5, 'B', 3) to stack. (node
    5 doesn't have children, so no additional entries)
-   stack = `[(node 1, 'B', 1), (node 2, 'C', 2), (node 5, 'B', 3)]`
-   max_depth remains 3.

**Step 8:**

-   Pop (node 5, 'B', 3) from stack.
-   node is not None and marker is 'B', so add (node 5, 'C', 3) to stack. (node
    5 doesn't have a right child, so no additional entries)
-   stack = `[(node 1, 'B', 1), (node 2, 'C', 2), (node 5, 'C', 3)]`
-   max_depth remains 3.

**Step 9:**

-   Pop (node 5, 'C', 3) from stack.
-   node is not None and marker is 'C', so max_depth is updated to
    max(max_depth, depth) = max(3, 3) = 3.
-   stack = `[(node 1, 'B', 1), (node 2, 'C', 2)]`
-   max_depth remains 3.

**Step 10:**

-   Pop (node 2, 'C', 2) from stack.
-   node is not None and marker is 'C', so max_depth remains max(max_depth,
    depth) = max(3, 2) = 3.
-   stack = `[(node 1, 'B', 1)]`
-   max_depth remains 3.

**Step 11:**

-   Pop (node 1, 'B', 1) from stack.
-   node is not None and marker is 'B', so add (node 1, 'C', 1) and (node 3,
    'A', 2) to stack.
-   stack = `[(node 1, 'C', 1), (node 3, 'A', 2)]`
-   max_depth remains 3.

**Step 12:**

-   Pop (node 3, 'A', 2) from stack.
-   node is not None and marker is 'A', so add (node 3, 'B', 2) to stack. (node
    3 doesn't have children, so no additional entries)
-   stack = `[(node 1, 'C', 1), (node 3, 'B', 2)]`
-   max_depth remains 3.

**Step 13:**

-   Pop (node 3, 'B', 2) from stack.
-   node is not None and marker is 'B', so add (node 3, 'C', 2) to stack. (node
    3 doesn't have a right child, so no additional entries)
-   stack = `[(node 1, 'C', 1), (node 3, 'C', 2)]`
-   max_depth remains 3.

**Step 14:**

-   Pop (node 3, 'C', 2) from stack.
-   node is not None and marker is 'C', so max_depth remains max(max_depth,
    depth) = max(3, 2) = 3.
-   stack = `[(node 1, 'C', 1)]`
-   max_depth remains 3.

**Step 15:**

-   Pop (node 1, 'C', 1) from stack.
-   node is not None and marker is 'C', so max_depth remains max(max_depth,
    depth) = max(3, 1) = 3.
-   stack is now empty.
-   max_depth remains 3.

Finally, the function will return max_depth, which is 3, as the maximum depth of
the binary tree.

**Final:**

-   The stack is finally empty.
-   max_depth is returned, which is 3. This is the maximum depth of the binary
    tree.

In this way, the iterative depth-first search traversal using a stack
successfully finds the maximum depth of the binary tree. It does so by emulating
the recursive depth-first search process using the stack and a marker to keep
track of where in the process it is for each node.

## Deprecated

```{code-cell} ipython3
:tags: [hide-input]

import time
from typing import Literal, Union

from rich.pretty import pprint

from omnivault.dsa.trees.binary import BinaryTreeNode
from omnivault.dsa.trees.utils import build_binary_tree_from_list_preorder
from omnivault.dsa.trees.utils import print_binary_tree

tree_values = [1, 2, 4, None, None, 5, None, None, 3, None, None]
root = build_binary_tree_from_list_preorder(tree_values)


class Solution:
    def maxDepth_stack(self, root: BinaryTreeNode) -> Union[int, Literal[0]]:
        if not root:
            return 0

        stack = [(1, root)]  # The stack holds tuples of a node and its depth
        max_depth = 0
        count = 0
        while stack:
            count += 1

            # Print the current state of the stack before popping
            print(f"\n\nIteration: {count}")
            pprint(
                f"Stack before popping: {[(node.value if node else None, depth) for depth, node in stack]}"
            )

            # Pop a node from the stack and print it
            depth, node = stack.pop()
            pprint(f"Popped node: {(node.value if node else None, depth)}")

            # Print the state of the stack after popping
            pprint(
                f"Stack after popping: {[(node.value if node else None, depth) for depth, node in stack]}"
            )

            if node:  # If the node is not None
                # Update max_depth if current depth is greater
                max_depth = max(max_depth, depth)
                pprint(f"Updated max depth: {max_depth}")

                # Add the right child and its depth to the stack
                stack.append((depth + 1, node.right))

                # Add the left child and its depth to the stack
                stack.append((depth + 1, node.left))

                # Print the state of the stack after adding the children
                pprint(
                    f"Stack after adding children: {[(node.value if node else None, depth) for depth, node in stack]}"
                )

        return max_depth


class Solution2:
    def maxDepth(self, root) -> int:
        if root is None:
            return 0

        node_stack = [(root, "process")]
        depth_stack = []

        count = 0
        while node_stack:

            node, action = node_stack.pop()

            if action == "process":
                node_stack.append((node, "post_process"))

                if node.right:
                    node_stack.append((node.right, "process"))
                if node.left:
                    node_stack.append((node.left, "process"))
                depth_stack.append(1)
            else:  # action == 'post_process'
                if node.left:
                    depth_stack[-1] = max(depth_stack[-1], 1 + depth_stack.pop())
                if node.right:
                    depth_stack[-1] = max(depth_stack[-1], 1 + depth_stack.pop())
            if count == 0:
                pprint(f"{node.value}: {depth_stack}")
                pprint([(node.value, action) for node, action in node_stack])

                time.sleep(10000)
            count += 1
        return depth_stack[0]


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# fmt: off
from enum import Enum

class State(Enum):
    PROCESS_LEFT = 1
    PROCESS_RIGHT = 2
    UPDATE_MAX_DEPTH = 3

def maxDepth(root):
    if not root:
        return 0

    stack = [(root, State.PROCESS_LEFT, 1)]
    max_depth = 0
    iteration = 1

    while stack:
        print(f"\n\nIteration: {iteration}")
        pprint(f"Stack before= {[(n.value if n else None, s.name, d) for n, s, d in stack]}")

        node, state, depth = stack.pop()

        if node is None:
            continue

        if state == State.PROCESS_LEFT:
            stack.append((node, State.PROCESS_RIGHT, depth))
            if node.left:
                stack.append((node.left, State.PROCESS_LEFT, depth + 1))
        elif state == State.PROCESS_RIGHT:
            stack.append((node, State.UPDATE_MAX_DEPTH, depth))
            if node.right:
                stack.append((node.right, State.PROCESS_LEFT, depth + 1))
        else:
            max_depth = max(max_depth, depth)

        print("Stack after:", [(n.value if n else None, s.name, d) for n, s, d in stack])
        iteration += 1

    return max_depth



# Construct the tree for testing
print(maxDepth(root))  # Output: 3


# print(Solution().maxDepth_stack(root))
# print(Solution2().maxDepth(root))
```

## Solution (Stack)

### Intuition

We can convert the recursion process into an iterative one using a stack data
structure. A stack data structure adheres to the principle of Last-In-First-Out
(LIFO), which means that the most recently added element is the first one to be
removed.

In the context of recursion, the function call stack also operates based on the
LIFO principle. When a function calls itself recursively, each recursive call
(along with any variables local to that call) gets pushed onto a call stack.
When a recursive call finishes, it's popped from the stack, and execution
continues in the previous call.

By utilizing a stack data structure, we can simulate this process iteratively,
effectively converting a recursive function into an iterative one. We manually
push elements onto the stack and pop them off, mirroring the push and pop
operations that would be performed automatically by the call stack in a
recursive function.

> The idea is to keep the next nodes to visit in a stack. Due to the FILO
> behavior of stack, one would get the order of visit same as the one in
> recursion.

### Visualization

TODO.

### Algorithm

### Claim

### Proof

### Implementation

```{code-cell} ipython3
class Solution:
    def maxDepth_stack(self, root: BinaryTreeNode) -> Union[int, Literal[0]]:
        if not root:
            return 0

        stack = [(1, root)] # The stack holds tuples of a node and its depth
        max_depth = 0
        while stack:
            depth, node = stack.pop()
            if node: # If the node is not None
                # Update max_depth if current depth is greater
                max_depth = max(max_depth, depth)

                # Add the right child and its depth to the stack
                stack.append((depth + 1, node.right))

                # Add the left child and its depth to the stack
                stack.append((depth + 1, node.left))

        return max_depth
```

#### The order of visit in the iterative solution is the same as the one in recursion

Earlier, we mentioned that the order of visit in the iterative solution is the
same as the one in recursion. Why and how so?

In the context of a depth-first search (DFS) implemented with a stack, the order
in which nodes are popped off the stack corresponds to the order in which nodes
are "visited" in the traversal.

When you use a stack to implement DFS, you're essentially treating the top of
the stack as your "current" position in the tree. Each time you pop a node off
the stack, you "visit" that node. Then you push its children onto the stack,
according to your chosen order for visiting children.

The "Last-In, First-Out" (LIFO) property of a stack ensures that the children of
the current node (which were the last to be added to the stack) will be the
first ones to be popped off and visited. This effectively simulates the behavior
of a recursive DFS, where you explore as deeply as possible along each branch
before backtracking, thanks to the call stack. In a stack-based DFS, the manual
stack replaces the call stack used in recursion.

So yes, in a DFS implemented with a stack, the order in which nodes are popped
from the stack is the order in which the nodes are visited.

#### Why do we push two children onto the stack when the recursive solution only recurses on one child at a time?

You're correct that the recursive solution doesn't explicitly "push" both
children onto a stack, but the process is similar. When you make a recursive
call, that call and its local context (including parameters and any other local
variables) are implicitly added (or "pushed") onto a system-managed stack known
as the call stack.

In a typical depth-first recursion of a binary tree, such as in the `maxDepth`
function, the process can be described like this:

1. Visit the current node.
2. Recursively visit the left child.
3. Once the left recursion finishes (meaning we've explored as deep as possible
   on the left), recursively visit the right child.

During the recursive process, the current execution context (including the
current node, its depth, and the yet-to-be-explored right child) is saved on the
call stack. This context is then recovered when the left recursive call is done,
which allows the algorithm to then explore the right child. In this sense, it's
like we're "pushing" both children onto the stack, with the left child being
immediately explored and the right child being saved for later.

The iterative solution with an explicit stack is doing essentially the same
thing, but the process is more explicit. It pushes both children onto the stack,
but because of the stack's LIFO nature, the right child is visited after all
nodes in the left subtree have been explored, which aligns with the order of
node visits in the recursive version.

So, although the iterative and recursive versions appear quite different in
code, the way they explore the tree -- and the order in which they visit nodes
-- is fundamentally the same.

Consequently, right after the first iteration we already have something like:

```text
[1, 3, 2]
```

One may be confused why we have `3` inside the stack already when we are just
supposed to be traversing from `1` to `2` first.

While it's true that the node `3` is added to the call stack only after all the
nodes in the left subtree of `1` have been processed, it's important to remember
that the call stack contains paused execution contexts. So even though
`maxDepth(3)` is not called until after `maxDepth(2)`, `maxDepth(1)` (and by
extension, `3`) is still on the call stack during the entire process.

### Tests

```{code-cell} ipython3
assert Solution().maxDepth_stack(root) == 3
```

### Time Complexity

### Space Complexity

## Solution (Top-down Recursive Preorder Traversal)

Add `depth` as an argument to the recursive function.

### Intuition

### Visualization

### Algorithm

### Claim

### Proof

### Implementation

### Time Complexity

### Space Complexity

## Tail Recursion

TODO.

### Intuition

### Visualization

### Algorithm

### Claim

### Proof

### Implementation

### Time Complexity

### Space Complexity

## References and Further Readings

- [Leetcode Solution](https://leetcode.com/problems/maximum-depth-of-binary-tree/editorial/)