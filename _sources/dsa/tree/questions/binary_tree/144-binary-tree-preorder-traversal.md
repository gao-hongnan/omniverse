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

# Preorder Traversal

<a href="https://leetcode.com/problems/binary-tree-preorder-traversal/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-144-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Easy-green)
![Tag](https://img.shields.io/badge/Tag-BinaryTree-orange)
![Tag](https://img.shields.io/badge/Tag-DFS-orange)
![Tag](https://img.shields.io/badge/Tag-Recursion-orange)
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

import sys
from pathlib import Path
parent_dir = str(Path().resolve().parents[3])
sys.path.append(parent_dir)

from omnivault.dsa.trees.binary import BinaryTreeNode
from omnivault.dsa.trees.utils import build_binary_tree_from_list_preorder
from omnivault.dsa.trees.utils import print_binary_tree
from omnivault.dsa.utils import compare_test_case, compare_test_cases
```

## Introduction

Preorder traversal is another depth-first traversal method in binary trees. The
nodes are visited in the following order:

1. **Visit** the root.
2. **Traverse** the left subtree in Preorder.
3. **Traverse** the right subtree in Preorder.

Here's how it works:

1. **Visit the root node:** In contrast to inorder and postorder traversals,
   preorder traversal starts by visiting the root node of the current subtree.

2. **Visit the left subtree:** After the root node, the traversal proceeds to
   the left subtree and follows the same set of rules: visiting the root, then
   its left child, and finally its right child. If the left subtree is empty,
   this step is skipped.

3. **Visit the right subtree:** After the left subtree has been completely
   traversed, the algorithm moves to the right subtree and applies the same
   rules: visiting the root, then its left child, and finally its right child.
   If the right subtree is empty, this step is skipped.

It's important to note that steps 2 and 3 are recursive. This means that they
will apply the same three steps to each subtree they visit. This recursion
continues until the algorithm reaches a leaf node, at which point it will return
to the previous node and continue traversing the tree.

## Example

For our example tree:

```bash
      1
   __/ \_
  2      6
 / \    / \
3   4  7   9
   /  /
  5  8
```

The order of nodes **visited** during a preorder traversal of this tree is:

```bash
1-2-3-4-5-6-7-8-9
```

Here's how you'd perform a preorder traversal on this example tree:

1. Visit the root node: `1`.

2. Visit all nodes in the left subtree (using the same method):

    - Visit the root of the left subtree: `2`.
        - Visit all nodes in the left subtree of `2`:
            - Visit the root of this subtree: `3`.
                - Visit all nodes in the left subtree of `3`. The left subtree
                  of `3` is `None`, so there are no nodes to visit.
                - Visit all nodes in the right subtree of `3`. The right subtree
                  of `3` is `None`, so there are no nodes to visit.
                - Now that all nodes in the left and right subtrees of `3` have
                  been visited, the visit to `3` is complete and we return to
                  `2`.
        - Visit all nodes in the right subtree of `2`:
            - Visit the root of this subtree: `4`.
                - Visit all nodes in the left subtree of `4`:
                    - Visit the root of this subtree: `5`.
                        - Visit all nodes in the left subtree of `5`. The left
                          subtree of `5` is `None`, so there are no nodes to
                          visit.
                        - Visit all nodes in the right subtree of `5`. The right
                          subtree of `5` is `None`, so there are no nodes to
                          visit.
                        - Now that all nodes in the left and right subtrees of
                          `5` have been visited, the visit to `5` is complete.
                - Visit all nodes in the right subtree of `4`. The right subtree
                  of `4` is `None`, so there are no nodes to visit.
                - Now that all nodes in the left and right subtrees of `4` have
                  been visited, the visit to `4` is complete.
        - Now that all nodes in the left and right subtrees of `2` have been
          visited, the visit to `2` is complete and we return to `1`.

3. Visit all nodes in the right subtree of `1`:

    - Visit the root of the right subtree: `6`.
        - Visit all nodes in the left subtree of `6`:
            - Visit the root of this subtree: `7`.
                - Visit all nodes in the left subtree of `7`:
                    - Visit the root of this subtree: `8`.
                        - Visit all nodes in the left subtree of `8`. The left
                          subtree of `8` is `None`, so there are no nodes to
                          visit.
                        - Visit all nodes in the right subtree of `8`. The right
                          subtree of `8` is `None`, so there are no nodes to
                          visit.
                        - Now that all nodes in the left and right subtrees of
                          `8` have been visited, the visit to `8` is complete.
                - Visit all nodes in the right subtree of `7`. The right subtree
                  of `7` is `None`, so there are no nodes to visit.
                - Now that all nodes in the left and right subtrees of `7` have
                  been visited, the visit to `7` is complete.
        - Visit all nodes in the right subtree of `6`:
            - Visit the root of this subtree: `9`.
                - Visit all nodes in the left subtree of `9`. The left subtree
                  of `9` is `None`, so there are no nodes to visit.
                - Visit all nodes in the right subtree of `9`. The right subtree
                  of `9` is `None`, so there are no nodes to visit.
                - Now that all nodes in the left and right subtrees of `9` have
                  been visited, the visit to `9` is complete.
        - Now that all nodes in the left and right subtrees of `6` have been
          visited, the visit to `6` is complete.

4. Now that all nodes in the left and right subtrees of `1` have been visited,
   the visit to `1` is complete. We have visited all nodes in the tree.

Therefore, the order of nodes **visited** during a preorder traversal of this
tree is `1-2-3-4-5-6-7-8-9`. The process is recursive, and the same set of rules
is applied to each subtree within the tree. Of course, any recursive process can
also be implemented iteratively, we can implement a preorder traversal of a
binary tree iteratively using a stack.

## Visualization

See
[leetcode's visualization of Preorder](https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/).

## Problem

Given the `root` of a binary tree, return the _preorder traversal of its nodes'
values_.

## Intuition

-   Parents tell children: I am going to introduce my name first, since in
    preorder traversal, the root of the current subtree is visited first. Then,
    if I have a left child, I will ask them to do the same thing: introduce
    themselves and then introduce their own children (if any), in the same
    order. This ensures that we complete the traversal of the left subtree
    before moving to the right. Finally, if I have a right child, I will ask
    them to do the same thing: introduce themselves and then introduce their own
    children (if any), in the same order. This ensures that we traverse the
    right subtree last.

-   Children follow their parent's instructions: Each child node in the tree
    considers itself as the root of its own subtree. They introduce themselves
    first (as instructed), and then if they have a left child, they ask that
    child to introduce themselves and their children (if any), following the
    same order. Finally, if they have a right child, they ask that child to
    introduce themselves and their children (if any), following the same order.

In essence, each node in the tree, whether a parent or a child, follows the same
process: they "introduce" themselves first (meaning they are visited first),
then they traverse their left subtree (if it exists), and finally they traverse
their right subtree (if it exists). This parent-child communication continues
until all nodes in the tree have been visited, ensuring a complete Preorder
Traversal of the tree.

## Assumptions

-   The input is a binary tree where a node can have at most two children.
-   The binary tree does not have to be balanced.
-   The binary tree can be empty, which means the root can be `None`.
    -   So make sure to handle this edge case in your code!
-   Each node contains an integer value.

## Constraints

-   The number of nodes in the tree is in the range:

    $$
    0 \leq \text{number of nodes} \leq 100
    $$

    This constraint specifies the minimum and maximum size of the binary tree
    that our algorithm needs to handle. Knowing this, we can evaluate whether
    our solution would scale efficiently for the largest possible input size.
    For example, a solution with a time complexity of $\mathcal{O}(n)$, where
    $n$ is the number of nodes in the tree, would likely be acceptable.

-   Each node's value is an integer in the range:

    $$
    -100 \leq \text{Node.val} \leq 100
    $$

    This constraint informs us about the range of values that each node in the
    tree can hold. It is important to take this into account when considering
    possible edge cases. However, for this particular problem of preorder
    traversal, the actual values of the nodes do not affect the traversal order.

### What are Constraints for?

In programming problems, **constraints** are given to define the scope and
limits of the problem. They help to determine the feasible approaches and
solutions for the problem by providing information about the range and
characteristics of the input data. They help us anticipate the worst-case
scenarios that our algorithm should be able to handle without leading to
inefficiencies or failures, like time limit exceeded or memory limit exceeded
errors.

## Test Cases

Note carefully `build_binary_tree_from_list_preorder` is a helper function that builds a binary
tree in **preorder**.

1. **Normal case**: A tree with more than one node.

    ```python
    root = build_binary_tree_from_list_preorder(
        [1, 2, 4, None, None, 5, None, None, 3, 6, None, None, 7, None, None]
    )
    assert preorder_traversal(root) == [1, 2, 4, 5, 3, 6, 7]
    ```

2. **Unbalanced tree**: A tree where one side has more nodes than the other
   side.

    ```python
    root = build_binary_tree_from_list_preorder([1, 2, None, 3, None, 4, None, None])
    assert preorder_traversal(root) == [1, 2, 3, 4]
    ```

3. **Tree with single path**: A tree where each parent has only one child.

    ```python
    root = build_binary_tree_from_list_preorder([1, 2, None, 3, None, 4, None, None])
    assert preorder_traversal(root) == [1, 2, 3, 4]
    ```

4. **Tree with duplicate values**: A tree with duplicate values in nodes.

    ```python
    root = build_binary_tree_from_list_preorder([1, 1, None, 1, None, 1, None, 1, None, None])
    assert preorder_traversal(root) == [1, 1, 1, 1, 1]
    ```

## Edge Cases

1. **Empty tree**: A tree with no nodes. This is a valid input and should return
   an empty list.

    ```python
    root = build_binary_tree_from_list_preorder([])
    assert preorder_traversal(root) == []
    ```

2. **Single node**: A tree with only one node. This edge case tests whether the
   function can handle the smallest non-empty tree.

    ```python
    root = build_binary_tree_from_list_preorder([1])
    assert preorder_traversal(root) == [1]
    ```

3. **Tree with maximum allowed nodes**: This edge case tests if the function can
   handle the largest possible tree within the constraints. Given the constraint
   that each node's value is unique, the input list will follow the pattern of
   descending to the leftmost node and then filling in the right subtree at each
   level before moving down to the next level.

    ```python
    root = build_binary_tree_from_list_preorder(list(range(99, -1, -1)))
    assert preorder_traversal(root) == list(range(99, -1, -1))
    ```

4. **Tree with minimum and maximum allowed node values**: This edge case tests
   if the function can handle the smallest and largest possible node values
   within the constraints.

    ```python
    root = build_binary_tree_from_list_preorder([-100, None, 100])
    assert preorder_traversal(root) == [-100, 100]
    ```

## Walkthrough / Whiteboarding

Consider the binary tree below:

```bash
    1
   / \
  2   3
 / \
4   5
```

The preorder traversal sequence for this tree is `1, 2, 4, 5, 3`.

1. First, visit the root node `1`.
2. Traverse its left subtree. Visit `2`, then go left and visit `4`, then go
   left and since the left is `None`, we return to `4` and go right and since
   the right is `None`, we return to `2` and go right and visit `5`.
3. Similarly, we found the left and right of `5` to be `None`, so we return to
   `2` and then since we are done with `2`, we return to `1`.
4. As of now, we completed the left subtree, return to the root node and now
   traverse the right subtree. Visit `3`.
5. No more nodes left to visit, hence the traversal is complete.

(144-binary-tree-preorder-traversal.html#theoretical-best-time-complexity)=
## Theoretical Best Time Complexity

In the context of tree traversal algorithms (inorder, preorder, postorder),
every node in the tree is encountered at most twice: once on the way down the
tree, and possibly once more on the way back up.

1. **On the way down:** As we traverse from the root of the tree to the leaves,
   we pass through each node once. At this time, we may perform some operation
   (such as printing the node's value, depending on the specific traversal
   algorithm).

2. **On the way back up:** After we reach a leaf node, we backtrack up the tree
   to find any remaining nodes that haven't been visited. When backtracking, we
   might pass through a node that we visited on the way down. However, when we
   backtrack through a node, we don't perform the operation again; we just pass
   through it on our way back to a node that we haven't fully processed.

Therefore, each node is encountered at most twice: once on the way down, and
possibly once more on the way back up. However, the operation (e.g., printing
the node's value) is only performed once for each node.

Because each node is processed once, and each processing operation takes
constant time[^constant] ($\mathcal{O}(1)$), the overall time complexity for the
traversal is linear in the number of nodes, i.e., $\mathcal{O}(n)$, where n is
the number of nodes in the tree. This is true regardless of whether the tree is
balanced or unbalanced, and whether it's a binary search tree or not.

To be more precise,

1. **Node Visit:** Each node in the tree needs to be visited exactly once to
   fulfill the preorder traversal. If there are $n$ nodes in the tree, this
   contributes $n$ operations.

2. **Backtracking:** Once the left and right subtrees of a node are fully
   traversed, we need to backtrack to the parent node to continue the traversal.
   For every node except the root, there will be a backtracking operation. So,
   this gives us $n-1$ operations.

Therefore, the total number of operations required for a binary tree preorder
traversal is

$$
\begin{aligned}
\mathcal{T}(n) &= \mathcal{O}(n) + \mathcal{O}(n-1) \\
&= \mathcal{O}(2n - 1) \\
&\approx \mathcal{O}(n)
\end{aligned}
$$

However, when considering time complexity, we focus on the highest order term
and ignore the constants and lower-order terms as they become insignificant when
$n$ becomes very large. Thus, we say that the time complexity of the binary tree
preorder traversal is $\mathcal{O}(n)$.

Consequently, the theoretical best time complexity for this problem is
$\mathcal{O}(n)$, where $n$ is the number of nodes in the binary tree.

## Theoretical Best Space Complexity

The theoretical best (auxiliary) space complexity for this problem is
$\mathcal{O}(n)$, where $n$ is the number of nodes in the binary tree. This is
due to the space required by the call stack (assuming a recursive
implementation), which grows proportional to the maximum height of the tree
(which in worst case could be $n$), and the space required to store the output,
which would be equal to the number of nodes in the binary tree.

## Space-Time Tradeoff

In this problem, there's not much of a traditional space-time tradeoff as in
some other problems. The traversal of the binary tree is a linear operation with
respect to the number of nodes and the space required is also proportional to
the number of nodes. The most efficient solution will always involve visiting
each node in the tree once, and the space used is necessary to store the output
and accommodate the recursion stack (in case of recursive approach). However, an
iterative solution using an explicit stack would have the same space complexity
due to the need of storing nodes in a stack data structure.

## Solution (Recursive)

### Implementation

```{code-cell} ipython3
class Solution1:
    def preorderTraversal(self, root: Optional[BinaryTreeNode]) -> List[int]:
        preorder: List[int] = []

        def traverse(root: Optional[BinaryTreeNode], preorder: List[int]) -> List[int]:
            if not root:
                # return [] is more accurate since it syncs up with the leetcode
                # test cases
                return preorder

            preorder.append(root.value)
            traverse(root.left, preorder)
            traverse(root.right, preorder)
            return preorder

        preorder = traverse(root, preorder)
        return preorder

class Solution2:
    def __init__(self) -> None:
        self.preorder: List[int] = []

    def reset(self) -> None:
        """To reset if the class instance were to traverse a new root."""
        self.preorder = []

    def preorderTraversal(self, root: Optional[BinaryTreeNode]) -> List[int]:
        if not root:
            return  # default returns None

        self.preorder.append(root.value)
        self.preorderTraversal(root.left)
        self.preorderTraversal(root.right)

        return self.preorder
```

### Tests

```{code-cell} ipython3
preorder_traversal = Solution1().preorderTraversal

# Helper function
def print_and_test(
    root: Optional[BinaryTreeNode],
    expected_output: List[int],
    case_name: str,
    print_tree: bool = True,
):
    if print_tree:
        print(case_name)
        lines = print_binary_tree(
            root, node_info=lambda n: (str(n.value), n.left, n.right), is_top=False
        )
        print("\n".join(lines))
    compare_test_case(preorder_traversal(root), expected_output, case_name)

# Normal case
print_and_test(
    build_binary_tree_from_list_preorder(
        [1, 2, 4, None, None, 5, None, None, 3, 6, None, None, 7, None, None]
    ),
     [1, 2, 4, 5, 3, 6, 7],
    "Normal case",
)

# Unbalanced tree
print_and_test(
    build_binary_tree_from_list_preorder([1, 2, None, 3, None, None, 4]),
    [1, 2, 3, 4],
    "Unbalanced tree",
)

# Tree with single path
print_and_test(
    build_binary_tree_from_list_preorder([1, 2, None, 3, None, None, 4]),
    [1, 2, 3, 4],
    "Tree with single path",
)

# Tree with duplicate values
print_and_test(
    build_binary_tree_from_list_preorder([1, 1, 1, None, None, 1, None, None, 1]),
    [1, 1, 1, 1, 1],
    "Tree with duplicate values",
)

# Empty tree
print_and_test(build_binary_tree_from_list_preorder([]), [], "Empty tree", print_tree=False)

# Single node
print_and_test(build_binary_tree_from_list_preorder([1]), [1], "Single node")

# Tree with maximum allowed nodes (0-99)
print_and_test(
    build_binary_tree_from_list_preorder(list(range(100))),
    list(range(100)),
    "Tree with maximum allowed nodes",
    print_tree=False
)

# Tree with minimum and maximum allowed node values
print_and_test(
    build_binary_tree_from_list_preorder([-100, None, 100]),
    [-100, 100],
    "Tree with minimum and maximum allowed node values",
)
```

(144-binary-tree-preorder-traversal.html#recursion-time-complexity)=
### Time Complexity

In the context of trees, let $n$ be the **size** of the tree, i.e., the number
of nodes in the tree.

This recursive formula attempt to represent the running time of a
recursive algorithm that splits its input size in half at each level of
recursion, similar to a binary search or a merge sort algorithm.

We can describe the time complexity of binary tree traversal using the equation

$$
\mathcal{T}(n) = 2\mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(1),
$$

where $n$ is the number of nodes, and we assume for simplicity that the tree is
balanced (i.e., each node has two children).

This equation states that to traverse a tree with $n$ nodes, we need to traverse
two smaller trees each of size $\frac{n}{2}$ (the left and right subtrees), and
perform a constant amount of work $\mathcal{O}(1)$ for the current node (adding
the node's value to the output list).

If we expand this equation, it becomes:

$$
\begin{aligned}
\mathcal{T}(n) &= 2\mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(1) \\
&= 2\left(2\mathcal{T}\left(\frac{n}{4}\right) + \mathcal{O}(1)\right) + \mathcal{O}(1) \\
&= 4\mathcal{T}\left(\frac{n}{4}\right) + 2 \cdot \mathcal{O}(1) + \mathcal{O}(1) \\
&= 4\left(2\mathcal{T}\left(\frac{n}{8}\right) + \mathcal{O}(1)\right) + 2 \cdot \mathcal{O}(1) + \mathcal{O}(1) \\
&= 8\mathcal{T}\left(\frac{n}{8}\right) + 4 \cdot \mathcal{O}(1) + 2 \cdot \mathcal{O}(1) + \mathcal{O}(1) \\
&= \vdots \\
&= n\mathcal{T}(1) + \log(n) \cdot \mathcal{O}(1) \\
&= n \cdot \mathcal{O}(1) + \log(n) \cdot \mathcal{O}(1) \\
&= \mathcal{O}(n) + \mathcal{O}(\log n) \\
&= \mathcal{O}(n)
\end{aligned}
$$

This shows that the time complexity of binary tree traversal is
$\mathcal{O}(n)$, because the $\mathcal{O}(n)$ term dominates the
$\mathcal{O}(\log n)$ term for large $n$.

```{list-table} Time Complexity of Preorder Traversal Recursive Algorithm
:header-rows: 1
:name: preorder-recursion-traversal-time-complexity

* - Case
  - Worst Case
  - Average Case
  - Best Case
* - Empty Tree
  - $\mathcal{O}(1)$
  - $\mathcal{O}(1)$
  - $\mathcal{O}(1)$
* - Balanced Tree
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
* - Degenerate Tree
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
* - Random Tree
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
  - $\mathcal{O}(n)$
```

As one can see, besides the special case of an empty tree, the time complexity
is $\mathcal{O}(n)$ for all cases.

### Space Complexity

To find the space complexity, we first let $n$ be the **size** of the tree, i.e.,
the number of nodes in the tree. Furthermore, let $h$ be the **height** of the
tree, i.e., the number of levels in the tree.

We will subsequently derive the input space complexity and auxiliary space
complexity, and then combine them to find the total space complexity.

#### Input Space Complexity

The input to a tree traversal problem is typically the root of the tree,
represented as an instance of a Node class or similar structure. Each node of
the tree contains a value and pointers (or references) to its child nodes.

However, when we discuss the "space complexity of the input", we're referring to
the total amount of space taken up by the input data in memory. For a tree, this
includes the space required to store all of the nodes in the tree, including
their values and the pointers to their child nodes.

So, in the case of a tree with $n$ nodes, the input space complexity is indeed
$\mathcal{O}(n)$. This is because you need space to store each of the nodes
themselves, which includes their value and the pointers to their children. Each
node contributes a constant amount of space, so the total space for $n$ nodes is
proportional to $n$, hence $\mathcal{O}(n)$.

It's important to note that this is separate from the auxiliary space
complexity, which is the extra space used by the algorithm (such as the space
used by the function call stack in a recursive traversal).

#### Auxiliary Space Complexity

Let's reconsider auxiliary space complexity in terms of the tree's height
$h$ and discuss worst-case scenarios:

The auxiliary space complexity concerns the extra space (apart from the input
space) that an algorithm needs to execute. In a recursive tree traversal, the
primary use of extra space comes from the function call stack, which grows with
the depth of the recursive calls.

-   **Balanced binary tree**: In a balanced binary tree, the tree's height is
    approximately $h=\log(n)$, where $n$ is the number of nodes in the tree. The
    depth of recursive calls, in this case, is equal to the height of the tree,
    so the auxiliary space complexity is $\mathcal{O}(h) = \mathcal{O}(\log n)$.

-   **Degenerate or "vine" tree**: In a degenerate tree, each parent node only
    has one child, so the tree resembles a linked list with $n$ nodes. Here, the
    tree's height equals the number of nodes in the tree, i.e., $h = n$. Because
    the recursive calls trace down this linked list-like structure, the maximum
    depth of recursion is $h$ (which equals $n$ in this case). Therefore, the
    auxiliary space complexity is $\mathcal{O}(h)$, which is also
    $\mathcal{O}(n)$.

-   **Empty tree**: There are no nodes to visit in an empty tree, hence no
    recursion, and the auxiliary space complexity is $\mathcal{O}(1)$.

In the worst-case scenario, which could occur in a degenerate tree, the height
of the tree is $h=n$, resulting in a space complexity of $\mathcal{O}(n)$. Even in
a balanced binary tree, if we consider a skewed distribution of nodes (where one
child subtree is minimal while the other has the bulk of the nodes), the
recursion depth could reach $n$, making the worst-case space complexity
$\mathcal{O}(n)$. This is the same as the space complexity for the degenerate
tree, reflecting that the worst-case space complexity for tree traversal is
dependent on the maximum depth of the tree, which is $h$ (and can be $n$ in the
worst case).

```{list-table} Auxiliary Space Complexity of Preorder Traversal Recursive Algorithm
:header-rows: 1
:name: preorder-recursion-traversal-auxiliary-space-complexity

* - Case
  - Space Complexity
* - Empty Tree
  - $\mathcal{O}(1)$
* - Balanced Tree
  - $\mathcal{O}(h) = \mathcal{O}(\log n)$
* - Degenerate Tree
  - $\mathcal{O}(h) = \mathcal{O}(n)$
```

#### Total Space Complexity

The total space complexity is the combination of input space complexity and
auxiliary space complexity. However, it's worth noting that since we are summing
two asymptotic notations, we should take the maximum of the two when presenting
the final complexity, not just the summation.

Let's consider each case for the preorder traversal:

-   For an **empty tree**, the total space complexity is $\mathcal{O}(1)$ (input
    space complexity, as there's no node to store) + $\mathcal{O}(1)$ (auxiliary
    space complexity, as there's no recursion) = $\mathcal{O}(1)$.

-   For a **balanced binary tree**, the total space complexity is
    $\mathcal{O}(n)$ (input space complexity, as there are $n$ nodes) +
    $\mathcal{O}(\log n)$ (auxiliary space complexity, as the recursion depth is
    equal to the tree height, which is $\log n$ for a balanced tree). As $n$
    grows, $\mathcal{O}(n)$ becomes the dominant term, hence the total space
    complexity is $\mathcal{O}(n)$.

-   For a **degenerate tree**, the total space complexity is $\mathcal{O}(n)$
    (input space complexity, as there are $n$ nodes) + $\mathcal{O}(n)$
    (auxiliary space complexity, as the recursion depth is equal to the tree
    height, which is $n$ for a degenerate tree). Hence the total space
    complexity is $\mathcal{O}(n)$.

So we can summarize these scenarios in a table:

```{list-table} Total Space Complexity of Preorder Traversal Recursive Algorithm
:header-rows: 1
:name: preorder-recursion-traversal-total-space-complexity

* - Case
  - Total Space Complexity
* - Empty Tree
  - $\mathcal{O}(1)$
* - Balanced Tree
  - $\mathcal{O}(n)$
* - Degenerate Tree
  - $\mathcal{O}(n)$
```

In all cases, we are taking the maximum of the input space complexity and
auxiliary space complexity to derive the total space complexity. Note that I've
simplified the table and didn't separate it into worst, average, and best case
because these complexities don't change based on the best/worst/average case
scenario in this context.

## Solution (Iterative)

TODO.

1. `[1]` - Initially, we have only the root node in the stack. We pop `1`.
2. `[6, 2]` - After visiting the root node, we push its right child (`6`) first,
   then its left child (`2`). We pop `2`.
3. `[6, 4, 3]` - We visit `2`, then push its right child (`4`), then its left
   child (`3`). We pop `3`.
4. `[6, 4]` - We visit `3` and it has no children, so the stack remains the
   same. We pop `4`.
5. `[6, 5]` - We visit `4`, it only has left child (`5`), so we push `5`. We pop
   `5`.
6. `[6]` - We visit `5` and it has no children, so the stack remains the same.
   We pop `6`.
7. `[9, 7]` - We visit `6`, then push its right child (`9`), then its left child
   (`7`). We pop `7`.
8. `[9, 8]` - We visit `7`, it only has left child (`8`), so we push `8`. We pop
   `8`.
9. `[9]` - We visit `8` and it has no children, so the stack remains the same.
   We pop `9`.
10. `[]` - Finally, we visit `9` and it has no children, so the stack is empty.

The output list would look like this after each iteration:

1. `[1]` - After visiting `1`
2. `[1, 2]` - After visiting `2`
3. `[1, 2, 3]` - After visiting `3`
4. `[1, 2, 3, 4]` - After visiting `4`
5. `[1, 2, 3, 4, 5]` - After visiting `5`
6. `[1, 2, 3, 4, 5, 6]` - After visiting `6`
7. `[1, 2, 3, 4, 5, 6, 7]` - After visiting `7`
8. `[1, 2, 3, 4, 5, 6, 7, 8]` - After visiting `8`
9. `[1, 2, 3, 4, 5, 6, 7, 8, 9]` - After visiting `9`

This accounts for the process of popping each node off the stack and visiting
it. The preorder traversal always visits the current node before its children,
and it always visits the left child before the right child.

### Time Complexity

### Space Complexity

## Solution (Morris Traversal)

TODO.

### Time Complexity

### Space Complexity

## References and Further Readings

- **[Leetcode Solution](https://leetcode.com/problems/binary-tree-preorder-traversal/editorial)**
- **[Leetcode Card: Traverse a Tree](https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/1036/)**
- **[Tree Traversal Time Complexity](https://www.baeldung.com/cs/tree-traversal-time-complexity)**
- **[Time Complexity of Binary Tree Preorder](https://stackoverflow.com/questions/59233720/time-complexity-of-binary-tree-traversal-pre-order)**

[^constant]:
    In our traversal, we are either printing or appending to a list, which are
    both constant time operations. If you do something else, like inserting into
    a list, then the time complexity would be different.
