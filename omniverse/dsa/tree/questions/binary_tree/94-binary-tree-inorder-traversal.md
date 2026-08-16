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

# Inorder Traversal

<a href="https://leetcode.com/problems/binary-tree-inorder-traversal/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-94-blue"/></a>
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

## Introduction

Inorder traversal is a depth-first traversal method used in binary trees where
nodes are visited in the following order:

1. **Traverse** the left subtree in Inorder.
2. **Visit** the root.
3. **Traverse** the right subtree in Inorder.

The process is as follows:

1. **Visit the left subtree:** The traversal begins with the left subtree,
   following the same set of rules: visiting its left child, then the root, and
   finally its right child. If the left subtree is empty, this step is skipped.

2. **Visit the root node:** After the left subtree has been completely
   traversed, the algorithm visits the root node of the current subtree.

3. **Visit the right subtree:** After visiting the root node, the traversal
   moves to the right subtree and applies the same set of rules: visiting its
   left child, then the root, and finally its right child. If the right subtree
   is empty, this step is skipped.

As with postorder traversal, the traversal of the left and right subtrees
involves recursion, applying the same three steps to each subtree visited. The
recursion continues until a leaf node is reached, marking the base case for the
recursion.

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

The order of nodes **visited** during a inorder traversal of this tree is:

```bash
3-2-5-4-8-1-7-6-9
```

Here's how you'd perform an inorder traversal on this example tree::

1. Visit all nodes in the left subtree of `1`:
    - Visit all nodes in the left subtree of `2`:
        - Visit all nodes in the left subtree of `3`. The left subtree of `3` is
          None, so there are no nodes to visit.
        - Visit the root of this subtree: `3`.
        - Visit all nodes in the right subtree of `3`. The right subtree of `3`
          is None, so there are no nodes to visit.
    - Visit the root of the left subtree: `2`.
    - Visit all nodes in the right subtree of `2`:
        - Visit all nodes in the left subtree of `4`.
            - Visit the root of this subtree: `5`.
                - Visit all nodes in the left subtree of `5`. The left subtree
                  of `5` is None, so there are no nodes to visit.
                - Visit the root of this subtree: `5`.
                - Visit all nodes in the right subtree of `5`. The right subtree
                  of `5` is None, so there are no nodes to visit.
        - Visit the root of this subtree: `4`.
        - Visit all nodes in the right subtree of `4`. The right subtree of `4`
          is None, so there are no nodes to visit.
2. Visit the root node: `1`.
3. Visit all nodes in the right subtree of `1`:
    - Visit all nodes in the left subtree of `6`:
        - Visit all nodes in the left subtree of `7`.
            - Visit the root of this subtree: `8`.
                - Visit all nodes in the left subtree of `8`. The left subtree
                  of `8` is None, so there are no nodes to visit.
                - Visit the root of this subtree: `8`.
                - Visit all nodes in the right subtree of `8`. The right subtree
                  of `8` is None, so there are no nodes to visit.
        - Visit the root of this subtree: `7`.
        - Visit all nodes in the right subtree of `7`. The right subtree of `7`
          is None, so there are no nodes to visit.
    - Visit the root of the right subtree: `6`.
    - Visit all nodes in the right subtree of `6`:
        - Visit all nodes in the left subtree of `9`. The left subtree of `9` is
          None, so there are no nodes to visit.
        - Visit the root of this subtree: `9`.
        - Visit all nodes in the right subtree of `9`. The right subtree of `9`
          is None, so there are no nodes to visit.
4. Now that all nodes in the left and right subtrees of `1` have been visited
   and `1` itself has been visited, the traversal is complete.

Therefore, the order of nodes **visited** during an in-order traversal of this
tree is `3-2-5-4-1-8-7-6-9`. The process is recursive, and the same set of rules
is applied to each subtree within the tree. As with preorder, any recursive
process can also be implemented iteratively; we can implement an in-order
traversal of a binary tree iteratively using a stack.

## Visualization

See
[leetcode's visualization of Inorder](https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/).

## Problem

Given the `root` of a binary tree, return the _inorder traversal of its nodes'
values_.

## Intuition

In an in-order traversal, the intuition can be understood through the family
gathering analogy again, but this time with a different rule of introduction.

-   Parents tell children: "Before I introduce myself, I will ask my left child
    (if any) to introduce themselves first, along with their own children (if
    any), following the same rule. **Only after the entire left subtree has been
    introduced will I introduce myself**. Then, if I have a right child, I will
    ask them to do the same thing: introduce themselves and their own children
    (if any), in the same order. This ensures that the right subtree
    introductions happen after I introduce myself."

-   Children follow their parent's instructions: Each child node in the tree
    considers itself as the root of its own subtree. They first ask their left
    child (if any) to introduce themselves, then they introduce themselves, and
    finally, they ask their right child (if any) to introduce themselves,
    following the same rules.

In essence, each node in the tree, whether a parent or a child, follows the same
process: they let their left subtree (if it exists) "introduce" themselves
first, then they "introduce" themselves, and finally, they let their right
subtree (if it exists) "introduce" themselves.

This parent-child communication continues until all nodes in the tree have been
visited, ensuring a complete In-order Traversal of the tree.

So in an In-order traversal, a node's left subtree corresponds to everyone who
introduces themselves before it, the node itself is the one introducing
themselves now, and the right subtree corresponds to everyone who introduces
themselves after it.

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

1. **Normal case**: A tree with more than one node.

    The list represents a binary tree as follows:

    ```python
    root = build_binary_tree_from_list_preorder(
        [1, 2, 4, None, None, 5, None, None, 3, 6, None, None, 7, None, None]
    )
    assert inorder_traversal(root) == [4, 2, 5, 1, 6, 3, 7]
    ```

2. **Unbalanced tree**: A tree where one side has more nodes than the other
   side.

    ```python
    root = build_binary_tree_from_list_preorder([1, 2, 3, None, 4, None, None, None, None])
    assert inorder_traversal(root) == [4, 3, 2, 1]
    ```

3. **Tree with single path**: A tree where each parent has only one child.

    ```python
    root = build_binary_tree_from_list_preorder([1, 2, 3, None, 4, None, None, None, None])
    assert inorder_traversal(root) == [4, 3, 2, 1]
    ```

4. **Tree with duplicate values**: A tree with duplicate values in nodes.

    ```python
    root = build_binary_tree_from_list_preorder([1, 1, 1, None, 1, None, 1, None, None, None, None])
    assert inorder_traversal(root) == [1, 1, 1, 1, 1]
    ```

## Edge Cases

1. **Empty tree**: A tree with no nodes. This is a valid input and should return
   an empty list.

    ```python
    root = build_binary_tree_from_list_preorder([])
    assert inorder_traversal(root) == []
    ```

2. **Single node**: A tree with only one node. This edge case tests whether the
   function can handle the smallest non-empty tree.

    ```python
    root = build_binary_tree_from_list_preorder([1])
    assert inorder_traversal(root) == [1]
    ```

3. **Tree with maximum allowed nodes**: This edge case tests if the function can
   handle the largest possible tree within the constraints. Given the constraint
   that each node's value is unique, the input list will follow the pattern of
   descending to the leftmost node and then filling in the right subtree at each
   level before moving down to the next level.

    ```python
    root = build_binary_tree_from_list_preorder(list(range(1, 101)))
    # Depends on the structure of the tree.
    assert inorder_traversal(root) == list(range(1, 101))
    ```

4. **Tree with minimum and maximum allowed node values**: This edge case tests
   if the function can handle the smallest and largest possible node values
   within the constraints.

    ```python
    root = build_binary_tree_from_list_preorder([-100, None, 100])
    assert inorder_traversal(root) == [-100, 100]
    ```

## Walkthrough / Whiteboarding

The whiteboarding and complexity analysis for in-order traversal will be very
similar to the one we provided for preorder traversal. The primary difference
will be the order in which nodes are visited and added to the output list.

Consider the binary tree below:

```bash
    1
   / \
  2   3
 / \
4   5
```

The in-order traversal sequence for this tree is `4, 2, 5, 1, 3`.

1. First, traverse the left subtree of root node `1`. Visit `2`, then go left
   and visit `4`.
2. For `4`, the left and right are `None`, so we return to `2`.
3. Since we are done with the left subtree of `2`, we visit `2`, then go right
   and visit `5`.
4. Similarly, for `5`, the left and right are `None`, so we return to `2`, and
   then since we are done with `2`, we return to `1`.
5. Now we visit the root node `1` as we've completed the left subtree.
6. Next, we traverse the right subtree. Visit `3`.
7. No more nodes left to visit, hence the traversal is complete.

## Theoretical Best Time Complexity

The theoretical best time complexity for in-order traversal of a binary tree is
$\mathcal{O}(n)$, where $n$ is the number of nodes in the binary tree. This is
because each node is visited exactly once during the traversal.

See
[preorder traversal](144-binary-tree-preorder-traversal.html#theoretical-best-time-complexity)
for a detailed explanation of why the time complexity is $\mathcal{O}(n)$.

## Theoretical Best Space Complexity

The theoretical best (auxiliary) space complexity for this problem is
$\mathcal{O}(n)$, where $n$ is the number of nodes in the binary tree. This is
due to the space required by the call stack (assuming a recursive
implementation), which grows proportional to the maximum height of the tree
(which in worst case could be $n$), and the space required to store the output,
which would be equal to the number of nodes in the binary tree.

## Space-Time Tradeoff

As with preorder traversal, there isn't much of a traditional space-time
tradeoff in this problem. The traversal is a linear operation with respect to
the number of nodes, and the space required is also proportional to the number
of nodes. The most efficient solution will always involve visiting each node in
the tree once, and the space used is necessary to store the output and
accommodate the recursion stack (in case of recursive approach). However, an
iterative solution using an explicit stack would have the same space complexity
due to the need for storing nodes in a stack data structure.

## Solution (Recursive)

### Implementation

```{code-cell} ipython3
class Solution1:
    def inorderTraversal(self, root: Optional[BinaryTreeNode]) -> List[int]:
        inorder: List[int] = []

        def traverse(root: Optional[BinaryTreeNode], inorder: List[int]) -> List[int]:
            if not root:
                return inorder

            traverse(root.left, inorder)
            inorder.append(root.value)
            traverse(root.right, inorder)
            return inorder

        inorder = traverse(root, inorder)
        return inorder

class Solution2:
    def __init__(self) -> None:
        self.inorder: List[int] = []

    def reset(self) -> None:
        """To reset if the class instance were to traverse a new root."""
        self.inorder = []

    def inorderTraversal(self, root: Optional[BinaryTreeNode]) -> List[int]:
        if not root:
            return

        self.inorderTraversal(root.left)
        self.inorder.append(root.value)
        self.inorderTraversal(root.right)

        return self.inorder
```

### Tests

```{code-cell} ipython3
inorder_traversal = Solution1().inorderTraversal

def print_and_test_inorder(
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
    compare_test_case(inorder_traversal(root), expected_output, case_name)


# Test Case 1: Normal case
print_and_test_inorder(
    build_binary_tree_from_list_preorder(
        [1, 2, 4, None, None, 5, None, None, 3, 6, None, None, 7, None, None]
    ),
    [4, 2, 5, 1, 6, 3, 7],
    "Normal case",
)

# Test Case 2: Unbalanced tree
print_and_test_inorder(
    build_binary_tree_from_list_preorder([1, 2, 3, None, 4, None, None, None, None]),
    [3, 4, 2, 1],
    "Unbalanced tree",
)

# Test Case 3: Tree with single path
print_and_test_inorder(
    build_binary_tree_from_list_preorder([1, 2, 3, None, 4, None, None, None, None]),
    [3, 4, 2, 1],
    "Tree with single path",
)

# Test Case 4: Tree with duplicate values
print_and_test_inorder(
    build_binary_tree_from_list_preorder([1, 1, 1, None, 1, None, 1, None, None, None, None]),
    [1, 1, 1, 1, 1],
    "Tree with duplicate values",
)

# Edge Case 1: Empty tree
print_and_test_inorder(build_binary_tree_from_list_preorder([]), [], "Empty tree", print_tree=False)

# Edge Case 2: Single node
print_and_test_inorder(build_binary_tree_from_list_preorder([1]), [1], "Single node")

# Edge Case 3: Tree with maximum allowed nodes
print_and_test_inorder(
    build_binary_tree_from_list_preorder(list(range(1, 101))),
    list(range(100, 0, -1)),
    "Tree with maximum allowed nodes",
    print_tree=False,
)

# Edge Case 4: Tree with minimum and maximum allowed node values
print_and_test_inorder(
    build_binary_tree_from_list_preorder([-100, None, 100]),
    [-100, 100],
    "Tree with minimum and maximum allowed node values",
)
```

### Time Complexity

In the case of inorder traversal, the time and space complexity would be the
same as for preorder traversal. This is because the order in which the nodes are
visited (preorder, inorder, or postorder) does not change the amount of work to
be done or the space required. The only difference is the order in which the
nodes are added to the output list.

The time complexity of inorder traversal, like preorder and postorder
traversals, is $\mathcal{O}(n)$, where $n$ is the number of nodes. This is
because the algorithm must visit every node in the tree once and only once.

```{list-table} Time Complexity of Inorder Traversal Recursive Algorithm
:header-rows: 1
:name: inorder-recursion-traversal-time-complexity

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

If you want to see a more detailed analysis of the time complexity of inorder,
see
[my post on preorder](144-binary-tree-preorder-traversal.html#recursion-time-complexity).

### Space Complexity

The space complexity analysis for inorder traversal is quite
similar to that for preorder traversal, with a few nuances.

As usual, define $n$ to be the number of nodes in the tree and
$h$ to be the height of the tree.

#### Input Space Complexity

The input space complexity remains at $\mathcal{O}(n)$.

#### Auxiliary Space Complexity

The auxiliary space complexity is dependent on the height of the tree (i.e., the
maximum depth of recursion), which in the worst case (a degenerate tree) is
$\mathcal{O}(n)$ and in the best case (a balanced tree) is
$\mathcal{O}(\log n)$.

```{list-table} Auxiliary Space Complexity of Inorder Traversal Recursive Algorithm
:header-rows: 1
:name: inorder-recursion-traversal-auxiliary-space-complexity

* - Case
  - Auxiliary Space Complexity
* - Empty Tree
  - $\mathcal{O}(1)$
* - Balanced Tree
  - $\mathcal{O}(\log n)$
* - Degenerate Tree
  - $\mathcal{O}(n)$
```

#### Total Space Complexity

The total space complexity, which includes both the input and auxiliary space
complexities, is as follows:

```{list-table} Total Space Complexity of Inorder Traversal Recursive Algorithm
:header-rows: 1
:name: inorder-recursion-traversal-total-space-complexity

* - Case
  - Total Space Complexity
* - Empty Tree
  - $\mathcal{O}(1)$
* - Balanced Tree
  - $\mathcal{O}(n)$
* - Degenerate Tree
  - $\mathcal{O}(n)$
```

## Solution (Iterative)

TODO.

## Solution (Morris Traversal)

TODO.

## References and Further Readings

- **[Leetcode Solution](https://leetcode.com/problems/binary-tree-inorder-traversal/editorial)**
- **[Leetcode Card: Traverse a Tree](https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/1036/)**
- **[Tree Traversal Time Complexity](https://www.baeldung.com/cs/tree-traversal-time-complexity)**
- **[Time Complexity of Binary Tree Preorder](https://stackoverflow.com/questions/59233720/time-complexity-of-binary-tree-traversal-pre-order)**
