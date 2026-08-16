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

# Postorder Traversal

<a href="https://leetcode.com/problems/binary-tree-postorder-traversal/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-145-blue"/></a>
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

Postorder traversal is a method used in binary trees to visit all the nodes in a
defined order: first visit the left child, then the right child, and finally the
root node. This method is applied recursively to all subtrees of the tree.

1. **Traverse** the left subtree in Postorder.
2. **Traverse** the right subtree in Postorder.
3. **Visit** the root.

Let's break down the process more:

1. **Visit the left subtree:** The traversal begins with the left subtree. This
   isn't just a single node visit; it involves the whole subtree, implying that
   the traversal follows the same set of rules, i.e., visiting its left child,
   right child, and then the root node. If the left subtree is empty (no nodes
   to visit), this step is skipped.

2. **Visit the right subtree:** After the left subtree has been traversed, the
   algorithm moves to the right subtree following the same set of rules:
   visiting its left child, right child, and then the root node. If the right
   subtree is empty, this step is also skipped.

3. **Visit the root node:** Only after visiting all the nodes in both subtrees
   does the algorithm visit the root node of the current subtree.

It's important to note that steps 1 and 2 involve recursion, meaning they will
apply the same three steps to each subtree they visit. This recursion continues
until it reaches a tree with no children (known as a leaf node), which marks the
base case for the recursion.

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

The order of nodes **visited** during a postorder traversal is:

```bash
3-5-4-2-8-7-9-6-1
```

Here's how you'd perform a postorder traversal on this example tree:

1. Visit all nodes in the left subtree of `1`:
    - Visit all nodes in the left subtree of `2`:
        - Visit all nodes in the left subtree of `3`. The left subtree of `3` is
          None, so there are no nodes to visit.
        - Visit all nodes in the right subtree of `3`. The right subtree of `3`
          is None, so there are no nodes to visit.
        - Visit the root of this subtree: `3`.
    - Visit all nodes in the right subtree of `2`:
        - Visit all nodes in the left subtree of `4`:
            - Visit all nodes in the left subtree of `5`. The left subtree of
              `5` is None, so there are no nodes to visit.
            - Visit all nodes in the right subtree of `5`. The right subtree of
              `5` is None, so there are no nodes to visit.
            - Visit the root of this subtree: `5`.
        - Visit all nodes in the right subtree of `4`. The right subtree of `4`
          is None, so there are no nodes to visit.
        - Visit the root of this subtree: `4`.
    - Visit the root of the left subtree: `2`.
2. Visit all nodes in the right subtree of `1`:
    - Visit all nodes in the left subtree of `6`:
        - Visit all nodes in the left subtree of `7`:
            - Visit all nodes in the left subtree of `8`. The left subtree of
              `8` is None, so there are no nodes to visit.
            - Visit all nodes in the right subtree of `8`. The right subtree of
              `8` is None, so there are no nodes to visit.
            - Visit the root of this subtree: `8`.
        - Visit all nodes in the right subtree of `7`. The right subtree of `7`
          is None, so there are no nodes to visit.
        - Visit the root of this subtree: `7`.
    - Visit all nodes in the right subtree of `6`:
        - Visit all nodes in the left subtree of `9`. The left subtree of `9` is
          None, so there are no nodes to visit.
        - Visit all nodes in the right subtree of `9`. The right subtree of `9`
          is None, so there are no nodes to visit.
        - Visit the root of this subtree: `9`.
    - Visit the root of the right subtree: `6`.
3. Visit the root node: `1`.

Therefore, the order of nodes **visited** during an post-order traversal of this
tree is `3-5-4-2-8-7-9-6-1`. The process is recursive, and the same set of rules
is applied to each subtree within the tree. As with preorder, any recursive
process can also be implemented iteratively; we can implement an post-order
traversal of a binary tree iteratively using a stack.

## Visualization

See
[leetcode's visualization of Inorder](https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/992/).

## Problem

Given the `root` of a binary tree, return the _postorder traversal of its nodes'
values_.

## Intuition

For postorder traversal, you can still use the family gathering analogy, but now
with an even different rule of introduction:

-   Parents tell children: "Before I introduce myself, **I want both of my
    children (if they exist) to introduce themselves first, along with their own
    children (if any), following the same rule**. My left child and their
    descendants will go first, then my right child and their descendants will
    go. **Only after everyone else in our family has introduced themselves will
    I introduce myself.**"

-   Children follow their parent's instructions: Each child node in the tree
    considers itself as the parent of its own subtree. They first ask their left
    child (if any) to introduce themselves and their descendants, then they ask
    their right child (if any) to introduce themselves and their descendants,
    following the same rules. Only after both of their "children" have gone will
    they introduce themselves.

This parent-child communication continues until all nodes in the tree have been
visited, ensuring a complete postorder traversal of the tree.

In essence, each node in a postorder traversal defers its own visit until after
all of its descendants (its entire left and right subtrees) have been visited. A
node's left subtree corresponds to everyone who introduces themselves before the
right subtree, and the right subtree corresponds to everyone who introduces
themselves before the node itself.

So, in a postorder traversal, everyone in the left subtree introduces themselves
first, followed by everyone in the right subtree, and finally the node itself.

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

    ```python
    root = build_binary_tree_from_list_preorder(
        [1, 2, 4, None, None, 5, None, None, 3, 6, None, None, 7, None, None]
    )
    assert postorder_traversal(root) == [4, 5, 2, 6, 7, 3, 1]
    ```

2. **Unbalanced tree**: A tree where one side has more nodes than the other
   side.

    ```python
    root = build_binary_tree_from_list_preorder([1, 2, None, 3, None, 4, None, None])
    assert postorder_traversal(root) == [2, 4, 3, 1]
    ```

3. **Tree with single path**: A tree where each parent has only one child.

    ```python
    root = build_binary_tree_from_list_preorder([1, 2, None, 3, None, 4, None, None])
    assert postorder_traversal(root) == [4, 3, 2, 1]
    ```

4. **Tree with duplicate values**: A tree with duplicate values in nodes.

    ```python
    root = build_binary_tree_from_list_preorder([1, 1, None, 1, None, 1, None, 1, None, None])
    assert postorder_traversal(root) == [1, 1, 1, 1, 1]
    ```

## Edge Cases

1.  **Empty tree**: A tree with no nodes. This is a valid input and should
    return an empty list.

    ```python
    root = build_binary_tree_from_list_preorder([])
    assert postorder_traversal(root) == []
    ```

2.  **Single node**: A tree with only one node. This edge case tests whether the
    function can handle the smallest non-empty tree.

    ```python
    root = build_binary_tree_from_list_preorder([1])
    assert postorder_traversal(root) == [1]
    ```

3.  **Tree with maximum allowed nodes**: This edge case tests if the function
    can handle the largest possible tree within the constraints. Given the
    constraint that each node's value is unique, the input list will follow the
    pattern of descending to the leftmost node and then filling in the right
    subtree at each level before moving down to the next level.

    ```python
    root = build_binary_tree_from_list_preorder(list(range(99, -1, -1)))
    assert postorder_traversal(root) == list(range(0, 100))
    ```

4.  **Tree with minimum and maximum allowed node values**: This edge case tests
    if the function can handle the smallest and largest possible node values
    within the constraints.

    ```python
    root = build_binary_tree_from_list_preorder([-100, None, 100])
    assert postorder_traversal(root) == [100, -100]
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

The postorder traversal sequence for this tree is `4, 5, 2, 3, 1`.

1. First, traverse the left subtree of root node `1`. Visit `2`, then go left
   and visit `4`.
2. For `4`, both left and right are `None`, so we are done with `4`.
3. We return to `2`, go right and visit `5`. For `5`, the left and right are
   `None`, so we are done with `5`.
4. Now, having traversed both left and right subtrees of `2`, we are done with
   `2`.
5. We return to `1`, and proceed to its right subtree. Visit `3`. As there are
   no children for `3`, we are done with `3`.
6. Finally, having traversed both left and right subtrees of `1`, we are done
   with `1`, hence the traversal is complete.

## Theoretical Best Time Complexity

The theoretical best time complexity for postorder traversal of a binary tree is
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

As with other tree traversals, there isn't much of a traditional space-time
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
    def postorderTraversal(self, root: Optional[BinaryTreeNode]) -> List[int]:
        postorder: List[int] = []

        def traverse(root: Optional[BinaryTreeNode], postorder: List[int]) -> List[int]:
            if not root:
                return postorder

            traverse(root.left, postorder)
            traverse(root.right, postorder)
            postorder.append(root.value)
            return postorder

        postorder = traverse(root, postorder)
        return postorder


class Solution2:
    def __init__(self) -> None:
        self.postorder: List[int] = []

    def reset(self) -> None:
        """To reset if the class instance were to traverse a new root."""
        self.postorder = []

    def postorderTraversal(self, root: Optional[BinaryTreeNode]) -> List[int]:
        if not root:
            return

        self.postorderTraversal(root.left)
        self.postorderTraversal(root.right)
        self.postorder.append(root.value)

        return self.postorder
```

### Tests

```{code-cell} ipython3
postorder_traversal = Solution1().postorderTraversal

def print_and_test_postorder(
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
    compare_test_case(postorder_traversal(root), expected_output, case_name)

# Test Case 1: Normal case
print_and_test_postorder(
    build_binary_tree_from_list_preorder(
        [1, 2, 4, None, None, 5, None, None, 3, 6, None, None, 7, None, None]
    ),
    [4, 5, 2, 6, 7, 3, 1],
    "Normal case",
)

# Test Case 2: Unbalanced tree
print_and_test_postorder(
    build_binary_tree_from_list_preorder([1, 2, None, 3, None, 4, None, None]),
    [4, 3, 2, 1],
    "Unbalanced tree",
)

# Test Case 3: Tree with single path
print_and_test_postorder(
    build_binary_tree_from_list_preorder([1, 2, None, 3, None, 4, None, None]),
    [4, 3, 2, 1],
    "Tree with single path",
)

# Test Case 4: Tree with duplicate values
print_and_test_postorder(
    build_binary_tree_from_list_preorder([1, 1, None, 1, None, 1, None, 1, None, None]),
    [1, 1, 1, 1, 1],
    "Tree with duplicate values",
)

# Edge Case 1: Empty tree
print_and_test_postorder(build_binary_tree_from_list_preorder([]), [], "Empty tree", print_tree=False)

# Edge Case 2: Single node
print_and_test_postorder(build_binary_tree_from_list_preorder([1]), [1], "Single node")

# Edge Case 3: Tree with maximum allowed nodes
print_and_test_postorder(
    build_binary_tree_from_list_preorder(list(range(1, 101))),
    list(range(100, 0, -1)),
    "Tree with maximum allowed nodes",
    print_tree=False,
)

# Edge Case 4: Tree with minimum and maximum allowed node values
print_and_test_postorder(
    build_binary_tree_from_list_preorder([-100, None, 100]),
    [100, -100],
    "Tree with minimum and maximum allowed node values",
)
```

### Time Complexity

The time complexity of postorder traversal, similar to the other tree
traversals, is $\mathcal{O}(n)$, where $n$ is the number of nodes in the tree.
This is because the algorithm has to visit every node in the tree once and only
once. The specific order of visiting (whether it's preorder, inorder, or
postorder) doesn't affect the time complexity.

```{list-table} Time Complexity of Postorder Traversal Recursive Algorithm
:header-rows: 1
:name: postorder-recursion-traversal-time-complexity

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

### Space Complexity

The space complexity for the postorder traversal follows the same pattern as
other depth-first tree traversals.

#### Input Space Complexity

The input space complexity remains the same, i.e., $\mathcal{O}(n)$.

#### Auxiliary Space Complexity

The auxiliary space complexity for a recursive solution is dependent on the
maximum depth of the recursion, which corresponds to the height of the tree. For
a balanced tree, the height is $\mathcal{O}(\log n)$, whereas for a degenerate
tree (worst case), the height is $\mathcal{O}(n)$.

```{list-table} Auxiliary Space Complexity of Postorder Traversal Recursive Algorithm
:header-rows: 1
:name: postorder-recursion-traversal-auxiliary-space-complexity

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

Summing up the input and auxiliary space complexity gives the total space
complexity:

```{list-table} Total Space Complexity of Postorder Traversal Recursive Algorithm
:header-rows: 1
:name: postorder-recursion-traversal-total-space-complexity

* - Case
  - Total Space Complexity
* - Empty Tree
  - $\mathcal{O}(1)$
* - Balanced Tree
  - $\mathcal{O}(n)$
* - Degenerate Tree
  - $\mathcal{O}(n)$
```

Thus, the time complexity of postorder traversal is linear with respect to the
number of nodes, and the space complexity is linear in the worst case (when the
tree degenerates into a linked list), but logarithmic in the case of a balanced
tree.

## Solution (Iterative)

TODO.

## Solution (Morris Traversal)

TODO.

## References and Further Readings

- **[Leetcode Solution](https://leetcode.com/problems/binary-tree-postorder-traversal/editorial)**
- **[Leetcode Card: Traverse a Tree](https://leetcode.com/explore/learn/card/data-structure-tree/134/traverse-a-tree/1036/)**
- **[Tree Traversal Time Complexity](https://www.baeldung.com/cs/tree-traversal-time-complexity)**
- **[Time Complexity of Binary Tree Preorder](https://stackoverflow.com/questions/59233720/time-complexity-of-binary-tree-traversal-pre-order)**
