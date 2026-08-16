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

# Count Good Nodes in Binary Tree

<a href="https://leetcode.com/problems/count-good-nodes-in-binary-tree/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-1448-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow) ![Tag](https://img.shields.io/badge/Tag-BinaryTree-orange)
![Tag](https://img.shields.io/badge/Tag-DFS-orange) ![Tag](https://img.shields.io/badge/Tag-Recursion-orange)

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Optional, TypeVar
from typing import List, Union, TypeVar, Optional, Tuple
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

```{code-cell} ipython3
tree_values = [3, 1, 4, 7, None, None, 6, None, None, 0, 5, None, None, None, 0, None, None]
root = build_binary_tree_from_list_preorder(tree_values)
print_binary_tree(root, node_info=lambda n: (str(n.value), n.left, n.right), is_top=True)
```

ToT

1. Count of good notes is not the focus of the problem, although it's the value
   to return
2. What is the core "problem" that we what to solve? --> The condition to
   increment a good node count/qualify a good node
3. Qualification Requirement: No prior parent nodes in lineage of current node
   is greater than it, for it to be counted as good node
4. Given this, we can "codify" the condition to increment the count
5. Back point 3, we need to think of ways to store pass info. Could it be a
   list? Or a state that is much easier to transmit
6. Max value of the current node's parents' lineage

This above is the starting point, to think about filling up the templates of DFS
done recursively.

## Intuition Preorder

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

## Intuition

The first step is to understand that such problems can be usually solved by DFS
(Depth First Search) or BFS (Breadth First Search). In this case, we will use
DFS.

The next step is to ask which DFS to use, preorder, inorder or postorder? We
know every preorder problem can be solved by postorder, but not vice versa. And
inorder is commonly used for BST (Binary Search Tree) problems.

To answer that, we need to ask ourselves, what is the decision we need to make
at each node? Do we need information from the children to make the decision? Or
do we need information from the parent to make the decision?

The tell-tale sign is I can make the decision at the node itself when I
traverse.

In this question, it is very much natural to land ourselves in preorder.

**Why?**

This is because the decision about whether a node is "good" (i.e., its value is
greater than or equal to the maximum value seen so far on the path from the root
to the node) is most naturally made at the node itself before visiting its
children. Preorder traversal aligns perfectly with this requirement as it
processes the node first. In contrast, postorder traversal processes the node
after visiting its children, which is not ideal for this problem and although
solvable, is counter-intuitive.

The next step is to fit ourself into the framework of DFS.

We need to answer three questions:

-   Base case: When do we stop?
-   Return statement: What do we return to our parents?
-   State: What do we need to remember when we recurse?

If we can answer the above three questions, then the recurse body will be much
easier:

-   Recurse statement: How do we recurse?

## Solution (Preorder/Top-Down Traversal)

### Solution Intuition

We decided on preorder. So we have a rough template to follow which can help us
to answer the three questions above.

```{code-cell} ipython3
class Solution1:
    def preorderTraversal(self, root: Optional[BinaryTreeNode]) -> List[int]:
        preorder: List[int] = []

        def traverse(root: Optional[BinaryTreeNode], preorder: List[int]) -> List[int]:
            if not root:
                # return [] is more accurate since it syncs up with the leetcode
                # test cases
                return preorder

            preorder.append(root.val)
            traverse(root.left, preorder)
            traverse(root.right, preorder)
            return preorder

        preorder = traverse(root, preorder)
        return preorder
```

We want to:

-   Do something at the node itself.
-   Then recurse to the left subtree
-   Then recurse to the right subtree

We take the example:

```python
        3
       / \
      1   0
   __/ \_
  4      0
 / \    /
7   6  5
```

We trace a few times to get a feel:

-   Initialize `self.good_nodes` to `0` as the global count.
-   At root `3`, we know by definition of the problem, it is a good node, so we
    add `1` to the global count `self.good_nodes`. Now `self.good_nodes` is `1`.
-   We visited the root, so we need to recurse to the left subtree of `3`.
-   As usual, we need to check if the left subtree is `None`, if it is, then we
    return `None` to the parent. In this case, it is `1` and not `None`, so
    since we are using preorder, we will "visit" the parent node of the left
    subtree first, which is `1`. What do we do here? We need to **compare** `1`
    with `3`, and we know that `1` is smaller than `3`, so we do not add `1` to
    the global count `self.good_nodes`. Now `self.good_nodes` is still `1`.
-   We visited the parent node of the left subtree, so we need to continue
    recursing left until leaf before we go right. So again, the left subtree of
    `1` is `4`, which is not a leaf (i.e. `None`), so we need to visit `4`
    first. What do we do here? We need to **compare** `4` with `1` and `3`, and
    we know that `4` is greater than `1` and `3`, so we add `1` to the global
    count `self.good_nodes`. Now `self.good_nodes` is `2`.

    This poses a problem, as we traverse down, we need to keep track of a
    `state` which allows the comparison to happen. The naive way is to maintain
    a list that holds the visited path so far `[1, 3]` and then we can compare
    `4` with the list to see if `4` is greater than all the values in the list.
    But this is not efficient, we can do better.

    We keep a variable `max_so_far_in_the_path` which is initialized to
    `-float("inf")` (negative infinity) at the root so our first comparison will
    be `3` vs `-float("inf")` and we know that `3` is greater than
    `-float("inf")`, so we add `1` to the global count `self.good_nodes`.

    Then subsequently, we update `max_so_far_in_the_path` to `3` and we compare
    `1` with `3` and we know that `1` is smaller than `3`, so we do not add `1`
    and do not update `max_so_far_in_the_path`.

    When we reach `4`, we compare `4` with `max_so_far_in_the_path` which is
    `3`, and we know that `4` is greater than `3`, so we add `1` to the global
    count `self.good_nodes` and update `max_so_far_in_the_path` to `4`.

    This way, as we traverse down the path, we always have the
    `max_so_far_in_the_path` to compare with the current node.

-   One key thing to realise is that `max_so_far_in_the_path` is a `state` that
    needs to be maintained as we recurse down the path. This is the children
    asking the parent. So for instance at the node `4`, the
    `max_so_far_in_the_path` is `4`, which will be shared by both the left node
    `7` and the right node `6`.

#### Base Case

-   Return statement: Parents asks (call) their two children and ask them, can
    you tell me how many good nodes you have in your subtree?

    -   In this context, you do not need to necessarily return me the count,
        instead, can you add your answer to the global count
        `self.global_count`?

-   Children asks (call) the parents, I need to know the max value so far, this
    means that I need to ask you, am I bigger than you? I do not have any
    knowledge of my parents, therefore I need to check with you, hey, am I
    bigger than you? If I am bigger than you, then I will return my value,
    otherwise I will return your value.
    -   In this context, this means updating `max_so_far` as a state.

Based on the tree below, you must have a few intuitions:

-   As we go down the path `3-1-4-6` there are already 3 good nodes, but
    **after** `6`, the `max_so_far` is `6`.

    -   As it rolls back to `4`, the `max_so_far` is NOT `6`, but `4`, because
        `4` is the parent of `6`.
    -   Then you need to visualize that at Node `4`, it will now check right
        subtree, but of course nothing happens since right is `None`.
    -   As it rolls back to `1`, the `max_so_far` is `3`, because `3` is the
        parent of `1`, then it proceeds to check the right subtree and finds `5`
        eventually.

### Algorithm

## Math

The function $f(v, m)$ calculates the number of good nodes in the subtree rooted
at node $v$, where $m$ is the maximum value encountered from the root to node
$v$. The definition is:

$$
f(v, m) =
    \begin{cases}
      0 & \text{if } v = \emptyset \\
      1 + f\big(\ell(v), \max(m, c(v))\big) + f\big(r(v), \max(m, c(v))\big) & \text{if } c(v) \geq m \\
      f\big(\ell(v), m\big) + f\big(r(v), m\big) & \text{otherwise}
    \end{cases}
$$

where

-   $\emptyset$ represents the empty tree, or `None` in Python.
-   $v$ is a node in the tree.
-   $c(v)$ denotes the value of node $v$. $c$ is the "center/mid" node.
-   $m$ is the maximum value observed on the path from the root to node $v
   $.
-   $\ell(v)$ and $r(v)$ are the left and right children of node $v
   $,
    respectively.
-   The function $f$ returns the count of good nodes in the subtree rooted at
    $v$, given the maximum value $m$ up to $v$.

This definition captures the recursive nature of the problem. It first checks if
the current node $v$ is null (represented by $\emptyset$). If $v$ is not null,
the function then checks whether $val(v)$ is at least $m$, the maximum value
seen so far. If so, it increments the count (represented by the "1 +" part of
the formula) and continues the recursion for both the left and right children,
updating the maximum value as necessary. If $val(v) < m$, it only continues the
recursion without incrementing the count.

## Solution

```python
# Definition for a binary tree node.
# class BinaryTreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: BinaryTreeNode) -> int:
        return self.countGoodNodes(root, max_so_far=-float("inf"))

    def countGoodNodes(self, root: BinaryTreeNode, max_so_far: int) -> int:
        if max_so_far is None:
            max_so_far = -float("inf") # negative infinity

        if not root:
            return 0 # for sure because if you reached
                     # the leaf, then you should stop recurse

        good_nodes = 0 # root is confirmed so we will add 1 later

        # when you write recurse call, think of finale, the final return in the func

        # max_so_far = max(root.val, max_so_far) # impt think of how recursive call gonna come here also
        # mistake is not using >= and use >, see samples, they should be clear to say about greater or equals than as well.
        if root.val >= max_so_far:
            good_nodes += 1
            max_so_far = root.val

        good_nodes += self.countGoodNodes(root.left, max_so_far)
        good_nodes += self.countGoodNodes(root.right, max_so_far)
        return good_nodes

class Solution:
    def __init__(self):
        self.global_count = 0
    def goodNodes(self, root: BinaryTreeNode) -> int:
        return self.countGoodNodes(root, max_so_far=-float("inf"))

    def countGoodNodes(self, root: BinaryTreeNode, max_so_far: int) -> int:
        if max_so_far is None:
            max_so_far = -float("inf") # negative infinity

        if not root:
            return None # for sure because if you reached
                     # the leaf, then you should stop recurse


        # when you write recurse call, think of finale, the final return in the func

        # max_so_far = max(root.val, max_so_far) # impt think of how recursive call gonna come here also
        # mistake is not using >= and use >, see samples, they should be clear to say about greater or equals than as well.
        if root.val >= max_so_far:
            self.global_count += 1
            max_so_far = root.val

        self.countGoodNodes(root.left, max_so_far)
        self.countGoodNodes(root.right, max_so_far)
        return self.global_count

class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def countGoodNodes(node, maxVal):
            if not node:
                return 0

            if node.val >= maxVal:
                good = 1
            else:
                good = 0
            maxVal = max(maxVal, node.val)
            good += countGoodNodes(node.left, maxVal)
            good += countGoodNodes(node.right, maxVal)

            return good

        return countGoodNodes(root, float("-inf"))
```

### Time Complexity

There are n nodes and n - 1 edges in a tree so if we traverse each once then the
total traversal is O(2n - 1) which is O(n).

## References and Further Readings

-   https://leetcode.com/problems/count-good-nodes-in-binary-tree/solutions/982782/easy-python-recursive-beats-90-with-comments/
-   https://leetcode.com/problems/count-good-nodes-in-binary-tree/solutions/1218157/python-3-dfs-with-a-detailed-explanation-suitable-for-novice-users/
-   https://leetcode.com/problems/count-good-nodes-in-binary-tree/solutions/3326049/python-dfs-iterative-beats-88/
-   https://algo.monster/problems/visible_tree_node
