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

# Traversal

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gao-hongnan/omniverse/blob/main/omniverse/dsa/tree/traversal.ipynb)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import rich
from rich.jupyter import print
from typing import Optional, List, Union


from omnivault.dsa.trees.binary import BinaryTreeNode
from omnivault.dsa.trees.utils import build_binary_tree_from_list_preorder
from omnivault.dsa.trees.utils import print_binary_tree
```

In the realm of
[**data structures**](https://en.wikipedia.org/wiki/Data_structure), a
[**tree**](<https://en.wikipedia.org/wiki/Tree_(data_structure)>) is a
nonlinear, hierarchical structure consisting of **nodes**, which hold data,
connected by **edges**, which define the relationship between the nodes. One of
the most fundamental operations we can perform on a tree is
[**traversal**](https://en.wikipedia.org/wiki/Tree_traversal), which is the
process of visiting each node in the tree exactly once in a specified order.

There are several common strategies for tree traversal, each with its own use
cases and properties. Broadly speaking, these strategies fall into two
categories:

1. [**Depth-First Traversal**](https://en.wikipedia.org/wiki/Depth-first_search):
   In this approach, we explore as far as possible along each branch before
   backtracking. The three most common variants of depth-first traversal are:

    - [**Preorder Traversal**](<https://en.wikipedia.org/wiki/Tree_traversal#Pre-order_(NLR)>):
      Visit the root node, traverse the left subtree, and finally traverse the
      right subtree.
    - [**Inorder Traversal**](<https://en.wikipedia.org/wiki/Tree_traversal#In-order_(LNR)>):
      Traverse the left subtree, visit the root node, and finally traverse the
      right subtree. This strategy is used commonly in binary search trees as it
      visits the nodes in ascending order.
    - [**Postorder Traversal**](<https://en.wikipedia.org/wiki/Tree_traversal#Post-order_(LRN)>):
      Traverse the left subtree, traverse the right subtree, and finally visit
      the root node. This strategy is often useful for operations such as
      deleting or freeing nodes of a tree from memory.

2. [**Breadth-First Traversal (or Level Order Traversal)**](https://en.wikipedia.org/wiki/Breadth-first_search):
   In this approach, we visit all the nodes of a level before going to the next
   level. This strategy is widely used in algorithms related to searching and
   more.

While these strategies cover the most common forms of tree traversal, it's worth
noting that trees are highly flexible structures, and these traversal methods
can be adapted or combined to suit specific problems. As always, the right
traversal method depends on the specific properties of the tree and the nature
of the problem you're trying to solve. Understanding the basics of each
traversal method and the scenarios in which they're most effective is a critical
part of mastering tree-based algorithms.


Let's consider our running example.

```{code-cell} ipython3
tree_values: List[Union[int, None]] = [
    1,
    2,
    3,
    None,
    None,
    4,
    5,
    None,
    None,
    None,
    6,
    7,
    8,
    None,
    None,
    None,
    9,
    None,
    None,
]

root = build_binary_tree_from_list_preorder(tree_values)
lines = print_binary_tree(
    root, node_info=lambda n: (str(n.value), n.left, n.right), is_top=False
)
print("\n".join(lines))
```

## Preorder Traversal

See [my write up on preorder traversal here](questions/binary_tree/144-binary-tree-preorder-traversal.md).

### Inorder Traversal

See [my write up on inorder traversal here](questions/binary_tree/94-binary-tree-inorder-traversal.md).

### Postorder Traversal

See [my write up on postorder traversal here](questions/binary_tree/145-binary-tree-postorder-traversal.md).

### Level Order Traversal

See [my write up on level order traversal here](questions/binary_tree/102-binary-tree-level-order-traversal.md).

## Tips on Tree Recursion (Backtracking)

In the context of recursion and the code you provided, when we say "return," we
are referring to the control flow returning to the previous level of the
recursive call stack.

When a recursive function is called, it starts a new level or instance of the
function with the new input. If the base case is not met (i.e., the tree is not
empty in this case), it keeps on calling itself, creating new levels in the
process. When it finally hits the base case (an empty tree), it doesn't have any
operations to perform, so it "returns" control back to the previous level of the
function where it was called. This returning cascades back through the previous
levels all the way to the first level (which was the first call to the recursive
function), at which point the recursion ends.

In our specific code, `return` is used to stop the execution of the current
function and go back to the previous function call (the previous level in the
recursive call stack). When the root is an empty list (`[]`), the function hits
the base case and immediately returns without doing anything else. This is how
it knows to stop making new recursive calls for this branch and to go back to
the previous level of recursion, effectively "traversing back up" the tree.

This is a key aspect of recursion: you always have a condition that breaks the
recursion, and when that condition is met, you return from the current level of
recursion. Without a base case that halts the recursion, you'd have an infinite
loop, or in the context of recursion, infinite recursive calls.

## Solve Tree Problems Recursively

- https://leetcode.com/explore/learn/card/data-structure-tree/17/solve-problems-recursively/534/

## References and Further Readings

