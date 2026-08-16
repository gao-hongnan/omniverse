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

# Generalizing to N-ary Trees

```{contents}
:local:
```

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
from typing import Generic, TypeVar, Optional, List
```

## Ternary Trees

```{code-cell} ipython3
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class TernaryTreeNode(Generic[T]):
    """This class represents a node in a ternary tree."""

    val: T
    child_1: Optional[TernaryTreeNode[T]] = None  # left
    child_2: Optional[TernaryTreeNode[T]] = None  # mid
    child_3: Optional[TernaryTreeNode[T]] = None  # right

    def __str__(self) -> str:
        return f"TernaryTreeNode({self.val})"

    def __repr__(self) -> str:
        return (
            f"TernaryTreeNode(val={self.val}, child_1={repr(self.child_1)},"
            f" child_2={repr(self.child_2)}, child_3={repr(self.child_3)})"
        )


def print_ternary_tree(node: Optional[TernaryTreeNode[T]], indent: str = "") -> None:
    """Render a ternary tree as an indented outline, left child first."""
    if node is None:
        return

    print(f"{indent}{node.val}")
    for child in (node.child_1, node.child_2, node.child_3):
        if child is not None:
            print_ternary_tree(child, indent + "    ")
```

Then you can build a ternary tree `1-2-3-4-6` in preorder as follows:

```{code-cell} ipython3
root = TernaryTreeNode(val=1)
root.child_1 = TernaryTreeNode(val=2)
root.child_1.child_1 = TernaryTreeNode(val=3)
# node 3 has no more children, leaf
root.child_1.child_1.child_1 = None
root.child_1.child_1.child_2 = None
root.child_1.child_1.child_3 = None
# backtrack to node 2 and no mid and right
root.child_1.child_2 = None
root.child_1.child_3 = None

root.child_2 = TernaryTreeNode(val=4)
root.child_3 = TernaryTreeNode(val=6)

print_ternary_tree(root)
```

This just means that with root node `1`, it has three children `2`, `4`, and
`6`. Node `2` has one child `3`, and node `4` has no children. Node `6` has no
children.

## N-ary Trees

Generalizing a ternary tree node to an N-ary tree node would involve changing
the fixed number of child pointers (i.e., `child_1`, `child_2`, `child_3`) to a
list or another collection that can hold an arbitrary number of children. Here's
an example of how you could define an N-ary tree node using a list to hold the
child pointers:

```{code-cell} ipython3
from dataclasses import dataclass

@dataclass
class NaryTreeNode(Generic[T]):
    """This class represents a node in an N-ary tree."""

    val: T
    children: Optional[List[NaryTreeNode[T]]] = None

    def __str__(self) -> str:
        return f"NaryTreeNode({self.val})"

    def __repr__(self) -> str:
        return f"NaryTreeNode(val={self.val}, children={repr(self.children)})"
```

This `NaryTreeNode` class can be used to represent trees with any number of
children per node. The children of a node are stored in a list, which can be of
any length, allowing for a flexible number of child nodes.

Below is a more complex example using the N-ary tree structure. This example
builds a tree with more levels and various numbers of children at each node.

```{code-cell} ipython3
root = NaryTreeNode(val=1)
root.children = [
    NaryTreeNode(val=2, children=[
        NaryTreeNode(val=5, children=[NaryTreeNode(val=9), NaryTreeNode(val=10)]),
        NaryTreeNode(val=6, children=[NaryTreeNode(val=11)]),
        NaryTreeNode(val=7)
    ]),
    NaryTreeNode(val=3, children=[
        NaryTreeNode(val=8, children=[
            NaryTreeNode(val=12),
            NaryTreeNode(val=13, children=[NaryTreeNode(val=14), NaryTreeNode(val=15)])
        ])
    ]),
    NaryTreeNode(val=4)
]
```

Here's an overview of the structure:

-   The root has value 1 and three children.
-   The first child of the root (value 2) has three children.
    -   Its first child (value 5) has two children (values 9 and 10).
    -   Its second child (value 6) has one child (value 11).
    -   Its third child (value 7) has no children.
-   The second child of the root (value 3) has one child (value 8).
    -   That child has two children:
        -   One with value 12 and no children.
        -   One with value 13 and two children (values 14 and 15).
-   The third child of the root (value 4) has no children.

This tree structure allows for a wide variety of shapes and sizes of trees,
reflecting the flexibility of the N-ary tree data structure.

```{prf:remark} A N-ary Tree with only 3 children is not a 3-ary tree
:label: narytree-ternarytree

The tree described above is not strictly a 3-ary tree, even though the root has
three children. An N-ary tree allows for any number of children per node, not
just N. A 3-ary tree would specifically restrict each node to have exactly three
children (or fewer if they are leaf nodes).

In the given example, the nodes have various numbers of children: some have two,
some have three, and some have none. The flexibility in the number of children
is what makes this an N-ary tree rather than a fixed-arity tree like a binary or
ternary tree. It's the concept of having a variable number of children that
defines an N-ary tree, not the specific number of children at any given node.
```

### Ternary Tree in terms of N-ary Tree

A ternary tree is a special case of an N-ary tree where each node has exactly
three positions for children. These positions correspond to the left, middle,
and right children.

Some restrictions:

-   The `children` list must have exactly three positions, corresponding to the
    left, middle, and right children. Missing children may be represented by
    `None` or another sentinel value.
-   Each child node must follow the same structure, having exactly three
    positions for its children.

This definition ensures that every node in the tree adheres to the ternary
structure, whether or not all three children are present.
