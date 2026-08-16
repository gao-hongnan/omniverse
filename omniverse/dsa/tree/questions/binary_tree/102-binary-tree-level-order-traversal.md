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

# Level Order Traversal

<a href="https://leetcode.com/problems/binary-tree-level-order-traversal/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-102-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)
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
```

Level order traversal, also known as breadth-first traversal, involves visiting
all the nodes of a level before moving on to the next level. In this traversal,
you start at the root, then visit all nodes at depth 1, then all nodes at depth
2, and so on.

Here's how you'd perform a level order traversal on your example tree:

1. Visit the root node: `1`.
2. Visit all nodes at depth 1: `2`, `6`.
3. Visit all nodes at depth 2: `3`, `4`, `7`, `9`.
4. Visit all nodes at depth 3: `5`, `8`.

The sequence of visited nodes would be: `1`, `2`, `6`, `3`, `4`, `7`, `9`, `5`,
`8`.
