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

# Balanced Binary Tree

<a href="https://leetcode.com/problems/balanced-binary-tree/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-110-blue"/></a>
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
```

## TODO

This question is not as easy as it seems. Read the solution and understood why
you need $-1$ as well as return `max(left, right) + 1` in the recursive function.

To revisit.

Let's consider our running example.

```{code-cell} ipython3
tree_values: List[Union[int, None]] = [
    1,
    2,
    4,
    None,
    7,
    None,
    None,
    5,
    None,
    None,
    3,
    None,
    6,
    8,
    None,
]

root = build_binary_tree_from_list_preorder(tree_values)
lines = print_binary_tree(
    root, node_info=lambda n: (str(n.value), n.left, n.right), is_top=False
)
rich.print("\n".join(lines))
```

## Intuition

Some intuition first on whether to use top-down or bottom-up approach (i.e. preorder vs postorder).

You want the parent to ask: can you both child check if you are balanced.
