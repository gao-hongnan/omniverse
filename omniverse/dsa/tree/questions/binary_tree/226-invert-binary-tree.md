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

# Invert Binary Tree

<a href="https://leetcode.com/problems/invert-binary-tree/">\
<img alt="Question Number" src="https://img.shields.io/badge/Question-226-blue"/></a>
![Difficulty](https://img.shields.io/badge/Difficulty-Easy-green)
![Tag](https://img.shields.io/badge/Tag-BinaryTree-orange)
![Tag](https://img.shields.io/badge/Tag-DFS-orange)
![Tag](https://img.shields.io/badge/Tag-Recursion-orange)

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

## Solution

Let's take a step-by-step walkthrough of your code using an example tree.

Suppose the input tree is:

```text
    4
   / \
  2   7
 / \ / \
1  3 6  9
```

- First, you call `invertTree` on the root node, which is 4.

- Since root is not `None`, you proceed to call `invertTree(root.left)`, which calls `invertTree` on node 2.

- For node 2, you proceed to call `invertTree(root.left)`, which calls `invertTree` on node 1.

- For node 1, it's a leaf node, so `root.left` and `root.right` are both `None`. Thus, `invertTree(root.left)` and `invertTree(root.right)` both return `None`.

- Since `root.left` and `root.right` for node 1 are both `None`, swapping them does nothing. The function `invertTree(1)` then returns node 1, which is assigned to `left` in the `invertTree(2)` call.

- Back in the `invertTree(2)` call, you now call `invertTree(root.right)`, which calls `invertTree` on node 3. Since node 3 is also a leaf node, the process is the same as for node 1, and `invertTree(3)` returns node 3, which is assigned to `right`.

- Now, in the `invertTree(2)` call, you have `left = 1` and `right = 3`, you swap `root.left` and `root.right`, so node 2's left child becomes node 3 and its right child becomes node 1. You then return node 2.

- The process continues similarly for the rest of the tree. For node 4, after the recursive calls to `invertTree(root.left)` and `invertTree(root.right)`, `left` becomes the root of the inverted left subtree and `right` becomes the root of the inverted right subtree. Then you swap `root.left` and `root.right` and return the root.

The inverted tree will be:

```
    4
   / \
  7   2
 / \ / \
9  6 3  1
```


```python
class Solution:
    def invertTree(self, root: Optional[BinaryTreeNode]) -> Optional[BinaryTreeNode]:
        if root is None:
            return None

        inverted_left_subtree = self.invertTree(root.left)
        inverted_right_subtree = self.invertTree(root.right)

        root.left = inverted_right_subtree
        root.right = inverted_left_subtree

        return root
```

Here, I'll keep track of the status and state of the variables `root`, `inverted_left_subtree`, and `inverted_right_subtree`:

1. **Line 2**: Call `invertTree(4)`. Here, `root = 4`. Variables `inverted_left_subtree` and `inverted_right_subtree` haven't been defined yet.

2. **Line 3**: We check if `root` is `None`, which it isn't. So we proceed to line 6.

3. **Line 6**: Call `invertTree(root.left)`, which is `invertTree(2)`. This pauses execution for `invertTree(4)` and starts `invertTree(2)`, setting `root = 2`.

4. **Line 6 of invertTree(2)**: Call `invertTree(root.left)`, which is `invertTree(1)`. This pauses execution for `invertTree(2)` and starts `invertTree(1)`, setting `root = 1`.

5. **Line 6 of invertTree(1)**: Call `invertTree(root.left)`, which is `invertTree(None)`. This starts a new execution and immediately returns `None`, setting `inverted_left_subtree = None`.

6. **Line 7 of invertTree(1)**: Call `invertTree(root.right)`, which is `invertTree(None)`. This also returns `None`, setting `inverted_right_subtree = None`.

7. **Line 9 and 10 of invertTree(1)**: Since both `inverted_left_subtree` and `inverted_right_subtree` are `None`, swapping `root.left` and `root.right` does nothing. `invertTree(1)` returns `root`, which is node 1.

8. **Resuming Line 6 of invertTree(2)**: The `invertTree(1)` call returns node 1, which is assigned to `inverted_left_subtree`.

9. **Line 7 of invertTree(2)**: Call `invertTree(root.right)`, which is `invertTree(3)`. This process is similar to the process for node 1, and `invertTree(3)` returns node 3, setting `inverted_right_subtree = Node(3)`.

10. **Line 9 and 10 of invertTree(2)**: Swap `root.left` and `root.right`, so node 2's left child becomes node 3 and its right child becomes node 1. Return `root`, which is node 2.

11. **Resuming Line 6 of invertTree(4)**: The `invertTree(2)` call returns node 2, which is assigned to `inverted_left_subtree`.

12. The process continues similarly for the right subtree of node 4 (i.e., `invertTree(7)`) and finally for swapping the left and right children of the root itself.

This step-by-step explanation keeps track of the status of the `root`, `inverted_left_subtree`, and `inverted_right_subtree` variables at each step of the function call. The recursive nature of the function allows us to keep track of the state at each level of recursion, and the function modifies and returns these variables accordingly.


1. **Line 2**: `invertTree` is first called on the root node, which is 4. So, `root=4`.

2. **Line 3**: We check if `root` is `None`, which is not, so we continue to the next line.

3. **Line 6**: We call `invertTree` on `root.left` (node 2). We pause the current execution (for root=4) and begin executing `invertTree` for `root=2`.

4. **Line 3**: For `root=2`, it is also not `None`, so we continue.

5. **Line 6**: Now we call `invertTree` on `root.left` (node 1). This will pause the execution for `root=2` and begin executing `invertTree` for `root=1`.

6. **Line 3**: For `root=1`, we check if it is `None`, which it's not.

7. **Line 6**: We call `invertTree` on `root.left` of node 1, which is `None`. This starts a new execution for `root=None`.

8. **Line 3**: For `root=None`, this condition is met, and so we return `None` in **Line 4**.

9. **Line 7**: Back in `invertTree(1)`, we're trying to execute `invertTree` on `root.right` of node 1, which is also `None`. This starts a new execution and immediately returns `None` as well.

10. **Line 9 and 10**: Since `root.left` and `root.right` for node 1 are both `None`, swapping them does nothing. The function `invertTree(1)` then returns node 1, which is assigned to `inverted_left_subtree` in the `invertTree(2)` call.

11. **Line 7**: Back in the `invertTree(2)` call, we now call `invertTree(root.right)`, which calls `invertTree` on node 3. This process is similar to the process for node 1, and `invertTree(3)` will return node 3, which is assigned to `inverted_right_subtree`.

12. **Line 9 and 10**: Now, in the `invertTree(2)` call, we have `inverted_left_subtree = Node(1)` and `inverted_right_subtree = Node(3)`, we swap `root.left` and `root.right`, so node 2's left child becomes node 3 and its right child becomes node 1. We then return node 2.

13. This process continues in a similar fashion for the right subtree of the original root (node 7) and finally for the root itself (node 4).

By the end of these steps, your tree is fully inverted.

Remember that each time we call `invertTree`, we start a new "execution context". That's why we're able to pause at **Line 6 or 7** and resume after the function call finishes. This is a key aspect of how recursion works.