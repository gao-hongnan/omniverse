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

# Find All Anagrams in a String

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
[![LeetCode Problem](https://img.shields.io/badge/LeetCode-26-FFA116?style=social&logo=leetcode)](https://leetcode.com/problems/find-all-anagrams-in-a-string)
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)
![Tag](https://img.shields.io/badge/Tag-Sliding_Window-orange)
![Tag](https://img.shields.io/badge/Tag-Hash_Table-orange)
![Tag](https://img.shields.io/badge/Tag-String-orange)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from IPython.display import display
from typing import Generator, List, Union, Any
from rich.pretty import pprint

import sys
from pathlib import Path

def find_root_dir(current_path: Path | None = None, marker: str = '.git') -> Path | None:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path | None
        The starting path to search from. If None, the current working directory
        `Path.cwd()` is used.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path | None
        The path to the root directory. Returns None if the marker is not found.
    """
    if not current_path:
        current_path = Path.cwd()
    current_path = current_path.resolve()
    for parent in [current_path, *current_path.parents]:
        if (parent / marker).exists():
            return parent
    return None

root_dir = find_root_dir(marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.dsa.utils import compare_test_case
else:
    raise ImportError("Root directory not found.")
```

## Problem

### Implementation

```{code-cell} ipython3
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_hashmap = self.get_string_hashmap(string=p)

        output: List[int] = []

        window_size: int = len(p)

        len_s = len(s)
        for index in range(len_s):
            if index == len_s:
                break

            substring = s[index:window_size]
            window_size += 1

            substring_hashmap = self.get_string_hashmap(string=substring)

            if substring_hashmap == p_hashmap:
                # a match
                output.append(index)
        return output


    def get_string_hashmap(self, string: str) -> Dict[str, int]:
        counter = {}

        for char in string:
            if char not in counter:
                counter[char] = 1
            else:
                counter[char] += 1
        return counter
```

### Tests

```{code-cell} ipython3
s = "cbaebabacd"
p = "abc"
expected = [0, 6]
actual = Solution().findAnagrams(s=s, p=p)
compare_test_case(actual=actual, expected=expected, description="Test Case 1")
```
