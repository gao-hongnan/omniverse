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

# Vector and Its Operations

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from IPython.display import display
from typing import Sequence, TypeVar, Optional
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

def find_root_dir(current_path: Path = Path.cwd(), marker: str = '.git') -> Optional[Path]:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path
        The starting path to search from.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path or None
        The path to the root directory. Returns None if the marker is not found.
    """
    current_path = current_path.resolve()
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    return None

root_dir = find_root_dir(marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.utils.visualization.style import use_svg_display
    from omnivault.linear_algebra.plotter import (
        VectorPlotter,
        Vector,
        add_vectors_to_plotter,
        add_text_annotations,
    )
else:
    raise ImportError("Root directory not found.")

use_svg_display()
```

## Vector Addition

```{code-cell} ipython3
:tags: [hide-input, remove-output]

# Create plot using VectorPlotter
fig, ax = plt.subplots(figsize=(9, 9))

plotter = VectorPlotter(
    fig=fig,
    ax=ax,
    ax_kwargs={
        "set_xlim": {"left": 0, "right": 15},
        "set_ylim": {"bottom": 0, "top": 15},
        "set_xlabel": {"xlabel": "x-axis", "fontsize": 16},
        "set_ylabel": {"ylabel": "y-axis", "fontsize": 16},
        "set_title": {"label": "Vector Addition", "size": 18},
    },
)


# Define vectors and colors
vectors = [
    Vector(origin=(0, 0), direction=(4, 7), color="r"),
    Vector(origin=(0, 0), direction=(8, 4), color="b"),
    Vector(origin=(0, 0), direction=(12, 11), color="g"),
    Vector(origin=(4, 7), direction=(8, 4), color="b"),
    Vector(origin=(8, 4), direction=(4, 7), color="r"),
]

add_vectors_to_plotter(plotter, vectors)
add_text_annotations(plotter, vectors)

# Plot and show
plotter.plot()
plotter.save("./assets/02-vector-operation-addition.svg")
```

```{figure} ./assets/02-vector-operation-addition.svg
---
name: 02-vector-operation-vector-addition
---

Vector addition; By Hongnan G.
```
