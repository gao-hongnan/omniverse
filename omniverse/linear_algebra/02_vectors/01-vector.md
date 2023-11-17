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

## Learning Objectives

- Definition of a Vector
  - Vector Operations with both Algebraic and Geometric understanding.
  - cohen2021linear

## Definition

### Geometric Definition

```{prf:definition} Geometric Definition of a Vector
:label: 01-vector-geometric-definition

A **vector** is a mathematical object that possesses both **magnitude** and
**direction**. Geometrically, it is represented as a directed line segment,
where the length of the line indicates its magnitude and its orientation in
space signifies its direction.
```

```{prf:example} Vector versus Coordinate
:label: 01-vector-vector-versus-coordinate

A key distinction in linear algebra is between a **vector** and a **coordinate
in space**:

> For example, in diagram 2.3, while the three coordinates (circles) are
> distinct, the three vectors (lines) are equivalent. This equivalence is
> because each vector represents a movement of 1 unit to the right and 2 units
> down in a two-dimensional space, denoted conventionally as a vector
> $\mathbf{v} = [1, -2]$ (bolded or $\vec{v}$). When positioned at the origin, the head of this vector
> aligns with the coordinate point $(1, -2)$. The takeaway is that all 3 vectors
> have the same magnitude and direction and can be represented by the vector
> $\mathbf{v} = [1, -2]$.
```

```{code-cell} ipython3
:tags: [hide-input, remove-output]

# Create plot using VectorPlotter
fig, ax = plt.subplots(figsize=(9, 9))
plotter = VectorPlotter(
    fig=fig,
    ax=ax,
    ax_kwargs={
        'set_xlim': {'left': -5, 'right': 5},
        'set_ylim': {'bottom': -5, 'top': 5},
        'set_xlabel': {'xlabel': 'X-axis', 'fontsize': 12},
        'set_ylabel': {'ylabel': 'Y-axis', 'fontsize': 12},
        'set_title': {'label': 'Vector Plot with Annotations', 'fontsize': 16},
    }
)

# Define vectors
vector1 = Vector(origin=(0, 0), direction=(1, -2), color="r", label="v1")
vector2 = Vector(origin=(2, 2), direction=(1, -2), color="g", label="v2")
vector3 = Vector(origin=(-2, -2), direction=(1, -2), color="b", label="v3")

# Add vectors and annotations to plotter
for vector in [vector1, vector2, vector3]:
    plotter.add_vector(vector)
    annotation_text = f"{vector.label}: ({vector.direction[0]}, {vector.direction[1]})"
    plotter.add_text(vector.origin[0] + vector.direction[0]/2,
                     vector.origin[1] + vector.direction[1]/2,
                     annotation_text, fontsize=12, color=vector.color)

# Plot and show
plotter.plot()
plotter.save("./assets/01-vector-vector-versus-coordinate.svg")
```

```{figure} ./assets/01-vector-vector-versus-coordinate.svg
---
name: 01-vector-vector-versus-coordinate
---

Three of the same vectors with different starting coordinates; By Hongnan G.
```

### Vector is Invariant under Coordinate Transformation

The **_geometric interpretation_** of vectors is crucial, and this aspect
deserves special emphasis. We often state that a **_vector is invariant under
coordinate
transformation_**[^vector-is-invariant-under-coordinate-transformation].

Consider a vector $\mathbf{v}$ in a vector space. Mathematically, this can be
represented as the following:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_{D}
\end{bmatrix} \subseteq \mathbb{R}^{D},
$$

where $v_d$ is the $d$th component of the vector. The vector $\mathbf{v}$ is in
an $D$-dimensional space. The important concept to grasp here is that the vector
$\mathbf{v}$ itself is an abstract entity, distinct from how it's represented in
any particular coordinate system.

```{prf:theorem} Vector is Invariant under Coordinate Transformation
:label: 01-vector-is-invariant-under-coordinate-transformation

A given vector, $\mathbf{v}$, remains the same entity, irrespective of the
coordinate system used to describe it. This property is referred to as the
_**invariance of a vector under coordinate transformations**_. In essence, the
vector's intrinsic properties—its direction and magnitude—do not change, even
though its coordinates might vary with different choices of basis.

For example, consider the vector $\mathbf{v}$ in a two-dimensional space. In
one coordinate system, $\mathbf{v}$ might have coordinates $(x, y)$, but
in a rotated coordinate system, its coordinates could appear different, say
$(x', y')$. Despite this change in representation, the vector $\mathbf{v}$
itself has not changed; it still has the same length and points in the same
direction in space. This is perfectly illustrated in {numref}`01-vector-vector-versus-coordinate`,
where the three vectors have different coordinates but are equivalent because
they have the same orientation (direction) and length (magnitude).

The concept of **_basis_** is central to this idea. A basis provides a frame
of reference for describing vectors. Changing the basis is akin to changing
the viewpoint but not the vector itself. At this juncture, delving deeply into
the concept of basis might be overwhelming, but it's helpful to think of a
basis as our point of reference. In our discussions, we are considering the
origin as this reference point.
```

This idea is fundamental in understanding how vectors behave in different
coordinate systems and underlines their abstract nature, independent of any
particular representation.

### Algebraic Definition

```{prf:definition} Algebraic Definition of a Vector
:label: 01-vector-algebraic-definition

In the context of linear algebra, a **vector** $\mathbf{v}$ within an
$D$-dimensional space over a field $\mathbb{F}$ is defined as an ordered
$D$-tuple of elements from $\mathbb{F}$. Specifically, if $\mathbb{F}$ is a
field (such as the real numbers $\mathbb{R}$ or the complex numbers
$\mathbb{C}$) and $D$ is a positive integer, then a vector $\mathbf{v}$ with $D$
entries $v_1, v_2, \cdots, v_D$, where each $v_d$ belongs to $\mathbb{F}$, is
termed an _$D$-vector_[^why-D] over $\mathbb{F}$.

Mathematically, $\mathbf{v}$ is represented as:

$$
\mathbf{v} = (v_1, v_2, \cdots, v_D), \text{ where } v_d \in \mathbb{F} \text{ for each } d = 1, 2, \cdots, D
$$

This notation emphasizes that $\mathbf{v}$ is an ordered collection of elements,
where the order of these elements is crucial to the definition of the vector.
The set of all such $D$-vectors over $\mathbb{F}$ is denoted by $\mathbb{F}^D$.

In the context of **vector spaces**, which will be explored in more detail
later, these $D$-vectors form the fundamental elements of the space, adhering to
specific rules of addition and scalar multiplication consistent with the
properties of the field $\mathbb{F}$. This algebraic perspective is essential in
understanding the structure and operations within vector spaces.
```

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
plotter.save("./assets/01-vector-addition.svg")
```

```{figure} ./assets/01-vector-addition.svg
---
name: 01-vector-vector-addition
---

Vector addition; By Hongnan G.
```

## Equality of Vectors

```{prf:definition} Equality of Vectors
:label: 01-vector-equality-of-vectors

- By definition of the geometrical interpretation of vectors, two vectors are
  **equal if and only if they have the same magnitude in the same direction**,
  which is why even though {numref}`01-vector-vector-versus-coordinate`'s 3 vectors look visually different, but
  are actually the same vector.
- By definition of the algebraical interpretation of vectors, two vectors $\mathbf{v}_1$
  and $\mathbf{v}_2$ are **equal if and only if each elements of** $\mathbf{v}_1$ is equal to
  $\mathbf{v}_2$.
```

[^vector-is-invariant-under-coordinate-transformation]:
    See
    [What does it mean for a vector to remain invariant under coordinate transformation?](https://www.physicsforums.com/threads/what-does-it-mean-for-a-vector-to-remain-invariant-under-coordinate-transformation.517681/)

[^why-D]:
    The choice of $D$ to represent dimensionality of a vector is chosen on
    purpose where $D$ is a common notation in deep learning to represent the
    dimensionality of a vector space in which the embedding resides.
