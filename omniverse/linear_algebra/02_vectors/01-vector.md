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

def find_root_dir(current_path: Path, marker: str) -> Optional[Path]:
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

current_file_path = Path("__file__")
root_dir          = find_root_dir(current_file_path, marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.utils.visualization.style import use_svg_display
    from omnivault.linear_algebra.plot import VectorPlotter
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

```{figure} https://storage.googleapis.com/reighns/reighns_ml_projects/docs/linear_algebra/linear_algebra_theory_intuition_code_chap2_fig_2.3.svg
---
name: 01-vector-vector-versus-coordinate
---

3 of the same vectors with different starting coordinates; By Hongnan G.
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
direction in space. This is perfectly illustrated in {numref}`01-vector-vector-versus-coordinate`.

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

[^1]:
    [What does it mean for a vector to remain invariant under coordinate transformation?](https://www.physicsforums.com/threads/what-does-it-mean-for-a-vector-to-remain-invariant-under-coordinate-transformation.517681/)

### Algebraic Definition (Vectors)

```{prf:definition} Algebraic Definition of a Vector
:label: 01-vector-algebraic-definition

For a field $\mathbb{F}$ and a positive integer $n$, a **vector** $\mathbf{v}$ with $n$ entries
$v_1, v_2, \cdots, v_n$, each $v_i$ belonging to $\mathbb{F}$, is called an $n$-vector
over $\mathbb{F}$. In particular, $\mathbf{v}$ is **ordered** and can be represented
mathematically as:

$$\mathbf{v} = \{(v_1, v_2, \cdots, v_n) ~|~ v_i \in \mathbb{F}\}$$

The set of $n$-vectors over $\mathbb{F}$ is denoted $\mathbb{F}^n$. We will deal with this more
in vector spaces, just keep a mental note here.
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize = (9,9))

vec = np.array([[[0, 0, 4, 7]],
               [[0, 0, 8, 4]],
               [[0, 0, 12, 11]],
               [[4, 7, 8, 4]],
               [[8, 4, 4, 7]]])
color = ['r','b','g','b','r']

for i in range(vec.shape[0]):
    X,Y,U,V = zip(*vec[i,:,:])
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', color = color[i], scale=1, alpha = .6)

ax.set_xlim([0, 15])
ax.set_ylim([0, 15])
ax.set_xlabel('x-axis', fontsize =16)
ax.set_ylabel('y-axis', fontsize =16)
ax.grid()

for i in range(3):
    ax.text(x = vec[i,0,2], y = vec[i,0,3], s = '(%.0d, %.0d)' %(vec[i,0,2],vec[i,0,3]), fontsize = 16)

ax.text(x= vec[0,0,2]/2, y = vec[0,0,3]/2, s= '$u$', fontsize = 16)
ax.text(x= 8, y = 9, s= '$v$', fontsize = 16)
ax.text(x= 6, y = 5.5, s= '$u+v$', fontsize = 16)

ax.set_title('Vector Addition', size = 18)
plt.show()
```

## Equality of Vectors

```{prf:definition} Equality of Vectors
:label: 01-vector-equality-of-vectors

- By definition of the geometrical interpretation of vectors, two vectors are
  **equal if and only if they have the same magnitude in the same direction**,
  which is why even though figure 2.3's 3 vectors look visually different, but
  are actually the same vector.
- By definition of the algebraical interpretation of vectors, two vectors $\v_1$
  and $\v_2$ are **equal if and only if each elements of** $\v_1$ is equal to
  $\v_2$.
```

[^vector-is-invariant-under-coordinate-transformation]:
    See
    [What does it mean for a vector to remain invariant under coordinate transformation?](https://www.physicsforums.com/threads/what-does-it-mean-for-a-vector-to-remain-invariant-under-coordinate-transformation.517681/)
