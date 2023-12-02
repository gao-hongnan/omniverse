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

# Systems of Linear Equations

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import rich
from IPython.display import display

def find_root_dir(current_path: Path = Path.cwd(), marker: str = '.git') -> Optional[Path]:
    """
    Find the root directory by searching for a directory or file that serves as
    a marker.

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
        VectorPlotter2D,
        VectorPlotter3D,
        add_vectors_to_plotter,
        add_text_annotations,
    )
    from omnivault.linear_algebra.vector import Vector2D, Vector3D
else:
    raise ImportError("Root directory not found.")

use_svg_display()
```

```{code-cell} ipython3
x = np.linspace(-5, 5, 100)
y1 = -x + 6
y2 = x + 4

fig, ax = plt.subplots(figsize = (12, 7))
ax.scatter(1, 5, s = 200, zorder=5, color = 'r', alpha = .8)

ax.plot(x, y1, lw =3, label = '$x+y=6$')
ax.plot(x, y2, lw =3, label = '$x-y=-4$')
ax.plot([1, 1], [0, 5], ls = '--', color = 'b', alpha = .5)
ax.plot([-5, 1], [5, 5], ls = '--', color = 'b', alpha = .5)
ax.set_xlim([-5, 5])
ax.set_ylim([0, 12])

ax.legend()
s = '$(1,5)$'
ax.text(1, 5.5, s, fontsize = 20)
ax.set_title('Solution of $x+y=6$, $x-y=-4$', size = 22)
ax.grid()
plt.show()
```

This is a first introduction to systems of linear equations. We'll start with a
motivating example, then define the general form of linear equations, and
finally discuss the geometric interpretation of linear equations. This is
definitely not the most rigorous treatment of the topic, but it should be
sufficient for our purposes. We will dive deeper into the topic in later
chapters, talking about it through the lens of vectors and matrices.

## Motivation

Let's explore a detailed example to better motivate systems of linear equations.
We'll examine Example 2.1 from Section 2.1, titled 'Systems of Linear
Equations', on page 19 of the book 'Mathematics for Machine Learning' by
Deisenroth, Faisal, and Ong (2020) {cite}`deisenroth2020mathematics`. This
example provides an insightful application of these concepts in a practical
context."

1. **Context**: We have a company that produces a set of products
   $N_1, \ldots, N_n$. These products require resources $R_1, \ldots, R_m$ to be
   produced.

2. **Resource Requirements**: Each product $N_j$ requires a certain amount of
   each resource $R_i$. This amount is denoted by $a_{ij}$. For instance,
   $a_{ij}$ is the amount of resource $R_i$ needed to produce one unit of
   product $N_j$.

3. **Objective**: The company wants to find an optimal production plan. This
   means deciding how many units $x_j$ of each product $N_j$ to produce, given
   the constraint of available resources.

4. **Available Resources**: The total available units of each resource $R_i$ is
   given by $b_i$.

5. **System of Linear Equations**: The heart of the problem is to determine the
   values of $x_j$ (the quantity of each product to produce) such that all
   resources are optimally used (ideally, no resources are left over).

    To do this, we set up a system of linear equations. For each resource $R_i$,
    the total consumption by all products should be equal to the available
    amount of that resource $b_i$. This leads to the equation:

    ```{math}
    :label: 02-systems-of-linear-equations-eq-1

    a_{i1}x_1 + a_{i2}x_2 + \cdots + a_{in}x_n = b_i
    ```

    for each $i = 1, \ldots, m$. Here, $a_{i1}x_1$ represents the amount of
    resource $R_i$ used by product $N_1$, $a_{i2}x_2$ by product $N_2$, and so
    on.

6. **Solution**: Consequently, an **_optimal production plan_**
   $\left(x_1^*,
   \ldots, x_n^*\right)$ is one that satisfies the system of
   linear equations in {eq}`02-systems-of-linear-equations-eq-1`:

    ```{math}
    :label: 02-systems-of-linear-equations-eq-2

    \begin{aligned}
        & a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n && = && \ b_1 \\
        & a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n && = && \ b_2 \\
        & \vdots && && \ \vdots \\
        & a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n && = && \ b_m
    \end{aligned}
    ```

    where

    - $a_{ij} \in \mathbb{R}$ is the amount of resource $R_i$ needed to produce
      one unit of product $N_j$.
    - $b_i \in \mathbb{R}$ is the total available units of resource $R_i$. This
      is the constraint on the amount of resource $R_i$ that can be used.

    Equation {eq}`02-systems-of-linear-equations-eq-2` is what we call the
    general form of a _system of linear equations_. The unknowns are the
    quantities $x_1, \ldots, x_n$ of each product to produce. The coefficients
    $a_{ij}$ represent the amount of resource $R_i$ needed to produce one unit
    of product $N_j$. The constants $b_i$ represent the total available units of
    resource $R_i$. Finally, every $n$-tuple $\left(x_1^*, \ldots, x_n^*\right)$
    that satisfies {eq}`02-systems-of-linear-equations-eq-2` is a solution to
    the system of linear equations.

    In other words, the optimal production plan is the solution to the system of
    linear equations. This solution is unique if the system is
    [**consistent**](https://en.wikipedia.org/wiki/Consistent_and_inconsistent_equations)
    (i.e., has at least one solution). If the system is
    [**inconsistent**](https://en.wikipedia.org/wiki/Consistent_and_inconsistent_equations)
    (i.e., has no solution), then there is no optimal production plan.

In summary, this example illustrates how a practical problem in production
planning can be modeled using linear algebra. The system of linear equations is
central to finding an optimal solution that maximizes resource utilization.

### Analogy: Bakery

1. **Products ($N_j$)**: Imagine a bakery that makes various types of bread and
   pastries. These could be:

    - $N_1$: Loaves of whole wheat bread
    - $N_2$: Baguettes
    - $N_3$: Croissants
    - ... and so on, up to $N_n$ being the nth type of bread or pastry.

2. **Resources ($R_i$)**: The resources are the ingredients and materials needed
   to make these breads and pastries. Examples include:

    - $R_1$: Flour
    - $R_2$: Yeast
    - $R_3$: Butter
    - $R_4$: Sugar
    - ... up to $R_m$, the mth resource.

3. **Resource Requirements ($a_{ij}$)**: Each type of bread or pastry requires
   specific amounts of these ingredients. For example:

    - To make one loaf of whole wheat bread ($N_1$), you might need 2 units of
      flour ($a_{11} = 2$), 1 unit of yeast ($a_{21} = 1$), and no sugar
      ($a_{41} = 0$).

4. **Optimal Production Plan**: The bakery needs to decide how many of each type
   of bread and pastry to bake each day. This decision is based on the available
   ingredients. For instance, if they have 100 units of flour, 50 units of
   yeast, 30 units of butter, and 20 units of sugar, how many loaves of whole
   wheat bread, baguettes, croissants, etc., should they bake to use all these
   ingredients efficiently without any waste?

5. **System of Linear Equations**: This situation can be modeled as a system of
   linear equations. Each equation corresponds to one resource, equating the
   total amount of that resource used by all products to the available amount.
   Solving these equations gives the bakery the optimal number of each type of
   bread and pastry to bake.

## General Form of Linear Equations

Without touching on the concept of vectors and matrices, we can define a system
of linear equations purely in terms of algebraic equations with a geometric
interpretation.

### System of Linear Equations (Algebraic Form)

````{prf:definition} System of Linear Equations (Algebraic Form)
:label: 02-systems-of-linear-equations-definition-algebraic-form

A general system of $N$ linear equations with $D$ unknowns is given by:

```{math}
:label: 02-systems-of-linear-equations-definition-algebraic-form-eq-1

\begin{aligned}
    & a_{11}x_1 + a_{12}x_2 + \cdots + a_{1D}x_D && = && \ b_1 \\
    & a_{21}x_1 + a_{22}x_2 + \cdots + a_{2D}x_D && = && \ b_2 \\
    & \ \ \vdots \\
    & a_{N1}x_1 + a_{N2}x_2 + \cdots + a_{ND}x_D && = && \ b_N
\end{aligned}
```

where $x_{n,d}$ (for $n = 1, \ldots, N$ and $d = 1, \ldots, D$) are the unknowns
and $a_{n,d}$ (for $n = 1, \ldots, N$ and $d = 1, \ldots, D$) and $b_n$ (for
$n = 1, \ldots, N$) are the coefficients and constants, respectively.
````

The choice of using $N$ and $D$ for the number of equations and unknowns instead
of $m$ and $n$ is intentional. In the context of machine learning, the notations
$N$ and $D$ are commonly used to represent the following:

-   $N$: The number of samples or observations in a dataset. In a dataset
    comprising multiple individual data points (like patients in a medical
    study, images in a computer vision task, or days in a time series analysis),
    $N$ is typically used to denote the total count of these data points.

-   $D$: The number of features or dimensions for each sample. In machine
    learning, each data point is often described by a set of features or
    attributes. For example, in a dataset of houses, the features might include
    the number of rooms, square footage, location, age of the building, etc. $D$
    represents the total count of these features.

Furthermore, in machine learning context, the $x_{n,d}$ presented in
{eq}`02-systems-of-linear-equations-definition-algebraic-form-eq-1` are often
referred to as **_features_** or **_attributes_**. The $a_{n,d}$ are referred to
as **_weights_** or **_coefficients_**. The $b_n$ are referred to as
**_labels_** or **_targets_**. For that reason, we often denote the linear
equations in {eq}`02-systems-of-linear-equations-definition-algebraic-form-eq-1`
as:

```{math}
:label: 02-systems-of-linear-equations-definition-algebraic-form-eq-2

\begin{aligned}
    & \theta_{1}x_{1,1} + \theta_{2}x_{1,2} + \cdots + \theta_{D}x_{1,D} && = && \ y_1 \\
    & \theta_{1}x_{2,1} + \theta_{2}x_{2,2} + \cdots + \theta_{D}x_{2,D} && = && \ y_2 \\
    & \ \ \vdots \\
    & \theta_{1}x_{N,1} + \theta_{2}x_{N,2} + \cdots + \theta_{D}x_{N,D} && = && \ y_N
\end{aligned}
```

where $\theta_{d} \in \mathbb{R}$ (for $d = 1, \ldots, D$) are the coefficients
or weights associated with each feature, $x_{n,d} \in \mathbb{R}$ represents the
value of the $d$-th feature for the $n$-th sample, and $y_n \in \mathbb{R}$ (for
$n = 1, \ldots, N$) are the target or outcome values for each sample. Also
notice that the coefficients $\theta_{d}$ are the same for each sample $n$.

### System of Linear Equations (Geometric Interpretation)

Consider a system of linear equations with two unknowns $x_1$ and $x_2$:

```{math}
:label: 02-systems-of-linear-equations-geometric-interpretation-eq-1

\begin{aligned}
    & 4x_1 + 4x_2 && = && \ 5 \\
    & 2x_1 - 4x_2 && = && \ 0
\end{aligned}
```

## References and Further Readings

-   [Wikipedia: System of linear equations](https://en.wikipedia.org/wiki/System_of_linear_equations)
-   Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). _Mathematics for
    Machine Learning_. Cambridge University Press. (Chapter 3.1, Norms).
-   https://github.com/weijie-chen/Linear-Algebra-With-Python/blob/master/Chapter%201%20-%20Linear%20Equation%20System.ipynb
-   https://github.com/fastai/numerical-linear-algebra
-   [A First Course in Linear Algebra by Ken Kuttler](<https://math.libretexts.org/Bookshelves/Linear_Algebra/A_First_Course_in_Linear_Algebra_(Kuttler)>)
-   https://math.stackexchange.com/questions/1634411/why-adding-or-subtracting-linear-equations-finds-their-intersection-point
