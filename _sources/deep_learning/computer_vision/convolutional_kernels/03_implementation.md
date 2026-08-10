---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.16.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
myst:
    html_meta:
        "description lang=en": >-
            Implement 2D cross-correlation and the feature-map shape formula
            from scratch in NumPy, then run a Sobel edge-detection kernel and
            verify the shape by hand.
        "keywords": >-
            cross-correlation implementation, numpy convolution, feature map,
            Sobel edge detection, calculate feature map shape, CNN from
            scratch, Python typing
---

# Implementing 2D Cross-Correlation From Scratch in NumPy

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Deep_Learning-blue)
![Tag](https://img.shields.io/badge/Level-Intermediate-yellow)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
```

The cleanest way to confirm you understand an operator is to write it in twenty
lines. This page does exactly that: it re-implements the author's two helper
functions — `cross_correlation_2d` and `calculate_feature_map_shape` — in
idiomatic, fully type-annotated NumPy, then runs them on a small numeric case
and on a real edge-detection kernel.

It is for you if you have read {doc}`the concept chapter <02_concept>` and want
the sliding-window operator under your fingers. By the end you will have a
`cross_correlation_2d` you can step through line by line, a
`calculate_feature_map_shape` that specialises the general output-shape formula
{eq}`eq:conv-output-shape` to stride $1$ and zero padding, and a printed
feature map showing a Sobel-style vertical-edge detector firing exactly on the
edge — the whole story of {doc}`the intro <01_intro>` in twenty executable
lines.

```{admonition} Prerequisites
:class: note

This page implements, as code, the operator defined in
{doc}`the concept chapter <02_concept>`; you will get more out of the
implementation if you have read the definition and the output-shape formula
there first. The notation follows
{doc}`the deep-learning notation page <../../../notations/deep_learning>`.
```

## The output shape, straight from the formula

The concept chapter gives the general shape rule as
$H_{out} = \lfloor (H_{in} + 2P - K_h) / S \rfloor + 1$. The author's helper
covers the common case this chapter starts from — no padding, unit stride — so
we set $P = 0$ and $S = 1$ and the floor disappears:

$$
H_{out} = H_{in} - K_h + 1, \qquad W_{out} = W_{in} - K_w + 1.
$$ (eq:shape-stride1-pad0)

That is the whole of `calculate_feature_map_shape`. It is deliberately tiny: it
exists to keep the sliding window in `cross_correlation_2d` from recomputing
the output size by hand.

```{code-cell} ipython3
def calculate_feature_map_shape(
    x: NDArray[np.floating],
    kernel: NDArray[np.floating],
) -> tuple[int, int]:
    """Feature-map shape for stride 1 and zero padding.

    Specialises the general formula
    ``(H_in + 2 * P - K_h) // S + 1`` to ``P = 0`` and ``S = 1`` — the
    convention this chapter uses until stride and padding are introduced.

    Args:
        x: Input grid of shape ``(H_in, W_in)``.
        kernel: Kernel grid of shape ``(K_h, K_w)``.

    Returns:
        The output shape ``(H_in - K_h + 1, W_in - K_w + 1)``.
    """
    h_in, w_in = x.shape
    k_h, k_w = kernel.shape
    return h_in - k_h + 1, w_in - k_w + 1
```

## The sliding-window operator

With the shape in hand, the operator writes itself. Allocate an output grid of
that shape, then for every anchor position slice the corresponding patch out
of the input and write its dot product with the kernel into one cell. This is
{prf:ref}`def:conv-cross-correlation-2d` translated line for line from
mathematics into NumPy — no flipping, no padding, unit stride, exactly as deep
learning means when it says "convolution".

```{code-cell} ipython3
def cross_correlation_2d(
    x: NDArray[np.floating],
    kernel: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Sliding-window 2D cross-correlation (the ``Conv2d`` operator).

    Slide ``kernel`` across ``x`` in unit strides with no padding. At each
    anchor position, return the dot product of the kernel with the
    ``K_h x K_w`` patch of ``x`` it covers.

    Args:
        x: Input grid of shape ``(H_in, W_in)``.
        kernel: Kernel grid of shape ``(K_h, K_w)``.

    Returns:
        Feature map of shape ``(H_in - K_h + 1, W_in - K_w + 1)``.
    """
    out_h, out_w = calculate_feature_map_shape(x, kernel)
    k_h, k_w = kernel.shape
    out = np.zeros((out_h, out_w), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[i : i + k_h, j : j + k_w]   # K_h x K_w window
            out[i, j] = np.sum(patch * kernel)    # scalar dot product
    return out
```

## A worked numeric call

Before trusting the function on anything interesting, reproduce
{prf:ref}`ex:conv-3x3-on-5x5` from the concept chapter: the $5 \times 5$ ramp
under a $3 \times 3$ all-ones kernel should return the $3 \times 3$ feature map
whose top-left entry is $63$ and whose centre entry is $117$. If those two
match, the sliding window is anchored correctly.

```{code-cell} ipython3
x = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, 20.0],
        [21.0, 22.0, 23.0, 24.0, 25.0],
    ]
)
box = np.ones((3, 3))

feature_map = cross_correlation_2d(x, box)
print("shape:  ", feature_map.shape)      # (3, 3)
print("Y[0,0]: ", feature_map[0, 0])      # 63.0
print("Y[1,1]: ", feature_map[1, 1])      # 117.0
print(feature_map)
```

## An edge-detection demo

Counting sums is fine for checking the anchor logic, but it hides what kernels
are *for*. A more honest demo is an edge detector: a kernel whose dot product
is large exactly when the patch straddles a boundary and near zero over flat
regions. Build a $6 \times 6$ image that is bright ($10$) on the left half and
dark ($0$) on the right — a single vertical edge down the middle — and slide a
classic vertical Sobel-style filter over it
({cite}`zhang2023dive`, Ch. 7.1, discusses exactly this family of hand-crafted
detectors before the network learns its own).

```{code-cell} ipython3
# A 6x6 image: bright block on the left, dark block on the right.
image = np.zeros((6, 6), dtype=np.float64)
image[:, :3] = 10.0

# Vertical-edge detector: +1 on the left column, -1 on the right.
vertical_edge = np.array(
    [
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]
)

edge_map = cross_correlation_2d(image, vertical_edge)
print("feature-map shape:", edge_map.shape)   # (4, 4)
print(edge_map)
```

The feature map is $4 \times 4$, exactly as {eq}`eq:shape-stride1-pad0`
predicts ($(6 - 3 + 1) \times (6 - 3 + 1)$). Its two middle columns read $30$
and the outer columns read $0$: the detector fires only where its $3 \times 3$
window straddles the boundary between the bright and dark halves, and stays
silent over the flat regions on either side. That is the whole job of a
convolutional layer in one picture — light up where the pattern is present,
stay dark where it is not — except that in a real network the kernel entries
are *learned* from data rather than hand-set as they are here.

What this page does not do is multi-channel input, padding, stride, or
backpropagation. Each is a small extension of the loop above: multi-channel
sums one cross-correlation per input channel (see {eq}`eq:conv-multichannel`),
padding surrounds `x` with zeros before the loop, stride skips indices in the
`range`, and backpropagation is the same dot product run in reverse. For all of
those at production speed, use `torch.nn.Conv2d` — but now you know what it is
doing underneath.

```{admonition} Summary
:class: tip

If this page had to be one sentence: **2D cross-correlation is twenty lines of
NumPy — one shape helper and a double loop that takes a patch–kernel dot
product at every anchor — and a hand-set Sobel filter on a synthetic edge
shows exactly the "fire on the pattern, stay dark elsewhere" behaviour that a
learned kernel generalises.**

- `calculate_feature_map_shape` specialises the general output-shape formula
  to stride $1$ and zero padding: $(H - K + 1) \times (W - K + 1)$.
- `cross_correlation_2d` is {prf:ref}`def:conv-cross-correlation-2d` translated
  line for line: slice a patch, multiply, sum, advance.
- The Sobel demo fires ($30$) only where the window straddles the edge;
  learned kernels do the same thing, but discover their weights from data.

For the formal operator, see {doc}`the concept chapter <02_concept>`; for the
three priors that make this worth doing, see {doc}`the intro <01_intro>`.
```

```{admonition} Further reading
:class: seealso

- {doc}`Cross-Correlation vs Convolution: the sliding-window math <02_concept>`
  — the formal definitions this page implements.
- {cite}`zhang2023dive`, Chapter 7.1, is the conceptual source for this
  implementation and covers the hand-crafted edge detectors demoed above.
- The [PyTorch `nn.Conv2d` documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  is the production interface that adds channels, padding, stride, and
  autodiff on top of the loop written here.
```
