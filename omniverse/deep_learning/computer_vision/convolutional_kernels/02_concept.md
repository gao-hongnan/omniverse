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
            CNNs say convolution but compute cross-correlation. Learn the
            sliding-window dot product, stride, padding, and the exact
            feature-map output-shape formula.
        "keywords": >-
            cross-correlation, convolution, 2D convolution, stride, padding,
            feature map, output shape, multi-channel convolution, CNN
---

# Cross-Correlation vs Convolution: The Sliding-Window Math

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

import numpy as np
```

Open PyTorch's `nn.Conv2d` or Keras's `Conv2D` and the documentation calls the
operation a *convolution*. It is not — not in the sense every
signal-processing textbook uses. Deep learning computes a **cross-correlation**:
the kernel slides over the image without being flipped, taking a dot product at
every position. The difference is a single sign on the kernel indices, and
because the kernel is *learned* it costs the network nothing. But the naming
is a real divergence, and naming it honestly is the point of this page.

This is the concept half of the chapter. It is for you if you understood
{doc}`the intro's three priors <01_intro>` and now want the precise
sliding-window operator behind them. By the end you can write down 2D
cross-correlation entry by entry, derive the output shape under any stride and
padding, explain the multi-channel weight tensor, and say exactly why a
"convolutional" layer that omits the flip is still a convolutional layer.

```{admonition} Prerequisites
:class: note

This page builds directly on {doc}`the convolutional-kernels intro <01_intro>`,
which motivates locality, parameter sharing, and translation equivariance.
The notation follows
{doc}`the deep-learning notation page <../../../notations/deep_learning>`.
The from-scratch code for every operator defined here lives in
{doc}`the implementation chapter <03_implementation>`.
```

## Cross-correlation: the sliding dot product

Intuition first. Place a small kernel $\mathbf{K}$ over the top-left corner of
the image $\mathbf{X}$, multiply overlapping entries element by element, sum
the products into a single number, and write that number to the top-left entry
of the output. Slide the kernel one column right and repeat; at the end of the
row, drop one row down and start again. The output grid you fill in is the
*feature map*: at every position it records how well the kernel matched the
patch underneath.

That is the entire operator. Formally:

```{prf:definition} 2D cross-correlation
:label: def:conv-cross-correlation-2d

Let $\mathbf{X} \in \R^{H \times W}$ be an input image and
$\mathbf{K} \in \R^{K_h \times K_w}$ a kernel. The **2D cross-correlation** of
$\mathbf{X}$ with $\mathbf{K}$ is the matrix
$\mathbf{Y} \in \R^{(H - K_h + 1) \times (W - K_w + 1)}$ whose $(i, j)$ entry
is

$$
Y_{i,j} \;\defeq\; \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1}
X_{i+u,\; j+v}\, K_{u,v}.
$$ (eq:cross-correlation-entry)

Each output element is the dot product of the kernel with the
$K_h \times K_w$ patch of $\mathbf{X}$ anchored at row $i$, column $j$.
Equivalently, $\mathbf{Y}$ is the matrix of inner products between $\mathbf{K}$
and every same-shaped window of $\mathbf{X}$.
```

Two conventions are hidden in that definition and are worth pulling out. There
is **no flip**: the kernel index $K_{u,v}$ multiplies $X_{i+u, j+v}$ with the
*same* sign on both axes, not $X_{i-u, j-v}$. And there is **no padding and
unit stride**: the kernel stops as soon as it no longer fits, which is why the
output is smaller than the input. Both are relaxed below.

## A worked example: a 3×3 kernel on a 5×5 input

```{prf:example} A 3×3 box kernel on a 5×5 input
:label: ex:conv-3x3-on-5x5

Take the $5 \times 5$ input and $3 \times 3$ all-ones kernel

$$
\mathbf{X} =
\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 10 \\
11 & 12 & 13 & 14 & 15 \\
16 & 17 & 18 & 19 & 20 \\
21 & 22 & 23 & 24 & 25
\end{bmatrix},
\qquad
\mathbf{K} =
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}.
$$

The output shape is $(5 - 3 + 1) \times (5 - 3 + 1) = 3 \times 3$. The
top-left entry is the dot product of $\mathbf{K}$ with the top-left $3 \times 3$
patch of $\mathbf{X}$:

$$
Y_{0,0} = 1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13 = 63.
$$ (eq:conv-example-y00)

One step down and one across, the window anchored at $X_{1,1} = 7$ sums to

$$
Y_{1,1} = 7 + 8 + 9 + 12 + 13 + 14 + 17 + 18 + 19 = 117.
$$ (eq:conv-example-y11)

Repeating for all nine anchors gives the full feature map

$$
\mathbf{Y} =
\begin{bmatrix}
63 & 72 & 81 \\
108 & 117 & 126 \\
153 & 162 & 171
\end{bmatrix}.
$$ (eq:conv-example-full)

Two patterns are worth noticing. Each step *right* adds $9$ — three rows times
the $+3$ gained by dropping column $k$ and adding column $k+3$ into the window.
Each step *down* adds $45$, three columns times the $+15$ between adjacent rows
of $\mathbf{X}$. A box kernel is a local mass detector, and the feature map
faithfully records how much mass sits under each window.
```

The arithmetic is mechanical, so let a machine check it. The cell below
reproduces every entry of {prf:ref}`ex:conv-3x3-on-5x5` using nothing but
NumPy — the same operator we will package as `cross_correlation_2d` in the
implementation chapter.

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

# Manual sliding-window cross-correlation.
out_h, out_w = x.shape[0] - box.shape[0] + 1, x.shape[1] - box.shape[1] + 1
y = np.zeros((out_h, out_w))
for i in range(out_h):
    for j in range(out_w):
        y[i, j] = np.sum(x[i : i + 3, j : j + 3] * box)

print("shape:", y.shape)        # (3, 3)
print("Y[0, 0]:", y[0, 0])      # 63.0
print("Y[1, 1]:", y[1, 1])      # 117.0
print(y)
```

## Stride and padding

The definition above shrinks the output by $K_h - 1$ rows and $K_w - 1$
columns. Two knobs restore control over the output size.

**Padding** $P$ surrounds the input with $P$ rings of zeros on every side, so
the kernel can sit against — and even hang over — the original border. With
$P = 1$ and a $3 \times 3$ kernel, the feature map keeps the input's height and
width (the so-called "same" padding).

**Stride** $S$ is how far the kernel steps between anchors. $S = 1$ visits
every position; $S = 2$ visits every other, halving each spatial dimension and
turning the convolutional layer into a learned downsampler.

Putting both into {prf:ref}`def:conv-cross-correlation-2d` gives the
output-shape formula that every framework's `Conv2d` obeys:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K_h}{S} \right\rfloor + 1,
\qquad
W_{out} = \left\lfloor \frac{W_{in} + 2P - K_w}{S} \right\rfloor + 1.
$$ (eq:conv-output-shape)

The floor is what handles strides that do not divide the padded input evenly;
the framework simply drops the sliver of input the last kernel position would
need. Sanity checks: with $P = 0$, $S = 1$ the formula recovers
$H - K_h + 1$ from {prf:ref}`def:conv-cross-correlation-2d`; with
$P = 1$, $S = 1$, $K_h = 3$ it gives $H_{out} = H_{in}$.

## Multiple input and output channels

Real images are not single-channel. An RGB input has $C_{in} = 3$ channels, and
a hidden layer deep in a network may have hundreds. The operator generalises by
giving the kernel its own depth.

Each **output channel** $c$ is produced by a stack of $C_{in}$ two-dimensional
kernels $\mathbf{W}_{c, c'} \in \R^{K_h \times K_w}$, one per input channel.
Cross-correlate each kernel with its matching input channel and **sum** the
results, then add a per-channel bias $b_c$:

$$
Y_{c,\, i,\, j} \;\defeq\; b_c + \sum_{c'=1}^{C_{in}}
\sum_{u, v} X_{c',\, i+u,\, j+v}\, W_{c,\, c',\, u,\, v}.
$$ (eq:conv-multichannel)

The full weight tensor is therefore
$\mathbf{W} \in \R^{C_{out} \times C_{in} \times K_h \times K_w}$, with a bias
vector $\mathbf{b} \in \R^{C_{out}}$. The spatial shape formula
{eq}`eq:conv-output-shape` is unchanged; it now describes each of the $C_{out}$
output channels independently. This is the tensor shape you will see reported
by any framework's `Conv2d`, and it is the reason a single convolutional layer
holds $C_{out} \cdot C_{in} \cdot K_h \cdot K_w$ kernel weights rather than
just $K_h \cdot K_w$.

## Why we still call it "convolution"

```{prf:remark} Convolution flips the kernel; deep learning does not
:label: rem:conv-vs-correlation

In signal processing, 2D *convolution* flips the kernel in both axes before
sliding it:

$$
(\mathbf{X} * \mathbf{K})_{i,j} \;\defeq\;
\sum_{u, v} X_{i-u,\; j-v}\, K_{u,v}.
$$ (eq:true-convolution)

The operator every framework calls `Conv2d` drops that flip — it computes
{prf:ref}`def:conv-cross-correlation-2d`, not {eq}`eq:true-convolution`. This
is a genuine divergence in notation, and it is worth saying plainly rather than
papering over.

It is also harmless in practice. Because the kernel entries are learned, a
network trained under cross-correlation with kernel $\mathbf{K}$ computes
exactly the family of functions it would under convolution with the flipped
$\widetilde{\mathbf{K}}_{u,v} = K_{-u,-v}$. Gradient descent simply learns the
flipped version if it needs to. So the misnomer changes nothing about
representational power — only the sign convention on the indices — which is why
the entire field has tolerated it for decades.
```

```{admonition} Summary
:class: tip

If this page had to be one sentence: **deep learning's "convolution" is an
unflipped cross-correlation — a shared kernel taking a dot product at every
sliding position — and its output shape is
$\lfloor (H_{in} + 2P - K)/S \rfloor + 1$ per spatial axis, with
$C_{out} \times C_{in}$ kernels turning one input stack into $C_{out}$ output
channels.**

- Cross-correlation (no flip) is what `Conv2d` actually computes; true
  convolution {eq}`eq:true-convolution` flips the kernel, but the flip is
  absorbed by learning ({prf:ref}`rem:conv-vs-correlation`).
- The output-shape formula {eq}`eq:conv-output-shape` unifies padding $P$ and
  stride $S$; it specialises to $H - K + 1$ when $P = 0, S = 1$.
- Multi-channel convolution {eq}`eq:conv-multichannel` sums one cross-correlation
  per input channel, producing a
  $\R^{C_{out} \times C_{in} \times K_h \times K_w}$ weight tensor.

Next: {doc}`the from-scratch NumPy implementation <03_implementation>`, which
packages the operator above into `cross_correlation_2d` and runs a Sobel-style
edge detector on a synthetic image. For the motivating priors, see
{doc}`the intro <01_intro>`.
```

```{admonition} Further reading
:class: seealso

- {doc}`Implementing 2D Cross-Correlation From Scratch in NumPy <03_implementation>`
  — the operators of this page as executable code.
- {cite}`zhang2023dive`, Chapter 7.1, develops the same operator and the
  output-shape formula in the standard textbook treatment.
- The [PyTorch `nn.Conv2d` documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  lists the exact `stride`, `padding`, and channel conventions used in
  production code.
```
