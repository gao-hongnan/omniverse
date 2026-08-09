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
            The four design principles behind convolutional networks —
            translation equivariance, locality, weight sharing, abstraction —
            plus a concise CNN glossary of kernel, filter, feature map,
            receptive field, padding, and stride.
        "keywords": >-
            CNN, convolutional neural network, translation equivariance,
            translation invariance, locality, weight sharing, receptive field,
            kernel, filter, feature map, padding, stride
---

# CNN Design Principles: Locality, Weight Sharing, Equivariance

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Deep_Learning-blue)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

A convolutional layer is not defined only by its arithmetic — there is a
{doc}`separate page for that <convolutional_kernels/02_concept>`. It is defined
by four design choices that distinguish it from a dense layer: it looks at
*local* neighborhoods, it *shares weights* across positions, it responds
*equivariantly* to shifts, and its representations grow more *abstract* with
depth. This page is the concise reference for those principles and the
vocabulary they introduce. Read it for the why; read
{doc}`the convolution chapter <convolutional_kernels/01_intro>` for the how.

By the end you will be able to state each principle in one sentence, explain
the difference between translation *equivariance* and translation *invariance*
(a distinction the literature routinely blurs), and decode the CNN vocabulary —
kernel versus filter, feature map, receptive field, padding, stride — without
confusion.

```{admonition} Prerequisites
:class: note

This assumes the
{doc}`computer-vision chapter introduction <01_intro>`. The full arithmetic of
cross-correlation, padding, stride, and the output-shape formula lives in
{doc}`the convolution concept page <convolutional_kernels/02_concept>`; this
page summarizes and links rather than re-deriving.
```

## The four design principles

```{prf:property} Translation equivariance
:label: prp:translation-equivariance

A layer is **translation equivariant** when a shift of the input produces the
same shift of the output: $f(\text{shift}(\mathbf{X})) =
\text{shift}(f(\mathbf{X}))$. A convolutional layer is translation equivariant
because the same kernel is applied at every position, so whatever a shift moves
in the input moves identically in the feature map {cite}`zhang2023dive`.
```

```{admonition} Equivariance is not invariance
:class: warning

The literature (and many course notes) use *translation invariance* and
*translation equivariance* interchangeably — they are not the same.
**Equivariance** means the output *moves with* the input; **invariance** means
the output is *unchanged* by the input shift. A convolutional layer is
equivariant. Translation *invariance* is introduced later, by **pooling** layers
that collapse spatial location, not by the convolution itself. Conflating the
two is the single most common CNN vocabulary error.
```

```{prf:property} Locality
:label: prp:locality

The earliest layers should attend to **local regions** of the input and ignore
distant content {cite}`zhang2023dive`. A kernel of size $K \times K$ enforces
this: each output element depends only on a $K \times K$ patch of the input.
Local patterns — edges, corners, textures — are aggregated across layers into
whole-image understanding, so locality at the bottom is what enables globally
meaningful features at the top.
```

```{prf:property} Weight (parameter) sharing
:label: prp:weight-sharing

The same kernel weights are **reused at every spatial position**. An edge
detector useful in the top-left corner is useful in the bottom-right too, so
the network learns one detector and applies it everywhere rather than
relearning it per location. This is what slashes the parameter count relative
to a dense layer and what makes the layer translation equivariant — both
consequences follow from the single decision to share.
```

```{prf:property} Increasing abstraction
:label: prp:abstraction

As depth grows, representations become **more abstract** and less tied to exact
pixel positions. Early layers detect edges and color blobs; deeper layers
compose them into textures, parts, and objects. Depth, combined with locality
and pooling, is what turns pixels into semantics.
```

```{admonition} Intuition — the four principles as one story
:class: tip

*Locality* says "look nearby"; *weight sharing* says "look the same way
everywhere"; *equivariance* is what follows mechanically from sharing;
*abstraction* is what depth builds on top. State them in that order and the
convolutional layer's design reads as a single argument, not four disconnected
rules.
```

## A CNN glossary

These terms recur throughout the chapter; the full derivations live in
{doc}`the convolution concept page <convolutional_kernels/02_concept>`.

```{list-table}
:header-rows: 1
:widths: 22 78

* - Term
  - Meaning
* - **Image**
  - A tensor $\mathbf{X} \in \R^{C \times H \times W}$: $C$ channels, $H$ rows,
    $W$ columns. A grayscale image has $C = 1$; RGB has $C = 3$.
* - **Kernel**
  - A small 2D weight matrix, typically $K \times K$, slid across the image.
* - **Filter**
  - A 3D weight tensor of shape $C_{\text{in}} \times K \times K$ — a stack of
    one kernel per input channel. See {prf:ref}`rem:filter-size`.
* - **Feature map**
  - The output of applying one filter across the image; "feature map" and
    "output channel" are used interchangeably {cite}`zhang2023dive`.
* - **Receptive field**
  - The region of the *input* that contributes to a single output element. It
    grows with depth: a unit two layers deep sees a patch of the original
    image larger than one layer's kernel.
* - **Stride** $s$
  - How many pixels the kernel shifts per step. $s > 1$ downsamples.
* - **Padding** $p$
  - Zeros added around the input. *Valid* padding adds none (output shrinks);
    *same* padding adds $p = (K - 1)/2$ (for odd $K$) so the output keeps the
    input's spatial size.
* - **Output size**
  - For input $n \times n$, kernel $k$, padding $p$, stride $s$:
    $\lfloor (n + 2p - k) / s \rfloor + 1$ per spatial dimension.
```

## Two distinctions worth memorizing

```{prf:remark} A "5×5 filter" is not a 5×5 matrix
:label: rem:filter-size

Saying "a $5 \times 5$ filter" does **not** mean the filter is a $5 \times 5$
matrix. It means the filter is a $C_{\text{in}} \times 5 \times 5$ tensor — one
$5 \times 5$ kernel per input channel. The $5 \times 5$ refers to the *spatial*
extent only; the channel dimension is implied by the previous layer.
```

```{prf:remark} The filters are learned, not designed
:label: rem:learned-filters

Hand-designing kernels (Sobel, Gaussian, etc.) is tedious and brittle. In a
CNN every entry of every filter is a **learnable parameter**, set by
backpropagation just like the weights of a dense layer. The network discovers
its own edge and texture detectors from the data.
```

## Two patterns the glossary implies

```{prf:remark} 1×1 convolutions mix channels
:label: rem:one-by-one

A $1 \times 1$ kernel still spans all $C_{\text{in}}$ channels, so a $1 \times
1$ convolution computes a learned weighted sum *across channels* at each
position — a per-pixel linear projection of the channel dimension. It changes
the number of channels without touching the spatial size, and is the standard
tool for cheap channel expansion and bottleneck blocks.
```

```{prf:remark} Parameters come from the filters, not the positions
:label: rem:param-count

Because weights are shared across positions, a layer's parameter count is the
filter size times the number of filters — independent of the image size. A
$3 \times 3$ filter bank on a $224 \times 224$ image has the same number of
weights as on a $32 \times 32$ image. This is the practical payoff of
{prf:ref}`prp:weight-sharing`.
```

## Summary

If this page had to be one sentence: **a convolutional layer is local, shares
weights across space, is translation equivariant (not invariant), and builds
abstraction with depth — and its vocabulary (kernel, filter, feature map,
receptive field, padding, stride) is just the language for those four
decisions.** For the arithmetic behind the glossary — the cross-correlation
slide, the output-shape formula, multi-channel volumes — continue to
{doc}`the convolution concept page <convolutional_kernels/02_concept>`.

```{admonition} Further reading
:class: seealso

- {cite}`zhang2023dive`, Ch. 7, whose treatment of the two principles
  (*translation invariance*, *locality*) this page sharpens and corrects.
- {doc}`Convolutional Kernels: How CNNs Learn Local Patterns <convolutional_kernels/01_intro>`
  — the mechanism this glossary refers to.
```
