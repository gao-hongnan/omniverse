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
            Convolutional kernels extract local patterns from images. Learn why
            parameter sharing, locality, and translation equivariance make CNNs
            beat dense layers.
        "keywords": >-
            convolutional kernel, convolutional neural network, CNN, feature
            map, parameter sharing, translation equivariance, locality, image
---

# Convolutional Kernels: How CNNs Learn Local Patterns

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

Take a $224 \times 224$ RGB image and hand it to a dense layer. Flattened,
that is $150{,}528$ inputs; a single $1{,}000$-unit hidden layer costs roughly
150 million weights, and not one of them is told that pixel $(i, j)$ has
anything to do with pixel $(i, j+1)$. Move the cat one pixel to the right and
the entire input vector changes, so the entire representation changes with it.
Why do convolutional layers — the workhorse of computer vision — sidestep both
the parameter explosion and the shift problem in one move?

This three-page chapter answers that. It is written for someone who has
trained a multilayer perceptron but never opened a convolutional network, and
it isolates the single idea that makes vision tractable: a small, learned
filter that slides across the image. The formal sliding-window operator lives
in {doc}`the concept chapter <02_concept>`; the from-scratch NumPy
implementation lives in {doc}`the implementation chapter <03_implementation>`.
By the end of the series you will be able to name the three structural priors
a convolutional layer bakes in — locality, parameter sharing, and translation
equivariance — explain why each cuts the dense layer's bill, and read a
feature map without panic.

```{admonition} Prerequisites
:class: note

You need comfort with matrix multiplication and the forward pass of a
one-hidden-layer MLP — we restate every equation we use, but we do not teach
linear layers from scratch. The notation follows
{doc}`the deep-learning notation page <../../../notations/deep_learning>`.
The discussion of *why* bad parameter scales harm training connects forward to
{doc}`the numerical-stability and initialization series <../../training_fundamentals/numerical_stability_and_initialization/01_intro>`,
but nothing here depends on having read it.
```

## Why a dense layer is the wrong tool for an image

A dense (fully connected) layer connects every input element to every output
unit through its own weight. For tabular data that is exactly what you want.
For an image it fails on two independent counts.

**Parameter explosion.** Spatial resolution is expensive. A megapixel image
has on the order of $10^6$ pixels per channel, so even a modest first hidden
layer carries upwards of a billion weights before it has learned anything.
Most of that capacity is spent rediscovering the same local edge at every
position in the frame.

**Loss of spatial structure.** Flattening a 2D grid into a vector throws away
the only fact that makes vision tractable — that nearby pixels are correlated
and distant ones mostly are not. After the flatten, the layer has no way to
know that input $42$ and input $43$ were once neighbours. It also has no
invariance to position: shift the input by one pixel and the network sees an
unrelated vector, so a cat at column $10$ and a cat at column $11$ require two
largely independent sets of weights to recognise.

A convolutional layer replaces the giant weight matrix with a small *kernel*
(also called a *filter*) that slides across the image, and in doing so it
bakes in three priors that address every problem above at once.

## What a convolutional layer does instead

At each spatial position, the layer places its kernel over a small
neighbourhood of the input, takes a dot product, and writes the result to one
pixel of the *feature map*. It then steps the kernel across by one position
and repeats, using the **same** kernel weights everywhere. Three structural
properties fall out of that single design choice.

```{list-table}
:header-rows: 1
:widths: 24 38 38

* - Property
  - Dense (fully connected) layer
  - Convolutional layer
* - **Parameters per output unit**
  - one weight per input pixel: $H \cdot W \cdot C$
  - one shared $K_h \times K_w \times C_{in}$ kernel, reused at every position
* - **Spatial structure**
  - flattened away before the layer sees it
  - preserved — each output sees one $K_h \times K_w$ neighbourhood
* - **Shift the input by one pixel**
  - a completely different input vector
  - the output shifts by one pixel — the *same* features, relocated
```

**Locality** says each output unit looks only at a small neighbourhood of the
input, of size $K_h \times K_w$. The layer never connects a single output to a
pixel on the far side of the image, which matches the empirical fact that
visual patterns are local: an edge is made of a few adjacent pixels, not the
whole frame.

**Parameter sharing** says the *same* kernel slides across every position. A
$3 \times 3$ kernel has nine weights regardless of image size, and a vertical
edge detector learned at the top-left corner is applied, for free, at the
bottom-right corner too. Sharing is what collapses the billion-weight first
layer into something with a few thousand parameters.

**Translation equivariance** is the mathematical payoff of sharing: if a
feature fires at position $(i, j)$ in the image, then shifting the image by
one pixel makes the same feature fire at $(i+1, j)$. Shifts propagate through
the layer instead of scrambling it.

```{prf:definition} Translation equivariance
:label: def:conv-translation-equivariance

Let $f$ be an image-to-image operator and let $T_\Delta$ denote a spatial
shift by the displacement $\Delta$. The operator $f$ is **translation
equivariant** when shifting the input and then applying $f$ gives the same
result as applying $f$ and then shifting the output:

$$
f\!\left(T_\Delta \mathbf{X}\right) \;=\; T_\Delta\, f(\mathbf{X}).
$$ (eq:translation-equivariance)

A 2D cross-correlation whose kernel is shared across all positions satisfies
this exactly. (Equivariance is weaker than translation *invariance*: an
equivariant layer moves the feature when the input moves; an invariant layer
would ignore the move altogether. Pooling, not convolution, is what buys
invariance.)
```

The scale of the saving is not subtle. The same single hidden layer that
needed ~150 million dense weights needs only a $3 \times 3 \times 3$ kernel
per output channel — $27$ weights per channel, or $27{,}000$ for $1{,}000$
output channels. That is a five-thousand-fold reduction, before any
cleverness in the rest of the network.

```{code-cell} ipython3
h_in, w_in, c_in = 224, 224, 3
units = 1000

dense_params = h_in * w_in * c_in * units
conv_params = 3 * 3 * c_in * units      # one 3x3 kernel per output channel

print(f"dense layer params: {dense_params:,}")   # 150,528,000
print(f"conv layer params:    {conv_params:,}")    # 27,000
print(f"reduction factor:     {dense_params // conv_params:,}x")
```

The next page, {doc}`the concept chapter <02_concept>`, makes the
sliding-window operator precise — entry by entry, then with stride, padding,
and multiple channels. The page after that,
{doc}`the implementation chapter <03_implementation>`, turns the operator into
twenty lines of NumPy you can step through yourself.

```{admonition} Summary
:class: tip

If this page had to be one sentence: **a convolutional layer beats a dense
layer for images because it bakes in three true priors — locality, parameter
sharing, and translation equivariance — and in doing so it replaces a
billion-weight first layer with a few thousand shared kernel weights.**

- A dense layer flattens the image, explodes the parameter count, and treats a
  shifted cat as unrelated to the original.
- A convolutional layer keeps the 2D structure, looks only at local
  neighbourhoods, reuses one kernel everywhere, and propagates shifts
  faithfully through the feature map.
- Equivariance ($f(T_\Delta \mathbf{X}) = T_\Delta f(\mathbf{X})$) is the
  formal statement of "shifts propagate"; it is weaker than, and different
  from, shift invariance.

Next: {doc}`cross-correlation, stride, and padding <02_concept>`; then
{doc}`the from-scratch NumPy implementation <03_implementation>`.
```

```{admonition} Further reading
:class: seealso

- {doc}`Cross-Correlation vs Convolution: the sliding-window math <02_concept>`
  — the formal operator this intro only sketches.
- {cite}`zhang2023dive`, Chapter 7.1, gives the standard textbook treatment of
  the same operator and is the conceptual source for this chapter.
- The [PyTorch `nn.Conv2d` documentation](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  shows the production interface whose internals the next two pages unpack.
```
