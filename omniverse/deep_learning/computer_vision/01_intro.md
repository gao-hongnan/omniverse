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
            How convolutional neural networks turn images into features —
            locality, weight sharing, and translation equivariance, plus the
            cross-correlation mechanism and input normalization.
        "keywords": >-
            computer vision, convolutional neural network, CNN, cross-correlation,
            convolutional kernel, image normalization, translation equivariance
---

# Computer Vision: Learning to See with Convolutions

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Deep_Learning-blue)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

A 224×224 color image is 150,528 numbers. A dense layer reading every pixel into
every hidden unit would need a weight matrix with tens of millions of entries —
and would have to relearn the notion of an "edge" independently at every spatial
location. Convolutional layers replace that wasteful wiring with a small
*kernel* slid across the image, reusing the same handful of weights everywhere.
That single change is why vision moved from hand-crafted features to learned
representations.

This chapter builds the convolutional layer from first principles: why the
design works (its four properties — locality, weight sharing, translation
equivariance, and increasing abstraction), how the sliding-window computation
actually proceeds (cross-correlation, padding, stride, channels), and how the
input scale is set up so that what follows trains well. It is the visual
counterpart to the
{doc}`initialization series <../training_fundamentals/numerical_stability_and_initialization/01_intro>`;
the two meet at the question of signal scale.

By the end of the chapter you will be able to compute a feature map by hand
from a kernel, predict the output shape under any padding and stride, explain
why the same edge detector is useful at every position, and standardize an
image batch before training.

```{admonition} Prerequisites
:class: note

Comfort with matrix multiplication and the MLP forward pass —
{doc}`permutation symmetry <../training_fundamentals/numerical_stability_and_initialization/02_permutation_symmetry>`
is useful context for why parameter sharing changes a layer's symmetry
structure. Notation follows
{doc}`the deep-learning notation page <../../notations/deep_learning>`.
```

## Roadmap

1. **{doc}`Convolutional Kernels: How CNNs Learn Local Patterns <convolutional_kernels/01_intro>`**
   — the mechanism: cross-correlation versus true convolution, the
   sliding-window dot product, padding, stride, multi-channel inputs and
   outputs, and the output-shape formula, with a from-scratch implementation.
2. **{doc}`Image Normalization vs Standardization <image_normalization/01_concept>`**
   — why the scale of the input pixels controls gradient scale and training
   stability, the difference between min-max normalization and zero-mean
   standardization, and when to use per-image, per-channel, or dataset
   statistics.
3. **{doc}`CNN Design Principles <helpsheet>`** — a concise reference for the
   four properties that make a layer "convolutional", plus a glossary of the
   vocabulary (kernel, filter, feature map, receptive field) the chapter uses.

## Summary

If this introduction had to be one sentence: **a convolutional layer sees an
image through a small, reused kernel, and that reuse is what makes vision
tractable — the rest of the chapter is the arithmetic of the slide and the
setup of the input scale.** Begin with
{doc}`the convolution mechanism <convolutional_kernels/01_intro>` for the
arithmetic, or {doc}`the design principles <helpsheet>` for the why.

```{admonition} Further reading
:class: seealso

- {cite}`zhang2023dive`, Ch. 7 *Convolutional Neural Networks*, the primary
  reference for the mechanism and properties covered here.
```
