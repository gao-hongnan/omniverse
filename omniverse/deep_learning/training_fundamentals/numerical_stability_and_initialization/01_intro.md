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
            Why neural network weight initialization decides whether training
            converges at all — symmetry breaking and scale-aware init (Xavier,
            Kaiming) explained from first principles.
        "keywords": >-
            deep learning, weight initialization, permutation symmetry, Xavier
            initialization, Kaiming initialization, constant weight
            initialization, symmetry breaking
---

# Why Neural Network Initialization Matters

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Deep_Learning-blue)
![Tag](https://img.shields.io/badge/Level-Intermediate-yellow)

```{contents}
:local:
```

Train the same network architecture twice from two different sets of initial
weights and you will sometimes see one run converge cleanly while the other
stalls at a poor solution — or fails to learn at all. The architecture is
identical, the data is identical, the optimizer is identical. The only thing
that differs is the starting point. Why should that decide whether training
works?

This series answers that question. It has two halves. The first,
{doc}`permutation symmetry <02_permutation_symmetry>`, proves a structural
fact about every multilayer network: the units within a hidden layer are
*interchangeable* — relabel them and the network computes exactly the same
function. The second, {doc}`weight initialization <03_weight_initialization>`,
turns that fact into a practical warning: if you initialize those
interchangeable units identically, they stay identical forever, and the layer
collapses to a single unit. The fix — random, scale-aware initialization —
exists precisely to *break* that symmetry and to keep activation and gradient
magnitudes stable as they cross each layer.

By the end of the series you will be able to predict, before training, whether
a given initialization scheme can learn at all; explain why initializing every
weight to the same constant paralyses a network; and choose between schemes
like Xavier/Glorot and Kaiming/He based on the activation function and depth of
your model, rather than copying a default and hoping.

```{admonition} Prerequisites
:class: note

You need comfort with matrix multiplication and the forward and backward
passes of a one-hidden-layer MLP — we restate every equation we use, but we do
not teach the chain rule from scratch. The notation follows
{doc}`the deep-learning notation page <../../../notations/deep_learning>`;
a passing familiarity with
{doc}`cross-entropy loss <../../../influential/loss_functions/04_cross_entropy_loss>`
helps for the gradient discussion but is not required.
```

## The two failures initialization must prevent

A poorly initialized network fails in one of two qualitatively different ways,
and the rest of the series is organized around distinguishing them.

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Failure
  - Mechanism
  - Where it is resolved
* - **Symmetry lock**
  - Units start identical, receive identical gradients, and so update
    identically — the layer never discovers that its units *could* specialize.
  - {doc}`02_permutation_symmetry` and
    {doc}`03_weight_initialization`
* - **Scale runaway**
  - Activation or gradient variance shrinks to zero (vanishing) or balloons
    (exploding) as it crosses layers, so deep networks lose signal or numerical
    stability.
  - {doc}`03_weight_initialization`
```

The two failures are independent. A network can be symmetry-locked yet
perfectly scaled, or well-symmetry-broken yet vanishing. Good initialization
prevents both at once.

```{admonition} Intuition — why "identical start" is a trap
:class: tip

Imagine two cooks in a kitchen handed the same recipe, the same ingredients,
and — crucially — told to make exactly the same moves. Whatever each learns,
they learn together; neither ever explores a move the other has not already
made. Two cooks collapse into one. Hidden units initialized identically are
those two cooks: identical inputs, identical weights, identical updates. The
*capacity* to specialize is there in the architecture, but the training
dynamics can never reach it. Symmetry must be *broken* by the initializer, not
earned by the optimizer — because, as the next chapter shows, the optimizer
provably preserves it.
```

## Roadmap

1. **{doc}`Permutation symmetry in Neural Networks <02_permutation_symmetry>`**
   establishes the interchangeable-units property rigorously: we permute the
   hidden units of a one-hidden-layer MLP and prove, forward *and* backward,
   that the output and every gradient are unchanged. The proof is short and
   concrete — a permutation matrix, four lines of algebra — but its consequence
   (permutation symmetry is an exact redundancy the optimizer can never
   remove) is the load-bearing fact for everything that follows.
2. **{doc}`Weight Initialization: Why Constant and Zero Init Fail <03_weight_initialization>`**
   applies that fact. We prove that constant initialization keeps a layer's
   units identical forever, then derive the scale-aware schemes — Xavier/Glorot
   for saturating activations, Kaiming/He for ReLU families — that both break
   symmetry *and* control variance across depth.

## Summary

If this introduction had to be one sentence: **initialization matters because
the optimizer cannot fix a starting point that is either symmetry-locked or
scale-broken — those are properties the initializer must guarantee up front.**
The next chapter proves the symmetry property that makes identical init fatal;
the chapter after turns that proof into the practical rules (Xavier, Kaiming)
you reach for every time you call `nn.Linear`.

```{admonition} Further reading
:class: seealso

- {cite}`zhang2023dive`, §5.4 *Numerical Stability and Initialization*, whose
  treatment of permutation symmetry and constant initialization motivates this
  series.
- {doc}`Why cosine annealing with warmup stabilizes training <../../../playbook/training/why_cosine_annealing_warmup_stabilize_training>`
  — a companion note on the *optimizer-side* of training stability; this series
  handles the *architecture-and-init* side.
```
