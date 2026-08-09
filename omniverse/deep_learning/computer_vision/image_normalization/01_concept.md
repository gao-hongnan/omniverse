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
            Image normalization vs standardization: how input scale drives
            gradient scale, when to min-max vs z-score per channel, and why it
            fixes optimization conditioning.
        "keywords": >-
            image normalization, image standardization, min-max scaling,
            per-channel mean std, input preprocessing, gradient conditioning
---

# Image Normalization vs Standardization: Why Scale Matters

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
```

A neural network's first layer sees your raw inputs. Feed it images whose
pixels span $[0, 255]$ and the activations in the first hidden layer inherit
that scale — no matter how carefully you tuned the weight initialization. The
init schemes you met in the weight-initialization chapter *preserve* variance
across depth; they do not *correct* a bad input scale. That correction is this
chapter's topic.

Two operations do the job, and they are not the same. **Normalization**
rescales pixel values into a fixed range, usually $[0, 1]$. **Standardization**
recenters each channel to zero mean and unit variance. Choosing between them —
and deciding whether to compute statistics per-image, per-channel, or across
the whole dataset — is the difference between a model that trains in twenty
epochs and one that crawls or diverges.

By the end you will know which operation to reach for, where its statistics
come from, and why skipping this step quietly undoes your initialization work.
The companion page {doc}`02_implementation` turns each formula into runnable
code.

```{admonition} Prerequisites
:class: note

This page assumes the forward/backward variance analysis from
{doc}`weight initialization <../../training_fundamentals/numerical_stability_and_initialization/03_weight_initialization>`.
Read {doc}`the deep learning notations <../../../notations/deep_learning>` if
any symbol is unfamiliar.
```

## The input scale your initializer cannot fix

Recall the variance budget from the weight-initialization series. For a linear
layer $\mathbf{z} = \mathbf{W}\mathbf{x}$ with fan-in $n_{\text{in}}$,
independent zero-mean weights and inputs, and
$\operatorname{Var}(w_{ij}) = \operatorname{Var}(w)$, the variance of one
pre-activation is

$$
\operatorname{Var}(z_j)
= n_{\text{in}}\operatorname{Var}(w)\operatorname{Var}(x).
$$ (eq:var-budget)

Xavier and Kaiming choose $\operatorname{Var}(w)$ so that
$\operatorname{Var}(z) \approx \operatorname{Var}(x)$ — they *preserve*
whatever input variance you hand them. Hand them $\operatorname{Var}(x)\approx 1$
and every layer stays near one. Hand them $\operatorname{Var}(x)\approx 5000$
(the variance of raw pixels spread over $[0,255]$) and every layer stays near
five thousand. The initializer is scale-equivariant: it inherits your input
scale, it never corrects it.

So the precondition for Xavier/Kaiming to do their job is that
$\operatorname{Var}(x)$ already be $O(1)$. That is exactly what
standardization arranges — and it is why a preprocessing step that looks
cosmetic is, in fact, load-bearing.

```{prf:remark} Input scale is also an optimizer-conditioning problem
:label: rem:norm-conditioning

There is a second, independent reason. For a smooth loss surface, the
eigenvalue spread of the Hessian is governed by the spread of the input
variances across features {cite:p}`lecun_efficient_1998`. If one channel has
variance $1$ and another $10^{6}$, every gradient step overshoots along the
high-variance direction and crawls along the low-variance one, forcing a
learning rate tuned to the worst-scaled channel. Per-channel standardization
collapses that spread to one, so the usable learning rate rises and
convergence speeds up — a conditioning gain, separate from the
variance-preservation argument above.
```

## Two operations, precisely

The two transforms are both affine, but they optimize different things.
Normalization pins the *range*; standardization pins the *first two moments*.

```{prf:definition} Min-max normalization (to $[0,1]$)
:label: def:norm-minmax

For a pixel value $x$ with channel extrema $x_{\min}$ and $x_{\max}$,

$$
x' \defeq \frac{x - x_{\min}}{x_{\max} - x_{\min}}.
$$ (eq:minmax)

The result is guaranteed to lie in $[0, 1]$, the darkest pixel mapping to $0$
and the brightest to $1$. The operation preserves the shape of the
distribution; it only rescales and shifts it.
```

```{prf:definition} Standardization (z-scoring)
:label: def:norm-standardize

For a pixel value $x$ with channel mean $\mu$ and standard deviation $\sigma$,

$$
x' \defeq \frac{x - \mu}{\sigma}.
$$ (eq:zscore)

The result has zero mean and unit variance per channel. Unlike normalization,
standardization does **not** guarantee a bounded range — an outlier can push
$x'$ well beyond $\pm 3$ — but it guarantees the first two moments, which is
what {eq}`eq:var-budget` and the conditioning remark need.
```

How to choose? If the downstream layer is a `BatchNorm`, standardization is
partly redundant (BatchNorm recenters anyway), though a standardized input
still helps the *first* layer before any normalization has run. For models
without internal normalization — or for the input layer of any network —
per-channel standardization is the safer default precisely because it secures
the variance budget the initializer assumes. Min-max normalization is the
right call when the model's inputs must be probabilities (e.g. an input that
feeds a cross-entropy over bins) or when a bounded range is a hard
contract.

## Per-image, per-channel, or dataset-wide?

The two definitions above say *what* to compute, not *over which pixels*.
That choice has three standard scopes, and they are not interchangeable.

```{list-table}
:header-rows: 1
:widths: 24 30 46

* - Scope
  - Statistics computed over
  - When it applies
* - **Per-image**
  - one image's own pixels
  - Each image must stand alone (e.g. contrast normalization that is robust to
    a per-image lighting shift). Discards the dataset's shared fingerprint.
* - **Per-channel** (dataset-wide)
  - all images, one channel at a time
  - The standard choice for image classification. Yields one mean and one std
    per channel — e.g. ImageNet's $(0.485, 0.456, 0.406)$ /
    $(0.229, 0.224, 0.225)$.
* - **Global** (all pixels, all channels pooled)
  - every pixel of every image
  - Rare for RGB; collapses channel structure. More common for single-channel
    audio/spectrograms or grayscale.
```

For RGB classification, per-channel dataset-wide statistics are the norm: they
remove both the per-channel mean (the reddish bias natural images share) and
the per-channel scale, giving each channel an equal voice in the first layer's
weighted sum.

## The train-only statistics rule

```{admonition} Data leakage warning
:class: warning

Compute mean and std on the **training split only**, then apply those same
constants to validation and test. Computing statistics over the full dataset
before splitting leaks information from the held-out sets into preprocessing —
a silent, common form of data leakage that inflates validation scores.
```

The transformed validation set will therefore *not* be exactly zero-mean — it
carries the training set's fingerprint, which is the point. See
{doc}`02_implementation` for the split-then-standardize pattern written out in
code.

## Summary

If this page had to be one sentence: **normalization pins the range and
standardization pins the moments, and either way the goal is to feed the
network inputs of $O(1)$ scale so that the variance-preservation your
initializer provides has a well-conditioned starting point to preserve.** The
two are affine rivals — min-max for bounded inputs, z-scoring for matched
channel scales — and the statistics that drive them come from the training
split alone.

The next page, {doc}`02_implementation`, computes per-channel mean and std
over an image tensor and applies both transforms to a batch, printing the
before/after statistics so the effect is visible. For the inverse story —
keeping variance near one *inside* the network rather than at its input — see
{doc}`weight initialization <../../training_fundamentals/numerical_stability_and_initialization/03_weight_initialization>`.

```{admonition} Further reading
:class: seealso

- {cite}`lecun_efficient_1998` — the canonical primary source for input
  standardization as an optimizer-conditioning fix (Section 4.3).
- {cite}`Goodfellow-et-al-2016`, §8.3 — preprocessing, including the
  per-pixel and per-feature normalization taxonomy.
- [CS231n — Data Preprocessing](https://cs231n.github.io/neural-networks-2/#datapre) —
  the concise notes that popularized the per-channel mean/std recipe.
- [fast.ai — computing your own image stats](https://forums.fast.ai/t/calculating-our-own-image-stats-imagenet-stats-cifar-stats-etc/40355) —
  community discussion of dataset-specific statistics.
```
