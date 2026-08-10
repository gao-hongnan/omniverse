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
            Compute per-channel mean and std over an image tensor, then apply
            standardization and min-max normalization to a batch, with the
            train-only statistics pattern that prevents data leakage.
        "keywords": >-
            image normalization, image standardization, calc mean std,
            per-channel mean std, pytorch preprocessing, data leakage
---

# Computing Image Statistics and Applying Both Transforms

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Deep_Learning-blue)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

from __future__ import annotations

import numpy as np
import torch
```

This page implements the two transforms defined in {doc}`01_concept`. The goal
is concrete: compute per-channel mean and standard deviation over an image
tensor, apply standardization ({eq}`eq:zscore`) and min-max normalization
({eq}`eq:minmax`) to a batch, and watch the before/after statistics change.
It closes with the train-only statistics rule that prevents the data leakage
warned about in the concept page.

By the end you will have a reusable `calc_mean_std`, the matching
`standardize` and `normalize_minmax` transforms, and a split-then-standardize
template you can drop into a data pipeline. Stats are computed in NumPy (the
analysis tool the original author used); the transforms run on `torch.Tensor`,
which is what a real training loop hands them.

```{admonition} Prerequisites
:class: note

Read {doc}`01_concept` first — this page implements its definitions
({prf:ref}`def:norm-minmax`, {prf:ref}`def:norm-standardize`) and assumes the
variance-budget motivation. Basic familiarity with `torch.Tensor` indexing is
enough.
```

## A synthetic RGB dataset on the $[0, 255]$ scale

Rather than download CIFAR-10, we synthesize an RGB tensor with the same
fingerprint natural images carry — red dominant, blue suppressed — so the
per-channel means are visibly different and the effect of standardization is
unambiguous.

```{code-cell} ipython3
torch.manual_seed(1930)

N, C, H, W = 2048, 3, 32, 32  # 2048 images, 3 channels, 32x32 px each
channel_means = torch.tensor([125.0, 123.0, 114.0]).view(1, C, 1, 1)
images = (channel_means + 25.0 * torch.randn(N, C, H, W)).clamp(0, 255)

print("dataset shape:", tuple(images.shape))
print("per-channel mean (raw [0,255]):", images.mean(dim=(0, 2, 3)).tolist())
print("per-channel std  (raw [0,255]):", images.std(dim=(0, 2, 3)).tolist())
```

The means sit near $(125, 123, 114)$ — the reddish bias natural photographs
share — and the raw scale is in the hundreds. Feeding this straight into a
network hands the first layer the $\operatorname{Var}(x)\approx 600$ input
that {eq}`eq:var-budget` will faithfully propagate forward.

## Computing per-channel mean and std

The function below mirrors the author's `calcMeanStd`: it rescales pixels to
$[0, 1]$ first, then reduces over every axis except the channel axis. Reducing
over $(N, H, W)$ is exactly equivalent to flattening all pixels of one channel
into a single array and calling `.mean()` — the pedagogical point of the
original — but without materializing the flattened copy.

```{code-cell} ipython3
def calc_mean_std(images: np.ndarray) -> dict[str, tuple[float, ...]]:
    """Per-channel mean and std after rescaling pixels to [0, 1].

    Args:
        images: array of shape ``(N, C, H, W)`` on the ``[0, 255]`` scale.

    Returns:
        ``{"mean": (per-channel means), "std": (per-channel stds)}``, one
        entry per channel, in $[0, 1]$ units.

    Raises:
        ValueError: if ``images`` is not 4-dimensional.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (N, C, H, W); got shape {images.shape}")
    images01 = images.astype(np.float64) / 255.0
    means = images01.mean(axis=(0, 2, 3))
    stds = images01.std(axis=(0, 2, 3))      # population std, matching the original
    return {
        "mean": tuple(float(m) for m in means),
        "std": tuple(float(s) for s in stds),
    }
```

```{code-cell} ipython3
stats = calc_mean_std(images.numpy())
print("calc_mean_std mean:", [round(m, 4) for m in stats["mean"]])
print("calc_mean_std std :", [round(s, 4) for s in stats["std"]])

# Idiomatic one-liner: reduce over every axis except the channel axis.
images01 = images.float() / 255.0
print("vectorized   mean :", images01.mean(dim=(0, 2, 3)).tolist())
print("vectorized   std  :", images01.std(dim=(0, 2, 3), correction=0).tolist())
```

Both forms agree. The per-channel means — roughly $(0.49, 0.48, 0.45)$ — are
in the same ballpark as the canonical CIFAR-10 constants $(0.491, 0.482,
0.447)$, which is no accident: it is the shared fingerprint of natural RGB
images.

## Applying the two transforms

The transforms operate on a `torch.Tensor` mini-batch. Both broadcast a
per-channel parameter over the $(N, C, H, W)$ layout by reshaping it to
$(1, C, 1, 1)$.

```{code-cell} ipython3
def standardize(
    batch: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Per-channel z-scoring, ``x' = (x - mean) / std``.

    Args:
        batch: shape ``(N, C, H, W)``, already on the ``[0, 1]`` scale.
        mean, std: shape ``(C,)`` per-channel statistics.
    """
    shape = (1, -1, 1, 1)  # broadcast over (N, C, H, W)
    return (batch - mean.view(shape)) / std.view(shape)


def normalize_minmax(batch: torch.Tensor) -> torch.Tensor:
    """Per-channel min-max rescaling to ``[0, 1]``.

    Args:
        batch: shape ``(N, C, H, W)`` on any common scale.
    """
    x_min = batch.amin(dim=(0, 2, 3), keepdim=True)
    x_max = batch.amax(dim=(0, 2, 3), keepdim=True)
    return (batch - x_min) / (x_max - x_min)
```

### Standardization: zero mean, unit variance

```{code-cell} ipython3
batch01 = images[:256].float() / 255.0        # a mini-batch on [0, 1]
mean_c = torch.tensor(stats["mean"])
std_c = torch.tensor(stats["std"])

standardized = standardize(batch01, mean_c, std_c)
print("before standardize — mean:", batch01.mean(dim=(0, 2, 3)).tolist())
print("after  standardize — mean:", standardized.mean(dim=(0, 2, 3)).tolist())
print("after  standardize — std :", standardized.std(dim=(0, 2, 3), correction=0).tolist())
```

After standardization each channel's mean is $\approx 0$ and each std is
$\approx 1$ — the $O(1)$ scale the variance budget
{eq}`eq:var-budget` assumes. That is the whole point: the initializer can now
preserve a unit scale because a unit scale is what it was handed.

### Min-max normalization: range pinned, shape preserved

```{code-cell} ipython3
normalized = normalize_minmax(batch01)
print("before normalize — min:", batch01.amin(dim=(0, 2, 3)).tolist())
print("before normalize — max:", batch01.amax(dim=(0, 2, 3)).tolist())
print("after  normalize — min:", normalized.amin(dim=(0, 2, 3)).tolist())
print("after  normalize — max:", normalized.amax(dim=(0, 2, 3)).tolist())
print("after  normalize — mean:", normalized.mean(dim=(0, 2, 3)).tolist())
```

Min-max nails every channel to exactly $[0, 1]$, but note what it leaves
untouched: the per-channel means are still offset (the red channel is still
brighter than blue). Standardization removed that offset; min-max did not.
That is the operational difference between {prf:ref}`def:norm-minmax` and
{prf:ref}`def:norm-standardize` — pin the range, or pin the moments.

## Split before you standardize

The data-leakage rule from {doc}`01_concept` in code: split first, compute
statistics on the training split alone, then apply those *same* constants to
both splits.

```{code-cell} ipython3
perm = torch.randperm(N)
train_idx, val_idx = perm[: 4 * N // 5], perm[4 * N // 5 :]
train_raw, val_raw = images[train_idx], images[val_idx]

train_stats = calc_mean_std(train_raw.numpy())   # train-only statistics
mean_t = torch.tensor(train_stats["mean"])
std_t = torch.tensor(train_stats["std"])

train_std = standardize(train_raw.float() / 255.0, mean_t, std_t)
val_std = standardize(val_raw.float() / 255.0, mean_t, std_t)   # same constants

print("train — standardized mean:", train_std.mean(dim=(0, 2, 3)).tolist())
print("val   — standardized mean:", val_std.mean(dim=(0, 2, 3)).tolist())
print("(val is NOT zero-mean — that is correct: it borrows train's fingerprint.)")
```

The training split's standardized mean is $\approx 0$ by construction, while
the validation split's is only *close* to $0$. That residual offset is not a
bug — it is the absence of leakage. The validation set is being judged on the
training set's yardstick, which is precisely what held-out evaluation
requires.

## Summary

If this page had to be one sentence: **`calc_mean_std` turns an image tensor
into the three-mean-three-std fingerprint, `standardize` and `normalize_minmax`
apply the two rival transforms, and the split-then-standardize pattern keeps
that fingerprint honest by fitting it on the training split alone.** Drop the
three functions into a data pipeline, compute the constants once on your
training set, and your first layer receives the $O(1)$-scale input that
{doc}`weight initialization <../../../influential/numerical_stability_and_initialization/03_weight_initialization>`
needs to do its job.

For the conceptual half — why input scale drives gradient scale, when to
prefer min-max over z-scoring, and the per-image/per-channel/dataset-wide
taxonomy — read {doc}`01_concept`.

```{admonition} Further reading
:class: seealso

- {doc}`01_concept` — the definitions and motivation this page implements.
- {cite}`lecun_efficient_1998` — the conditioning argument that justifies
  per-channel z-scoring at the input.
- [Kaggle — computing dataset mean and std](https://www.kaggle.com/kozodoi/computing-dataset-mean-and-std) —
  an efficient batched implementation for datasets too large to hold in
  memory.
- [CS231n — Data Preprocessing](https://cs231n.github.io/neural-networks-2/#datapre) —
  the notes that codified the per-channel recipe.
```
