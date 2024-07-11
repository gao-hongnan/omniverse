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

# The Concept of Low-Rank Adaptation Of Large Language Models

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import os
import random
import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, TypeVar

import numpy as np
import torch
from IPython.display import display
from rich.pretty import pprint
from torch import nn
```

```{code-cell} ipython3
:tags: [hide-input]

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn

__all__ = [
    "seed_all",
    "seed_worker",
    "configure_deterministic_mode",
    "raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer",
]

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer(value: int) -> None:
    if not (min_seed_value <= value <= max_seed_value):
        raise ValueError(f"Seed must be within the range [{min_seed_value}, {max_seed_value}]")


def configure_deterministic_mode() -> None:
    r"""
    Activates deterministic mode in PyTorch and CUDA to ensure reproducible
    results at the cost of performance and potentially higher CUDA memory usage.
    It sets deterministic algorithms, disables cudnn benchmarking and enables,
    and sets the CUBLAS workspace configuration.

    References
    ----------
    - `PyTorch Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_
    - `PyTorch deterministic algorithms <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>`_
    - `CUBLAS reproducibility <https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>`_
    """

    # fmt: off
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark        = False
    torch.backends.cudnn.deterministic    = True
    torch.backends.cudnn.enabled          = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # fmt: on
    warnings.warn(
        "Deterministic mode is activated. This will negatively impact performance and may cause increase in CUDA memory footprint.",
        category=UserWarning,
        stacklevel=2,
    )


def seed_all(
    seed: int = 1992,
    seed_torch: bool = True,
    set_torch_deterministic: bool = True,
) -> int:
    """
    Seeds all relevant random number generators to ensure reproducible
    outcomes. Optionally seeds PyTorch and activates deterministic
    behavior in PyTorch based on the flags provided.

    Parameters
    ----------
    seed : int, default=1992
        The seed number for reproducibility.
    seed_torch : bool, default=True
        If True, seeds PyTorch's RNGs.
    set_torch_deterministic : bool, default=True
        If True, activates deterministic mode in PyTorch.

    Returns
    -------
    seed : int
        The seed number used for reproducibility.
    """
    raise_error_if_seed_is_negative_or_outside_32_bit_unsigned_integer(seed)

    # fmt: off
    os.environ["PYTHONHASHSEED"] = str(seed)       # set PYTHONHASHSEED env var at fixed value
    np.random.seed(seed)                           # numpy pseudo-random generator
    random.seed(seed)                              # python's built-in pseudo-random generator

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)           # pytorch (both CPU and CUDA)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        if set_torch_deterministic:
            configure_deterministic_mode()
    # fmt: on
    return seed
```

## Motivation

Consider that you have access to an open source large language model that is of
175 billion parameters, and you want to fine-tune it for your domain specific
task for your company. Let's say you succeed in fine-tuning the model and it
works well for your task and you spent a lot of time and resources on it.
However your data used for fine-tuning is non-stationary and the model's
performance degrades over time. So you have to retrain the model again and again
to keep up with the data distribution changes. To further exacerbate the
problem, your other departments across the company wants to also fine-tune the
large language model for their own domain specific tasks.

This will become a problem because performing a _full fine-tuning_ for each
domain specific task will be computationally expensive and time consuming simply
because full fine-tuning requires _adjusting all the parameters_ of the model -
which means if your model has 175 billion _trainable_ parameters, you will have
to adjust all of them for each domain specific task. Such prohibitive
computational cost and time consumption is not feasible for most companies and
this is where _Low Rank Adaptation_ comes in - in which it _freezes_ the
**backbone** of the model weights and _inject trainable rank decomposition
matrices into selected layers_ {cite}`hu2021loralowrankadaptationlarge` of the
said model (often a transformer model).

To take things into perspective, GPT-3 175B fine-tuned with Adam (note adaptive
optimizer like this takes up a lot of memory because it stores the first and
second moments of the gradients) and LoRA can reduce the trainable parameters by
10,000 times and the GPU memory by 3 folds - all while maintaining the on-par
performance with suitable hyperparameters.

## Rank And Low-Rank Decomposition Via Matrix Factorization

Firstly, we assume that the reader has a basic understanding of the
[transformer-based models](https://arxiv.org/abs/1706.03762) and how they work,
as well as the definition of
[_rank_](<https://en.wikipedia.org/wiki/Rank_(linear_algebra)>) and
[_low-rank decomposition_ ](https://en.wikipedia.org/wiki/Low-rank_approximation)in
linear algebra.

In simple terms, given a matrix $\mathbf{W}$ with dimensions $d \times k$, the
rank of a matrix is the number of linearly independent rows or columns it
contains, denoted as $r$, where $r \leq \min (d, k)$. Intuitively, if a matrix
is full rank, it represents a wider array of linear transformations, indicating
a diverse set of information. Conversely, a low-rank matrix, having fewer
linearly independent rows or columns, suggests that it contains redundant
information due to dependencies among its elements. For instance, an image of a
person can be represented as a low-rank matrix because the pixels in the image
often show strong spatial correlations. Techniques like principal component
analysis (PCA) exploit this property to compress images, reducing dimensionality
while retaining essential features.

A low-rank approximation of a matrix $\mathbf{W}$ is another matrix
$\overset{\sim}{\mathbf{W}}$ with the same dimensions as $\mathbf{W}$, which
approximates $\mathbf{W}$ while having a lower rank, aimed at minimizing the
approximation error $\| \mathbf{W} - \overset{\sim}{\mathbf{W}} \|$ under a
specific norm. This is actually a minimization problem and is well-defined and
used commonly in various applications. A common way to find such an
approximation is to use the
[singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
(SVD) of $\mathbf{W}$, which decomposes $\mathbf{W}$ into three matrices
$\mathbf{U}$, $\mathbf{\Sigma}$, and $\mathbf{V}^{T}$ such that
$\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^{T}$, where $\mathbf{U}$ and
$\mathbf{V}$ are orthogonal matrices (assume
$\mathbf{W} \in \mathbb{R}^{d \times k}$) and $\mathbf{\Sigma}$ is a diagonal
matrix with non-negative real numbers on the diagonal. However, for our specific
purpose, we will mention another form of low-rank decomposition via matrix
factorization. More concretely, we will use two matrices $\mathbf{A}$ and
$\mathbf{B}$ to approximate a given matrix $\mathbf{W}$.

$$
\mathbf{A}, \mathbf{B}=\underset{\mathbf{A}, \mathbf{B}}{\operatorname{argmin}} \frac{1}{2}\|\mathbf{A} \mathbf{B}-\mathbf{W}\|_F^2
$$

where $\|\mathbf{B} \mathbf{A}-\mathbf{W}\|_F^2$ is the objective function to
minimize. The Frobenius norm $\| \cdot \|_F$ is used to measure the error
between the original matrix $\mathbf{W}$ and the approximation
$\mathbf{A} \mathbf{B}$. Lastly, it is important to note that if $\mathbf{W}$
has rank $r$, then $\mathbf{A}$ and $\mathbf{B}$ will have dimensions
$d \times r$ and $r \times k$, respectively, where $r \lt \min (d, k)$ (or in
our LoRA case, $r \ll \min (d, k)$ to emphasise that the $r$ is much smaller
than $\min (d, k)$).

## Key Idea 1. The Update Weights Of Fine-Tuning Has A Low Intrinsic Rank

First, we use very rough notations to describe the update weights of fine-tuning
as a matrix $\mathbf{W} \in \mathbb{R}^{d \times k}$ in a gradient-based
optimization process. For simplicity we call $\mathbf{W}$ as the _pre-trained
weights_ and $\nabla \mathbf{W}$ as the _update weights_ at a given iteration of
a gradient-based optimization process. More concretely, we have the below update
process:

$$
\begin{aligned}
\mathbf{W}  &\leftarrow \underbrace{\mathbf{W} - \alpha \nabla \mathbf{W}}_{\mathbf{W} - \alpha \frac{\partial \mathcal{J}}{\partial \mathbf{W}}}
\end{aligned}
$$

where $\mathcal{J}$ is the objective function, $\alpha$ is the learning rate,
and $\nabla \mathbf{W}$ is the gradient of the objective function with respect
to the pre-trained weights $\mathbf{W}$.

To ease the notations, we further differentiate $\mathbf{W}^{(t)}$ as the
pre-trained weights at iteration $t$ and $\nabla \mathbf{W}^{(t)}$ as the update
weights at iteration $t$. We can then rewrite the above equation as:

$$
\begin{aligned}
\mathbf{W}^{(t+1)} &= \mathbf{W}^{(t)} - \alpha \nabla \mathbf{W}^{(t)}
\end{aligned}
$$

to indicate that $\mathbf{W}^{(t+1)}$ is the updated weights after
$\nabla \mathbf{W}^{(t)}$ is applied to $\mathbf{W}^{(t)}$.

Empirical evidence suggests that deep learning models (often large language
models) $\mathcal{M}$ are over-parametrized with respect to their parameter
space $\Theta$ (i.e. the weights of the model). This means the model contains
more parameters than are necessary to achieve the minimum error on the training
data. This redundancy often implies that many parameters are either not
essential or are correlated with others. While the "full space" offers maximal
degrees of freedom for the model parameters, allowing complex representations
and potentially capturing intricate patterns in the data, it can lead to
overfitting, and in our context, it can lead to high computational costs and
memory usage.

As a result, the authors hypothesize that $\mathcal{M}$ can operate within a
much lower-dimensional subspace $\mathcal{S}$ which means that we can reduce the
effective degrees of freedom of the model $\mathcal{M}$ by projecting the
weights of the model into a lower-dimensional subspace while _maintaining the
performance of the model_ - and this is what the author mean by "model resides
in a low intrinsic dimension".

More concretely, suppose the weight matrix
$\mathbf{W} \in \mathbb{R}^{d \times k}$ has rank $r$ meaning the maximum number
of linearly independent rows or columns it contains is $r$. We say that this
weight matrix reside in a subspace $\mathcal{S}_{W} \subset \mathbb{R}^{d}$
(column space/range of $\mathbf{W}$) and the dimension of this subspace is $r$.
The authors argue that the update weights $\nabla \mathbf{W}$ of the model
$\mathcal{M}$ at a given iteration of the optimization process also reside in a
low-dimensional subspace
$\mathcal{S}_{\nabla \mathbf{W}} \subset \mathbb{R}^{d}$ with
$\dim\left(\mathcal{S}_{\nabla \mathbf{W}}\right) \ll r$. Note that without
LoRA, the update weights $\nabla \mathbf{W}$ typically have a high rank (not
guaranteed to be of same rank of $\mathbf{W}$), and so the authors intelligently
proposed an approximation of the update weights $\nabla \mathbf{W}$ using a
low-rank decomposition because of the hypothesis that the update weights reside
in a low-dimensional subspace and is sufficient to represent the
over-parametrized model $\mathcal{M}$.

$$
\begin{aligned}
\mathbf{W}^{(t+1)} &= \mathbf{W}^{(t)} - \alpha \nabla \mathbf{W}^{(t)} \\
&= \mathbf{W}^{(t)} - \alpha \mathbf{B}^{(t)} \mathbf{A}^{(t)}
\end{aligned}
$$

where $\mathbf{A}^{(t)}$ and $\mathbf{B}^{(t)}$ are the low-rank decomposition
matrices of the update weights $\nabla \mathbf{W}^{(t)}$ at iteration $t$. In
other words, the update weights $\nabla \mathbf{W}$ are approximated by the
product of two low-rank matrices $\mathbf{A} \in \mathbb{R}^{r \times k}$ and
and $\mathbf{B} \in \mathbb{R}^{d \times r}$, where $r \ll \min (d, k)$.

$$
\begin{aligned}
\nabla \mathbf{W} \approx \mathbf{B} \mathbf{A}
\end{aligned}
$$

Now, let's do some quick math. Earlier we said our model is of size
$\| \Theta_{\mathcal{M}} \| = 175,000,000,000$ parameters. Then for simplicity
case we assume our weight $\mathbf{W} \in \mathbb{R}^{d \times k}$ of the
pretrained model $\mathcal{M}$ to be of size
$d = k = \sqrt{175,000,000,000} = 418,330$.

If we assume that the rank of the

If we assume that the rank of the model is $r = 10,000$, then the number of
parameters in the low-rank decomposition is
$r \times (d + k) = 10,000 \times (d + k)$. If we assume that
$d = k = 175,000,000,000$, then the number of parameters in the low-rank
decomposition is $10,000 \times 350,000,000,000 = 3,500,000,000,000$ parameters.
This is a reduction of $175,000,000,000 / 3,500,000,000,000 = 0.05\%$ of the
original model size. This is a significant reduction in the number of parameters
in the model.

```{figure} ./assets/lora_weights_visual.png
---
name: lora-weights-visual-paper
---
sss
```

```{figure} ./assets/lora_weights_visual_seb.png
---
name: low-rank-weights-visual-seb
---
sss
```

## Citations

-   https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html
-   https://en.wikipedia.org/wiki/Low-rank_approximation
-   https://en.wikipedia.org/wiki/Rank_(linear_algebra)

-   [1] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and
    W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," _arXiv
    preprint arXiv:2106.09685_, submitted Jun. 17, 2021, revised Oct. 16, 2021.
    [Online]. Available: https://arxiv.org/abs/2106.09685
-   https://www.ethanepperly.com/index.php/2021/10/26/big-ideas-in-applied-math-low-rank-matrices/
-   https://pyproximal.readthedocs.io/en/stable/tutorials/matrixfactorization.html
-   https://rendazhang.medium.com/matrix-decomposition-series-6-low-rank-matrix-factorization-5a3b96832bad
