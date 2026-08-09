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
            Why initializing all neural network weights to the same constant
            paralyses a layer, and why random scale-aware init (Xavier,
            Kaiming) fixes both symmetry and signal scale — with the math.
        "keywords": >-
            weight initialization, constant weight initialization, Xavier
            initialization, Glorot initialization, Kaiming initialization, He
            initialization, symmetry breaking, vanishing gradient, deep learning
---

# Weight Initialization: Why Constant and Zero Init Fail

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

import torch
```

The previous chapter proved that a hidden-unit permutation is a *training
invariant*: if two parameterizations are related by relabeling the units, plain
gradient descent keeps them related, forever ({prf:ref}`cor:symmetry-invariant`).
This chapter cashes that fact in. It has one warning and one fix.

The warning: an initialization that is *symmetric* — identical across the hidden
units — is a permutation-invariant starting point, so by the corollary it stays
symmetric, and the layer collapses to the power of a single unit. The most
blunt version, setting every weight to the same constant, is a textbook mistake
precisely because the optimizer cannot rescue it. The fix has two parts. First,
**randomize**: draw the weights independently so no two units begin life
identical, breaking the symmetry the corollary would otherwise preserve. Second
— and this is where most of the subtlety lives — **scale** the random weights so
that activation and gradient variance neither vanishes nor explodes as signals
cross the layers. That second requirement is what the Xavier/Glorot and
Kaiming/He schemes solve.

By the end you will know why `nn.Linear` does not initialize to a constant or to
zero, why it draws from a distribution whose variance depends on the layer
width, and how to pick the variance from the activation function and the
direction (forward or backward) you want to stabilize.

```{admonition} Prerequisites
:class: note

This page assumes {doc}`permutation symmetry <02_permutation_symmetry>` —
specifically {prf:ref}`cor:symmetry-invariant`, that gradient descent preserves
a hidden-unit permutation — and the forward/backward equations of a one-hidden-
layer MLP. Read {doc}`the intro <01_intro>` first if you are new to the series.
```

## The constant-initialization trap

Recall the corollary: if a parameter setting is invariant under some hidden-unit
permutation $\mathbf{P}$ at step $t$, it is invariant under $\mathbf{P}$ at
step $t+1$. A *symmetric* initialization is exactly such a setting.

```{prf:definition} Symmetric (constant) initialization
:label: def:constant-init

A hidden layer is **symmetrically initialized** when its parameters are
invariant under at least one nontrivial permutation of the hidden units. The
extreme case — **constant initialization** — sets every entry of
$\mathbf{W}^{(1)}$, $\mathbf{b}^{(1)}$, $\mathbf{W}^{(2)}$, $\mathbf{b}^{(2)}$
to the same scalar $c$. Such a setting is invariant under *every* hidden-unit
permutation.
```

```{prf:theorem} Symmetric initialization is a permanent trap
:label: thm:constant-trap

If the parameters are invariant under a nontrivial hidden-unit permutation
$\mathbf{P}$ at initialization, then by {prf:ref}`cor:symmetry-invariant` they
remain invariant under $\mathbf{P}$ after every gradient step. In particular,
under constant initialization the hidden units compute identical activations
and receive identical updates forever; the layer is permanently equivalent to
one with a single hidden unit.
```

The proof is one line: invariance under $\mathbf{P}$ is a fixed point of the
update, because the update preserves permutation equivalence. Intuitively, the
two cooks from the {doc}`intro <01_intro>` are still making the same moves on
step 1000 as on step 0 — the optimizer never gives them a reason to diverge,
because by construction they never have one. We watch it happen:

```{code-cell} ipython3
torch.manual_seed(1992)

# Toy nonlinear target: y = sum of squares of 4 features. Needs hidden units.
X = torch.randn(256, 4)
Y = (X**2).sum(dim=1, keepdim=True)

def make_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),   # hidden layer, 8 units
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )

def constant_init(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.constant_(m.weight, 0.37)   # every weight the same
        torch.nn.init.constant_(m.bias, 0.0)

const_model = make_model().apply(constant_init)
rand_model = make_model()                          # PyTorch default (random, scaled)

H_const = const_model[0](X)                        # hidden pre-activations, constant init
print("constant init — all 8 hidden units identical?",
      torch.allclose(H_const[:, 0:1], H_const))   # every column == column 0
```

```{code-cell} ipython3
def train(model: torch.nn.Module, steps: int = 400) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    for _ in range(steps):
        loss = loss_fn(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        H = model[0](X)
    return loss.item(), torch.allclose(H[:, 0:1], H)

const_loss, const_symmetric = train(const_model)
rand_loss, rand_symmetric = train(rand_model)
print(f"constant init: final loss = {const_loss:.3f}, units still identical = {const_symmetric}")
print(f"random  init : final loss = {rand_loss:.3f}, units still identical = {rand_symmetric}")
```

The constant-initialized layer finishes with its hidden units still cloned from
one another and a loss stuck far above the random layer's. That gap is not bad
luck; it is {prf:ref}`thm:constant-trap` made visible. Random initialization
finishes lower precisely because it broke the symmetry on step zero.

```{admonition} Zero initialization is a different, equally fatal, trap
:class: warning

Setting $\mathbf{W}^{(2)} = \mathbf{0}$ zeroes the backpropagated error
$\boldsymbol{\delta}_{H} = (\boldsymbol{\delta}_{O}\mathbf{W}^{(2)\top}) \odot
\sigma'(\mathbf{Z})$ regardless of $\boldsymbol{\delta}_{O}$, so
$\partial \mathcal{L}/\partial \mathbf{W}^{(1)} = \mathbf{0}$ and the hidden
weights never move. Every hidden unit then receives the same (zero) gradient
and stays at zero — symmetry again, reached by a different route. This is why
the bias-only edge case and zero weight init both fail: they are symmetric
fixed points.
```

## Random initialization breaks the symmetry

The cure for the trap is to refuse to enter it. Draw each weight independently
from a zero-mean distribution so that, with probability one, no two hidden units
share identical incoming weights: there is no permutation $\mathbf{P}$ relating
them, so {prf:ref}`cor:symmetry-invariant` has nothing to preserve, and the
units are free to specialize.

But "random" alone leaves a second, independent problem unsolved — the *scale*
of the random draw — and that problem is where depth enters the story.

## The scale problem: vanishing and exploding activations

Symmetry breaking handles the *direction* of training (it can start); scale
handles the *magnitude* (it can keep going). Consider a linear layer
$\mathbf{z} = \mathbf{W}\mathbf{x}$ before the activation, with $n_{\text{in}}$
inputs, weights and inputs independent and zero-mean, and
$\operatorname{Var}(w_{ij}) = \operatorname{Var}(w)$. The variance of one
pre-activation is

$$
\operatorname{Var}(z_j)
= \operatorname{Var}\!\left(\sum_{i=1}^{n_{\text{in}}} w_{ji} x_i\right)
= \sum_{i=1}^{n_{\text{in}}} \operatorname{Var}(w_{ji})\operatorname{Var}(x_i)
= n_{\text{in}}\operatorname{Var}(w)\operatorname{Var}(x).
$$

So the activation variance is multiplied by $n_{\text{in}}\operatorname{Var}(w)$
at every layer. If that factor exceeds 1, variance explodes exponentially with
depth; if it is below 1, variance vanishes. Either way a deep network loses
numerical meaning long before the signal reaches the output — the vanishing- and
exploding-gradient symptoms. To keep $\operatorname{Var}(z) \approx
\operatorname{Var}(x)$ across a forward pass, set

$$
\operatorname{Var}(w) = \frac{1}{n_{\text{in}}}.
$$ (eq:forward-variance)

The backward pass makes the symmetric demand with the *output* fan-in
$n_{\text{out}}$: to keep gradient variance stable flowing backward,
$\operatorname{Var}(w) = 1/n_{\text{out}}$. A layer cannot generally satisfy
$1/n_{\text{in}}$ and $1/n_{\text{out}}$ at once, which forces a compromise.

## Xavier/Glorot and Kaiming/He

```{prf:definition} Xavier (Glorot) initialization
:label: def:xavier

For saturating activations (tanh, sigmoid), {cite}`glorot_bengio_2010` propose
the harmonic-mean compromise between the forward and backward variance
constraints,

$$
\operatorname{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}},
$$

realized either as a Gaussian $\mathcal{N}(0,\, 2/(n_{\text{in}}+n_{\text{out}}))$
or a uniform $\mathcal{U}(-a, a)$ with $a = \sqrt{6/(n_{\text{in}}+n_{\text{out}})}$.
```

Xavier averages the two demands because tanh-like activations *saturate* — their
gradient shrinks as the activation flattens — so neither the forward nor the
backward scale can be allowed to run away. For the ReLU family, the picture
differs: ReLU zeroes roughly half its inputs, halving the variance passed
forward, so the forward constraint must compensate.

```{prf:definition} Kaiming (He) initialization
:label: def:kaiming

For ReLU and its variants, {cite}`he_zhang_ren_sun_2015` set the forward
variance to

$$
\operatorname{Var}(w) = \frac{2}{n_{\text{in}}},
$$

the factor of 2 canceling the half of the signal ReLU discards. The matching
backward-preserving form is $\operatorname{Var}(w) = 2/n_{\text{out}}$.
```

The $2/n_{\text{in}}$ is just {eq}`eq:forward-variance` with a correction for
ReLU's zeroing; it is the reason deep ReLU networks became trainable in the
first place.

## Choosing in practice

```{list-table}
:header-rows: 1
:widths: 28 30 42

* - Activation
  - Scheme
  - PyTorch
* - tanh / sigmoid
  - Xavier/Glorot
  - `torch.nn.init.xavier_normal_(m.weight)`
* - ReLU / LeakyReLU
  - Kaiming/He
  - `torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")`
* - SELU (self-normalizing)
  - fixed by the math
  - `torch.nn.init.kaiming_normal_` with the SELU-prescribed variance
```

Two rules of thumb survive all of the above: **never initialize symmetrically**
(constant or zero), and **always tie the weight variance to the fan-in** (and,
for saturating activations, the fan-out). Everything else is choosing which
variance formula matches your activation.

## Summary

If this chapter had to be one sentence: **initialization must break the
permutation symmetry the optimizer cannot, and must scale the weights so
activation and gradient variance stay near one across depth — random alone does
the first, Xavier and Kaiming do both.** The constant-init trap
({prf:ref}`thm:constant-trap`) is the direct consequence of
{prf:ref}`cor:symmetry-invariant`; the variance analysis leading to Xavier
{cite}`glorot_bengio_2010` and Kaiming {cite}`he_zhang_ren_sun_2015` is the
scale half. Together they are why every modern layer draws its initial weights
from a fan-in-scaled distribution rather than a constant.

```{admonition} Further reading
:class: seealso

- {cite}`glorot_bengio_2010` — the original Xavier/Glorot analysis of forward
  and backward signal scale in deep feedforward networks.
- {cite}`he_zhang_ren_sun_2015` — the Kaiming/He initialization for ReLU
  networks.
- {cite}`zhang2023dive`, §5.4 — the textbook motivation for symmetric-init
  failure that this series formalizes.
```
