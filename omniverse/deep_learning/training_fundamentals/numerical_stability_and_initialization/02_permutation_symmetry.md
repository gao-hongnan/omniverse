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
            Hidden units in a neural network are interchangeable: permuting
            them leaves the output and every gradient unchanged. The full
            permutation-symmetry derivation, forward and backward.
        "keywords": >-
            permutation symmetry, neural network, hidden units,
            over-parametrization, weight symmetry, deep learning, backpropagation
---

# Permutation Symmetry in Neural Networks Explained

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
```

Rename the third hidden unit of a trained MLP to be "the first" and shuffle the
others to match. Does the network's behavior change? It does not — and not by
coincidence. The units inside a hidden layer are *interchangeable*: any
relabeling of them, accompanied by the matching relabeling of the weights, is
an exact symmetry of the network. The function computed is identical, and —
this is the part that matters for training — so is every gradient.

This chapter proves that statement, called **permutation symmetry**, for a
one-hidden-layer network. The proof has two halves. Forward, we show the output
is unchanged when the units are relabeled; backward, we show the gradients
relabel in lockstep, so a gradient step *preserves* the relabeling. The
consequence is sharp: permutation symmetry is an invariant of training itself.
The optimizer cannot break it, which is exactly why identical initialization is
fatal — the topic of {doc}`the next chapter <03_weight_initialization>`.

By the end you will be able to state precisely *which* weight matrices must be
permuted together, prove output-invariance in four lines, and explain why a
network whose units start out identical can never grow them apart.

```{admonition} Prerequisites
:class: note

This is the second page of the
{doc}`initialization series <01_intro>`. You need the forward and backward
equations of a one-hidden-layer MLP and the definition of a permutation
matrix; both are restated below.
```

## Hidden units are interchangeable

A hidden layer holds $h$ units. Nothing in the layer's definition ordains an
ordering on them: unit 3 is not structurally prior to unit 7. The only thing
that distinguishes two units is the column of incoming weights that feeds each
one and the row of outgoing weights that reads each one. Swap those together —
the incoming column *and* the outgoing row — and you have renamed the unit
without altering a single computation.

```{prf:definition} Permutation symmetry of a hidden layer
:label: def:perm-symmetry

A hidden layer with $h$ units has **permutation symmetry**: for every
permutation (relabeling) $\pi$ of the $h$ units, there is a relabeling of the
weight matrices that leaves the network's input-output function exactly
unchanged.
```

That is the claim. The rest of the chapter makes it precise and proves it.

## Setting: a one-hidden-layer MLP

We work with a single hidden layer so every step is explicit; the result
extends layer-by-layer to any depth. With a minibatch
$\mathbf{X} \in \R^{n \times d}$ of $n$ samples and $d$ features, $h$ hidden
units, and $q$ outputs, the forward pass is

$$
\begin{aligned}
\mathbf{H} &= \sigma\!\left(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}\right), \\
\mathbf{O} &= \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)},
\end{aligned}
$$

where $\mathbf{W}^{(1)} \in \R^{d \times h}$, $\mathbf{b}^{(1)} \in \R^{1 \times h}$,
$\mathbf{W}^{(2)} \in \R^{h \times q}$, $\mathbf{b}^{(2)} \in \R^{1 \times q}$, and
$\sigma$ is applied element-wise. Write the hidden pre-activation as
$\mathbf{Z} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$, so
$\mathbf{H} = \sigma(\mathbf{Z})$.

Two index conventions to fix in mind, because the whole proof turns on them:

- The $h$ **columns** of $\mathbf{W}^{(1)}$ and $\mathbf{b}^{(1)}$ index the
  hidden units. Column $j$ is the incoming weight vector for unit $j$.
- The $h$ **rows** of $\mathbf{W}^{(2)}$ index the hidden units. Row $j$ is the
  outgoing weight vector *from* unit $j$.

So a hidden unit $j$ lives in *column* $j$ of the first weight matrix and *row*
$j$ of the second. Relabeling unit $j$ therefore means permuting columns of
$\mathbf{W}^{(1)}$ and rows of $\mathbf{W}^{(2)}$ — in the same order.

## Permutation matrices, briefly

```{prf:definition} Permutation matrix
:label: def:perm-matrix

A **permutation matrix** $\mathbf{P} \in \R^{h \times h}$ has exactly one entry
equal to $1$ in each row and each column and $0$ everywhere else.
Right-multiplying $\mathbf{P}$ permutes columns; left-multiplying permutes
rows. Permutation matrices are orthogonal: $\mathbf{P}^{\top}\mathbf{P} =
\mathbf{P}\mathbf{P}^{\top} = \mathbf{I}$.
```

For example,

$$
\mathbf{P} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

swaps the first two columns of anything it right-multiplies and the first two
rows of anything it left-multiplies. Orthogonality is the key: it lets us move
$\mathbf{P}$ through an expression and cancel $\mathbf{P}^{\top}\mathbf{P}$ to
$\mathbf{I}$.

## Permuting the weights

Applying a hidden-unit permutation $\mathbf{P}$ to the parameters means
relabeling the units consistently across both weight matrices and the hidden
bias. Using the convention above:

$$
\mathbf{W}^{(1)}_{\pi} \defeq \mathbf{W}^{(1)}\mathbf{P}, \qquad
\mathbf{b}^{(1)}_{\pi} \defeq \mathbf{b}^{(1)}\mathbf{P}, \qquad
\mathbf{W}^{(2)}_{\pi} \defeq \mathbf{P}^{\top}\mathbf{W}^{(2)}.
$$

The output bias $\mathbf{b}^{(2)}$ is indexed by outputs $q$, not hidden units
$h$, so it is untouched. Read the subscripts literally:
$\mathbf{W}^{(1)}\mathbf{P}$ permutes the *columns* of $\mathbf{W}^{(1)}$ (the
hidden units as seen by layer 1), and $\mathbf{P}^{\top}\mathbf{W}^{(2)}$
permutes the *rows* of $\mathbf{W}^{(2)}$ (the hidden units as seen by layer 2).
The transpose appears because left-multiplication by $\mathbf{P}^{\top}$ does
the same relabeling as right-multiplication by $\mathbf{P}$.

```{prf:remark} Why the transpose, and why one convention throughout
:label: rem:why-transpose

Because $\mathbf{W}^{(2)}$'s hidden-unit index is its *row* index, permuting
those rows is a left-multiplication. To perform the *same* relabeling that
$\mathbf{P}$ performs as a right-multiplication, we left-multiply by
$\mathbf{P}^{\top}$ (the inverse of a permutation matrix is its transpose).
We use this column/right, row/left convention for $\mathbf{W}^{(1)}$ and
$\mathbf{W}^{(2)}$ respectively throughout — including the gradient and
update steps — so every "permuted" quantity stays a faithful relabeling.
```

## The forward pass is invariant

We prove two things: the hidden activations merely get permuted, and the output
does not change at all.

The first step needs a small lemma — that an element-wise activation commutes
with a column permutation.

```{prf:lemma} Element-wise functions commute with permutation
:label: lem:elemwise-permute

For any element-wise function $\sigma$, any matrix $\mathbf{Z}$, and any
permutation matrix $\mathbf{P}$ of compatible width,
$\sigma(\mathbf{Z}\mathbf{P}) = \sigma(\mathbf{Z})\,\mathbf{P}$.
```

The reason: $\mathbf{Z}\mathbf{P}$ just reorders the columns of $\mathbf{Z}$,
and because $\sigma$ acts independently on each entry, reordering the columns
before or after applying $\sigma$ gives the same result. We verify it
numerically before trusting the algebra:

```{code-cell} ipython3
rng = np.random.default_rng(1992)
Z = rng.standard_normal((4, 3))                 # any pre-activation matrix
P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # a column permutation

sigmoid = lambda m: 1 / (1 + np.exp(-m))

# sigma(Z P)  vs.  sigma(Z) P
np.allclose(sigmoid(Z @ P), sigmoid(Z) @ P)
```

With the lemma in hand, the hidden activation under the permuted weights is

$$
\mathbf{H}_{\pi}
= \sigma\!\left(\mathbf{X}\mathbf{W}^{(1)}_{\pi} + \mathbf{b}^{(1)}_{\pi}\right)
= \sigma\!\left(\mathbf{X}\mathbf{W}^{(1)}\mathbf{P} + \mathbf{b}^{(1)}\mathbf{P}\right)
= \sigma\!\left(\bigl(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}\bigr)\mathbf{P}\right)
= \sigma(\mathbf{Z}\mathbf{P})
= \sigma(\mathbf{Z})\,\mathbf{P}
= \mathbf{H}\mathbf{P}.
$$ (eq:hidden-permuted)

The activations are not different — they are the same activations, columns
reordered. Now feed them through the permuted second layer:

$$
\mathbf{O}_{\pi}
= \mathbf{H}_{\pi}\mathbf{W}^{(2)}_{\pi} + \mathbf{b}^{(2)}
= (\mathbf{H}\mathbf{P})(\mathbf{P}^{\top}\mathbf{W}^{(2)}) + \mathbf{b}^{(2)}
= \mathbf{H}(\mathbf{P}\mathbf{P}^{\top})\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
= \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}
= \mathbf{O}.
$$ (eq:output-invariant)

```{prf:theorem} Forward invariance
:label: thm:forward-invariance

Permuting the hidden units of a layer — $\mathbf{W}^{(1)}_{\pi} =
\mathbf{W}^{(1)}\mathbf{P}$, $\mathbf{b}^{(1)}_{\pi} = \mathbf{b}^{(1)}\mathbf{P}$,
$\mathbf{W}^{(2)}_{\pi} = \mathbf{P}^{\top}\mathbf{W}^{(2)}$ — permutes the
hidden activations ($\mathbf{H}_{\pi} = \mathbf{H}\mathbf{P}$) and leaves the
output exactly unchanged ($\mathbf{O}_{\pi} = \mathbf{O}$).
```

The output is identical, so the *loss* is identical. The forward pass cannot
tell the two parameterizations apart.

## The backward pass is invariant

If the gradients also relabeled consistently, then a training step cannot
distinguish the parameterizations either — which is what makes the symmetry
ungovernable. We show exactly that. Take squared error
$\mathcal{L} = \tfrac{1}{2}\lVert \mathbf{O} - \mathbf{Y} \rVert^{2}$, so the
output-layer error is $\boldsymbol{\delta}_{O} = \mathbf{O} - \mathbf{Y}$,
unchanged by the permutation because $\mathbf{O}$ is.

**Output-layer weights.** Since $\partial \mathbf{O}/\partial \mathbf{W}^{(2)}$
contributes $\mathbf{H}$,

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}} = \mathbf{H}^{\top}\boldsymbol{\delta}_{O}.
$$

Under the permutation, $\mathbf{H}$ becomes $\mathbf{H}_{\pi} = \mathbf{H}\mathbf{P}$,
so

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}_{\pi}}
= \mathbf{H}_{\pi}^{\top}\boldsymbol{\delta}_{O}
= (\mathbf{H}\mathbf{P})^{\top}\boldsymbol{\delta}_{O}
= \mathbf{P}^{\top}\mathbf{H}^{\top}\boldsymbol{\delta}_{O}
= \mathbf{P}^{\top}\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(2)}}.
$$ (eq:grad-W2)

The gradient's *rows* are permuted by $\mathbf{P}^{\top}$ — precisely the
relabeling that $\mathbf{W}^{(2)}_{\pi} = \mathbf{P}^{\top}\mathbf{W}^{(2)}$
applied to the weights. Weight and gradient relabel together.

**Hidden-layer weights.** Let
$\boldsymbol{\delta}_{H} = (\boldsymbol{\delta}_{O}\mathbf{W}^{(2)\top}) \odot \sigma'(\mathbf{Z})$,
the backpropagated error at the hidden layer, with $\odot$ the element-wise
(Hadamard) product. Then
$\partial \mathcal{L}/\partial \mathbf{W}^{(1)} = \mathbf{X}^{\top}\boldsymbol{\delta}_{H}$.
We need $\boldsymbol{\delta}_{H}$ under the permutation. Using
$\mathbf{W}^{(2)\top}_{\pi} = (\mathbf{P}^{\top}\mathbf{W}^{(2)})^{\top} =
\mathbf{W}^{(2)\top}\mathbf{P}$, $\mathbf{Z}_{\pi} = \mathbf{Z}\mathbf{P}$, and
{prf:ref}`lem:elemwise-permute` again
($\sigma'(\mathbf{Z}\mathbf{P}) = \sigma'(\mathbf{Z})\mathbf{P}$):

$$
\boldsymbol{\delta}_{H,\pi}
= \bigl(\boldsymbol{\delta}_{O}\mathbf{W}^{(2)\top}_{\pi}\bigr) \odot \sigma'(\mathbf{Z}_{\pi})
= \bigl(\boldsymbol{\delta}_{O}\mathbf{W}^{(2)\top}\mathbf{P}\bigr) \odot \bigl(\sigma'(\mathbf{Z})\mathbf{P}\bigr)
= \bigl((\boldsymbol{\delta}_{O}\mathbf{W}^{(2)\top}) \odot \sigma'(\mathbf{Z})\bigr)\mathbf{P}
= \boldsymbol{\delta}_{H}\mathbf{P},
$$

where the Hadamard product pulls through the column permutation for the same
reason $\sigma$ did — it is element-wise. Therefore

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}_{\pi}}
= \mathbf{X}^{\top}\boldsymbol{\delta}_{H,\pi}
= \mathbf{X}^{\top}\boldsymbol{\delta}_{H}\mathbf{P}
= \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}}\mathbf{P},
$$ (eq:grad-W1)

and similarly $\partial \mathcal{L}/\partial \mathbf{b}^{(1)}_{\pi} =
(\partial \mathcal{L}/\partial \mathbf{b}^{(1)})\mathbf{P}$, while
$\partial \mathcal{L}/\partial \mathbf{b}^{(2)}$ is unchanged. Every gradient
that is indexed by hidden units is relabeled by the *same* $\mathbf{P}$ that
relabeled the weights.

```{prf:theorem} Backward invariance
:label: thm:backward-invariance

Under the hidden-unit permutation, the gradients relabel exactly as the
weights do: $\partial \mathcal{L}/\partial \mathbf{W}^{(1)}_{\pi} =
(\partial \mathcal{L}/\partial \mathbf{W}^{(1)})\mathbf{P}$ and
$\partial \mathcal{L}/\partial \mathbf{W}^{(2)}_{\pi} =
\mathbf{P}^{\top}(\partial \mathcal{L}/\partial \mathbf{W}^{(2)})$.
```

## The optimizer preserves the symmetry

This is the punchline. Take one SGD step of size $\alpha$. For the first layer,

$$
\mathbf{W}^{(1)}_{\pi} \leftarrow \mathbf{W}^{(1)}_{\pi} - \alpha\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}_{\pi}}
= \mathbf{W}^{(1)}\mathbf{P} - \alpha\!\left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}}\mathbf{P}\right)
= \left(\mathbf{W}^{(1)} - \alpha\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(1)}}\right)\!\mathbf{P}.
$$

The updated $\mathbf{W}^{(1)}_{\pi}$ is *still* the updated $\mathbf{W}^{(1)}$
right-multiplied by $\mathbf{P}$. The same algebra gives
$\mathbf{W}^{(2)}_{\pi} \leftarrow \mathbf{P}^{\top}(\mathbf{W}^{(2)} - \alpha\,\partial \mathcal{L}/\partial \mathbf{W}^{(2)})$.
The permutation relationship survives the step unchanged — and so, by
{prf:ref}`thm:forward-invariance`, does $\mathbf{O}_{\pi} = \mathbf{O}$.

```{prf:corollary} Permutation symmetry is a training invariant
:label: cor:symmetry-invariant

If two parameterizations are related by a hidden-unit permutation at step $t$,
they are related by the same permutation at step $t+1$, and produce the same
output at every step. Plain gradient descent cannot break permutation
symmetry.
```

```{admonition} What *can* break it
:class: tip

If the optimizer cannot, what does? Asymmetric initialization (random weights)
starts the units out *non*-identical, so there is no permutation relating them
to preserve. Stochastic regularization such as **dropout** perturbs units
independently each step, which is part of why it helps beyond its explicit
regularization role. Batch noise and data augmentation help for the same
reason. The deterministic, full-batch gradient descent of the proof above does
not.
```

## Why this matters: exact redundancy

Permutation symmetry is a form of **over-parametrization** (equivalently,
non-identifiability): many distinct settings of the parameters compute the
same function and attain the same loss. The practical consequences:

- **The loss surface has many equivalent minima.** For a layer of $h$ units
  there are $h!$ parameterizations of any solution, related by permutation.
  Comparing two trained networks parameter-for-parameter is therefore
  meaningless without accounting for this symmetry.
- **Symmetric initializations are traps.** If the initializer happens to place
  units in a symmetric configuration — most dramatically, identical — the
  corollary says training stays there. This is not a loose intuition; it is the
  content of {prf:ref}`cor:symmetry-invariant`, and it is the bridge to
  {doc}`weight initialization <03_weight_initialization>`.

## Summary

If this chapter had to be one sentence: **hidden units are interchangeable, and
because the gradient relabels exactly as the weights do, a gradient step
preserves that interchangeability — so plain training can never break a
symmetry the initializer created.** The forward invariance
($\mathbf{O}_{\pi} = \mathbf{O}$, {prf:ref}`thm:forward-invariance`) and the
gradient relabeling ({prf:ref}`thm:backward-invariance`) are the two facts to
carry forward. The next chapter uses the second of them — the
corollary — to prove that constant initialization is fatal, and then derives
the random, scale-aware schemes (Xavier, Kaiming) that break symmetry *and*
control signal scale across depth.

```{admonition} Further reading
:class: seealso

- {cite}`zhang2023dive`, §5.4, motivates the permutation-symmetry argument for
  why symmetric (constant) initialization fails; this chapter formalizes it
  fully, forward and backward.
- {doc}`Weight Initialization: Why Constant and Zero Init Fail <03_weight_initialization>`
  — the direct continuation, applying {prf:ref}`cor:symmetry-invariant`.
```
