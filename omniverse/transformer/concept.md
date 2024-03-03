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

# Concept

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

def configure_deterministic_mode() -> None:
    """
    See https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    and https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
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
    Seed all random number generators.

    Parameters
    ----------
    seed : int
        Seed number to be used, by default 1992.
    seed_torch : bool
        Whether to seed PyTorch or not, by default True.

    Returns
    -------
    seed: int
        The seed number.
    """
    # fmt: off
    os.environ["PYTHONHASHSEED"] = str(seed)       # set PYTHONHASHSEED env var at fixed value
    np.random.default_rng(seed)                    # numpy pseudo-random generator
    random.seed(seed)                              # python's built-in pseudo-random generator

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)           # pytorch (both CPU and CUDA)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        if set_torch_deterministic:
            configure_deterministic_mode()
    # fmt: on
    return seed
```

## Self-Attention

### Intuition of Attention Mechanism

Attention is not a new concept, and one of the most influencial papers came from
_Neural Machine Translation by Jointly Learning to Align and Translate_
{cite}`bahdanau2014neural`, a paper published during 2014. In the context of our
post, we would stick to one intuitive interpretation, that _the attention
mechanism describes a **weighted average** of (sequence) elements with the
weights **dynamically** computed based on an input query and elements’ keys_
{cite}`lippe2023uvadlc`. In other words, we want contextually relevant
information to be weighted more heavily than less relevant information. For
example, the sentence _the cat walks by the river bank_ would require the word
_bank_ to be weighted more heavily than the word _the_ when the word _cat_ is
being processed. The dynamic portion is also important because this allows the
model to adjust the weights based on an input sequence (note that the learned
weights are static but the interaction with the input sequence is dynamic). When
attending to the first token _cat_ in the sequence, we would want the token
_cat_ to be a **weighted average** of all the tokens in the sequence, including
itself. This is the essence of the self-attention mechanism.

### Token Embedding and Vector Representation Process

Given an input sequence $\mathbf{x} = \left(x_1, x_2, \ldots, x_T\right)$, where
$T$ is the length of the sequence, and each $x_t \in \mathcal{V}$ is a token in
the sequence, we use a generic embedding function $h_{\text{emb}}$ to map each
token to a vector representation in a continuous vector space:

$$
\begin{aligned}
h_{\text{emb}} : \mathcal{V}  &\rightarrow \mathbb{R}^{D} \\
x_t                           &\mapsto \mathbf{z}_t
\end{aligned}
$$

where $\mathcal{V}$ is the vocabulary of tokens (discrete space $\mathbb{Z}$),
and $D$ is the dimension of the embedding space (continuous space). The output
of the embedding function $h_{\text{emb}}$ is a sequence of vectors
$\mathbf{Z} = \left(\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_T\right)$,
where each $\mathbf{z}_t \in \mathbb{R}^{D}$ is the vector representation of the
token $x_t$ in the sequence. As seen earlier, we represent the sequence of
vectors $\mathbf{Z}$ as a matrix $\mathbf{Z} \in \mathbb{R}^{T \times D}$, where
each row of the matrix represents the vector representation of each token in the
sequence.

### Queries, Keys, and Values

#### Database Analogy

Let's draw an analogy to understand the concept of queries, keys, and values in
the context of the attention mechanism. Consider a database $\mathcal{D}$
consisting of tuples of keys and values. For instance, the database
$\mathcal{D}$ might consist of tuples
`{("Zhang", "Aston"), ("Lipton", "Zachary"), ("Li", "Mu"), ("Smola", "Alex"), ("Hu", "Rachel"), ("Werness", "Brent")}`
with the last name being the key and the first name being the value
{cite}`zhang2023dive`. Operations on the database $\mathcal{D}$ can be performed
using queries $q$ that operate on the keys and values in the database. More
concretely, if our query is "Li", or more verbosely, "What is the first name
associated with the last name Li?", the answer would be "Mu" - the **key**
associated with the **query** "What is the first name associated with the last
name Li?" is "Li", and the **value** associated with the key "Li" is "Mu".
Furthermore, if we also allowed for approximate matches, we would retrieve
("Lipton", "Zachary") instead.

More rigorously, we denote
$\mathcal{D} \stackrel{\text { def }}{=}\left\{\left(\mathbf{k}_1, \mathbf{v}_1\right), \ldots\left(\mathbf{k}_m, \mathbf{v}_m\right)\right\}$
a database of $m$ tuples of _keys_ and _values_, as well as a query
$\mathbf{q}$. Then we can define the attention over $\mathcal{D}$ as

$$
\operatorname{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\operatorname{def}}{=} \sum_{t=1}^T \alpha\left(\mathbf{q}, \mathbf{k}_t\right) \mathbf{v}_t
$$

where
$\alpha\left(\mathbf{q}, \mathbf{k}_t\right) \in \mathbb{R}(t=1, \ldots, T)$ are
scalar attention weights {cite}`zhang2023dive`. The operation itself is
typically referred to as
[_attention pooling_](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-pooling.html).
The term "attention" is used because this operation focuses specifically on
those terms that have a substantial weight, denoted as $\alpha$, meaning it
gives more importance to these terms. Consequently, the attention over
$\mathcal{D}$ generates a linear combination of values contained in the
database. In fact, this contains the above example as a special case where all
but one weight is zero. Why so? Because the query is an exact match for one of
the keys.

To illustrate why in the case of an exact match within a database the attention
weights ($\alpha$) are all zero except for one, let's use the attention formula
provided and consider a simplified example with vectors.

```{prf:example} Exact Match Scenario
:label: decoder-concept-attention-exact-match-scenario

Imagine a simplified database $\mathcal{D}$ consisting of 3 key-value pairs,
where each key $\mathbf{k}_t$ and the query $\mathbf{q}$ are represented as
vectors in some high-dimensional space, and the values $\mathbf{v}_t$ are also
vectors (or can be scalar for simplicity in this example). For simplicity, let's
assume our vectors are in a 2-dimensional space and represent them as follows:

-   Keys (representing $3$ keys in the database):
    -   $\mathbf{k}_1 = [1, 0]$,
    -   $\mathbf{k}_2 = [0, 1]$,
    -   $\mathbf{k}_3 = [1, 1]$
-   Values (corresponding to the keys):
    -   $\mathbf{v}_1 = [0.1, 0.9]$,
    -   $\mathbf{v}_2 = [0.2, 0.8]$,
    -   $\mathbf{v}_3 = [0.3, 0.7]$
-   Query (looking for an item/concept similar to $\mathbf{k}_1$):
    -   $\mathbf{q} = [1, 0]$

The attention weights $\alpha(\mathbf{q}, \mathbf{k}_t)$ indicate how similar or
relevant each key is to the query. In an exact match scenario, the similarity
calculation will result in a high value (e.g., $1$) when the query matches a key
exactly, and low values (e.g., $0$) otherwise. For simplicity, let's use a
simple matching criterion where the weight is $1$ for an exact match and $0$
otherwise:

-   $\alpha(\mathbf{q}, \mathbf{k}_1) = 1$ (since
    $\mathbf{q} =
    \mathbf{k}_1$, exact match)
-   $\alpha(\mathbf{q}, \mathbf{k}_2) = 0$ (since
    $\mathbf{q} \neq
    \mathbf{k}_2$, no match)
-   $\alpha(\mathbf{q}, \mathbf{k}_3) = 0$ (since
    $\mathbf{q} \neq
    \mathbf{k}_3$, no match)

Using the attention formula:

$$
\begin{aligned}
\operatorname{Attention}(\mathbf{q}, \mathcal{D})
   &= \sum_{t=1}^3 \alpha(\mathbf{q}, \mathbf{k}_t) \mathbf{v}_t \\
   &= (1 \cdot [0.1, 0.9]) + (0 \cdot [0.4, 0.6]) + (0 \cdot [0.7, 0.3]) \\
   &= [0.1, 0.9]
\end{aligned}
$$

This calculation shows that because the attention weights for $\mathbf{k}_2$ and
$\mathbf{k}_3$ are zero (due to no exact match), they don't contribute to the
final attention output. Only $\mathbf{k}_1$, which exactly matches the query,
has a non-zero weight (1), making it the sole contributor to the attention
result. This is a direct consequence of the query being an exact match for one
of the keys, leading to a scenario where "all but one weight is zero."
```

#### Queries, Keys, and Values in Attention Mechanism

The database example is a neat analogy to understand the concept of queries,
keys, and values in the context of the attention mechanism. To put things into
perspective, each token $x_t$ in the input sequence $\mathbf{x}$ emits three
vectors through projecting its corresponding token and positional embedding
output $\mathbf{z}_t$, a query vector $\mathbf{q}_t$, a key vector
$\mathbf{k}_t$, and a value vector $\mathbf{v}_t$. Consider the earlier example
_cat walks by the river bank_, where each word is a token in the sequence. When
we start to process the first token $\mathbf{z}_1$, _cat_, we would consider a
query vector $\mathbf{q}_1$, projected from $\mathbf{z}_1$, to be used to
interact with the key vectors $\mathbf{k}_t$ for $t \in \{1, 2, \ldots, T\}$, in
the sequence - determining how much _attention_ "cat" should pay to every other
token in the sequence (including itself). Consequently, it will also emit a key
vector $\mathbf{k}_1$ so that other tokens can interact with it. Subsequently,
the attention pooling will form a linear combination of the query vector
$\mathbf{q}_1$ with every other key vector $\mathbf{k}_t$ in the sequence,

$$
\alpha(\mathbf{q}_1, \mathbf{k}_t) \in \mathbb{R} = \mathbf{q}_1 \cdot \mathbf{k}_t \quad \text{for } t \in \{1, 2, \ldots, T\}
$$

and each $\alpha(\mathbf{q}_1, \mathbf{k}_t)$ will indicate how much attention
the token "cat" should pay to the token at position $t$ in the sequence. We
would later see that we would add a softmax normalization to the attention
scores to obtain the final attention weights.

We would then use the attention scores $\alpha(\mathbf{q}_1, \mathbf{k}_t)$ to
create a weighted sum of the value vectors $\mathbf{v}_t$ to form the new
representation of the token "cat".

$$
\operatorname{Attention}(\mathbf{q}_1, \mathbf{k}_t, \mathbf{v}_t) = \sum_{t=1}^T \alpha(\mathbf{q}_1, \mathbf{k}_t) \mathbf{v}_t
$$

Consequently, the first token must also emit a value vector $\mathbf{v}_1$. You
can think of the value vector as carrying the actual information or content that
will be aggregated based on the attention scores.

To reiterate, the output
$\operatorname{Attention}(\mathbf{q}_1, \mathbf{k}_t, \mathbf{v}_t)$ will be the
new representation of the token "cat" in the sequence, which is a weighted sum
of the value vectors $\mathbf{v}_t$ based on the attention scores
$\alpha(\mathbf{q}_1, \mathbf{k}_t)$ and now not only holds semantic and
positional information about the token "cat" itself but also contextual
information about the other tokens in the sequence. This allows the token "cat"
to have a better understanding of itself in the context of the whole sentence.
In this whole input sequence, the most ambiguous token is the token "bank" as it
can refer to a financial institution or a river bank. The attention mechanism
will help the token "bank" to understand its context in the sentence - likely
focusing more on the token "river" than the token "cat" or "walks" to understand
its context.

The same process will be repeated for each token in the sequence, where each
token will emit a query vector, a key vector, and a value vector. The attention
scores will be calculated for each token in the sequence, and the weighted sum
of the value vectors will be used to form the new representation of each token
in the sequence.

To end this off, we can intuitively think of the query, key and value as
follows:

-   **Query**: What does the token want to know? Maybe to the token _bank_, it
    is trying to figure out if it is a financial institution or a river bank.
    But obviously, when considering the token "bank" within such an input
    sequence, the query vector generated for "bank" would not actually ask "Am I
    a financial institution or a river bank?" but rather would be an abstract
    feature vector in a $D$ dimensional subspace that somehow captures the
    potential and context meanings of the token "bank" and once it is used to
    interact with the key vectors, it will help to determine later on how much
    attention the token "bank" should pay to the other tokens in the sequence.
-   **Key**: Carrying on from the previous point, if the query vector for the
    token "bank" is being matched with the key vectors of the other tokens in
    the sequence, the key "river" will be a good match for the query "bank" as
    it will help the token "bank" to understand its context in the sentence. In
    this subspace, the key vector for "river" will be a good match for the query
    because it is more of an "offering" service to the query vector, and it will
    know when it is deemed to be important to the query vector. As such, the
    vectors in this subspace are able to identify itself as important or not
    based on the query vector.

-   **Value**: The value vector is the actual information or content that will
    be aggregated based on the attention scores. If the attention mechanism
    determines that "river" is highly relevant to understanding the context of
    "bank" within the sentence, the value vector associated with "river" will be
    given more weight in the aggregation process. This means that the
    characteristics or features encoded in the "river" value vector
    significantly influence the representation of the sentence or the specific
    context being analyzed.

### Linear Projections

We have discussed the concept of queries, keys, and values but have not yet
discussed how these vectors are obtained. As we have continuously emphasized,
the query, key, and value vectors lie in a $D$-dimensional subspace, and they
encode various abstract information about the tokens in the sequence.
Consequently, it is no surprise that these vectors are obtained through linear
transformations/projections of the token embeddings $\mathbf{Z}$ using learned
weight matrices $\mathbf{W}^{\mathbf{Q}}$, $\mathbf{W}^{\mathbf{K}}$ and
$\mathbf{W}^{\mathbf{V}}$.

````{prf:definition} Linear Projections for Queries, Keys, and Values
:label: decoder-concept-linear-projections-queries-keys-values

In the self-attention mechanism, each token embedding
$\mathbf{z}_t \in \mathbb{R}^{D}$ is projected into a new context vector across
different **subspaces**. This projection is accomplished through three distinct
**linear transformations**, each defined by a unique weight matrix:

$$
\mathbf{W}^{\mathbf{Q}} \in \mathbb{R}^{D \times d_q}, \quad
\mathbf{W}^{\mathbf{K}} \in \mathbb{R}^{D \times d_k}, \quad
\mathbf{W}^{\mathbf{V}} \in \mathbb{R}^{D \times d_v}
$$

where $d_q, d_k, d_v \in \mathbb{Z}^+$ are the hidden dimensions of the
subspaces for the query, key, and value vectors, respectively.


```{prf:remark} Dimensionality of the Subspaces
:label: decoder-concept-linear-projections-queries-keys-values-remark

It is worth noting that this post is written in the context of understand
GPT models, and the dimensionality of the query, key, and value vectors are
the same and usually equal to the dimensionality of the token embeddings.
Thus, we may use $D$ interchangeably to indicate $d_k, d_v$ and $d_q$. This
is not always the case, as encoder-decoder models might have different
dimensionalities for the query, key, and value vectors. However, query and key
must have the same dimensionality for the dot product to work.
```

Each token embedding $\mathbf{z}_t$ is transformed into three vectors:

-   The **query vector** $\mathbf{q}_t$, representing what the token is looking
    for in other parts of the input,
-   The **key vector** $\mathbf{k}_t$, representing how other tokens can be
    found or matched,
-   The **value vector** $\mathbf{v}_t$, containing the actual information to be
    used in the output.

These transformations are formally defined as:

$$
\mathbf{q}_t = \mathbf{z}_t \mathbf{W}^{Q}, \quad
\mathbf{k}_t = \mathbf{z}_t \mathbf{W}^{K}, \quad
\mathbf{v}_t = \mathbf{z}_t \mathbf{W}^{V}
$$

with each residing in $d_q, d_k, d_v$-dimensional subspaces, respectively.

Given an input sequence of $T$ tokens, the individual vectors for each token can
be stacked into matrices:

$$
\mathbf{Q} = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \\ \vdots \\
\mathbf{q}_T \end{bmatrix} \in \mathbb{R}^{T \times d_q}, \quad \mathbf{K} =
\begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \\ \vdots \\ \mathbf{k}_T
\end{bmatrix} \in \mathbb{R}^{T \times d_k}, \quad \mathbf{V} = \begin{bmatrix}
\mathbf{v}_1 \\ \mathbf{v}_2 \\ \vdots \\ \mathbf{v}_T \end{bmatrix} \in
\mathbb{R}^{T \times d_v}
$$

where each row of these matrices corresponds to the query, key, and value
vectors for each token, respectively.

These matrices are generated through simple matrix multiplication of the token
embedding matrix $\mathbf{Z} \in \mathbb{R}^{T \times D}$ with the weight
matrices
$\mathbf{W}^{\mathbf{Q}}, \mathbf{W}^{\mathbf{K}}$ and $\mathbf{W}^{\mathbf{V}}$:

$$
\mathbf{Q} = \mathbf{Z} \mathbf{W}^{\mathbf{Q}}, \quad
\mathbf{K} = \mathbf{Z} \mathbf{W}^{\mathbf{K}}, \quad
\mathbf{V} = \mathbf{Z} \mathbf{W}^{\mathbf{V}}
$$
````

### Scaled Dot-Product Attention

#### Definition

```{prf:definition} Scaled Dot-Product Attention
:label: decoder-concept-scaled-dot-product-attention

The attention mechanism is a function that maps a set of queries, keys, and
values to an output, all of which are represented as matrices in a
$D$-dimensional space. Specifically, the function is defined as:

$$
\begin{aligned}
    \text{Attention}: \mathbb{R}^{T \times d_q} \times \mathbb{R}^{T \times d_k}
    \times \mathbb{R}^{T \times d_v}       & \rightarrow \mathbb{R}^{T \times d_v}                          \\
    (\mathbf{Q}, \mathbf{K}, \mathbf{V}) & \mapsto \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
\end{aligned}
$$

where given a query matrix $\mathbf{Q} \in \mathbb{R}^{T \times d_q}$, a key
matrix $\mathbf{K} \in \mathbb{R}^{T \times d_k}$, and a value matrix
$\mathbf{V} \in \mathbb{R}^{T \times d_v}$, the attention mechanism computes the
the output matrix as follows:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V} \in \mathbb{R}^{T \times d_v}
$$

where:

-   $\mathbf{Q}\mathbf{K}^{\top}$ represents the dot product between the query
    and key matrices, resulting in a matrix of scores that indicate the degree
    of alignment or relevance between each query and all keys.
-   $\sqrt{d_k}$ is a scaling factor used to normalize the scores, preventing them
    from becoming too large and ensuring a stable gradient during training. This
    scaling factor is particularly important as it helps maintain the softmax
    output in a numerically stable range {cite}`vaswani2017attention`.
-   $\text{softmax}(\cdot)$ is applied row-wise to convert scores into attention
    weights, ensuring that for each query, the weights across all keys sum up
    to 1. This normalization step allows the mechanism to effectively distribute
    focus across the keys according to their relevance to each query.
-   The resulting matrix of attention weights is then used to compute a weighted
    sum of the values in $\mathbf{V}$, producing the output matrix. This output
    represents a series of context vectors, each corresponding to a query and
    containing aggregated information from the most relevant parts of the input
    sequence as determined by the attention weights.
```

In what follows, we will break down the components of the attention mechanism
and explain how it works in detail:

-   What is Attention Scoring Function?
-   Why Softmax?
-   Why Scale by $\sqrt{d_k}$?
-   What is Context Vector?

#### Attention Scoring Function

In order to know which tokens in the sequence are most relevant to the current
token, we need to calculate the attention scores between the query and key
vectors. Consequently, we would need a scoring function that measures the
influence or contribution of the $j$-th position on the $i$-th position in the
sequence. This is achieved through the dot product between the query and key
vectors, the reasoning through a
[Gaussian kernel](https://en.wikipedia.org/wiki/Gaussian_filter) is rigorous and
provides a good
[intuition](https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html)
why we chose the dot product as the scoring function (other than the fact that
it is a measure of similarity).

```{prf:definition} Attention Scoring Function
:label: decoder-concept-attention-scoring-function

Define the attention scoring function $\alpha(\cdot)$ as a function
that calculates the relevance or influence of each position $t$ in the sequence
on position $i$, known as the attention scores. The attention scoring function
$\alpha(\cdot)$ is defined using the dot product between query and key
vectors, leveraging its property as a similarity measure.

$$
\begin{aligned}
\alpha: \mathbb{R}^{d_q} \times \mathbb{R}^{d_k} & \rightarrow \mathbb{R} \\
(\mathbf{q}, \mathbf{k}_t)                   & \mapsto \alpha(\mathbf{q}, \mathbf{k}_t)
\end{aligned}
$$

Specifically, the function is expressed as:

$$
\alpha(\mathbf{q}, \mathbf{k}_t) = \langle \mathbf{q}, \mathbf{k}_t \rangle = \mathbf{q} \cdot \mathbf{k}_t \in \mathbb{R}
$$

where:

-   $\mathbf{q}$ is a query vector representing in the sequence, seeking
    information or context.
-   $\mathbf{k}_t$ is the key vector representing the $t$-th position in the
    sequence, offering context or information.
-   $\langle \mathbf{q}, \mathbf{k}_t \rangle$ denotes the dot product between
    the query vector $\mathbf{q}$ and the key vector $\mathbf{k}_t$, which
    quantifies the level of similarity or alignment between the current position
    that $\mathbf{q}$ is at (say $i$-th) and $t$-th positions in the sequence.

The expression $\mathbf{q} \cdot \mathbf{k}_t$ is a scalar value that indicates
the degree of alignment or relevance between the query at $i$-th position and
the key at $t$-th position in the sequence. We would need to calculate the
attention scores for each token in the sequence with respect to the query vector
$\mathbf{q}$, and the key vectors $\mathbf{k}_t$ for
$t \in \{1, 2, \ldots, T\}$.

So this leads us to:

$$
\alpha(\mathbf{q}, \mathbf{K}) = \mathbf{q}\mathbf{K}^{\top} \in \mathbb{R}^{1 \times T}
$$

where

$$
\mathbf{K} = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \\ \vdots \\
\mathbf{k}_T \end{bmatrix} \in \mathbb{R}^{T \times d_k}
$$

is the matrix of key vectors for each token in the sequence, and the output
$\alpha(\mathbf{q}, \mathbf{K}) \in \mathbb{R}^{1 \times T}$ is a row
vector of attention scores for the query vector $\mathbf{q}$ with respect to
each key vector $\mathbf{k}_t$ for $t \in \{1, 2, \ldots, T\}$.

Lastly, there are $T$ such queries in the input sequence $\mathbf{Q}$, and we
can stack all the query vectors $\mathbf{q}_t$ into a matrix
$\mathbf{Q} \in \mathbb{R}^{T \times d_q}$ to calculate the attention scores for
all the queries in the sequence with respect to all the key vectors in the
sequence.

$$
\alpha(\mathbf{Q}, \mathbf{K}) = \mathbf{Q}\mathbf{K}^{\top} \in \mathbb{R}^{T \times T}
$$

To this end, each row of the matrix $\mathbf{Q}\mathbf{K}^{\top}$ represents the
attention scores for each query vector at position $i$ in the sequence with
respect to all the key vectors in the sequence.
```

#### Scaling Down the Dot Product of Query and Key Vectors

```{prf:definition} Query and Key are Independent and Identically Distributed (i.i.d.)
:label: decoder-concept-query-key-iid

Under the assumption of the query $\mathbf{q}$ and key $\mathbf{k}_t$ are
_**independent and identically distributed**_ (i.i.d.) random variables with a
gaussian distribution of mean $0$ and variance $\sigma^2$:

$$
\mathbf{q} \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2), \quad \mathbf{k}_t \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2)
$$
```

```{prf:definition} Variance of Dot Product
:label: decoder-concept-variance-dot-product


Given that $\mathbf{q} \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2), \quad \mathbf{k}_t \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2)$,
the variance of the dot product between the query vector $\mathbf{q}$ and the key
vector $\mathbf{k}_t$ is:

$$
\mathbb{V}[\mathbf{q} \cdot \mathbf{k}_t] = \sum_{i=1}^{d_k} \mathbb{V}[q_i
k_{ti}] = d_k \cdot \sigma^4.
$$
```

```{prf:proof}
The dot product between $\mathbf{q}$ and $\mathbf{k}_t$ can be expressed as the
sum of the products of their components:

$$\mathbf{q} \cdot \mathbf{k}_t = \sum_{i=1}^{d_k} q_i k_{ti},$$

where $q_i$ and $k_{ti}$ are the $i$-th components of $\mathbf{q}$ and
$\mathbf{k}_t$, respectively.

The variance of the sum of random variables (when these variables are
independent, which is our case since components are iid) is the sum of their
variances. The product $q_i k_{ti}$ is a new random variable, and its variance
can be calculated as follows for a single pair of components:

$$
\mathbb{V}[q_i k_{ti}] = \mathbb{E}[(q_i k_{ti})^2] - (\mathbb{E}[q_i
k_{ti}])^2.
$$

Given that $q_i$ and $k_{ti}$ are independent and both have mean 0:

$$\mathbb{E}[q_i k_{ti}] = \mathbb{E}[q_i] \cdot \mathbb{E}[k_{ti}] = 0.$$

The expectation of the square of the product is:

$$
\mathbb{E}[(q_i k_{ti})^2] = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_{ti}^2] =
\sigma^2 \cdot \sigma^2 = \sigma^4.
$$

Since $\mathbb{E}[q_i k_{ti}] = 0$, the variance of the product $q_i k_{ti}$ is
simply $\sigma^4$.

For the dot product, we sum across all $d_k$ components, and since the variance
of the sum of independent random variables is the sum of their variances:

$$
\mathbb{V}[\mathbf{q} \cdot \mathbf{k}_t] = \sum_{i=1}^{d_k} \mathbb{V}[q_i
k_{ti}] = d_k \cdot \sigma^4.
$$
```

We want to ensure that the variance of the dot product still remains the same as
the variance of the query and key vectors at $\sigma^2$ regardless of the vector
dimensions. To do so, we scale down the dot product by $\sqrt{d_k}$, which is
the square root of the dimensionality of the key vectors, this operation would
scale the variance of the dot product down by $\sqrt{d_k}^2 = d_k$ (since
variance of a scaled random variable is the square of the scale factor times the
original variance).

Now our variance would be $\sigma^4$ - but it is still not the same as the
variance of the query and key vectors. This is okay because the original paper
assume the variance $\sigma^2 = 1$ {cite}`vaswani2017attention`, and therefore
it does not matter since $\sigma^2 = \sigma^4$ when $\sigma^2 = 1$.

```{prf:definition} Attention Scoring Function with Scaling
:label: decoder-concept-attention-scoring-function-with-scaling

To this end, the updated scoring function is:

$$
\alpha(\mathbf{Q}, \mathbf{K}) = \frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}
$$
```

Before we look into the reason why we scale down the dot product, let's first
complete the final block of the attention mechanism, which is the softmax
normalization.

#### Softmax

```{prf:definition} Attention Scores
:label: decoder-concept-attention-scores

Currently the attention scores $\alpha(\mathbf{Q}, \mathbf{K})$ are raw scores
that indicate the degree of alignment or relevance between each query and all
keys. They can be negative or positive, and they can be large or small. We
denote them as the raw **attention scores** $\alpha(\mathbf{Q}, \mathbf{K}) \in
\mathbb{R}^{T \times T}$.
```

```{prf:definition} Softmax Normalization and Attention Weights
:label: decoder-concept-softmax-normalization-attention-weights

It is common in deep learning to form a convex combination {cite}`zhang2023dive`
of the attention scores $\alpha(\mathbf{Q}, \mathbf{K})$ to obtain the
**attention weights**, denoted as $\text{softmax}(\alpha(\mathbf{Q}, \mathbf{K}))$, which
are non-negative and sum to $1$. This is achieved through the softmax
normalization function, which is defined as:

$$
\text{softmax}(\alpha(\mathbf{Q}, \mathbf{K})) = \frac{\exp(\alpha(\mathbf{Q}, \mathbf{K}))}{\sum_{t=1}^T \exp(\alpha(\mathbf{Q}, \mathbf{k}_t))} \in \mathbb{R}^{T \times T}
$$

where:

-   $\exp(\cdot)$ is the exponential function, which is applied element-wise to
    the raw attention scores $\alpha(\mathbf{Q}, \mathbf{K})$.
-   The denominator is the sum of the exponentials of the raw attention scores
      across the $T$ keys, ensuring that the attention weights sum to $1$ for
      each query, allowing the mechanism to effectively distribute focus across
      the keys according to their relevance to each query.
```

The choice of softmax is a convenient choice, but not the only choice. However,
it is convenient because it is both _differentiable_, which is often a desirable
property for training deep learning models that are optimized using
gradient-based methods, and it is also _monotonic_, which means that the
**attention weights** are preserved exactly in the order as the raw **attention
scores**.

```{prf:definition} Attention Scoring Function with Scaling and Softmax
:label: decoder-concept-attention-scoring-function-with-scaling-softmax

To this end, our final attention scoring function is:

$$
\alpha(\mathbf{Q}, \mathbf{K}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}
$$
```

#### Context Vector/Matrix

Consequently, we complete the walkthrough of the scaled dot-product attention
mechanism by calculating the context vector, which is the weighted sum of the
value vectors based on the attention weights obtained from the softmax
normalization.

```{prf:definition} Context Vector/Matrix
:label: decoder-concept-context-vector-matrix

Given the attention weights $\alpha(\mathbf{Q}, \mathbf{K})$ and the value
matrix $\mathbf{V}$, the context vector $\mathbf{C}$ is defined as the output
of the scaled dot-product attention mechanism:

$$
\mathbf{C} := \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V} \in \mathbb{R}^{T \times d_v}
$$

where each row $\mathbf{c}_t$ of the context matrix $\mathbf{C}$ is the
new embedding of the token at position $t$ in the sequence, containing
not only the semantic and positional information of the token itself, but also
contextual information from the other tokens in the sequence.
```

#### Numerical Stability and Gradient Saturation

We can now revisit on the underlying reason why we scale down the dot product
$\mathbf{Q}\mathbf{K}^{\top}$ by $\sqrt{d_k}$.

First, the softmax function has all the desirable properties we want,
_smoothness_, _monotonicity_, and _differentiability_, but it is _sensitive_ to
large input values.

The softmax function is defined as follows for a given logit $z_i$ among a set
of logits $Z$:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

If the variance of the logits before applying softmax is too large (not scaled
down to a more manageable range), the exponential function $e^{z_i}$ can lead to
extremely large output values for any $z_i$ that is even slightly larger than
others in the set. This is due to the exponential function's rapid growth with
respect to its input value.

```{prf:remark} Gradient Saturation
:label: decoder-concept-gradient-saturation

-   **For one random element:** If one of the logits $z_i$ is significantly
    larger than the others (which is more likely when the variance of the logits
    is high), $e^{z_i}$ will dominate the numerator and denominator of the
    softmax function for this logit. This will cause the softmax output for this
    logit to approach 1, as it essentially overshadows all other $e^{z_j}$ terms
    in the denominator.

-   **For all others:** Simultaneously, the softmax outputs for all other logits
    $z_j$ (where $j \neq i$) will approach 0, because their $e^{z_j}$
    contributions to the numerator will be negligible compared to $e^{z_i}$ in
    the denominator. Thus, the attention mechanism would almost exclusively
    focus on the token corresponding to the dominant logit, ignoring valuable
    information from other parts of the input sequence.
-   Furthermore, the gradients through the softmax function will be very small
    (close to zero) for all logits except the dominant one, which can lead to
    _gradient saturation_ and even _vanishing gradients_ during training.
```

```{code-cell} ipython3

def softmax(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(z) / torch.sum(torch.exp(z), axis=0)

# Without scaling: large inputs
logits_large = torch.tensor([10, 20, 30], dtype=torch.float32)
softmax_large = softmax(logits_large)

d_k = 512
scaling_factor = torch.sqrt(torch.tensor(d_k))
scaled_logits = logits_large / scaling_factor
softmax_scaled = softmax(scaled_logits)

print("Softmax without scaling:", softmax_large)
print("Softmax with scaling:", softmax_scaled)
```

As we can see, a vector with large inputs can lead to a _sharpening_ effect on
the output of the softmax function, essentially causing the output to be too
peaky, converging to 1 for the largest input and 0 for the rest (one-hot).

```{prf:remark} Numerical Stability
:label: decoder-concept-numerical-stability

We know the importance of weight initialization in deep learning models,
this is because it dictates the variance of the activations and gradients
throughout the network. Without going into the theory, it is intuitive
to think that having similar variance across all layer activations is
a desirable property for numerical stability.
By doing so, the model helps to ensure that the gradients are stable
during backpropagation, avoiding the vanishing or exploding gradients problem
and enabling effective learning.

In the specific context of the attention mechanism, the variance of the dot
products used to calculate attention scores is scaled down by the factor
$\frac{1}{\sqrt{d_k}}$ to prevent softmax saturation. This allows each element
to have a chance to influence the model's learning, rather than having a single
element dominate because of the variance scaling with $d_k$.
```

#### Visualizing Variance of Dot Product

If we set $d_k = 512$, and mean $0$ with unit variance, we will see in action
that indeed the scaled dot product has a variance of $1$ while the unscaled dot
product has a variance of $512$, which coincides with our theoretical analysis.

```{code-cell} ipython3
seed_all(92, True, False)

# Set the dimensionality of the keys and queries
d_k = 512
# Set the batch size, number of heads, and sequence length
B, H, L = 4, 8, 32
# Standard deviation for initialization
sigma = 1.0

# Initialize Q and K with variance sigma^2
Q = torch.randn(B, H, L, d_k) * sigma
K = torch.randn(B, H, L, d_k) * sigma

# Calculate dot products without scaling
unscaled_dot_products = torch.matmul(Q, K.transpose(-2, -1))

# Calculate the variance of the unscaled dot products
unscaled_variance = unscaled_dot_products.var(unbiased=False)

# Apply the scaling factor 1 / sqrt(d_k)
scaled_dot_products = unscaled_dot_products / torch.sqrt(torch.tensor(d_k).float())

# Calculate the variance of the scaled dot products
scaled_variance = scaled_dot_products.var(unbiased=False)

print(f"Unscaled Variance: {unscaled_variance}")
print(f"Scaled Variance: {scaled_variance}")

# Apply softmax to the scaled and unscaled dot products
softmax_unscaled = torch.nn.functional.softmax(unscaled_dot_products, dim=-1)
softmax_scaled = torch.nn.functional.softmax(scaled_dot_products, dim=-1)
```

#### Projections Lead to Dynamic Context Vectors

From the start, we mentioned _the attention mechanism describes a **weighted
average** of (sequence) elements with the weights **dynamically** computed based
on an input query and elements’ keys_. We can easily see the **weighted
average** part through self-attention. The **dynamic** part comes from the fact
that the context vectors are computed based on the input query and its
corresponding keys. There should be no confusion that all the learnable weights
in this self-attention mechanism are the weight matrices
$\mathbf{W}^{\mathbf{Q}}$, $\mathbf{W}^{\mathbf{K}}$ and
$\mathbf{W}^{\mathbf{V}}$, but the dynamic is really because the scoring
function uses a dot product $\mathbf{Q}\mathbf{K}^{\top}$, which is **dynamic**
because it is solely decided by the full input sequence $\mathbf{x}$. Unlike
static embeddings, where the word "cat" will always have the same embedding
vector, the context vector for the word "cat" will be different in different
sentences because it now depends on the full input sequence $\mathbf{x}$.

Consequently, the projection of the token embeddings into the query and key
space is needed.

#### Implementation

```{code-cell} ipython3
class Attention(ABC, nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("The `forward` method must be implemented by the subclass.")


class ScaledDotProductAttention(Attention):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fmt: off
        d_q               = query.size(dim=-1)

        attention_scores  = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / torch.sqrt(torch.tensor(d_q).float())
        attention_scores  = attention_scores.masked_fill(mask == 0, float("-inf")) if mask is not None else attention_scores

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector    = torch.matmul(attention_weights, value)
        # fmt: on
        return context_vector, attention_weights

torch.manual_seed(42)

B, H, L, D = 4, 8, 32, 512  # batch size, head, context length, embedding dimension
Q = torch.rand(B, H, L, D)  # query
K = torch.rand(B, H, L, D)  # key
V = torch.rand(B, H, L, D)  # value

# Scaled Dot-Product Attention
attention = ScaledDotProductAttention(dropout=0.0)
context_vector, attention_weights = attention(Q, K, V)

assert context_vector.shape == (B, H, L, D)
assert attention_weights.shape == (B, H, L, L)
pprint(context_vector.shape)
pprint(attention_weights.shape)

# assert each row of attention_weights sums to 1
# assert each element of attention_weights is between 0 and 1
attention_weights_summed_over_sequences = attention_weights.sum(dim=-1)
assert torch.allclose(
    attention_weights_summed_over_sequences, torch.ones(B, H, L)
), "The attention weights distribution induced by softmax should sum to 1."
assert torch.all(
    (0 <= attention_weights) & (attention_weights <= 1)
), "All attention weights should be between 0 and 1."
```

#### Heatmap

...

## Multi-Head Attention

We will keep this section brief as many of the concepts have been covered in the
previous section. Furthermore, there are many good illustrations out there with
more detailed explanations.

### Definition

```{prf:definition} Multi-Head Attention
:label: decoder-concept-multi-head-attention

```

The multi-head attention is a function that maps a query matrix
$\mathbf{Q} \in \mathbb{R}^{T \times d_q}$, a key matrix
$\mathbf{K} \in \mathbb{R}^{T \times d_k}$, and a value matrix
$\mathbf{V} \in \mathbb{R}^{T \times d_v}$ to an output matrix defined as
$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \in \mathbb{R}^{T \times d_v}$.
The function is defined as:

$$
\begin{aligned}
    \text{MultiHead}: \mathbb{R}^{T \times d_q} \times \mathbb{R}^{T \times d_k}
    \times \mathbb{R}^{T \times d_v}       & \rightarrow \mathbb{R}^{T \times d_v}                          \\
    (\mathbf{Q}, \mathbf{K}, \mathbf{V}) & \mapsto \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
\end{aligned}
$$

where the explicit expression for the multi-head attention mechanism is:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\mathbf{C}_1, \mathbf{C}_2, \ldots, \mathbf{C}_H)\mathbf{W}^O
$$

where:

-   $H$ is the number of attention heads, which is a hyperparameter of the
    multi-head attention mechanism.
-   $\mathbf{W}_{h}^{\mathbf{Q}} \in \mathbb{R}^{D \times d_q}$: The learnable
    query weight matrix for the $h$-th head.
    -   Note that $d_q = \frac{D}{H}$, where $D$ is the hidden dimension of the
        token embeddings.
-   $\mathbf{W}_{h}^{\mathbf{K}} \in \mathbb{R}^{D \times d_k}$: The key weight
    matrix for the $h$-th head.
    -   Note that $d_k = \frac{D}{H}$, where $D$ is the hidden dimension of the
        token embeddings.
-   $\mathbf{W}_{h}^{\mathbf{V}} \in \mathbb{R}^{D \times d_v}$: The value
    weight matrix for the $h$-th head.
    -   Note that $d_v = \frac{D}{H}$, where $D$ is the hidden dimension of the
        token embeddings.
-   $\mathbf{C}_h = \text{Attention}(\mathbf{Q}\mathbf{W}^Q_h, \mathbf{K}\mathbf{W}^K_h, \mathbf{V}\mathbf{W}^V_h) \in \mathbb{R}^{T \times d_v}$
    for $h = 1, 2, \ldots, H$ is the context matrix obtained from the $h$-th
    head of the multi-head attention mechanism.
    -   We also often denote $\mathbf{C}_h$ as $\text{head}_h$.
-   $\text{Concat}(\cdot)$ is the concatenation operation that concatenates the
    context matrices $\mathbf{C}_1, \mathbf{C}_2, \ldots, \mathbf{C}_H$ along
    the feature dimension, resulting in a matrix of context vectors of shape
    $\mathbb{R}^{T \times H \cdot d_v} = \mathbb{R}^{T \times D}$.
-   $\mathbf{W}^O \in \mathbb{R}^{d_v \times H \cdot d_v}$ is a learnable weight
    matrix that projects the concatenated context vectors back to the original
    dimensionality $D$.

### ???

```
H = num_heads = 1
d_q = d_k = d_v = D // H
pprint(d_q)

# W_q = nn.Linear(D, d_q)
W_q = torch.randn(D, d_q, requires_grad=True)
pprint(W_q)

Q = Z @ W_q

W_k = torch.randn(D, d_k, requires_grad=True)
K = Z @ W_k

W_v = torch.randn(D, d_v, requires_grad=True)
V = Z @ W_v

pprint(Q)
pprint(K)
pprint(V)
```

## Casual Attention/Masked Self-Attention

In the context of GPT models, which is a decoder-only architecture, the
self-attention mechanism is often referred to as **masked self-attention** or
**causal attention**. The reason is that the attention mechanism is masked to
prevent information flow from future tokens to the current token. Given the
autoregressive and self-supervised nature of the GPT models, the prediction for
the current token should not be influenced by future tokens, as they are not
known during inference.

Consequently, the self-attention mechanism in the GPT models is designed to
allow each token to attend to itself and all preceding tokens in the sequence,
but not to future tokens. We would need to change the earlier explanation of the
self-attention mechanism to reflect this constraint.

### Intuition

One sentence to summarize the understanding.

**Casual attention (masked self attention) in decoder reduces to self attention
for the last token in the input sequence.**

Causal attention in a decoder architecture, such as the one used in Transformer
models, effectively reduces to self-attention for the last token in the input
sequence.

1. **Causal Attention Mechanism**: In a causal attention mechanism, each token
   is allowed to attend to itself and all preceding tokens in the sequence. This
   is enforced by masking future tokens to prevent information flow from future
   tokens into the current or past tokens. This mechanism is crucial in
   generative models where the prediction for the current token should not be
   influenced by future tokens, as they are not known during inference.

2. **Self-Attention Mechanism**: In self-attention, each token computes
   attention scores with every other token in the sequence, including itself.
   These attention scores are used to create a weighted sum of the values (token
   representations), which becomes the new representation of the token.

3. **Last Token in the Sequence**: When considering the last token in the
   sequence, the causal attention mechanism's nature implies that this token has
   access to all previous tokens in the sequence, including itself. There are no
   future tokens to mask. Therefore, the attention mechanism for this token
   becomes identical to the standard self-attention mechanism where it is
   attending to all tokens up to itself.

## Perplexity

-   https://keras.io/api/keras_nlp/metrics/perplexity/
-   https://lightning.ai/docs/torchmetrics/stable/text/perplexity.html
-   https://huggingface.co/docs/transformers/perplexity

## References and Further Readings
