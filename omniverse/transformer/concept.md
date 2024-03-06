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

## Introduction

...

## GPT-1 and GPT-2

In Natural Language Understanding (NLU), there are a wide range of tasks, such
as textual entailment, question answering, semantic similarity assessment, and
document classification. These tasks are inherently labeled, but given the
scarcity of such data, it makes
[discriminative](https://en.wikipedia.org/wiki/Discriminative_model) models such
as Bidirectional Long Short-Term Memory (Bi-LSTM) underperform
{cite}`radford2018improving`, likely leading to poor performance on these tasks.

In the paper
[_Improving Language Understanding by Generative Pre-Training_](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf),
the authors demonstrated that _generative pre-training_ of a language model on a
diverse corpus of unlabeled text, followed by _discriminative fine-tuning_ on
each specific task, can overcome the constraints of the small amount of
annotated data for these specific tasks. The process is collectively termed as
[semi-supervised learning](https://en.wikipedia.org/wiki/Semi-supervised_learning)
and the goal is to learn an **_universal representation_** of the natural
language space that can be used across a wide range of tasks.

The pretraining objective is to predict the next token in a sequence, in an
**_autoregressive_** manner, given the previous tokens. The pretrained model,
often known as the **_foundational model_** (or _backbone_), serves as a base
from which specialized capabilities can be added through _fine-tuning_ on
specific tasks. In the fine-tuning phase, task-specific adaptations are
necessary: the input format must be adjusted to align with the particular
requirements of the task at hand, and the model's final layer—or "head"—needs to
be replaced to accommodate the task's specific class structure. The author
showed that this approach yielded state-of-the-art results on a wide range of
NLU tasks.

Notwithstanding the success of this approach, the same set of authors came up
with a new paper in the following year, titled
[_Language Models are Unsupervised Multitask Learners_](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf),
where they introduced a new model, _GPT-2_, that was larger in model capacity,
and trained on a much larger unlabeled corpus, **WebText**. However, the key
innovation was to void the supervised fine-tuning step, and instead, they
demonstrated that GPT-2 could be used directly on a wide range of NLU tasks
directly, with what they termed as the _zero-shot transfer_. The motivation is
that the authors think that foundational language models should be competent
generalists, rather than narrowly experts {cite}`radford2019language`. They call
for the need to shift the language model paradigm to one that is generic enough
to handle NLU tasks without the need to curate specific training data for each
specific task.

In what follows, we take a look how the authors formalized the framework. We
start by defining certain definitions and notations that will be used throughout
this article.

## Autoregressive Self-Supervised Learning Paradigm

Let $\mathcal{D}$ be the true but unknown distribution of the natural language
space. In the context of unsupervised learning with self-supervision, such as
language modeling, we consider both the inputs and the implicit labels derived
from the same data sequence. Thus, while traditionally we might decompose the
distribution $\mathcal{D}$ of a supervised learning task into input space
$\mathcal{X}$ and label space $\mathcal{Y}$, in this scenario, $\mathcal{X}$ and
$\mathcal{Y}$ are intrinsically linked, because $\mathcal{Y}$ is a shifted
version of $\mathcal{X}$, and so we can consider $\mathcal{D}$ as a distribution
over $\mathcal{X}$ only.

Since $\mathcal{D}$ is a distribution, we also define it as a probability
distribution over $\mathcal{X}$, and we can write it as:

$$
\begin{aligned}
\mathcal{D} &= \mathbb{P}(\mathcal{X} ; \boldsymbol{\Theta}) \\
            &= \mathbb{P}_{\{\mathcal{X} ; \boldsymbol{\Theta}\}}(\mathbf{x})
\end{aligned}
$$

where $\boldsymbol{\Theta}$ is the parameter space that defines the distribution
$\mathbb{P}(\mathcal{X} ; \boldsymbol{\Theta})$ and $\mathbf{x}$ is a sample
from $\mathcal{X}$ generated by the distribution $\mathcal{D}$. It is common to
treat $\mathbf{x}$ as a sequence of tokens (i.e. a sentence is a sequence of
tokens), and we can write $\mathbf{x} = \left(x_1, x_2, \ldots, x_T\right)$,
where $T$ is the length of the sequence.

Given such a sequence $\mathbf{x}$, the joint probability of the sequence can be
factorized into the product of the conditional probabilities of each token in
the sequence via the
[chain rule of probability](<https://en.wikipedia.org/wiki/Chain_rule_(probability)>):

$$
\mathbb{P}(\mathbf{x} ; \boldsymbol{\Theta}) = \prod_{t=1}^T \mathbb{P}(x_t \mid x_1, x_2, \ldots, x_{t-1} ; \boldsymbol{\Theta})
$$

We can do this because natural language are _inherently ordered_. Such
decomposition allows for _tractable sampling_ from and _estimation_ of the
distribution $\mathbb{P}(\mathbf{x} ; \boldsymbol{\Theta})$ as well as any
conditionals in the form of
$\mathbb{P}(x_{t-k}, x_{t-k+1}, \ldots, x_{t} \mid x_{1}, x_{2}, \ldots, x_{t-k-1} ; \boldsymbol{\Theta})$
{cite}`radford2019language`.

To this end, consider a corpus $\mathcal{S}$ with $N$ sequences
$\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{N}\right\}$ that are
sampled $\text{i.i.d.}$ from the distribution $\mathcal{D}$ and let GPT model
$\mathcal{G}$ be an _autoregressive_ and _self-supervised learning_ model that
is trained to maximize the likelihood of the sequences in the corpus
$\mathcal{S}$, which is defined as the objective function
$\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$ where
$\hat{\boldsymbol{\Theta}}$ is the estimated parameter space that approximates
the true parameter space $\boldsymbol{\Theta}$.

### Autoregressive Self-Supervised Learning

The learning paradigm of an autoregressive self-supervised learning framework
can be formalized as a learning algorithm $\mathcal{A}$ that is trained to
predict the next token $x_t$ in a sequence given the previous tokens
$x_{<t} = \left(x_1, x_2, \ldots, x_{t-1}\right)$ in the sequence $\mathbf{x}$
(_autoregressive_), where $t \in \{1, 2, \ldots, T\}$ is the position of the
token in the sequence, and _self-supervised_ because the "label" $x_t$ is
derived from the input sequence $\mathbf{x}$ itself. The model $\mathcal{G}$
then uses $\mathcal{A}$ to learn a **_conditional probability distribution_**
$\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$ over the vocabulary
$\mathcal{V}$ of tokens, conditioned on the contextual preciding tokens
$x_{<t} = \left(x_1, x_2, \ldots, x_{t-1}\right)$, where $\boldsymbol{\Theta}$
is the parameter space that defines the distribution
$\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$.

The distinction between $\mathcal{V}$ and $\mathcal{X}$ is that $\mathcal{V}$ is
the vocabulary of tokens, which is a discrete space, and $\mathcal{X}$ is the
natural language space, which is a combinatorial discrete space. We can think of
$\mathcal{X}$ as the natural language space of _**all possible sequences**_
$\mathbf{x}$ that can be formed from the vocabulary $\mathcal{V}$ (an
enumeration over $\mathcal{V}$). Consequently, there is no confusion that a
_sequence_ $\mathbf{x}$ is a member of $\mathcal{X}$, and a _token_ $x_t$ is a
member of $\mathcal{V}$.

Through this learning algorithm, we can recover all chained conditional
probabilities of the form $\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$,
which implicitly defines the joint probability distribution
$\mathbb{P}(\mathbf{x}
; \boldsymbol{\Theta})$ over the natural language space
$\mathcal{X}$[^1].

### Estimation of the Conditional Probability Distribution

In practice, we can only _**estimate**_ the conditional probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$ from the corpus
$\mathcal{S}$, and we can write the process of estimating as:

$$
\hat{\mathbb{P}}(x_t \mid x_{<t} ; \hat{\boldsymbol{\Theta}}) \approx \mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})
$$

where $\hat{\boldsymbol{\Theta}}$ is the estimated parameter space that
approximates the true parameter space $\boldsymbol{\Theta}$.

To facilitate the notational burden, we denote the estimated conditional
probability distribution
$\hat{\mathbb{P}}(x_t \mid x_{<t} ; \hat{\boldsymbol{\Theta}})$ as a function
$f_{\hat{\boldsymbol{\Theta}}}(\cdot)$, and equate them as:

$$
\begin{aligned}
f_{\hat{\boldsymbol{\Theta}}}(x_t \mid x_{<t}) &:= \mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta}) \\
\end{aligned}
$$

where $f_{\hat{\boldsymbol{\Theta}}}(x_t \mid x_{<t})$ can be realised as our
GPT model $\mathcal{G}$.

To this end, we should be clear that this learning process is to approximate the
true distribution $\mathcal{D}$ of the natural language space $\mathcal{X}$, but
instead of modeling over the entire space $\mathcal{X}$, consisting of all
sequences $\mathbf{x}$, we model over the vocabulary $\mathcal{V}$ of tokens,
which is to generate the next token in a sequence given the previous tokens in
the sequence.

### Initial Condition of Conditional Probability Distribution

While the earlier conditional distribution seems correct by definition of the
[chain rule of probability](<https://en.wikipedia.org/wiki/Chain_rule_(probability)>),
it is worth noting that we are being a bit loose when $t=1$. Firstly, when
$t=1$, we are actually conditioning on nothing, and so it is the case that we
are estimating $\mathbb{P}(x_1 ; \boldsymbol{\Theta})$. But this is not part of
the learning process because we would need something to condition on. For the
sake of completeness, we can treat the initial token $x_1$ as the initial
condition, and we can write the chain rule as:

$$
\mathbb{P}(\mathbf{x} ; \boldsymbol{\Theta}) = \mathbb{P}(x_1 ; \boldsymbol{\Theta}) \prod_{t=2}^T \mathbb{P}(x_t \mid x_1, x_2, \ldots, x_{t-1} ; \boldsymbol{\Theta})
$$

where $\mathbb{P}(x_1 ; \boldsymbol{\Theta})$ can be thought of the "initial
prompt" or "initial condition" of the sequence $\mathbf{x}$.

For further reading, one can find more details below:

-   [Working with Sequences - Dive Into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/sequence)
-   [How do LLMs learn to be "Generative", as we often describe them?](https://github.com/huggingface/transformers/issues/28860)

### Markov Assumption

Now suppose that we wish to employ the strategy mentioned above, where we
condition only on the $\tau$ previous time steps, i.e.,
$x_{t-1}, \ldots, x_{t-\tau}$, rather than the entire sequence history
$x_{t-1}, \ldots, x_1$. Whenever we can throw away the history beyond the
previous $\tau$ steps without any loss in predictive power, we say that the
sequence satisfies a Markov condition, i.e., that the future is conditionally
independent of the past, given the recent history. When $\tau=1$, we say that
the data is characterized by a first-order Markov model, and when $\tau=k$, we
say that the data is characterized by a $k^{\text {th }}$-order Markov model
{cite}`zhang2023dive`.

More formally, a discrete-time Markov chain is a sequence of
[random variables](https://en.wikipedia.org/wiki/Random_variable)
$X_1, X_2, X_3, \ldots$ with the
[Markov property](https://en.wikipedia.org/wiki/Markov_property), namely that
the probability of moving to the next state depends only on the present state
and not on the previous states:

$$
\mathbb{P}\left(X_{t+1} \mid X_{1}, X_{2}, \ldots, X_{t}\right) = \mathbb{P}\left(X_{t+1} \mid X_{t-k+1}, X_{t-k+2}, \ldots, X_{t}\right)
$$

for all $t \in \mathbb{N}$ and all states
$X_{t+1}, X_{t}, X_{1}, X_{2}, \ldots$.

The [Markov assumption](https://en.wikipedia.org/wiki/Markov_property) is more
of an implicit assumption in the autoregressive self-supervised learning
framework where we can draw parallels to. We often find it useful to work with
models that proceed as though a Markov condition were satisfied, even when we
know that this is only approximately true. With real text documents we continue
to gain information as we include more and more leftwards context. But these
gains diminish rapidly. Thus, sometimes we compromise, obviating computational
and statistical difficulties by training models whose validity depends on a
$k^{\text {th }}$-order Markov condition. Even today's massive RNN- and
Transformerbased language models seldom incorporate more than thousands of words
of context {cite}`zhang2023dive`. In short, the Markov assumption is a
convenient assumption to simplify the modeling of the joint probability
distribution of the token sequences.

Further readings on the Markov assumption can be found in the following:

-   [GPT-4 absolutely isn’t a Markov chain](https://news.ycombinator.com/item?id=35551452)
-   [GPT is a Finite State Markov Chain - Andrej Karpathy](https://twitter.com/karpathy/status/1645115622517542913)
-   [Working with Sequences - Dive Into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/sequence.html)
-   [Why GPT model is a higher order hidden markov model](https://cs.stackexchange.com/questions/160891/why-gpt-model-is-a-higher-order-hidden-markov-model)

### The Estimator Function is Smooth with Respect to the Parameters

This assumption is a common one in the context of deep learning, because for
when we say that the estimator function $f_{\hat{\boldsymbol{\Theta}}}(\cdot)$
is _smooth_ with respect to the parameter space $\hat{\boldsymbol{\Theta}}$, we
state a simplified definition as follows.

The estimator function $f_{\hat{\boldsymbol{\Theta}}}(\cdot)$ is _smooth_ with
respect to the parameter space $\hat{\boldsymbol{\Theta}}$ if the function is
continuous and differentiable with respect to the parameter space
$\hat{\boldsymbol{\Theta}}$ up to a certain order (usually the first for SGD
variants and second order for Newton).

What this implies is that the derivative of the function with respect to the
parameter space $\hat{\boldsymbol{\Theta}}$, denoted as
$\nabla_{\hat{\boldsymbol{\Theta}}} f_{\hat{\boldsymbol{\Theta}}}(\cdot)$ is
continuous. Loosely, you can think of that a small perturbation in the parameter
space $\hat{\boldsymbol{\Theta}}$ will result in a small change in the output of
the function $f_{\hat{\boldsymbol{\Theta}}}(\cdot)$ - enabling gradient-based
optimization algorithms to work effectively as if not, then taking a step in the
direction of the gradient would not guarantee a decrease in the loss function,
slowing down convergence.

However, this is also not a strict assumption as in practice, piece-wise linear
activation functions are not smooth because the derivative is not continuous at
$0$, and consequently, $f_{\hat{\boldsymbol{\Theta}}}(\cdot)$ is
[not smooth with respect to the parameter space](https://stats.stackexchange.com/questions/473643/why-are-neural-networks-smooth-functions)
$\hat{\boldsymbol{\Theta}}$.

### Context Length and Token Context Window

Given a coherent sequence of tokens $\mathbf{x}$, say, _the tabby cat walks by
the river bank_, we may not always pass the full sequence to the model. Based on
a _context length_ $\tau$, we can pass a _token context window_ of length $\tau$
to the model. For instance, if $\tau=4$, then the token context window would be
$\left(x_{t-3}, x_{t-2}, x_{t-1}, x_{t}\right)$, and the model would be trained
to predict the next token $x_{t+1}$ given the token context window. In other
words, the sentence above would be broken down into the following token context
windows:

-   _the tabby cat walks_
-   _by the river bank_

And the longer the context length, the model would be able to capture
longer-range dependenciees in the sequence, but also may increase the
computational complexity of the model {cite}`math11112451`.

More formally, we can define the token context window as a function
$C_{\tau}(\mathbf{x}, t)$ that maps a sequence $\mathbf{x}$ and a position $t$
to a token context window of length $\tau$:

$$
\begin{aligned}
C_{\tau} : \mathcal{X} \times \mathbb{N} &\rightarrow \mathcal{X}^{\tau} \\
(\mathbf{x}, t) &\mapsto \left(x_{t-\tau+1}, x_{t-\tau+2}, \ldots, x_{t}\right)
\end{aligned}
$$

### Conditional Entropy and Perplexity as Loss Function

Having defined the basis of the autoregressive self-supervised learning
framework, we can now define the loss function $\mathcal{L}$ that is used to
train the model $\mathcal{G}$ to maximize the likelihood of the sequences in the
corpus $\mathcal{S}$. In order to transit towards the final objective/loss
function, we would need to define the notion of
[_**conditional entropy**_](https://en.wikipedia.org/wiki/Conditional_entropy).

#### Conditional Entropy

Define $X_t$ as a random variable representing the token at position $t$ in the
sequence $\mathbf{x}$ and $X_{<t} = \left(X_1, X_2, \ldots, X_{t-1}\right)$ as
random variables representing the tokens at positions $1, 2, \ldots, t-1$ in the
sequence $\mathbf{x}$. Then the conditional
[entropy](https://en.wikipedia.org/wiki/Shannon_Entropy) of the token $X_t$
given a specific realization of $X_{<t}$ is defined as:

$$
\begin{aligned}
H\left(X_t \mid X_{<t} = x_{<t} \right) &= -\sum_{x_t \in \mathcal{V}} \mathbb{P}\left(x_t \mid x_{<t} ; \boldsymbol{\Theta}\right) \log \mathbb{P}\left(x_t \mid x_{<t} ; \boldsymbol{\Theta}\right) \\
\end{aligned}
$$

This calculates the conditional entropy given a specific realization of the
context $X_{<t} = x_{<t}$, where we see that summation sums over all
possibilities of the token $x_t$ in the vocabulary $\mathcal{V}$, considering
the probability of the token $x_t$ given _a particular preceding_ sequence of
tokens.

To account for all possible realizations of the context $X_{<t}$, we simply sum
over all possible realizations of the context $X_{<t}$, and we can write the
conditional entropy as:

$$
\begin{aligned}
H\left(X_t \mid X_{<t}\right) = -\sum_{x_{t} \in \mathcal{V}} \sum_{x_{<t} \in \mathcal{V}^{<t}} \mathbb{P}\left(x_t, x_{<t} ; \boldsymbol{\Theta}\right) \log \mathbb{P}\left(x_t \mid x_{<t} ; \boldsymbol{\Theta}\right)
\end{aligned}
$$

where $\mathbb{P}\left(x_t, x_{<t} ; \boldsymbol{\Theta}\right)$ is the joint
probability distribution of observing the sequence $(x_{<t}, x_t)$,
$\mathbb{P}\left(x_t \mid x_{<t} ; \boldsymbol{\Theta}\right)$ is the
conditional probability distribution of observing the token $x_t$ given the
context $x_{<t}$, and $\mathcal{V}^{<t}$ is the set of all possible realizations
of the context $X_{<t}$.

It is worth noting that the conditional entropy $H\left(X_t \mid X_{<t}\right)$
is also the conditional expectation of the negative log-likelihood of the token
$X_t$ given the context $X_{<t}$, and we can write it as:

$$
H\left(X_t \mid X_{<t}\right) = -\mathbb{E}_{\mathcal{D}}\left[\log \mathbb{P}\left(X_t \mid X_{<t} ; \boldsymbol{\Theta}\right)\right]
$$

One can more details on the concept of conditional entropy in the following:

-   [Conditional Entropy - Wikipedia](https://en.wikipedia.org/wiki/Conditional_entropy)
-   [Conditional Expectation - Wikipedia](https://en.wikipedia.org/wiki/Conditional_expectation)

#### Perplexity

Language model has a standing history of using
[**Perplexity**](https://en.wikipedia.org/wiki/Perplexity) as a measure of the
quality of a language model. It is a measure of how well a probability
distribution or probability model predicts a sample. Without going into the
details, we define the perplexity of a probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$ as the exponential of the
conditional entropy of the distribution, and we can write it as:

$$
\begin{aligned}
\operatorname{Perplexity}\left(\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})\right) &= \exp\left(H\left(X_t \mid X_{<t}\right)\right) \\
\end{aligned}
$$

To read more about perplexity, one can find more details in the following:

-   [Perplexity - Wikipedia](https://en.wikipedia.org/wiki/Perplexity)
-   [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity)

#### Loss Function

Given the definitions of the conditional entropy and perplexity, we can define
the loss and objective function $\mathcal{L}$:

$$
\begin{aligned}
\mathcal{L}\left(\mathcal{D} ; \boldsymbol{\Theta}\right) &= -\sum_{\mathbf{x} \in \mathcal{D}} \sum_{t=1}^T \log \mathbb{P}\left(x_t \mid C_{\tau}(\mathbf{x}, t) ; \boldsymbol{\Theta}\right) \\
\end{aligned}
$$

However, we do not know the true distribution $\mathcal{D}$, and so we can only
estimate the loss function $\mathcal{L}$ from the corpus $\mathcal{S}$, and we
can write the process of estimating as:

$$
\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right) = -\sum_{\mathbf{x} \in \mathcal{S}} \sum_{t=1}^T \log \mathbb{P}\left(x_t \mid C_{\tau}(\mathbf{x}, t) ; \hat{\boldsymbol{\Theta}}\right)
$$

#### Convergence

It can be shown that the given the Markov assumption and a token context window
size of $\tau$, the loss function $\mathcal{L}$ is a
[consistent estimator](https://en.wikipedia.org/wiki/Consistent_estimator) of
the true distribution $\mathcal{D}$, and the the objective
$\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$
converges to the true conditional probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$ over $\mathcal{D}$ as the
size of the corpus $\mathcal{S}$ goes to infinity, if the model has sufficient
capacity and the optimization algorithm is appropriate {cite}`math11112451`.

Furthermore, the proposition that the conditional entropy
$H\left(X_t \mid X_{<t}\right)$ of the true data-generating process is upper
bounded by the by the logarithm of the size of the vocabulary $\mathcal{V}$,
i.e., $H\left(X_t \mid X_{<t}\right) \leq \log |\mathcal{V}|$
{cite}`math11112451`.

The proposition that the conditional entropy has an upper limit, carries
significant implications for optimizing autoregressive self-supervised learning
models. Specifically, because the conditional entropy cannot exceed the
logarithm of the vocabulary size $\mathcal{V}$, we infer a similar upper limit
on perplexity. This cap on perplexity offers a valuable benchmark for evaluating
and comparing different models, establishing a theoretical maximum for model
performance based on the size of the vocabulary {cite}`math11112451`.

### GPT is a Autoregressive Self-Supervised Learning Model

Finally, we can piece together the autoregressive self-supervised learning
framework to define the GPT model $\mathcal{G}$ as a model that is trained to
maximize the likelihood of the sequences in the corpus $\mathcal{S}$ via the
objective function
$\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$ where
$\hat{\boldsymbol{\Theta}}$ is the estimated parameter space that approximates
the true parameter space $\boldsymbol{\Theta}$, and $\mathcal{S}$ is the corpus
of sequences that are sampled $\text{i.i.d.}$ from the distribution
$\mathcal{D}$.

In pseudo-code, the GPT model $\mathcal{G}$ consists of decoder blocks, each
block consisting of a multi-head self-attention mechanism and a position-wise
feed-forward neural network, with a head layer to produce a probability
distribution over the vocabulary $\mathcal{V}$ of tokens.

$$
\begin{aligned}
h_0 &= \mathcal{S} \cdot \mathbf{W}_{e}+ \mathbf{W}_{p} \\
h_{\ell} &= \text{DecoderBlock}(h_{\ell-1}) \quad \text{for} \quad \ell = 1, 2, \ldots, L \\
\mathbb{P}(x_t \mid C_{\tau}(\mathbf{x}, t) ; \boldsymbol{\Theta}) &= \text{softmax}(h_{L} \cdot \mathbf{W}_{e}^{\top})
\end{aligned}
$$

where:

-   $\mathbf{W}_{e}$ is the embedding matrix that maps the token to a vector
    representation in a continuous vector space,
-   $\mathbf{W}_{p}$ is the positional encoding matrix that encodes the position
    of the token in the sequence,
-   $\text{DecoderBlock}$ is a function that applies a multi-head self-attention
    mechanism and a position-wise feed-forward neural network to the input
    sequence,
-   $\mathbf{W}_{e}^{\top}$ is the transpose of the embedding matrix that maps
    the vector representation of the token back to the vocabulary space.

Note that it is only a pseudo-code because notations like $\mathbf{W}_{e}$ are
used to denote both the token embedding matrix in $h_0$ and the transformed
contextual embedding matrix in the head/linear/last layer. The actual
implementation of the GPT model is more complex, and we will take a look at it
in later sections.

### Supervised Fine-Tuning

Though GPT-2 has demonstrated that it can be used directly on a wide range of
NLU without the need for supervised fine-tuning, it is worth taking a detour
back to how GPT-1 was fine-tuned immediately after the pretraining phase.

In the paper _Improving Language Understanding by Generative Pre-Training_,
after the pretrained (foundational) model was trained with the objective
function
$\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$, we
would then fine-tune the model on a specific task by replacing the final layer
of the model with a task-specific layer, and then train the model on the
specific task with the task-specific layer. The authors showed that this
approach yielded state-of-the-art results on a wide range of NLU tasks.

## Multi-Head Attention

We will keep this section brief as many of the concepts have been covered in the
previous section. Furthermore, there are many good illustrations out there with
more detailed explanations.

### Definition

```{prf:definition} Multi-Head Attention
:label: decoder-concept-multi-head-attention

abc
```

The multi-head attention is a function that maps a query matrix
$\mathbf{Q} \in \mathbb{R}^{T \times d_q}$, a key matrix
$\mathbf{K} \in \mathbb{R}^{T \times d_k}$, and a value matrix
$\mathbf{V} \in \mathbb{R}^{T \times d_v}$ to an output matrix defined as
$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \in \mathbb{R}^{T \times d_v}$.
The function is defined as:

$$
\begin{aligned} \text{MultiHead}: \mathbb{R}^{T \times d_q} \times \mathbb{R}^{T
\times d_k} \times \mathbb{R}^{T \times d_v} & \rightarrow \mathbb{R}^{T \times
d_v} \\ (\mathbf{Q}, \mathbf{K}, \mathbf{V}) & \mapsto
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \end{aligned}
$$

where the explicit expression for the multi-head attention mechanism is:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{Concat}(\mathbf{C}\_1, \mathbf{C}\_2, \ldots, \mathbf{C}\_H)\mathbf{W}^O
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

[^1]:
    This part is not concrete as the formalization is not rigorous in the
    statistical learning framework, but the general idea is there.

[^2]: https://d2l.ai/chapter_recurrent-neural-networks/sequence.html

    $$
    $$
