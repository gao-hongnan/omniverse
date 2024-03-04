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

## Framework

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
distribution {cite}`radford2019language`
$\mathbb{P}(\mathbf{x} ; \boldsymbol{\Theta})$ as well as any conditionals in
the form of
$\mathbb{P}(x_{t-k}, x_{t-k+1}, \ldots, x_{t} \mid x_{1}, x_{2}, \ldots, x_{t-k-1} ; \boldsymbol{\Theta})$.

To this end, consider a corpus $\mathcal{S}$ with $N$ sequences
$\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{N}\right\}$ that are
sampled $\text{i.i.d.}$ from the distribution $\mathcal{D}$ and let GPT model
$\mathcal{G}$ be an _autoregressive_ and _self-supervised learning_ model that
is trained to maximize the likelihood of the sequences in the corpus
$\mathcal{S}$, which is defined as the objective function
$\mathcal{L}\left(\mathcal{S} ; \boldsymbol{\Theta}\right)$.

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
parameter space $\hat{\boldsymbol{\Theta}}$
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

### GPT Autoregressive Self-Supervised Learning Model

Finally, we can piece together the autoregressive self-supervised learning
framework to define the GPT model $\mathcal{G}$ as a model that is trained to
maximize the likelihood of the sequences in the corpus $\mathcal{S}$ via the
objective function
$\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$ where
$\hat{\boldsymbol{\Theta}}$ is the estimated parameter space that approximates
the true parameter space $\boldsymbol{\Theta}$, and $\mathcal{S}$ is the corpus
of sequences that are sampled $\text{i.i.d.}$ from the distribution
$\mathcal{D}$.

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

\begin{aligned} h\_{\text{emb}} : \mathcal{V} &\rightarrow \mathbb{R}^{D} \\ x_t
&\mapsto \mathbf{z}\_t \end{aligned}


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

\operatorname{Attention}(\mathbf{q}, \mathcal{D})
\stackrel{\operatorname{def}}{=} \sum\_{t=1}^T \alpha\left(\mathbf{q},
\mathbf{k}\_t\right) \mathbf{v}\_t


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

\begin{aligned} \operatorname{Attention}(\mathbf{q}, \mathcal{D}) &=
\sum\_{t=1}^3 \alpha(\mathbf{q}, \mathbf{k}\_t) \mathbf{v}\_t \\ &= (1 \cdot
[0.1, 0.9]) + (0 \cdot [0.4, 0.6]) + (0 \cdot [0.7, 0.3]) \\ &= [0.1, 0.9]
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

\alpha(\mathbf{q}\_1, \mathbf{k}\_t) \in \mathbb{R} = \mathbf{q}\_1 \cdot
\mathbf{k}\_t \quad \text{for } t \in \{1, 2, \ldots, T\}


$$

and each $\alpha(\mathbf{q}_1, \mathbf{k}_t)$ will indicate how much attention
the token "cat" should pay to the token at position $t$ in the sequence. We
would later see that we would add a softmax normalization to the attention
scores to obtain the final attention weights.

We would then use the attention scores $\alpha(\mathbf{q}_1, \mathbf{k}_t)$ to
create a weighted sum of the value vectors $\mathbf{v}_t$ to form the new
representation of the token "cat".

$$

\operatorname{Attention}(\mathbf{q}_1, \mathbf{k}\_t, \mathbf{v}\_t) =
\sum_{t=1}^T \alpha(\mathbf{q}\_1, \mathbf{k}\_t) \mathbf{v}\_t


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

\mathbf{q}\_t = \mathbf{z}\_t \mathbf{W}^{Q}, \quad \mathbf{k}\_t =
\mathbf{z}\_t \mathbf{W}^{K}, \quad \mathbf{v}\_t = \mathbf{z}\_t \mathbf{W}^{V}

$$

with each residing in $d_q, d_k, d_v$-dimensional subspaces, respectively.

Given an input sequence of $T$ tokens, the individual vectors for each token can
be stacked into matrices:


$$

\mathbf{Q} = \begin{bmatrix} \mathbf{q}\_1 \\ \mathbf{q}\_2 \\ \vdots \\
\mathbf{q}\_T \end{bmatrix} \in \mathbb{R}^{T \times d_q}, \quad \mathbf{K} =
\begin{bmatrix} \mathbf{k}\_1 \\ \mathbf{k}\_2 \\ \vdots \\ \mathbf{k}\_T
\end{bmatrix} \in \mathbb{R}^{T \times d_k}, \quad \mathbf{V} = \begin{bmatrix}
\mathbf{v}\_1 \\ \mathbf{v}\_2 \\ \vdots \\ \mathbf{v}\_T \end{bmatrix} \in
\mathbb{R}^{T \times d_v}

$$

where each row of these matrices corresponds to the query, key, and value
vectors for each token, respectively.

These matrices are generated through simple matrix multiplication of the token
embedding matrix $\mathbf{Z} \in \mathbb{R}^{T \times D}$ with the weight
matrices
$\mathbf{W}^{\mathbf{Q}}, \mathbf{W}^{\mathbf{K}}$ and $\mathbf{W}^{\mathbf{V}}$:


$$

\mathbf{Q} = \mathbf{Z} \mathbf{W}^{\mathbf{Q}}, \quad \mathbf{K} = \mathbf{Z}
\mathbf{W}^{\mathbf{K}}, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}^{\mathbf{V}}

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

\begin{aligned} \text{Attention}: \mathbb{R}^{T \times d_q} \times \mathbb{R}^{T
\times d_k} \times \mathbb{R}^{T \times d_v} & \rightarrow \mathbb{R}^{T \times
d_v} \\ (\mathbf{Q}, \mathbf{K}, \mathbf{V}) & \mapsto
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \end{aligned}

$$

where given a query matrix $\mathbf{Q} \in \mathbb{R}^{T \times d_q}$, a key
matrix $\mathbf{K} \in \mathbb{R}^{T \times d_k}$, and a value matrix
$\mathbf{V} \in \mathbb{R}^{T \times d_v}$, the attention mechanism computes the
the output matrix as follows:


$$

\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}
\in \mathbb{R}^{T \times d_v}

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

\begin{aligned} \alpha: \mathbb{R}^{d_q} \times \mathbb{R}^{d_k} & \rightarrow
\mathbb{R} \\ (\mathbf{q}, \mathbf{k}\_t) & \mapsto \alpha(\mathbf{q},
\mathbf{k}\_t) \end{aligned}

$$

Specifically, the function is expressed as:


$$

\alpha(\mathbf{q}, \mathbf{k}\_t) = \langle \mathbf{q}, \mathbf{k}\_t \rangle =
\mathbf{q} \cdot \mathbf{k}\_t \in \mathbb{R}

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

\alpha(\mathbf{q}, \mathbf{K}) = \mathbf{q}\mathbf{K}^{\top} \in \mathbb{R}^{1
\times T}

$$

where


$$

\mathbf{K} = \begin{bmatrix} \mathbf{k}\_1 \\ \mathbf{k}\_2 \\ \vdots \\
\mathbf{k}\_T \end{bmatrix} \in \mathbb{R}^{T \times d_k}

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

\alpha(\mathbf{Q}, \mathbf{K}) = \mathbf{Q}\mathbf{K}^{\top} \in \mathbb{R}^{T
\times T}

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

\mathbf{q} \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2), \quad
\mathbf{k}\_t \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2)

$$
```

```{prf:definition} Variance of Dot Product
:label: decoder-concept-variance-dot-product


Given that $\mathbf{q} \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2), \quad \mathbf{k}_t \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, \sigma^2)$,
the variance of the dot product between the query vector $\mathbf{q}$ and the key
vector $\mathbf{k}_t$ is:


$$

\mathbb{V}[\mathbf{q} \cdot \mathbf{k}_t] = \sum*{i=1}^{d_k} \mathbb{V}[q_i
k*{ti}] = d_k \cdot \sigma^4.

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

\mathbb{V}[\mathbf{q} \cdot \mathbf{k}_t] = \sum*{i=1}^{d_k} \mathbb{V}[q_i
k*{ti}] = d_k \cdot \sigma^4.

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

\alpha(\mathbf{Q}, \mathbf{K}) = \frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}
\in \mathbb{R}^{T \times T}

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

\text{softmax}(\alpha(\mathbf{Q}, \mathbf{K})) = \frac{\exp(\alpha(\mathbf{Q},
\mathbf{K}))}{\sum\_{t=1}^T \exp(\alpha(\mathbf{Q}, \mathbf{k}\_t))} \in
\mathbb{R}^{T \times T}

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

\alpha(\mathbf{Q}, \mathbf{K}) =
\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right) \in
\mathbb{R}^{T \times T}

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

\mathbf{C} :=
\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}
\in \mathbb{R}^{T \times d_v}

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
