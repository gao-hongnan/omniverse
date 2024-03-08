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

## Motivation

...

## Abstract

-   All previous pretrained language models require a second stage supervised
    fine-tuning to adapt to a specific task.
-   The authors demonstrated that language models, given enough capacity, and
    enough data, can be adapted to a wide range of tasks without the need for
    task-specific architectures.
-   When conditioned on a document and questions using the
    [CoQA dataset](https://huggingface.co/datasets/stanfordnlp/coqa) of 127,700+
    training samples, essentially a question-answering task, the model is able
    to match or exceed the performance of the 3 baselines.
-   The author emphasized that the capacity of the model is crucial for the
    success of the zero-shot transfer and that the model's performance scales
    log-linearly with the number of parameters. This means that as the model
    capacity increases logarithmically, the performance of the model increases
    linearly.

## Introduction

### Key 1. Competent Generalists over Narrow Experts (1)

-   The mention of works by Krizhevsky et al. (2012), Sutskever et al. (2014),
    and Amodei et al. (2016) points to machine learning systems with enough
    capacity and trained on large datasets, and supervised fine-tuning, is very
    successful in task specific domains such as image recognition, nlp tasks
    etc.
-   However, such systems, termed as "narrow experts", are fragile [Recht et al.
    (2018)], as they are highly dependent on the specific training regime and
    task. A slight pertubation to the input distribution can cause the model to
    perform poorly.
-   The authors then expressed the desire for "competent generalists" that can
    perform well across a wide range of tasks without the need for task-specific
    architectures or supervised fine-tuning.

### Key 2. IID Assumption Fails in Real World (2, 3)

-   It goes without saying that the main machine learning goal is to generalize
    on the unseen data points. However, to simplify machine learning objective
    modeling, we often assume that the training and test data are drawn from the
    same distribution, otherwise termed as the Independent and Identically
    Distributed (i.i.d.) assumption.
    -   As an aside, the i.i.d. assumption has deep roots in statistical
        modeling because it allows simplification of modeling, notably, we can
        express
        [joint probability distributions as the product of marginal distributions](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables).
    -   And techniques like resampling and cross validation with a holdout set
        to evaluate performance are also based on the assumption that the
        training and test data are drawn from the same distribution.
-   However, as the authors highlighted, the i.i.d. assumption fails in the real
    world. The distribution of the test data is often different from the
    training data, and the model's performance degrades significantly when the
    test data distribution is different from the training data distribution.
-   They attribute this to the prevalence of **single** task training on
    **single** domain datasets.

More readings:

-   https://stats.stackexchange.com/questions/213464/on-the-importance-of-the-i-i-d-assumption-in-statistical-learning
-   https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables

### Key 3. Multi-Task Learning is Nacent (4)

-   The author then highlighted that multi-task learning is a promising
    framework where by training a single model on multiple tasks simultaneously,
    the model can then leverage generalizable latent space embeddings and
    representations to perform well on multiple tasks.
-   It was then highlighted that recent work uses, for example, 10 (dataset,
    objective) pairs to train a single model (this is
    [meta-learning](<https://en.wikipedia.org/wiki/Meta-learning_(computer_science)>)).
    -   What this means is that each dataset and objective is distinct.
    -   For example, 1 dataset could be on sentiment data, with the objective of
        predicting the sentiment of a sentence, while another dataset could be
        on named entity recognition, with the objective of predicting the named
        entities in a sentence.
-   The challenge is then still rooted in the compilation, curation and
    annotation of the datasets and objectives for the model to be generalizable.
    So it still reduces to the original problem of single task training on
    single domain datasets as we may need just as much curated data to train a
    multi-task model as we would need to train multiple single task model. And
    it may not scale since we are only looking at 10 (dataset, objective) pairs.

### Key 4. From Word Embeddings to Contextual Embeddings (5,6)

-   Initially, word embeddings (Word2Vec, GloVe) were used to represent words as
    dense fixed-dimensional vectors in a continuous $D$ dimensional space,
    hinging on the fact that words occuring in similar contexts/documents are
    similar semantically. These vectors were then used as input to a model to
    perform a specific task.
-   The next advancement is capturing more contextual information by using
    contextual embeddings, where the word embeddings are conditioned on the
    entire context of the sentence. Recurrent Neural Networks (RNNs) is one
    example and the context embeddings can be "transferred" to other downstream
    tasks.

    From what I understand, unidirectional RNNs can only capture the context
    information from the past, while bidirectional RNNs can capture the context
    information from both the past and the future. However, both methods have
    its limitations in capturing long-range dependencies.

    Furthermore, RNN has a
    [well known problem in gradient vanishing](https://ai.stackexchange.com/questions/20075/why-does-the-transformer-do-better-than-rnn-and-lstm-in-long-range-context-depen)
    which means that the model is biased by the most recent tokens in the
    sequence, and the model's performance degrades as the sequence length
    increases.

-   Full contextual capture via self-attention, allow each token in the sequence
    to simultaneously attend to all other tokens in the sequence, and the
    attention weights are learned during training. This allows the model to
    capture long-range dependencies and is the basis for the Transformer
    architecture. Consequently, self-attention is non-sequential by design and
    operates over a _set_ of tokens, and not a _sequence_ of tokens. This calls
    for the need to introduce positional encodings to the input embeddings to
    capture the sequential nature of the tokens.

    This offers a significant leap over word embedding era where the embeddings
    are static. Now, given two sentences, _I went to the river bank_ versus _i
    went to the bank to withdraw money_, the word "bank" in the first sentence
    is semantically different from the word "bank" in the second sentence. The
    contextual embeddings can capture this difference.

-   The authors then went on to mention that the above methods would still
    require supervised fine-tuning to adapt to a specific task.

    If there are minimal or no supervised data is available, there are other
    lines of work using language model to handle it - commonsense reasoning
    (Schwartz et al., 2017) and sentiment analysis (Radford et al., 2017).

Further readings:

-   https://ai.stackexchange.com/questions/20075/why-does-the-transformer-do-better-than-rnn-and-lstm-in-long-range-context-depen
-   https://stackoverflow.com/questions/55158554/how-transformer-is-bidirectional-machine-learning

### Key 5. Zero Shot Learning and Zero Shot Transfer (7)

-   The authors then went on to mention that continuing from the two lines of
    work above, they continue the trend of using general methods of transfer to
    demonstrate that language models can perform down-stream tasks in a
    zero-shot manner, without any parameter or architecture modification.

-   Zero-shot learning (ZSL) refers to the ability of a model to correctly
    perform tasks or recognize categories it has never encountered during its
    training phase. The essence of ZSL is to generalize from seen to unseen
    classes or tasks by leveraging side information or semantic relationships.
    For example, a model trained to recognize on a set of animals (including
    horses) but not on zebra, should be able to recognize a zebra as something
    close to horse, given the semantic relationship between the two animals.

    TODO: may need further clarification: Furthermore, we can actually pass an
    unseen label and a zebra image. How? Con

-   Zero-shot transfer, often discussed within the context of transfer learning,
    involves applying a model trained on one set of tasks or domains to a
    completely new task or domain without any additional training. Here, the
    focus is on the transferability of learned features or knowledge across
    different but related tasks or domains. Zero-shot transfer extends the
    concept of transfer learning by not requiring any examples from the target
    domain during training, relying instead on the model's ability to generalize
    across different contexts based on its pre-existing knowledge.

Further readings:

-   [Zero shot learning - Wikipedia](https://en.wikipedia.org/wiki/Zero-shot_learning)
-   https://ai.stackexchange.com/questions/21719/what-is-the-difference-between-one-shot-learning-transfer-learning-and-fine-tun
-   https://joeddav.github.io/blog/2020/05/29/ZSL.html
-   https://arxiv.org/abs/1301.3666
-   https://ai.stackexchange.com/questions/23527/zero-shot-learning-available-labels-in-testing-set
-   https://www.theaidream.com/post/zero-shot-learning-can-you-classify-an-object-without-seeing-it-before
-   https://dl.acm.org/doi/10.1145/3293318

## Approach

### Key 1. Modeling Language Models over Joint Probability Distributions (1)

### Key 2. Conditional Distributions (2)

### Key 3. Conditional on Task (3)

### Key 4. Optimizing Unsupervised is the same as Optimizing Supervised (4)

### Key 5. Large Language Models has Capacity to Infer and Generalize (5)

## 2.1. Training Dataset

### Key 1. Rejection of CommonCrawl (1,2)

-   Prior work usually involves training language model on single domain
    datasets (link back to narrow experts).
-   But in order to achieve competent generalists, the authors argue that the
    model should be trained on a diverse range of tasks and domains.
-   CommonCrawl, hosts a huge amount of webscrapes (basically the entire
    internet) and is considered a diverse dataset.
-   However, the authors rejected CommonCrawl because it has severe data quality
    issues.

### Key 2. Construction of WebText Dataset

-   The authors aimed to create a web scrape that emphasizes document quality
    over quantity.
-   To ensure a level of document quality without incurring the prohibitive
    costs of manual filtering, the authors leveraged human curation indirectly
    by scraping all outbound links from Reddit that received at least 3 karma.
    Karma in this context serves as a heuristic for content that is interesting,
    educational, or entertaining as judged by Reddit users.
    -   What this outbound link mean is if a reddit post has an outbound link to
        another website, the authors scraped the content of the website if the
        post received at least 3 karma.
-   The resulting dataset, named WebText, contains of text from about 45 million
    links curated by Reddit users.
-   Further preprocessing such as de-duplication, heuristic based cleaning, as
    well as filtering out Wikipedia links resulted in about 40gb of text (8
    million documents).
-   The snapshot of the dataset is December 2017.
-   The exclusion of Wikipedia is made because the authors believe it is a
    common source of training data for other works, and in order to
    evaluate/test their model on other datasets with little leakage, they
    excluded Wikipedia.

## 2.2. Input Representation

### Key 1. Byte Pair Encoding (BPE) (1,2,3)

-   The authors highlighted traditional ways of tokenization, includes
    lower-casing, punctuation stripping, and splitting on white spaces, as well
    as encode unknown vocabulary with a special token so that the model can
    learn to handle unseen words in eval/test time.
    -   An example is LM are bad at analyzing emojis.
-   However, the authors mentioned this way restricts the natural language input
    space $\mathcal{X}$ and therefore limit the model space $\mathcal{H}$, as
    the latter is a function of the former.
-   To resolve this, the idea of byte-level encoding can be used - since you
    theoretically can encode any character in the world in UTF-8.
-   However, the limitation is current byte-level LMs tend to perform poorly on
    word level tasks.
-   The authors then introduced the BPE algorithm (is "byte-level" because it
    operates on UTF-8 encoded strings) where they striked a balance between
    character-level and word-level tokenization.
-   So in summary, BPE is the tokenizer used to encode the input text into a
    sequence of tokens - which form the input representation to the model.

More readings (honestly not well-versed with BPE):

-   https://github.com/karpathy/minbpe
-   https://huggingface.co/learn/nlp-course/en/chapter6/5

## 2.3. Model

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
$\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{N}\right\}$,

$$
\mathcal{S} = \left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{N}\right\} \underset{\text{i.i.d.}}{\sim} \mathcal{D}
$$

where each sequence $\mathbf{x}_{n}$ is a sequence of tokens that are sampled
$\text{i.i.d.}$ from the distribution $\mathcal{D}$.

Then, we can frame the
[likelihood function](https://gao-hongnan.github.io/gaohn-galaxy/probability_theory/08_estimation_theory/maximum_likelihood_estimation/concept.html)
$\hat{\mathcal{L}}(\cdot)$ as the likelihood of observing the sequences in the
corpus $\mathcal{S}$,

$$
\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right) = \prod_{n=1}^N \mathbb{P}(\mathbf{x}_{n} ; \hat{\boldsymbol{\Theta}})
$$

where $\hat{\boldsymbol{\Theta}}$ is the estimated parameter space that
approximates the true parameter space $\boldsymbol{\Theta}$.

Subsequently, the objective function is now well-defined, to be the maximization
of the likelihood of the sequences in the corpus $\mathcal{S}$,

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}^{*} &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right) \\
                              &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \prod_{n=1}^N \mathbb{P}(\mathbf{x}_{n} ; \hat{\boldsymbol{\Theta}}) \\
                              &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \prod_{n=1}^N \prod_{t=1}^{T_n} \mathbb{P}(x_{n, t} \mid x_{n, 1}, x_{n, 2}, \ldots, x_{n, t-1} ; \hat{\boldsymbol{\Theta}}) \\
\end{aligned}
$$

where $T_n$ is the length of the sequence $\mathbf{x}_{n}$.

Owing to the fact that multiplying many probabilities together can lead to
[numerical instability](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html#numerical-optimization-and-the-negative-log-likelihood)
because the product of many probabilities can be very small, it is common and
necessary to use the log-likelihood as the objective function, because it can be
proven that maximizing the log-likelihood is equivalent to maximizing the
likelihood itself.

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}^{*} &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \log\left(\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)\right) \\
&= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \sum_{n=1}^N \sum_{t=1}^{T_n} \log \mathbb{P}(x_{n, t} \mid x_{n, 1}, x_{n, 2}, \ldots, x_{n, t-1} ; \hat{\boldsymbol{\Theta}}) \\
\end{aligned}
$$

Furthermore, since we are treating the the loss function as a form of
minimization, we can simply negate the log-likelihood to obtain the negative
log-likelihood as the objective function to be minimized,

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}^{*} &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmin}} \left(-\log\left(\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)\right)\right) \\
&= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmin}} \left(-\sum_{n=1}^N \sum_{t=1}^{T_n} \log \mathbb{P}(x_{n, t} \mid x_{n, 1}, x_{n, 2}, \ldots, x_{n, t-1} ; \hat{\boldsymbol{\Theta}})\right) \\
\end{aligned}
$$

It is worth noting that the objective function is a function of the parameter
space $\hat{\boldsymbol{\Theta}}$, and not the data $\mathcal{S}$, so all
analysis such as convergence and consistency will be with respect to the
parameter space $\hat{\boldsymbol{\Theta}}$.

To this end, we denote the GPT model $\mathcal{G}$ to be an _autoregressive_ and
_self-supervised learning_ model that is trained to maximize the likelihood of
observing all data points $\mathbf{x} \in \mathcal{S}$ via the objective
function $\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$
by learning the conditional probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \hat{\boldsymbol{\Theta}})$ over the vocabulary
$\mathcal{V}$ of tokens, conditioned on the contextual preciding tokens
$x_{<t} = \left(x_1, x_2, \ldots, x_{t-1}\right)$. We are clear that although
the goal is to model the joint probability distribution of the token sequences,
we can do so by estimating the joint probability distribution via the
conditional probability distributions.

```{prf:remark} Simplification of the Objective Function
:label: decoder-simplified-objective-function

In what follows, we will mostly focus on the inner summand of the objective
function, namely, we will look at the loss function for a single sequence
$\mathbf{x}$. And in particular the conditional probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \hat{\boldsymbol{\Theta}})$. It should be clear
that the objective function is over all $N$ sequences in the corpus $\mathcal{S}$,
where each sequence $\mathbf{x}_n$ can be decomposed into the product of the
conditional probabilities of each token in the sequence.
```

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
Transformer based language models seldom incorporate more than thousands of
words of context {cite}`zhang2023dive`. In short, the Markov assumption is a
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

Given the definitions of the conditional entropy and perplexity, we can
formalize the loss function $\mathcal{L}$ as:

$$
\begin{aligned}
\mathcal{L}\left(\mathcal{D} ; \boldsymbol{\Theta}\right) &= -\sum_{\mathbf{x} \in \mathcal{D}} \sum_{t=1}^T \log \mathbb{P}\left(x_t \mid C_{\tau}(\mathbf{x}, t) ; \boldsymbol{\Theta}\right) \\
\end{aligned}
$$

and the objective function is to minimize the loss function $\mathcal{L}$,

$$
\begin{aligned}
\boldsymbol{\theta}^{*} &= \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\text{argmin}} \mathcal{L}\left(\mathcal{D} ; \boldsymbol{\Theta}\right) \\
                        &= \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\text{argmin}} -\sum_{\mathbf{x} \in \mathcal{D}} \sum_{t=1}^T \log \mathbb{P}\left(x_t \mid C_{\tau}(\mathbf{x}, t) ; \boldsymbol{\Theta}\right) \\
\end{aligned}
$$

However, we do not know the true distribution $\mathcal{D}$, and so we can only
estimate the loss function $\mathcal{L}$ from the corpus $\mathcal{S}$, and we
can write the process of estimating via the negative log-likelihood as:

$$
\begin{aligned}
\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right) &= -\sum_{\mathbf{x} \in \mathcal{S}} \sum_{t=1}^T \log \mathbb{P}\left(x_t \mid C_{\tau}(\mathbf{x}, t) ; \hat{\boldsymbol{\Theta}}\right) \\
    &= -\sum_{n=1}^N \sum_{t=1}^{T_n} \log \mathbb{P}\left(x_{n, t} \mid C_{\tau}(\mathbf{x}_{n}, t) ; \hat{\boldsymbol{\Theta}}\right) \\
\end{aligned}
$$

and consequently, the objective function is to minimize the estimated loss
function
$\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$,

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}^{*} &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmin}} \hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right) \\
                              &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmin}} -\sum_{n=1}^N \sum_{t=1}^{T_n} \log \mathbb{P}\left(x_{n, t} \mid C_{\tau}(\mathbf{x}_{n}, t) ; \hat{\boldsymbol{\Theta}}\right) \\
\end{aligned}
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

### Conditional on Task

In the GPT-2 paper, _Language Models are Unsupervised Multitask Learners_, the
authors introduced the concept of _conditional on task_ where the GPT model
$\mathcal{G}$ theoretically should not only learn the conditional probability
distribution $\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$ but also learn
the conditional probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta}, \mathcal{T})$ where
$\mathcal{T}$ is the task that the model should implicitly learn
{cite}`radford2019language`. This is a powerful concept because if such a
hypothesis is correct, then the GPT model $\mathcal{G}$ can indeed be a
multi-task learner, and can be used directly on a wide range of NLU tasks
without the need for supervised fine-tuning for downstream domain-specific
tasks.

In practice, the authors mentioned that task conditioning is often implemented
at an architectural level, via task specific encoder and decoder in the paper
[_One Model To Learn Them All_](https://arxiv.org/abs/1706.05137)
{cite}`kaiser2017model`, for instance, or at an algorithmic level, such as the
inner and outer loop optimization framework, as seen in the paper
[_Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks_](https://arxiv.org/abs/1703.03400)
{cite}`finn2017modelagnostic`.

However, the authors further mentioned that without task-specific architectural
changes, one can leverage the sequential nature of the natural language space
where we can construct a tasks, inputs and outputs all as a sequence of symbols
{cite}`radford2019language`. For example, a translation task can be formulated
as a sequence of symbols via
`(translate to french, english sequence, french sequence)`, where the model can
now learn to also condition on the task `(translate to french)` in addition to
the sequence of tokens. The paper _The Natural Language Decathlon: Multitask
Learning as Question Answering_ exemplifies this concept with their model
**Multitask Question Answering Network (MQAN)**, where a single model is trained
to perform many diverse natural language processing tasks simultaneously.

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

#### Objective Function for Fine-Tuning

More concretely, now our dataset $\mathcal{S}$ is a dataset of labeled examples
$\mathcal{S} = \left\{\left(\mathbf{x}_n, y_n\right)\right\}_{n=1}^N$, where it
may be sampled together from a new underlying distribution $\mathcal{D}$,
usually a cartesian product $\mathcal{X} \times \mathcal{Y}$ where $\mathcal{Y}$
is the label space. Each input sequence $\mathbf{x}_n$ is a sequence of tokens,
and each output label $y_n$ is a label from the set of labels $\mathcal{Y}$.

A task specific layer is often used to replace the original head layer, for
instance, if we are training the model on a text classification task with
$\mathcal{C}$ number of classes, then the task specific layer would be a linear
layer with $\mathcal{C}$ number of output units. Of course, the output of this
layer, being the logits, will usually pass into appropriate loss functions such
as the cross-entropy loss with a softmax layer on top of the logits to induce a
_not so well-calibrated_ probability distribution over the classes
$\mathcal{C}$.

If we denote the loss function (or the negative log-likelihood) of the
pre-training phase as
$\hat{\mathcal{L}}_{1}\left(\mathcal{S}_{1} ; \hat{\boldsymbol{\Theta}}_{1}\right)$,
then the objective in this second phase is simply to maximize the likelihood of
the labeled examples $\mathcal{S}$ via the objective function
$\hat{\mathcal{L}}_{2}\left(\mathcal{S}_{2} ; \hat{\boldsymbol{\Theta}}_{2}\right)$
where $\hat{\boldsymbol{\Theta}}_{1}$ is the estimated parameter space for the
pre-training phase, and $\hat{\boldsymbol{\Theta}}_{2}$ is the estimated
parameter space for the fine-tuning phase. Note that the
$\hat{\boldsymbol{\Theta}}_{2}$ is initialized with partial weights from the
pre-trained model $\mathcal{G}$, so it naturally should overlap with the
$\hat{\boldsymbol{\Theta}}_{1}$ up to the number of _frozen_ layers.

We denote the maximization as a minimization of the negative log-likelihood:

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}_{2}^{*} &= \underset{\hat{\boldsymbol{\theta}}_{2} \in \boldsymbol{\Theta}_{2}}{\text{argmin}} \hat{\mathcal{L}}_{2}\left(\mathcal{S}_{2} ; \hat{\boldsymbol{\Theta}}_{2}\right) \\
                                    &= \underset{\hat{\boldsymbol{\theta}}_{2} \in \boldsymbol{\Theta}_{2}}{\text{argmin}} -\sum_{n=1}^N \log \mathbb{P}\left(y_n \mid \mathbf{x}_n ; \hat{\boldsymbol{\Theta}}_{2}\right) \\
\end{aligned}
$$

It is also customary to find the expected loss over the dataset $\mathcal{S}$,

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}_{2}^{*} &= \underset{\hat{\boldsymbol{\theta}}_{2} \in \boldsymbol{\Theta}_{2}}{\text{argmin}} \mathbb{E}_{\mathcal{S}}\left[\hat{\mathcal{L}}_{2}\left(\mathcal{S}_{2} ; \hat{\boldsymbol{\Theta}}_{2}\right)\right] \\
                                    &= \underset{\hat{\boldsymbol{\theta}}_{2} \in \boldsymbol{\Theta}_{2}}{\text{argmin}} -\mathbb{E}_{\mathcal{S}}\left[\sum_{n=1}^N \log \mathbb{P}\left(y_n \mid \mathbf{x}_n ; \hat{\boldsymbol{\Theta}}_{2}\right)\right] \\
                                    &= -\frac{1}{N} \sum_{n=1}^N \log \mathbb{P}\left(y_n \mid \mathbf{x}_n ; \hat{\boldsymbol{\Theta}}_{2}\right) \\
\end{aligned}
$$

where $N$ is the number of samples in the dataset $\mathcal{S}$.

#### Auxiliary Loss Function

In the context of fine-tuning GPT-1 or similar models for specific tasks, the
term "auxiliary (supplementary) loss" refers to additional objectives or loss
functions that are incorporated into the fine-tuning process alongside the
primary loss function. This approach is based on the idea that including
auxiliary tasks or losses can help improve the model's performance on the main
task by leveraging the knowledge gained during pre-training. The author also
mentioned that this method (a) improving generalization of the supervised model,
and (b) accelerating convergence {cite}`radford2018improving`.

During pre-training, models like GPT-1 learn to predict the next token in a
sequence, which is a form of auxiliary task. When fine-tuning these models on
downstream tasks, the authors of the GPT-1 paper found it beneficial to include
the pre-training loss (the auxiliary loss) in the fine-tuning loss function.
This is done by calculating the primary loss for the specific task (e.g.,
classification, named-entity recognition) and then combining it with the
auxiliary loss, often with a weighting factor to balance their contributions.
The weighting factor, denoted as $\alpha$ in the fine-tuning loss function,
allows for adjusting the relative importance of the primary and auxiliary losses
during the fine-tuning process.

To this end, the final loss function for fine-tuning the GPT-1 model on a
specific task is a combination of the primary loss and the auxiliary loss, and
we can write it as:

$$
\begin{aligned}
\hat{\mathcal{L}}_{3}\left(\mathcal{S}_{2} ; \hat{\boldsymbol{\Theta}}_{3}\right) &= \alpha \hat{\mathcal{L}}_{2}\left(\mathcal{S}_{2} ; \hat{\boldsymbol{\Theta}}_{2}\right) + (1 - \alpha) \hat{\mathcal{L}}_{1}\left(\mathcal{S}_{1} ; \hat{\boldsymbol{\Theta}}_{1}\right) \\
\end{aligned}
$$

and we can minimize the new auxiliary loss function in the same way.

## References and Further Readings

-   https://developers.google.com/machine-learning/gan/generative
-   https://probmlcourse.github.io/csc412/lectures/week_2/
-   speech and recognition chapter 3 important
-   https://stackoverflow.com/questions/66451430/changes-in-gpt2-gpt3-model-during-few-shot-learning
-   https://ai.stackexchange.com/questions/12579/why-can-we-approximate-the-joint-probability-distribution-using-the-output-vecto
-   https://datascience.stackexchange.com/questions/65806/why-joint-probability-in-generative-models
-   https://d2l.ai/chapter_recurrent-neural-networks/language-model.html
-   https://stanford-cs324.github.io/winter2022/lectures/introduction/
-   https://www.probabilitycourse.com/chapter5/5_1_1_joint_pmf.php
-   https://math.stackexchange.com/questions/1566215/difference-between-joint-probability-distribution-and-conditional-probability-di
-   https://eugeneyan.com/writing/attention/
-   https://d2l.ai/chapter_convolutional-modern/resnet.html
-   https://songhuiming.github.io/pages/2023/05/28/gpt-1-gpt-2-gpt-3-instructgpt-chatgpt-and-gpt-4-summary/
-   https://keras.io/api/keras_nlp/metrics/perplexity/
-   https://lightning.ai/docs/torchmetrics/stable/text/perplexity.html
-   https://huggingface.co/docs/transformers/perplexity

[^1]:
    This part is not concrete as the formalization is not rigorous in the
    statistical learning framework, but the general idea is there.

[^2]:
    [Working with Sequences - Dive Into Deep Learning](https://d2l.ai/chapter_recurrent-neural-networks/sequence.html)

    $$
    $$
