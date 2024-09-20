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

## Likelihood

### Some Intuition

**_This section is adapted from {cite}`chan_2021`._**

Consider a set of $N$ data points
$\mathcal{S}=\left\{x^{(1)}, x^{(2)}, \ldots, x^{(n)}\right\}$. We want to
describe these data points using a probability distribution. What would be the
most general way of defining such a distribution?

Since we have $N$ data points, and we do not know anything about them, the most
general way to define a distribution is as a high-dimensional probability
density function (PDF) $f_{\mathbf{X}}(\mathbf{x})$. This is a PDF of a random
vector $\mathbf{X}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$. A particular
realization of this random vector is
$\mathbf{x}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$.

$f_{\mathbf{X}}(\mathbf{x})$ is the most general description for the $N$ data
points because $f_{\mathbf{X}}(\mathbf{x})$ is the **joint** PDF of all
variables. It provides the complete statistical description of the vector
$\mathbf{X}$. For example, we can compute the mean vector
$\mathbb{E}[\mathbf{X}]$, the covariance matrix
$\operatorname{Cov}(\mathbf{X})$, the marginal distributions, the conditional
distribution, the conditional expectations, etc. In short, if we know
$f_{\mathbf{X}}(\mathbf{x})$, we know everything about $\mathbf{X}$.

The joint PDF $f_{\mathbf{X}}(\mathbf{x})$ is always **parameterized** by a
certain parameter $\boldsymbol{\theta}$. For example, if we assume that
$\mathbf{X}$ is drawn from a joint Gaussian distribution, then
$f_{\mathbf{X}}(\mathbf{x})$ is parameterized by the mean vector
$\boldsymbol{\mu}$ and the covariance matrix $\boldsymbol{\Sigma}$. So we say
that the parameter $\boldsymbol{\theta}$ is
$\boldsymbol{\theta}=(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. To state the
dependency on the parameter explicitly, we write

$$
f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})=\mathrm{PDF} \text { of the random vector } \mathbf{X} \text { with a parameter } \boldsymbol{\theta} .
$$

When you express the joint PDF as a function of $\mathbf{x}$ and
$\boldsymbol{\theta}$, you have two variables to play with. The first variable
is the **observation** $\mathbf{x}$, which is given by the measured data. We
usually think about the probability density function
$f_{\mathbf{X}}(\mathbf{x})$ in terms of $\mathbf{x}$, because the PDF is
evaluated at $\mathbf{X}=\mathbf{x}$. In estimation, however, $\mathbf{x}$ is
something that you cannot control. When your boss hands a dataset to you,
$\mathbf{x}$ is already fixed. You can consider the probability of getting this
particular $\mathbf{x}$, but you cannot change $\mathbf{x}$.

The second variable stated in $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$
is the **parameter** $\boldsymbol{\theta}$. This parameter is what we want to
find out, and it is the subject of interest in an estimation problem. Our goal
is to find the optimal $\boldsymbol{\theta}$ that can offer the "best
explanation" to data $\mathbf{x}$, in the sense that it can maximize
$f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$.

The likelihood function is the PDF that shifts the emphasis to
$\boldsymbol{\theta}$, let's define it formally.

### Definition

```{prf:definition} Likelihood Function
:label: def:likelihood

Let $\mathbf{X}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$ be a random vector drawn from a joint PDF $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$, and let $\mathbf{x}=\left[x^{(1)}, \ldots, x^{(n)}\right]^{T}$ be the realizations. The likelihood function is a
function of the parameter $\boldsymbol{\theta}$ given the realizations $\mathbf{x}$ :

$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x}) \stackrel{\text { def }}{=} f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})
$$ (eq:likelihood)
```

```{prf:remark} Likelihood is not Conditional PDF
:label: rem:likelihood

A word of caution: $\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})$ is not a conditional PDF because $\boldsymbol{\theta}$ is not a random variable. The correct way to interpret $\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x})$ is to view it as a function of $\boldsymbol{\theta}$.
```

### Independence and Identically Distributed (IID)

While $f_{\mathbf{X}}(\mathbf{x})$ provides us with a complete picture of the
random vector $\mathbf{X}$, using $f_{\mathbf{X}}(\mathbf{x})$ is tedious. We
need to describe how each $x^{(n)}$ is generated and describe how $x^{(n)}$ is
related to $X_{m}$ for all pairs of $n$ and $m$. If the vector $\mathbf{X}$
contains $N$ entries, then there are $N^{2} / 2$ pairs of correlations we need
to compute. When $N$ is large, finding $f_{\mathbf{X}}(\mathbf{x})$ would be
very difficult if not impossible.

What does this mean? Two things.

1. There is no assumption of **independence** between the data points. This
   means that describing the joint PDF $f_{\mathbf{X}}(\mathbf{x})$ is very
   difficult.
2. Each data point _can_ be drawn from a different distribution
   $f_{X^{(n)}}(x^{(n)})$ for each $n$.

Hope is not lost.

Enter the **independence and identically distributed (IID)** assumption. This
assumption states that the data points $\mathbf{x}^{(n)}$ are independent and
identically distributed.

In other words, each data point $\mathbf{x}^{(n)}$ is drawn from **identical**
distribution $f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})$ parameterized by
$\boldsymbol{\theta}$ and each pair of data points $\mathbf{x}^{(n)}$ and
$\mathbf{x}^{(m)}$ are **independent** of each other.

Now, we can write the problem in a much simpler way, where the joint PDF
$f_{\mathbf{X}}(\mathbf{x})$ is replaced by the product of the PDFs of each data
point $f_{x^{(n)}}(x^{(n)})$.

$$
f_{\mathbf{X}}(\mathbf{x})=f_{x^{(1)}, \ldots, x^{(n)}}\left(x^{(1)}, \ldots, x^{(n)}\right)=\prod_{n=1}^{N} f_{x^{(n)}}\left(x^{(n)}\right) .
$$

or in our context, we can add the **parameter** $\boldsymbol{\theta}$ to the
PDFs.

$$
f_{\mathbf{X}}(\mathbf{x} ; \boldsymbol{\theta})=f_{x^{(1)}, \ldots, x^{(n)}}\left(x^{(1)}, \ldots, x^{(n)}\right)=\prod_{n=1}^{N} f_{x^{(n)}}\left(x^{(n)} ; \boldsymbol{\theta}\right) .
$$

Let's formally redefine the likelihood function with the IID assumption. Note
this is an ubiquitous assumption in machine learning and therefore we will stick
to this unless otherwise stated.

```{prf:definition} Likelihood Function with IID Assumption
:label: def:likelihood-iid

Given $\textrm{i.i.d.}$ random variables $x^{(1)}, \ldots, x^{(n)}$ that all have the same PDF $f_{x^{(n)}}\left(x^{(n)}\right)$, the **likelihood function** is defined as:


$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathbf{x}) \stackrel{\text { def }}{=} \prod_{n=1}^{N} f_{x^{(n)}}\left(x^{(n)} ; \boldsymbol{\theta}\right)
$$
```

Notice that in the previous sections, there was an implicit assumption that the
random vector $\mathbf{X}$ is a vector of **univariate\*** random variables.
This is not always the case. In fact, most of the time, the random vector
$\mathbf{X}$ is a "vector" (collection) of **multivariate** random variables in
the machine learning realm.

Let's redefine the likelihood function for the higher dimensional case, and also
take the opportunity to introduce the definition in the context of machine
learning.

### Likelihood in the Context of Machine Learning

```{prf:definition} Likelihood Function with IID Assumption (Higher Dimension)
:label: def:likelihood-iid-higher-dim

Given a dataset $\mathcal{S} = \left\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right\}$ where each $\mathbf{x}^{(n)}$ is a vector of $D$-dimensional drawn $\textrm{i.i.d.}$ from the same underlying distribution
$\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)$, the **likelihood function** is defined as:

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) &\stackrel{\text { def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)} ; \boldsymbol{\theta}\right) \\
&= \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(n)} ; \boldsymbol{\theta}\right)
\end{aligned}
$$ (eq:likelihood-machine-learning-1)

which means what is the probability of observing a sequence of $N$ data points $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}$?

Think of it as flipping $N$ coins, and what is the sequence of observing the permutation of, say, $HHTTTHH$?
```

### Likelihood in the Context of Supervised Learning

In supervised learning, there is often a label $y$ associated with each data
point $\mathbf{x}$.

```{prf:remark} Where's the $y$?
:label: rem:where-y

Some people may ask, isn't the setting in our classification problem a supervised one with labels?
Where are the $y$ in the likelihood function? Good point, the $y$ is not included in our current section
for simplicity. However, the inclusion of $y$ can be merely thought as denoting "an additional"
random variable in the likelihood function above.

For example, let's say $y$ is the target column of a classification problem on breast cancer, denoting
whether the patient has cancer or not. Then, the likelihood function can be written as:

$$
\mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(n)}, y^{(n)} ; \boldsymbol{\theta}\right)
$$

where $\mathcal{S}$ is defined as:

$$
\mathcal{S} = \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \ldots, \left(\mathbf{x}^{(n)}, y^{(n)}\right)\right\}
$$
```

```{prf:definition} Likelihood Function with IID Assumption (Supervised Learning)
:label: def:likelihood-iid-supervised-learning

More concretely, for a typical supervised learning problems, the learner $\mathcal{A}$ receives a labeled sample dataset $\mathcal{S}$
containing $N$ i.i.d. samples $\left(\mathbf{x}^{(n)}, y^{(n)}\right)$ drawn from $\mathbb{P}_{\mathcal{D}}$:

$$
\mathcal{S} = \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \left(\mathbf{x}^{(2)}, y^{(2)}\right), \ldots, \left(\mathbf{x}^{(N)}, y^{(N)}\right)\right\} \subset \mathbb{R}^{D} \quad \overset{\small{\text{i.i.d.}}}{\sim} \quad \mathbb{P}_{\mathcal{D}}\left(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\beta}\right)
$$ (eq-dataset-machine-learning)

where $\mathbb{P}_{\mathcal{D}}$ is assumed to be the underlying (joint) distribution that generates the dataset $\mathcal{S}$.

So in this setting, we generally assume that the tuple $\left(\mathbf{x}^{(n)}, y^{(n)}\right)$ is drawn from the joint distribution $\mathbb{P}_{\mathcal{D}}$
and not just $\mathbf{x}^{(n)}$ alone.

Note carefully that this does not say anything about the independence of $\mathbf{x}^{(n)}$ and $y^{(n)}$.
It only states that each pair $\left(\mathbf{x}^{(n)}, y^{(n)}\right)$ is independent, leading to this:

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) &\stackrel{\text { def }}{=} \mathbb{P}_{\mathcal{D}}\left(\left(\mathbf{x}^{(1)}, y^{(1)}\right), \ldots, \left(\mathbf{x}^{(n)}, y^{(n)}\right); \boldsymbol{\theta}\right) \\
&= \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(n)}, y^{(n)} ; \boldsymbol{\theta}\right)
\end{aligned}
$$ (eq:likelihood-machine-learning-2)

which answers the question of what is the probability of observing a sequence of $N$
data points $\left(\mathbf{x}^{(1)}, y^{(1)}\right), \ldots, \left(\mathbf{x}^{(n)}, y^{(n)}\right)$?
```

### Conditional Likelihood in the Context of Machine Learning

Now in discriminative algorithms such as
[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), we are
interested in the conditional likelihood function.

```{prf:definition} Conditional Likelihood Function (Machine Learning)
:label: def:conditional-likelihood-machine-learning

The conditional likelihood function is defined as:

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) &\stackrel{\text { def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathcal{Y}\mid \mathcal{X} ; \boldsymbol{\theta}\right) \\
&= \mathbb{P}_{\mathcal{D}}\left(y^{(1)}, y^{(2)}, \ldots, y^{(N)} \mid \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(N)} ; \boldsymbol{\theta}\right) \\
&= \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(y^{(n)} \mid \mathbf{x}^{(n)} ; \boldsymbol{\theta}\right)
\end{aligned}
$$ (eq:conditional-likelihood-machine-learning-1)

where we abuse notation and define $\mathcal{X} = \left\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(N)}\right\}$
and $\mathcal{Y} = \left\{y^{(1)}, y^{(2)}, \ldots, y^{(N)}\right\}$.

We are instead interested in the probability of observing a sequence of $N$
conditional data points $\left(y^{(1)} \mid \mathbf{x}^{(1)}, y^{(2)} \mid \mathbf{x}^{(2)}, \ldots, y^{(N)} \mid \mathbf{x}^{(N)}\right)$.
```

Now why does {eq}`eq:conditional-likelihood-machine-learning-1` still hold for
the conditional likelihood function to be able to factorize into a product of
conditional probabilities?

Did we also assume that the conditional data points
$\left(y^{(1)} \mid \mathbf{x}^{(1)}, y^{(2)} \mid \mathbf{x}^{(2)}, \ldots, y^{(N)} \mid \mathbf{x}^{(N)}\right)$
are $\textrm{i.i.d.}$ as well?

We can prove that this equation holds via
[**marginilization**](https://en.wikipedia.org/wiki/Marginal_distribution) and
the [**Fubini's Theorem**](https://en.wikipedia.org/wiki/Fubini%27s_theorem).

```{prf:proof}
The proof is as follows:

$$
\begin{aligned}
\mathbb{P}_{\mathcal{D}}\left(\mathcal{Y} \mid \mathcal{X} ; \boldsymbol{\theta}\right) &= \dfrac{\mathbb{P}_{\mathcal{D}}\left(\mathcal{Y}, \mathcal{X} ; \boldsymbol{\theta}\right)}{\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)} \\
&= \dfrac{\mathbb{P}_{\mathcal{D}}\left(\mathcal{Y}, \mathcal{X} ; \boldsymbol{\theta}\right)}{\int \mathbb{P}_{\mathcal{D}}\left(\mathcal{Y}, \mathcal{X} ; \boldsymbol{\theta}\right) d\mathcal{Y}}  &&\text{ Marginalization} \\
&= \dfrac{\mathbb{P}_{\mathcal{D}}\left(\mathcal{Y}, \mathcal{X} ; \boldsymbol{\theta}\right)}{\int \mathbb{P}_{\mathcal{D}}\left(\mathcal{Y} \mid \mathcal{X} ; \boldsymbol{\theta}\right) \mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right) d\mathcal{Y}}  &&\text{ Marginalization} \\
&= \dfrac{\mathbb{P}_{\mathcal{D}}\left(\mathcal{Y}, \mathcal{X} ; \boldsymbol{\theta}\right)}{\int \int \cdots \int \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(y^{(n)} \mid \mathbf{x}^{(n)} ; \boldsymbol{\theta}\right) \mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right) d\mathcal{Y}}  &&\text{ Fubini's Theorem} \\
&= \dfrac{\mathbb{P}_{\mathcal{D}}\left(\mathcal{Y}, \mathcal{X} ; \boldsymbol{\theta}\right)}{\int \int \cdots \int \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(y^{(n)} \mid \mathbf{x}^{(n)} ; \boldsymbol{\theta}\right) \mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right) d\mathbf{x}^{(n)}}  &&\text{ Fubini's Theorem} \\
&= \prod_{n=1}^{N} \mathbb{P}_{\mathcal{D}}\left(y^{(n)} \mid \mathbf{x}^{(n)} ; \boldsymbol{\theta}\right)
\end{aligned}
$$

See [proof here](https://stats.stackexchange.com/questions/331215/defining-conditional-likelihood).
```

Therefore, the conditional likelihood function is a product of conditional
probabilities, a consequence of the
[**i.i.d. assumption**](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
for the data points in $\mathcal{S}$.

### The Log-Likelihood Function

We will later see in an example that why the log-likelihood function is useful.
For now, let's just say that due to numerical reasons (underflow), we will use
the log-likelihood function instead of the likelihood function. The intuition is
that the likelihood defined in {eq}`eq:likelihood-machine-learning-1` is a
product of individual PDFs. If we have 1 billion samples (i.e.
$N = 1,000,000,000$), then the likelihood function will be a product of 1
billion PDFs. This is a very small number and will cause
[**arithmetic underflow**](https://en.wikipedia.org/wiki/Arithmetic_underflow).
The log-likelihood function is a solution to this problem.

```{prf:definition} Log-Likelihood Function
:label: def:log-likelihood

Given a dataset $\mathcal{S} = \left\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right\}$ where each $\mathbf{x}^{(n)}$ is a vector of $D$-dimensional drawn $\textrm{i.i.d.}$ from the same underlying distribution
$\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)$, the **log-likelihood function** is defined as:

$$
\log \mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \sum_{n=1}^{N} \log \mathbb{P}_{\mathcal{D}}\left(\mathbf{x}^{(n)} ; \boldsymbol{\theta}\right)
$$ (e:log-likelihood-machine-learning)
```

One will soon see that **maximization** of the log-likelihood function is
equivalent to **maximization** of the likelihood function. They give the same
result.

Let's walk through an example:

```{prf:example} Log-Likelihood of Bernoulli Distribution
:label: ex:log-likelihood-bernoulli

The log-likelihood of a sequence of $\textrm{i.i.d.}$ Bernoulli *univariate* random variables
$x^{(1)}, \ldots, x^{(n)}$ with parameter $\theta$.

If $x^{(1)}, \ldots, x^{(n)}$ are i.i.d. Bernoulli random variables, we have

$$
f_{\mathbf{X}}(\mathbf{x} ; \theta)=\prod_{n=1}^{N}\left\{\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right\} .
$$

Taking the log on both sides of the equation yields the log-likelihood function:

$$
\begin{aligned}
\log \mathcal{L}(\theta \mid \mathbf{x}) & =\log \left\{\prod_{n=1}^{N}\left\{\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right\}\right\} \\
& =\sum_{n=1}^{N} \log \left\{\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right\} \\
& =\sum_{n=1}^{N} x^{(n)} \log \theta+\left(1-x^{(n)}\right) \log (1-\theta) \\
& =\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)
\end{aligned}
$$
```

Now there will be more examples of higher-dimensional log-likelihood functions
in the next section. Furthermore, the section Maximum Likelihood Estimation for
Priors in [Naive Bayes](../../../influential/naive_bayes/02_concept.md),details
one example of log-likelihood function for a higher-dimensional multivariate
Bernoulli (Catagorical) distribution.

### Visualizing the Likelihood Function

This section mainly details how the likelihood function, despite being a
function of $\boldsymbol{\theta}$, also depends on the underlying dataset
$\mathcal{S}$. The presence of both should be kept in mind when we talk about
the likelihood function.

For a more detailed analysis, see page 471-472 of Professor Stanley Chan's book
"Introduction to Probability for Data Science" (see references section).

## Maximum Likelihood Estimation

After rigorously defining the likelihood function, we can now talk about the
term **maximum** in maximum likelihood estimation.

The action of maximization is in itself under
[optimization theory](https://en.wikipedia.org/wiki/Mathematical_optimization),
a branch in mathematics. Consequently, the maximum likelihood estimation problem
is an optimization problem that seeks to find the parameter
$\boldsymbol{\theta}$ that maximizes the likelihood function.

```{prf:definition} Maximum Likelihood Estimation
:label: def:maximum-likelihood-estimation

Given a dataset $\mathcal{S}$ consisting of $N$ samples defined as:

$$
\mathcal{S} = \left\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\right\},
$$

where $\mathcal{S}$ is $\textrm{i.i.d.}$ generated from the distribution $\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)$, parametrized by $\boldsymbol{\theta}$, where the parameter $\boldsymbol{\theta}$ can be a vector of parameters defined as:

$$
\boldsymbol{\theta} = \left\{\theta_{1}, \ldots, \theta_{k}\right\}.
$$


We define the likelihood function to be:

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right),
$$


then the maximum-likelihood estimate of the parameter $\boldsymbol{\theta}$ is a parameter that maximizes the likelihood function:

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}} &\stackrel{\text { def }}{=} \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\operatorname{argmax}} \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}\right) \\
&\stackrel{\text{ def }}{=} \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\operatorname{argmax}}\mathbb{P}_{\mathcal{D}}\left(\mathcal{X} ; \boldsymbol{\theta}\right)
\end{aligned}
$$ (eq:maximum-likelihood-estimation)
```

```{prf:remark} Maximum Likelihood Estimation for $\mathcal{S}$ with Label $y$
:label: rmk:maximum-likelihood-estimation

To be more verbose, let's also define the maximum likelihood estimate of the parameter $\boldsymbol{\theta}$ for a dataset $\mathcal{S}$ with label $y$.

First, we redefine $\mathcal{S}$ to be:

$$
\mathcal{S} = \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \ldots, \left(\mathbf{x}^{(n)}, y^{(n)}\right)\right\}
$$

where $\mathcal{S}$ is generated from the distribution $\mathbb{P}_{\mathcal{D}}\left(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\right)$. The likelihood function is then defined as:

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta} \mid \mathcal{S}) \stackrel{\text { def }}{=} \mathbb{P}_{\mathcal{D}}\left(\mathcal{X}, \mathcal{Y}; \boldsymbol{\theta}\right),
$$

then the maximum-likelihood estimate of the parameter $\boldsymbol{\theta}$ is a parameter that maximizes the likelihood function:

$$
\begin{aligned}
\widehat{\boldsymbol{\theta}} &\stackrel{\text { def }}{=} \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\operatorname{argmax}} \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}, y\right) \\
&\stackrel{\text{ def }}{=} \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\operatorname{argmax}} \mathbb{P}_{\mathcal{D}}\left(\mathcal{X}, \mathcal{Y}; \boldsymbol{\theta}\right)
\end{aligned}
$$ (eq:maximum-likelihood-estimation-for-label-y)
```

## Coin Toss Example

```{figure} ../assets/coin_generator.jpg
---
name: coin_generator
height: 500px
---
A coin generator that generates a sequence of coin flips.
```

Let's see how this works in a concrete example. Suppose we have a coin generator
as shown in {numref}`coin_generator`. We know for a fact that:

1. The coin generator generates coins **independently**.
2. The coin generator generates coins with an **identical** probability $\theta$
   (denoted $p$ in the diagram) of being heads.

Consequently, the probability that a coin generated by the coin generator is
heads is $\theta$, and the probability that a coin generated by the coin
generator is tails is $1-\theta$.

Let's say we press the button on the coin generator $N$ times, and we observe
$N$ coin flips. Let's denote the observed coin flips as
$X^{(1)}, \ldots, X^{(N)}$, where the realizations of each $X^{(n)}$ are
$x^{(n)} = 1$ if the $n$th flip is heads and $x^{(n)} = 0$ if the $n$th flip is
tails. This sequence of observations can be further collated into our familiar
dataset $\mathcal{S}$:

$$
\mathcal{S} = \left\{x^{(1)}, \ldots, x^{(N)}\right\}.
$$

Then the probability of observing this dataset $\mathcal{S}$ is equivalent to
asking what is the probability of observing this sequence of random variables
$X^{(1)}, \ldots, X^{(N)}$ and is given by:

```{math}
:label: eq:coin-toss-likelihood-1

\mathbb{P}(X ; \theta) = \prod_{n=1}^N \theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}.
```

Our goal is to find the value of $\theta$. How? Maximum likelihood estimation!
We want to find the value of $\theta$ that maximizes the **joint probability**
of observing the sequence of coin flips. This is equivalent to maximizing the
**likelihood** of observing the sequence of coin flips. The likelihood function
is given by the exact same equation as the joint probability, except that we do
a notational change:

$$
\mathcal{L}(\theta \mid \mathcal{S}) = \prod_{n=1}^N \theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}.
$$

Notice that this equation is none other than the product of $N$ Bernoulli random
variables, each with parameter $\theta$. This is not surprising since coin toss
is usually modelled as a Bernoulli random variable.

If we flip $13$ coins and get the sequence "HHHTHTTHHHHHT", then the probability
of observing this sequence is:

$$
\begin{aligned}
\mathbb{P}(X ; \theta) &= \theta^{x^{(1)}}(1-\theta)^{1-x^{(1)}} \times \theta^{x^{(2)}}(1-\theta)^{1-x^{(2)}} \times \theta^{x^{(3)}}(1-\theta)^{1-x^{(3)}} \times \theta^{x^{(4)}}(1-\theta)^{1-x^{(4)}} \times \theta^{x^{(5)}}(1-\theta)^{1-x^{(5)}} \times \theta^{x^{(6)}}(1-\theta)^{1-x^{(6)}} \times \theta^{x^{(7)}}(1-\theta)^{1-x^{(7)}} \times \theta^{x^{(8)}}(1-\theta)^{1-x^{(8)}} \times \theta^{x^{(9)}}(1-\theta)^{1-x^{(9)}} \times \theta^{x^{(10)}}(1-\theta)^{1-x^{(10)}} \times \theta^{x^{(11)}}(1-\theta)^{1-x^{(11)}} \times \theta^{x^{(12)}}(1-\theta)^{1-x^{(12)}} \times \theta^{x^{(13)}}(1-\theta)^{1-x^{(13)}} \\
&= \theta^{9}(1-\theta)^{4}.
\end{aligned}
$$

One nice thing about this example will be that we know the answer going in.
Indeed, if we said verbally, "I flipped 13 coins, and 9 came up heads, what is
our best guess for the probability that the coin comes us heads?, " everyone
would correctly guess $9/13$. What this maximum likelihood method will give us
is a way to get that number from first principals in a way that will generalize
to vastly more complex situations {cite}`zhang2023dive`.

For our example, the plot of $P(X \mid \theta)$ is as follows:

We know that a coin toss

```{code-cell} ipython3
:tags: [hide-input]

import sys
from pathlib import Path
parent_dir = str(Path().resolve().parents[3])
sys.path.append(parent_dir)

import matplotlib.pyplot as plt

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Optional
import rich
from IPython.display import HTML, display

import math

import matplotlib.pyplot as plt
from IPython.display import display

%config InlineBackend.figure_format = 'svg'
%matplotlib inline

def find_root_dir(current_path: Path | None = None, marker: str = '.git') -> Path | None:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path | None
        The starting path to search from. If None, the current working directory
        `Path.cwd()` is used.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path | None
        The path to the root directory. Returns None if the marker is not found.
    """
    if not current_path:
        current_path = Path.cwd()
    current_path = current_path.resolve()
    for parent in [current_path, *current_path.parents]:
        if (parent / marker).exists():
            return parent
    return None

current_file_path = Path("__file__")
root_dir          = find_root_dir(current_file_path, marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.utils.visualization.style import use_svg_display
    from omnivault.utils.reproducibility.seed import seed_all
else:
    raise ImportError("Root directory not found.")

use_svg_display()

import numpy as np

theta = np.arange(0, 1, 0.001)
likelihood = theta**9 * (1-theta)**4

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(theta, likelihood)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\mathbb{P}(X ; \theta)$')
plt.show();
```

This has its maximum value somewhere near our expected $9/13 \approx 0.7\ldots$.
To see if it is exactly there, we can turn to calculus. Notice that at the
maximum, the gradient of the function is flat. Thus, we could find the maximum
likelihood estimate by finding the values of $\theta$ where the derivative is
zero, and finding the one that gives the highest probability. We compute:

$$
\begin{aligned}
0 & = \frac{d}{d\theta} \mathbb{P}(X ; \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

This has three solutions: $0$, $1$ and $9/13$. The first two are clearly minima,
not maxima as they assign probability $0$ to our sequence. The final value does
_not_ assign zero probability to our sequence, and thus must be the maximum
likelihood estimate $\hat \theta = 9/13$ {cite}`zhang2023dive`.

We can justify this intuition by deriving the maximum likelihood estimate for
$\theta$ if we assume that the coin generator follows a Bernoulli distribution.
The more generic case of the maximum likelihood estimate for $\theta$ is given
by the following.

```{prf:definition} Maximum Likelihood Estimation for Bernoulli Distribution
:label: def:maximum-likelihood-estimation-for-bernoulli-distribution

The Maximum Likelihood estimate for a set of $\textrm{i.i.d.}$ Bernoulli random variables $\left\{x^{(1)}, \ldots, x^{(n)}\right\}$ with $x^{(n)} \sim \operatorname{Bernoulli}(\theta)$ for $n=1, \ldots, N$ is derived as follows.

We know that the log-likelihood function of a set of i.i.d. Bernoulli random variables is given by

$$
\log \mathcal{L}(\theta \mid \mathbf{x})=\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)
$$

Thus, to find the ML estimate, we need to solve the optimization problem

$$
\widehat{\theta}=\underset{\theta \in \Theta}{\operatorname{argmax}}\left\{\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)\right\} .
$$

Taking the derivative with respect to $\theta$ and setting it to zero, we obtain

$$
\frac{d}{d \theta}\left\{\left(\sum_{n=1}^{N} x^{(n)}\right) \cdot \log \theta+\left(N-\sum_{n=1}^{N} x^{(n)}\right) \cdot \log (1-\theta)\right\}=0 .
$$

This gives us

$$
\frac{\left(\sum_{n=1}^{N} x^{(n)}\right)}{\theta}-\frac{N-\sum_{n=1}^{N} x^{(n)}}{1-\theta}=0
$$

Rearranging the terms yields

$$
\widehat{\theta}=\frac{1}{N} \sum_{n=1}^{N} x^{(n)}
$$
```

Indeed, since we have $9$ heads and $4$ tails, we have:

$$
\begin{aligned}
\widehat{\theta} &= \frac{1}{N} \sum_{n=1}^{N} x^{(n)} \\
&= \frac{1}{13} \sum_{n=1}^{13} x^{(n)} \\
&= \frac{1}{13} \cdot 9 \\
&= \frac{9}{13}.
\end{aligned}
$$

since there are $9$ heads and $4$ tails, resulting in a sum of $9$ when you sum
up the $x^{(n)}$'s. Thus, the maximum likelihood estimate for $\theta$ is
$\frac{9}{13}$.

## Visualizing Likelihood and Maximum Likelihood Estimation as $N$ Increases

Read section 8.1.1 aznd 8.1.2 of Introduction to Probability for Data Science
written by Stanley H. Chan {cite}`chan_2021` for more details.

## Numerical Optimization and the Negative Log-Likelihood

**_The following section is adapted from section 22.7 from Dive Into Deep
Learning, {cite}`zhang2023dive`._**

The example on coin toss is nice, but what if we have billions of parameters and
data examples?

### Numerical Underflow

First, notice that if we make the assumption that all the data examples are
independent, we can no longer practically consider the likelihood itself as it
is a product of many probabilities. Indeed, each probability is in $[0,1]$, say
typically of value about $1/2$, and the product of $(1/2)^{1000000000}$ is far
below machine precision. We cannot work with that directly.

Let's check the smallest representable positive number greater than zero for the
`float32` data type:

```{code-cell} ipython3
:tags: [hide-input]

print(np.finfo(np.float32).eps)
```

Let's cook up a simple example to illustrate this. We will generate a random
sequence of $1000000000$ coin tosses, and compute the likelihood of the sequence
given that the coin is fair. We will then compute the log-likelihood of the
sequence given that the coin is fair.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

seed_all(42)

N = 1000000000
X = np.random.randint(0, 2, size=N)
theta = 0.5
likelihood = (theta ** np.sum(X)) * ((1 - theta) ** (N - np.sum(X)))
print(f"Likelihood: {likelihood}")
```

So the likelihood is $0$ because the $1000000000$ when multiplied is less than
the smallest representable positive number greater than zero for the `float32`
data type.

However, recall that the logarithm turns products to sums, in which case

$$
\log\left(\left(1/2\right)^{1000000000}\right) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

This number fits perfectly within even a single precision $32$-bit float. Thus,
we should consider the _log-likelihood_, which is

$$
\log(\mathbb{P}(X ; \boldsymbol{\theta})).
$$

Since the function $x \mapsto \log(x)$ is increasing, maximizing the likelihood
is the same thing as maximizing the log-likelihood.

We often work with loss functions, where we wish to minimize the loss. We may
turn maximum likelihood into the minimization of a loss by taking
$-\log(\mathbb{P}(X ; \boldsymbol{\theta}))$, which is the _negative
log-likelihood_.

To illustrate this, consider the coin flipping problem from before, and pretend
that we do not know the closed form solution. We may compute that

$$
-\log(\mathbb{P}(X ; \boldsymbol{\theta})) = -\log\left(\theta^{x^{(n)}}(1-\theta)^{1-x^{(n)}}\right) = -\left(n_H\log(\theta) + n_T\log(1-\theta)\right)
$$

where $n_H$ is the number of heads and $n_T$ is the number of tails. This form
is just like in
{prf:ref}`def:maximum-likelihood-estimation-for-bernoulli-distribution`.

This can be written into code, and freely optimized even for billions of coin
flips.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np

# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our parameters
theta = np.array(0.5)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    # Compute the gradient of the loss function with respect to theta
    grad_loss = -(n_H / theta - n_T / (1 - theta))

    # Update theta using the gradient
    theta -= lr * grad_loss

# Check output
print(f"Estimated theta: {theta}")
print(f"Empirical theta: {n_H / (n_H + n_T)}")
```

### Mathematical Convenience

Numerical convenience is not the only reason why people like to use negative
log-likelihoods. There are several other reasons why it is preferable.

The second reason we consider the log-likelihood is the simplified application
of calculus rules. As discussed above, due to independence assumptions, most
probabilities we encounter in machine learning are products of individual
probabilities.

$$
\mathbb{P}(X ; \boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

This means that if we directly apply the product rule to compute a derivative we
get

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} \mathbb{P}(X ; \boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}\mathbb{P}(x_1\mid\boldsymbol{\theta})\right)\cdot \mathbb{P}(x_2\mid\boldsymbol{\theta})\cdots \mathbb{P}(x_n\mid\boldsymbol{\theta}) \\
& \quad + \mathbb{P}(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}\mathbb{P}(x_2\mid\boldsymbol{\theta})\right)\cdots \mathbb{P}(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + \mathbb{P}(x_1\mid\boldsymbol{\theta})\cdot \mathbb{P}(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}\mathbb{P}(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

This requires $n(n-1)$ multiplications, along with $(n-1)$ additions, so it is
proportional to quadratic time in the inputs! Sufficient cleverness in grouping
terms will reduce this to linear time, but it requires some thought. For the
negative log-likelihood we have instead

$$
-\log\left(\mathbb{P}(X ; \boldsymbol{\theta})\right) = -\log(\mathbb{P}(x_1\mid\boldsymbol{\theta})) - \log(\mathbb{P}(x_2\mid\boldsymbol{\theta})) \cdots - \log(\mathbb{P}(x_n\mid\boldsymbol{\theta})),
$$

which then gives

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(\mathbb{P}(X ; \boldsymbol{\theta})\right) = \frac{1}{\mathbb{P}(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}\mathbb{P}(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{\mathbb{P}(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}\mathbb{P}(x_n\mid\boldsymbol{\theta})\right).
$$

This requires only $n$ divides and $n-1$ sums, and thus is linear time in the
inputs.

### Information Theory

The third and final reason to consider the negative log-likelihood is the
relationship to information theory. We will discuss this separately in the
section on information theory. This is a rigorous mathematical theory which
gives a way to measure the degree of information or randomness in a random
variable. The key object of study in that field is the entropy which is

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

which measures the randomness of a source. Notice that this is nothing more than
the average $-\log$ probability, and thus if we take our negative log-likelihood
and divide by the number of data examples, we get a relative of entropy known as
cross-entropy. This theoretical interpretation alone would be sufficiently
compelling to motivate reporting the average negative log-likelihood over the
dataset as a way of measuring model performance.

## Maximum Likelihood for Continuous Variables

**_The following section is adapted from section 22.7 from Dive Into Deep
Learning, {cite}`zhang2023dive`._**

Everything that we have done so far assumes we are working with discrete random
variables, but what if we want to work with continuous ones?

The short summary is that nothing at all changes, except we replace all the
instances of the probability with the probability density. Recalling that we
write densities with lower case $p$, this means that for example we now say

$$
-\log\left(p(X ; \boldsymbol{\theta})\right) = -\log(p(x^{(1)} ; \boldsymbol{\theta})) - \log(p(x^{(2)} ; \boldsymbol{\theta})) \cdots - \log(p(x_n ; \boldsymbol{\theta})) = -\sum_i \log(p(x^{(n)} ; \theta)).
$$

The question becomes, "Why is this OK?" After all, the reason we introduced
densities was because probabilities of getting specific outcomes themselves was
zero, and thus is not the probability of generating our data for any set of
parameters zero?

Indeed, this is the case, and understanding why we can shift to densities is an
exercise in tracing what happens to the epsilons.

Let's first re-define our goal. Suppose that for continuous random variables we
no longer want to compute the probability of getting exactly the right value,
but instead matching to within some range $\epsilon$. For simplicity, we assume
our data is repeated observations $x^{(1)}, \ldots, x^{(N)}$ of identically
distributed random variables $X^{(1)}, \ldots, X^{(N)}$. As we have seen
previously, this can be written as

$$
\begin{aligned}
&P(X^{(1)} \in [x^{(1)}, x^{(1)}+\epsilon], X^{(2)} \in [x^{(2)}, x^{(2)}+\epsilon], \ldots, X^{(N)} \in [x^{(N)}, x^{(N)}+\epsilon] ;\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x^{(1)} ; \boldsymbol{\theta})\cdot p(x^{(2)} ; \boldsymbol{\theta}) \cdots p(x_n ;\boldsymbol{\theta}).
\end{aligned}
$$

Thus, if we take negative logarithms of this we obtain

$$
\begin{aligned}
&-\log(P(X^{(1)} \in [x^{(1)}, x^{(1)}+\epsilon], X^{(2)} \in [x^{(2)}, x^{(2)}+\epsilon], \ldots, X^{(N)} \in [x^{(N)}, x^{(N)}+\epsilon] ; \boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x^{(n)} ; \boldsymbol{\theta})).
\end{aligned}
$$

If we examine this expression, the only place that the $\epsilon$ occurs is in
the additive constant $-N\log(\epsilon)$. This does not depend on the parameters
$\boldsymbol{\theta}$ at all, so the optimal choice of $\boldsymbol{\theta}$
does not depend on our choice of $\epsilon$! If we demand four digits or
four-hundred, the best choice of $\boldsymbol{\theta}$ remains the same, thus we
may freely drop the epsilon to see that what we want to optimize is

$$
- \sum_{i} \log(p(x^{(n)} ; \boldsymbol{\theta})).
$$

Thus, we see that the maximum likelihood point of view can operate with
continuous random variables as easily as with discrete ones by replacing the
probabilities with probability densities.

(estimation-theory-mle-common-distributions)=

## Maximum Likelihood Estimation for Common Distributions

### Maximum Likelihood for Univariate Gaussian

Suppose that we are given a set of i.i.d. univariate Gaussian random variables
$X^{(1)}, \ldots, X^{(N)}$, where both the mean $\mu$ and the variance
$\sigma^2$ are unknown.

Let $\boldsymbol{\theta}=\left[\mu, \sigma^2\right]^T$ be the parameter. Find
the maximum likelihood estimate of $\boldsymbol{\theta}$.

**First**, we define the likelihood and log-likelihood functions. Since the
random variables are i.i.d., the likelihood function is given by:

```{math}
:label: eq:gaussian-likelihood-1

\begin{aligned}
\overbrace{\mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S} = \left\{X^{(1)}, \ldots, X^{(N)}\right\}\right)}^{\mathbb{P}\left(\mathcal{S} = \left\{X^{(1)}, \ldots, X^{(N)}\right\} ; \boldsymbol{\theta}\right)} &= \prod_{n=1}^N \overbrace{f_{\boldsymbol{\theta}}\left(x^{(n)}\right)}^{\mathcal{N}\left(x^{(n)} ; \mu, \sigma^2\right) \\} \\
&= \prod_{n=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\left(x^{(n)} - \mu\right)^2}{2\sigma^2}\right).
\end{aligned}
```

The log-likelihood function is given by:

```{math}
:label: eq:gaussian-likelihood-2

\begin{aligned}
\overbrace{\log\mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S} = \left\{X^{(1)}, \ldots, X^{(N)}\right\}\right)}^{\log\mathbb{P}\left(\mathcal{S} = \left\{X^{(1)}, \ldots, X^{(N)}\right\} ; \boldsymbol{\theta}\right)} &= \log\left(\prod_{n=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\left(x^{(n)} - \mu\right)^2}{2\sigma^2}\right)\right) \\
&= \sum_{n=1}^N \log\left(\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\left(x^{(n)} - \mu\right)^2}{2\sigma^2}\right)\right) &&(*)\\
&= \sum_{n=1}^N \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) + \sum_{n=1}^N \log\left(\exp\left(-\frac{\left(x^{(n)} - \mu\right)^2}{2\sigma^2}\right)\right) &&(**)\\
&= \sum_{n=1}^N \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) + \sum_{n=1}^N -\frac{\left(x^{(n)} - \mu\right)^2}{2\sigma^2} \cdot \underbrace{\log(e)}_{1} &&(***)\\
&= \left(\sum_{n=1}^N -\frac{1}{2} \log(2\pi\sigma^2)\right) - \sum_{n=1}^N \frac{\left(x^{(n)} - \mu\right)^2}{2\sigma^2} &&(****)\\
&= -\frac{N}{2} \log(2\pi\sigma^2) - \sum_{n=1}^N \frac{\left(x^{(n)} - \mu\right)^2}{2\sigma^2}.
\end{aligned}
```

where

-   $(*)$ is the log-product rule, meaning that the logarithm of the product of
    $N$ terms is the sum of the logarithms of the $N$ terms.
-   $(**)$ is again the log-product rule.
-   $(***)$ is the log-exponential rule, meaning that the logarithm of the
    exponential of a term is the term multiplied by the logarithm of $e$.
-   $(****)$ is just writing
    $\log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) = \log\left(\sqrt{2\pi\sigma^2}^{-\frac{1}{2}}\right)$
    and therefore the log of it is just $-\frac{1}{2} \log(2\pi\sigma^2)$.

Then, we take the derivative of the log-likelihood function with respect to
$\mu$ and $\sigma^2$ and set them to zero to find the maximum likelihood
estimates.

$$
\begin{aligned}
\frac{\partial}{\partial \mu} \log \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}=\left\{X^{(1)}, \ldots, X^{(N)}\right\}\right) & =\frac{\partial}{\partial \mu}\left(-\frac{N}{2} \log \left(2 \pi \sigma^2\right)-\sum_{n=1}^N \frac{\left(x^{(n)}-\mu\right)^2}{2 \sigma^2}\right) \\
& =0-\frac{\partial}{\partial \mu}\left(\sum_{n=1}^N \frac{\left(x^{(n)}-\mu\right)^2}{2 \sigma^2}\right) \\
& =-\frac{1}{2 \sigma^2} \sum_{n=1}^N \frac{\partial}{\partial \mu}\left(\left(x^{(n)}-\mu\right)^2\right) \\
& =-\frac{1}{2 \sigma^2} \sum_{n=1}^N 2\left(x^{(n)}-\mu\right)(-1) \\
& =\frac{1}{\sigma^2}\sum_{n=1}^N \left(x^{(n)}- \mu\right) \\
\end{aligned}
$$

So the partial derivative of the log-likelihood function with respect to $\mu$
is:

$$
\frac{\partial}{\partial \mu} \log \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}=\left\{X^{(1)}, \ldots, X^{(N)}\right\}\right)=\frac{1}{\sigma^2}\sum_{n=1}^N \left(x^{(n)}- \mu\right)
$$

and setting it to zero gives:

$$
\begin{aligned}
\frac{1}{\sigma^2}\sum_{n=1}^N \left(x^{(n)}- \mu\right) = 0 &\iff \sum_{n=1}^N x^{(n)}- \mu = 0 \\
&\iff \sum_{n=1}^N x^{(n)} = \mu N \\
&\iff \mu = \frac{1}{N}\sum_{n=1}^N x^{(n)}.
\end{aligned}
$$

resulting in the maximum likelihood estimate for $\mu$ to be:

$$
\hat{\mu} = \frac{1}{N}\sum_{n=1}^N x^{(n)}.
$$

Similarly, we have for $\sigma^2$:

$$
\begin{aligned}
\frac{\partial}{\partial \sigma^2} \log \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}=\left\{X^{(1)}, \ldots, X^{(N)}\right\}\right) & =\frac{\partial}{\partial \sigma^2}\left(-\frac{N}{2} \log \left(2 \pi \sigma^2\right)-\sum_{n=1}^N \frac{\left(x^{(n)}-\mu\right)^2}{2 \sigma^2}\right) \\
& =-\frac{N}{2} \frac{\partial}{\partial \sigma^2}\left(\log \left(2 \pi \sigma^2\right)\right)-\sum_{n=1}^N \frac{\partial}{\partial \sigma^2}\left(\frac{\left(x^{(n)}-\mu\right)^2}{2 \sigma^2}\right) \\
& =-\frac{N}{2} \frac{1}{\sigma^2}-\sum_{n=1}^N \frac{-1}{2\left(\sigma^2\right)^2}\left(x^{(n)}-\mu\right)^2 \\
& =-\frac{N}{2 \sigma^2}+\frac{1}{2\left(\sigma^2\right)^2} \sum_{n=1}^N\left(x^{(n)}-\mu\right)^2
\end{aligned}
$$

So, the partial derivative with respect to $\sigma^2$ is:

$$
\frac{\partial}{\partial \sigma^2} \log \mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S}=\left\{X^{(1)}, \ldots, X^{(N)}\right\}\right)=-\frac{N}{2 \sigma^2}+\frac{1}{2\left(\sigma^2\right)^2} \sum_{n=1}^N\left(x^{(n)}-\mu\right)^2
$$

and setting it to zero gives:

$$
\begin{aligned}
-\frac{N}{2 \sigma^2}+\frac{1}{2\left(\sigma^2\right)^2} \sum_{n=1}^N\left(x^{(n)}-\mu\right)^2 = 0 &\iff \frac{N}{2 \sigma^2} = \frac{1}{2\left(\sigma^2\right)^2} \sum_{n=1}^N\left(x^{(n)}-\mu\right)^2 \\
&\iff \sigma^2 = \frac{1}{N}\sum_{n=1}^N\left(x^{(n)}-\mu\right)^2.
\end{aligned}
$$

resulting in the maximum likelihood estimate for $\sigma^2$ to be:

$$
\hat{\sigma}^2 = \frac{1}{N}\sum_{n=1}^N\left(x^{(n)}-\hat{\mu}\right)^2.
$$

Note in particular that we placed $\mu$ by $\hat{\mu}$.

Overall, the maximum likelihood estimate for the parameters of a Gaussian
distribution is:

$$
\hat{\boldsymbol{\theta}} = \begin{bmatrix} \hat{\mu} \\ \hat{\sigma}^2 \end{bmatrix} = \begin{bmatrix} \frac{1}{N}\sum_{n=1}^N x^{(n)} \\ \frac{1}{N}\sum_{n=1}^N\left(x^{(n)}-\hat{\mu}\right)^2 \end{bmatrix}
$$

### Maximum Likelihood Estimation for Multivariate Gaussian

See
[my proof on multiple linear regression](../../../influential/linear_regression/02_concept.md),
they have similar vein of logic. See
[here](https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian)
also.

Suppose that we are given a set of $\textrm{i.i.d.}$ $D$-dimensional Gaussian
random vectors $\mathbf{X}^{(1)}, \ldots, \mathbf{X}^{(N)}$ such that:

$$
\mathbf{X}^{(n)} = \begin{bmatrix} X^{(n)}_1 \\ \vdots \\ X^{(n)}_D \end{bmatrix} \sim \mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma}\right)
$$

where the mean vector $\boldsymbol{\mu}$ and the covariance matrix
$\boldsymbol{\Sigma}$ are given by

$$
\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \vdots \\ \mu_D \end{bmatrix}, \quad \boldsymbol{\Sigma} = \begin{bmatrix} \sigma_1^2 & \cdots & \sigma_{1D} \\ \vdots & \ddots & \vdots \\ \sigma_{D1} & \cdots & \sigma_D^2 \end{bmatrix}
$$

Now the $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ are unknown, and we want to
find the maximum likelihood estimate for them.

As usual, we find the likelihood function for the parameters $\boldsymbol{\mu}$
and $\boldsymbol{\Sigma}$, and then find the maximum likelihood estimate for
them.

$$
\begin{aligned}
\mathcal{L}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma} \mid \mathcal{S}=\left\{\mathbf{X}^{(1)}, \ldots, \mathbf{X}^{(N)}\right\}\right) & =\prod_{n=1}^N f_{\mathbf{X}^{(n)}}\left(\mathbf{x}^{(n)} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}\right) \\
& =\prod_{n=1}^N \frac{1}{\sqrt{(2 \pi)^{D}|\boldsymbol{\Sigma}|}} \exp \left\{-\frac{1}{2}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)\right\} \\
& =\left(\frac{1}{\sqrt{(2 \pi)^{D}|\boldsymbol{\Sigma}|}}\right)^N \exp \left\{-\frac{1}{2}\sum_{n=1}^N\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^{T} \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)\right\} \\
\end{aligned}
$$

and consequently the log-likelihood function is:

$$
\begin{aligned}
\log \mathcal{L}\left(\boldsymbol{\mu}, \mathbf{\Sigma} \mid \mathcal{S}=\left\{\mathbf{x}^{(1)}, \ldots, \mathbf{X}^{(N)}\right\}\right) & =\log \left(\prod_{n=1}^N \frac{1}{\sqrt{(2 \pi)^D|\boldsymbol{\Sigma}|}} \exp \left\{-\frac{1}{2}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^T \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)\right\}\right) \\
& =\sum_{n=1}^N \log \left(\frac{1}{\sqrt{(2 \pi)^D|\mathbf{\Sigma}|}}\right)+\sum_{n=1}^N \log \left(\exp \left\{-\frac{1}{2}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^T \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)\right\}\right) \\
& =\sum_{n=1}^N\left(-\frac{D}{2} \log (2 \pi)-\frac{1}{2} \log (|\mathbf{\Sigma}|)-\frac{1}{2}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^T \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)\right) \\
& =-\frac{N D}{2} \log (2 \pi)-\frac{N}{2} \log (|\boldsymbol{\Sigma}|)-\frac{1}{2} \sum_{n=1}^N\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^T \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)
\end{aligned}
$$

Finding the ML estimate requires taking the derivative with respect to both
$\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ :

$$
\begin{aligned}
& \frac{d}{d \boldsymbol{\mu}}\left\{-\frac{N D}{2} \log (2 \pi)-\frac{N}{2} \log (|\boldsymbol{\Sigma}|)-\frac{1}{2} \sum_{n=1}^N\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^T \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)\right\}=0 \\
& \frac{d}{d \boldsymbol{\Sigma}}\left\{-\frac{N D}{2} \log (2 \pi)-\frac{N}{2} \log (|\boldsymbol{\Sigma}|)-\frac{1}{2} \sum_{n=1}^N\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)^T \boldsymbol{\Sigma}^{-1}\left(\mathbf{x}^{(n)}-\boldsymbol{\mu}\right)\right\}=0 .
\end{aligned}
$$

After some tedious algebraic steps (see Duda et al., Pattern Classification,
Problem 3.14), we have that

$$
\begin{aligned}
& \widehat{\boldsymbol{\mu}}=\frac{1}{N} \sum_{n=1}^{N} \mathbf{x}^{(n)}, \\
& \widehat{\boldsymbol{\Sigma}}=\frac{1}{N} \sum_{n=1}^{N}\left(\mathbf{x}^{(n)}-\widehat{\boldsymbol{\mu}}\right)\left(\mathbf{x}^{(n)}-\widehat{\boldsymbol{\mu}}\right)^{T} .
\end{aligned}
$$

## References and Further Readings

-   Chan, Stanley H. "Chapter 8.1. Maximum-Likelihood Estimation." In
    Introduction to Probability for Data Science. Ann Arbor, Michigan: Michigan
    Publishing Services, 2021.
