# Concept

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
```

## Notations

```{prf:definition} Underlying Distributions
:label: underlying-distributions

- $\mathcal{X}$: Input space consists of all possible inputs $\mathbf{x} \in \mathcal{X}$.

- $\mathcal{Y}$: Label space = $\{1, 2, \ldots, K\}$ where $K$ is the number of classes.

- The mapping between $\mathcal{X}$ and $\mathcal{Y}$ is given by $c: \mathcal{X} \rightarrow \mathcal{Y}$ where $c$ is called *concept* according to the PAC learning theory.

- $\mathcal{D}$: The fixed but unknown distribution of the data. Usually, this refers
to the joint distribution of the input and the label,

  $$
  \begin{aligned}
  \mathcal{D} &= \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) \\
  &= \mathbb{P}_{\{\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\}}(\mathbf{x}, y)
  \end{aligned}
  $$

  where $\mathbf{x} \in \mathcal{X}$ and $y \in \mathcal{Y}$, and $\boldsymbol{\theta}$ is the
  parameter vector of the distribution $\mathcal{D}$.
```


```{prf:definition} Dataset
:label: dataset-definition

Now, consider a dataset $\mathcal{D}_{\{\mathbf{x}, y\}}$ consisting of $N$ samples (observations) and $D$ predictors (features) drawn **jointly** and **indepedently and identically distributed** (i.i.d.) from $\mathcal{D}$. Note we will refer to the dataset $\mathcal{D}_{\{\mathbf{x}, y\}}$ with the same notation as the underlying distribution $\mathcal{D}$ from now on.

- The training dataset $\mathcal{D}$ can also be represented compactly as a set:

    $$
    \begin{align*}
    \mathcal{D} \overset{\mathbf{def}}{=} \mathcal{D}_{\{\mathbf{x}, y\}} &= \left\{\mathbf{x}^{(n)}, y^{(n)}\right\}_{n=1}^N \\
    &= \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \left(\mathbf{x}^{(2)}, y^{(2)}\right), \cdots, \left(\mathbf{x}^{(N)}, y^{(N)}\right)\right\} \\
    &= \left\{\mathbf{X}, \mathbf{y}\right\}
    \end{align*}
    $$

  where we often subscript $\mathbf{x}$ and $y$ with $n$ to denote the $n$-th sample from the dataset, i.e.
  $\mathbf{x}^{(n)}$ and $y^{(n)}$. Most of the times, $\mathbf{x}^{(n)}$ is bolded since
  it represents a vector of $D$ number of features, while $y^{(n)}$ is not bolded since it is a scalar, though
  it is not uncommon for $y^{(n)}$ to be bolded as well if you represent it with K-dim one-hot vector.

- For the n-th sample $\mathbf{x}^{(n)}$, we often denote the $d$-th feature as $x_d^{(n)}$ and the representation of $\mathbf{x}^{(n)}$ as a vector as:

  $$
  \mathbf{x}^{(n)} \in \mathbb{R}^{D} = \begin{bmatrix} x_1^{(n)} & x_2^{(n)} & \cdots & x_D^{(n)} \end{bmatrix}_{D \times 1}
  $$

  is a sample of size $D$, drawn (jointly with $y$) $\textbf{i.i.d.}$ from $\mathcal{D}$.

- We often add an extra feature $x_0^{(n)} = 1$ to $\mathbf{x}^{(n)}$ to represent the bias term.
i.e.

  $$
  \mathbf{x}^{(n)} \in \mathbb{R}^{D+1} = \begin{bmatrix} x_0^{(n)} & x_1^{(n)} & x_2^{(n)} & \cdots & x_D^{(n)} \end{bmatrix}_{(D+1) \times 1}
  $$

- For the n-th sample's label $y^{(n)} \overset{\mathbf{def}}{=} c(\mathbf{x}^{(n)})$, if we were to represent it as K-dim one-hot vector, we would have:

  $$
  y^{(n)} \in \mathbb{R}^{K} = \begin{bmatrix} 0 & 0 & \cdots & 1 & \cdots & 0 \end{bmatrix}_{K \times 1}
  $$

  where the $1$ is at the $k$-th position, and $k$ is the class label of the n-th sample.

- Everything defined above is for **one single sample/data point**, to represent it as a matrix, we can define
a design matrix $\mathbf{X}$ and a label vector $\mathbf{y}$ as follows,

  $$
  \begin{aligned}
  \mathbf{X} \in \mathbb{R}^{N \times D} &= \begin{bmatrix} \mathbf{x}^{(1)} \\ \mathbf{x}^{(2)} \\ \vdots \\ \mathbf{x}^{(N)} \end{bmatrix} = \begin{bmatrix} x_1^{(1)} & x_2^{(1)} & \cdots & x_D^{(1)} \\ x_1^{(2)} & x_2^{(2)} & \cdots & x_D^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ x_1^{(N)} & x_2^{(N)} & \cdots & x_D^{(N)} \end{bmatrix}_{N \times D} \\
  \end{aligned}
  $$

  as the matrix of all samples. Note that each row is a sample and each column is a feature. We can append a column of 1's to the first column of $\mathbf{X}$ to represent the bias term.

  **In this section, we also talk about random vectors $\mathbf{X}$ so we will replace the design matrix $\mathbf{X}$ with $\mathbf{A}$ to avoid confusion.**

  Subsequently, for the label vector $\mathbf{y}$, we can define it as follows,

  $$
  \begin{aligned}
  \mathbf{y} \in \mathbb{R}^{N} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(N)} \end{bmatrix}
  \end{aligned}
  $$
```

```{prf:example} Joint Distribution Example
:label: joint-distribution-example

For example, if the number of features, $D = 2$, then let's say

$$
\mathbf{X}^{(n)} = \begin{bmatrix} X^{(n)}_1 & X^{(n)}_2 \end{bmatrix} \in \mathbb{R}^2
$$

consists of two Gaussian random variables,
with $\mu_1$ and $\mu_2$ being the mean of the two distributions,
and $\sigma_1$ and $\sigma_2$ being the variance of the two distributions;
furthermore, $Y^{(n)}$ is a Bernoulli random variable with parameter $\boldsymbol{\pi}$, then we have

$$
\begin{aligned}
\boldsymbol{\theta} &= \begin{bmatrix} \mu_1 & \sigma_1 & \mu_2 & \sigma_2 & \boldsymbol{\pi}\end{bmatrix} \\
&= \begin{bmatrix} \boldsymbol{\mu} & \boldsymbol{\sigma} & \boldsymbol{\pi} \end{bmatrix}
\end{aligned}
$$

where $\boldsymbol{\mu} = \begin{bmatrix} \mu_1 & \mu_2 \end{bmatrix}$ and $\boldsymbol{\sigma} = \begin{bmatrix} \sigma_1 & \sigma_2 \end{bmatrix}$.
```

```{prf:remark} Some remarks
:label: some-remarks

- From now on, we will refer the realization of $Y$ as $k$ instead.
- For some sections, when I mention $\mathbf{X}$, it means the random vector which resides in the
$D$-dimensional space, not the design matrix. This also means that this random vector refers
to a single sample, not the entire dataset.
```

```{prf:definition} Joint and Conditional Probability
:label: joint-and-conditional-probability

We are often interested in finding the probability of a label given a sample,

$$
\begin{aligned}
\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}) &= \mathbb{P}(Y = k \mid \mathbf{X} = \left(x_1, x_2, \ldots, x_D\right))
\end{aligned}
$$

where

$$
\mathbf{X} \in \mathbb{R}^{D} = \begin{bmatrix} X_1 & X_2 & \cdots & X_D \end{bmatrix}
$$

is a random vector and its realizations,

$$
\mathbf{x} = \begin{bmatrix} x_1 & x_2 & \cdots & x_D \end{bmatrix}
$$

and therefore, $\mathbf{X}$ can be characterized by an $D$-dimensional PDF

$$
f_{\mathbf{X}}(\mathbf{x}) = f_{X_1, X_2, \ldots, X_D}(x_1, x_2, \ldots, x_D ; \boldsymbol{\theta})
$$

and

$$
Y \in \mathbb{Z} \quad \text{and} \quad k \in \mathbb{Z}
$$

is a discrete random variable (in our case classification) and its realization respectively, and therefore, $Y$ can be characterized by a discrete PDF (PMF)

$$
f_{Y}(k ; \boldsymbol{\pi}) \sim \text{Categorical}(\boldsymbol{\pi})
$$


**Note that we are talking about one single sample tuple $\left(\mathbf{x}, y\right)$ here. I did not
index the sample tuple with $n$ because this sample can be any sample in the unknown distribution $\mathbb{P}_{\mathcal{X}, \mathcal{Y}}(\mathbf{x}, y)$
and not only from our given dataset $\mathcal{D}$.**
```

```{prf:definition} Likelihood
:label: likelihood

We denote the likelihood function as $\mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = k)$,
which is the probability of observing $\mathbf{x}$ given that the sample belongs to class $Y = k$.
```

```{prf:definition} Prior
:label: prior

We denote the prior probability of class $k$ as $\mathbb{P}(Y = k)$, which usually
follows a discrete distribution such as the Categorical distribution.
```

```{prf:definition} Posterior
:label: posterior

We denote the posterior probability of class $k$ given $\mathbf{x}$ as $\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x})$.
```

```{prf:definition} Marginal Distribution and Normalization Constant
:label: marginal-distribution-and-normalization-constant


We denote the normalizing constant as $\mathbb{P}(\mathbf{X} = \mathbf{x}) = \sum_{k=1}^K \mathbb{P}(Y = k) \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = k)$.
```

### Discriminative vs Generative

- Discriminative classifiers model the conditional distribution $\mathbb{P}(Y = k \mid \mathbf{X} =  \mathbf{x})$.
  This means we are modelling the conditional distribution of the target $Y$ given the input $\mathbf{x}$.
  This also means that we are using **conditional maximum likelihood** to estimate the parameters $\boldsymbol{\theta}$.
- Generative classifiers model the conditional distribution $\mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = k)$.
  This means we are modelling the conditional distribution of the input $\mathbf{X}$ given the target $Y$.
  Then we can use Bayes' rule to compute the conditional distribution of the target $Y$ given the input $\mathbf{X}$.
  This also means that we are using **joint maximum likelihood** to estimate the parameters $\boldsymbol{\theta}$.
- Both the target $Y$ and the input $\mathbf{X}$ are random variables in the generative model.
  In the discriminative model, only the target $Y$ is a random variable as the input $\mathbf{X}$ is fixed (we do not need to estimate anything about the input $\mathbf{X}$).
- For example, Logistic Regression models the target $Y$ as a function of predictor's $\mathbf{X} = \begin{bmatrix}X_1 \\ X_2 \\ \vdots \\X_D \end{bmatrix}$.
- Naive bayes models both the target $Y$ and the predictors $\mathbf{X}$ as a function of each other.
  This means we are modelling the joint distribution of the target $Y$ and the predictors $\mathbf{X}$.

## Naive Bayes Setup

Let

$$
\mathcal{D} = \left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \}_{n=1}^N = \left \{ \left(\mathrm{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^N
$$

be the dataset with $N$ samples and $D$ predictors.

All samples are assumed to be **independent and identically distributed (i.i.d.)** from the unknown but fixed joint distribution
$\mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$,

$$
\left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \} \overset{\small{\text{i.i.d.}}}{\sim} \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) \quad \text{for } n = 1, 2, \cdots, N
$$

where $\boldsymbol{\theta}$ is the parameter vector of the joint distribution. See {prf:ref}`joint-distribution-example` for an example of such.

(naive-bayes-inference-prediction)=
## Inference/Prediction

Before we look at the fitting/estimating process, let's look at the inference/prediction process.

Suppose the problem at hand has $K$ classes, $k = 1, 2, \cdots, K$, where $k$ is the index of the class.

Then, to find the class of a new test sample $\mathbf{x}^{(q)} \in \mathbb{R}^{D}$ with $D$ features,
we can compute the conditional probability of each class $Y = k$ given the sample $\mathbf{x}^{(q)}$:


```{prf:algorithm} Naive Bayes Inference Algorithm
:label: naive-bayes-inference-algorithm

- Compute the conditional probability of each class $Y = k$ given the sample $\mathbf{x}^{(q)}$:

  $$
  \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)}) = \dfrac{\mathbb{P}(Y = k) \mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k)}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)})} \quad \text{for } k = 1, 2, \cdots, K
  $$ (eq:conditional-naive-bayes)

- Choose the class $k$ that maximizes the conditional probability:

  $$
  \hat{y}^{(q)} = \arg\max_{k=1}^K \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)})
  $$ (eq:argmax-naive-bayes-1)

- The observant reader would have noticed that the normalizing constant
$\mathbb{P}\left(X = \mathbf{x}^{(q)}\right)$ is the same for all $k$.
Therefore, we can ignore it and simply choose the class $k$ that maximizes
the numerator of the conditional probability in {eq}`eq:conditional-naive-bayes`:

  $$
  \hat{y}^{(q)} = \arg\max_{k=1}^K \mathbb{P}(Y = k) \mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k)
  $$ (eq:argmax-naive-bayes-2)

  since where the normalizing constant is ignored, the conditional probability

  $$
  \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)}) \propto \mathbb{P}(Y = k) \mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k)
  $$ (eq:proportional-naive-bayes)

  by a constant factor $\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)})$.

  Note however, to recover the normalizing constant is easy, since the numerator $\mathbb{P}(Y = k) \mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k) $
  must sum up to 1 over all $k$, and therefore, the normalizing constant is simply $\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)}) = \sum_{k=1}^K \mathbb{P}(Y = k) \mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k)$.

- Expressing it in vector form, we have

  $$
  \begin{aligned}
  \hat{\mathbf{y}} &= \arg\max_{k=1}^K \begin{bmatrix} \mathbb{P}(Y=1) \mathbb{P}(\mathbf{X} = \mathbf{x}\mid Y = 1) \\ \mathbb{P}(Y=2) \mathbb{P}(\mathbf{X} = \mathbf{x}\mid Y = 2) \\ \vdots \\ \mathbb{P}(Y=K) \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = K) \end{bmatrix}_{K \times 1} \\
  &= \arg\max_{k=1}^K \begin{bmatrix} \mathbb{P}(Y=1) \\ \mathbb{P}(Y=2) \\ \cdots \\ \mathbb{P}(Y=K) \end{bmatrix}\circ \begin{bmatrix} \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = 1) \\ \mathbb{P}(\mathbf{X} = \mathbf{x}\mid Y = 2) \\ \vdots \\ \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = K) \end{bmatrix} \\
  &= \arg\max_{k=1}^K \mathbf{M_1} \circ \mathbf{M_2} \\
  &= \arg\max_{k=1}^K \mathbf{M_1} \circ \mathbf{M_3} \\
  \end{aligned}
  $$ (eq:argmax-naive-bayes-3)

  where

  $$
  \mathbf{M_1} = \begin{bmatrix}
  \mathbb{P}(Y = 1) \\
  \mathbb{P}(Y = 2) \\
  \vdots \\
  \mathbb{P}(Y = K)
  \end{bmatrix}_{K \times 1}
  $$ (eq:naive-bayes-m1)

  $$
  \mathbf{M_2} = \begin{bmatrix}
  \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = 1) \\
  \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = 2) \\
  \vdots \\
  \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = K)
  \end{bmatrix}_{K \times 1}
  $$ (eq:naive-bayes-m2)

  and

  $$
  \begin{aligned}
  \mathbf{M_3} &= \begin{bmatrix}
  \mathbb{P}(X_1 = x_1 \mid Y = 1 ; \theta_{11}) & \mathbb{P}(X_2 = x_2 \mid Y = 1 ; \theta_{12}) & \cdots & \mathbb{P}(X_D = x_D \mid Y = 1 ; \theta_{1D}) \\
  \mathbb{P}(X_1 = x_1 \mid Y = 2 ; \theta_{21}) & \mathbb{P}(X_2 = x_2 \mid Y = 2 ; \theta_{22}) & \cdots & \mathbb{P}(X_D = x_D \mid Y = 2 ; \theta_{2D}) \\
  \vdots & \vdots & \ddots & \vdots \\
  \mathbb{P}(X_1 = x_1 \mid Y = K ; \theta_{K1}) & \mathbb{P}(X_2 = x_2 \mid Y = K ; \theta_{K2}) & \cdots & \mathbb{P}(X_D = x_D \mid Y = K ; \theta_{KD})
  \end{bmatrix}_{K \times D}
  \end{aligned}
  $$ (eq:naive-bayes-m3)

  Note superscript $q$ is removed for simplicity, and $\circ$ is the element-wise (Hadamard) product.
  We will also explain why we replace $\mathbf{M_2}$ with $\mathbf{M_3}$ in {ref}`naive-bayes-conditional-independence`.
```

Now if we just proceed to estimate the conditional probability $\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)})$, we will need to estimate the joint probability $\mathbb{P}(X = \mathbf{x}^{(q)}, Y = k)$, since by definition, we have

$$
\mathbb{P}(X = \mathbf{x}^{(q)}, Y = k) = \mathbb{P}(Y = k) \mathbb{P}(X = \mathbf{x}^{(q)} \mid Y = k)
$$ (eq:joint-naive-bayes-1)

which is intractable[^intractable].

However, if we can ***estimate*** the conditional probability (likelihood) $\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k)$
and the prior probability $\mathbb{P}(Y = k)$, then we can use Bayes' rule to
compute the posterior conditional probability $\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)})$.

## The Naive Bayes Form

Quoted from [Wikipedia](https://en.wikipedia.org/wiki/Bayes%27_theorem#Simple_form_2), it is worth noting that
there's a few forms of Naive Bayes:

### Simple Form

If $X$ is continuous and $Y$ is discrete,

$$
f_{X \mid Y=y}(x)=\frac{P(Y=y \mid X=x) f_X(x)}{P(Y=y)}
$$

where each $f$ is a density function.

If $X$ is discrete and $Y$ is continuous,

$$
P(X=x \mid Y=y)=\frac{f_{Y \mid X=x}(y) P(X=x)}{f_Y(y)} .
$$

If both $X$ and $Y$ are continuous,

$$
f_{X \mid Y=y}(x)=\frac{f_{Y \mid X=x}(y) f_X(x)}{f_Y(y)} .
$$

### Extended form

A continuous event space is often conceptualized in terms of the numerator terms. It is then useful to eliminate the denominator using the law of total probability. For $f_Y(y)$, this becomes an integral:

$$
f_Y(y)=\int_{-\infty}^{\infty} f_{Y \mid X=\xi}(y) f_X(\xi) d \xi
$$

## The Naive Bayes Assumptions

In this section, we talk about some implicit and explicit assumptions of the Naive Bayes model.

### Independent and Identically Distributed (i.i.d.)

In supervised learning, implicitly or explicitly, one *always* assumes that the training set

$$
\begin{aligned}
\mathcal{D} &= \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \left(\mathbf{x}^{(2)}, y^{(2)}\right), \cdots, \left(\mathbf{x}^{(N)}, y^{(N)}\right)\right\} \\
\end{aligned}
$$

is composed of $N$ input/response tuples

$$
\left({\mathbf{X}}^{(n)} = \mathbf{x}^{(n)}, Y^{(n)} = y^{(n)}\right)
$$

that are ***independently drawn from the same (identical) joint distribution***

$$
\mathbb{P}_{\{\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\}}(\mathbf{x}, y)
$$

with

$$
\mathbb{P}(\mathbf{X} = \mathbf{x}, Y = y ; \boldsymbol{\theta}) = \mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x}) \mathbb{P}(\mathbf{X} = \mathbf{x})
$$

where $\mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x})$ is the conditional probability of $Y$ given $\mathbf{X}$,
the relationship that the learner algorithm/concept $c$ is trying to capture.

```{prf:definition} The i.i.d. Assumption
:label: iid-assumption

Mathematically, this i.i.d. assumption writes (also defined in {prf:ref}`def_iid`):

$$
\begin{aligned}
\left({\mathbf{X}}^{(n)}, Y^{(n)}\right) &\sim \mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y) \quad \text{and}\\
\left({\mathbf{X}}^{(n)}, Y^{(n)}\right) &\text{ independent of } \left({\mathbf{X}}^{(m)}, Y^{(m)}\right) \quad \forall n \neq m \in \{1, 2, \ldots, N\}
\end{aligned}
$$

and we sometimes denote

$$
\begin{aligned}
\left(\mathbf{x}^{(n)}, y^{(n)}\right) \overset{\text{i.i.d.}}{\sim} \mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y)
\end{aligned}
$$
```

The confusion in the **i.i.d.** assumption is that we are not talking about the individual random variables
$X_1^{(n)}, X_2^{(n)}, \ldots, X_D^{(n)}$ here, but the entire random vector $\mathbf{X}^{(n)}$.

This means there is no assumption of $X_1^{(n)}, X_2^{(n)}, \ldots, X_D^{(n)}$ being **i.i.d.**. Instead, the samples
$\mathbf{X}^{(1)}, \mathbf{X}^{(2)}, \ldots, \mathbf{X}^{(N)}$ are **i.i.d.**.

(naive-bayes-conditional-independence)=
### Conditional Independence

The core assumption of the Naive Bayes model is that the predictors $\mathbf{X}$
are conditionally independent given the class label $Y$.

But how did we arrive at the conditional independence assumption? Let's look at what we wanted to achieve in the first place.

Recall that our goal in {ref}`naive-bayes-inference-prediction` is to find the class $k \in \{1, 2, \cdots, K\}$ that maximizes the **posterior** probability
$\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})$.

$$
\begin{aligned}
\hat{y}^{(q)} &= \arg \max_{k} \mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta}) \\
              &= \arg \max_{k} \frac{\mathbb{P}(Y = k, \mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})} \\
              &= \arg \max_{k} \frac{\mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}})}{\mathbb{P}(\mathbf{X} = \mathbf{x}^{(q)} ; \boldsymbol{\theta})}\\
\end{aligned}
$$ (eq:argmax-naive-bayes-4)

We have seen earlier in {prf:ref}`naive-bayes-inference-algorithm` that since the denominator
is constant for all $k$, we can ignore it and just maximize the numerator.

$$
\begin{aligned}
\hat{y}^{(q)} &= \arg \max_{k} \mathbb{P}\left(Y = k ; \boldsymbol{\pi}\right) \mathbb{P}\left(\mathbf{X} = \mathbf{x}^{(q)} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right) \\
\end{aligned}
$$ (eq:argmax-naive-bayes-5)

This suggests we need to find estimates for both the **prior** and the **likelihood**. This of course
involves us finding the $\boldsymbol{\pi}$ and $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ that maximize the likelihood function[^likelihood-1], which we will talk about later.

In order to meaningfully optimize the expression, we need to decompose the expression {eq}`eq:argmax-naive-bayes-5`
into its components that contain the parameters we want to estimate.

$$
\begin{aligned}
\mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}) &= \mathbb{P}((Y, \mathbf{X}) ; \boldsymbol{\theta}, \boldsymbol{\pi}) \\
&= \mathbb{P}(Y, X_1, X_2, \ldots X_D)
\end{aligned}
$$ (eq:joint-distribution)

which is actually the joint distribution of $\mathbf{X}$ and $Y$[^joint-distribution].

This joint distribution expression {eq}`eq:joint-distribution` can be further decomposed by the chain rule of probability[^chain-rule-of-probability] as

$$
\begin{aligned}
\mathbb{P}(Y, X_1, X_2, \ldots X_D) &= \mathbb{P}(Y) \mathbb{P}(X_1, X_2, \ldots X_D \mid Y) \\
&= \mathbb{P}(Y) \mathbb{P}(X_1 \mid Y) \mathbb{P}(X_2 \mid Y, X_1) \cdots \mathbb{P}(X_D \mid Y, X_1, X_2, \ldots X_{D-1}) \\
&= \mathbb{P}(Y) \prod_{d=1}^D \mathbb{P}(X_d \mid Y, X_1, X_2, \ldots X_{d-1}) \\
&= \mathbb{P}(Y) \prod_{d=1}^{D} \mathbb{P}\left(X_d \middle \vert \bigcap_{d'=1}^{d-1} X_{d'}\right)
\end{aligned}
$$ (eq:joint-distribution-decomposed)

This alone does not get us any further, we still need to estimate roughly $2^{D}$ parameters[^2dparameters],
which is computationally expensive. Not to forget that we need to estimate for each class $k \in \{1, 2, 3, \ldots, K\}$
which has a complexity of $\sim \mathcal{O}(2^DK)$.

```{prf:remark} Why $2^D$ parameters?
:label: 2dparameters

Let's simplify the problem by assuming each feature $X_d$ and the class label $Y$ are binary random variables,
i.e. $X_d \in \{0, 1\}$ and $Y \in \{0, 1\}$.

Then $\mathbb{P}(Y, X_1, X_2, \ldots X_D)$ is a joint distribution of $D+1$ random variables, each with $2$ values.

This means the sample space of $\mathbb{P}(Y, X_1, X_2, \ldots X_D)$ is

$$
\begin{aligned}
\mathcal{S} &= \{(0, 1)\} \times \{(0, 1)\} \times \{(0, 1)\} \times \cdots \times \{(0, 1)\} \\
&= \{(0, 0, 0, \ldots, 0), (0, 0, 0, \ldots, 1), (0, 0, 1, \ldots, 0), \ldots, (1, 1, 1, \ldots, 1)\}
\end{aligned}
$$

which has $2^{D+1}$ elements.
To really get the exact joint distribution, we need to estimate the probability of each element in the sample space, which is $2^{D+1}$ parameters.

This has two caveats:

1. There are too many parameters to estimate, which is computationally expensive. Imagine if $D$ is 1000, we need to estimate $2^{1000}$ parameters, which is infeasible.
2. Even if we can estimate all the parameters, we are essentially overfitting the data by memorizing the training data. There is no learning involved.
```

This is where the "Naive" assumption comes in. The Naive Bayes' classifier assumes that
the features are conditionally independent[^conditional-independence] given the class label.

More formally stated,

```{prf:definition} Conditional Independence
:label: conditional-independence

$$
\mathbb{P}(X_d \mid Y = k, X_{d^{'}}) = \mathbb{P}(X_d \mid Y = k) \quad \text{for all } d \neq d^{'}
$$ (eq:conditional-independence)
```

with this assumption, we can further simplify expression {eq}`eq:joint-distribution-decomposed` as

$$
\begin{aligned}
\mathbb{P}(Y, X_1, X_2, \ldots X_D) &= \mathbb{P}(Y ; \boldsymbol{\pi}) \prod_{d=1}^D \mathbb{P}(X_d \mid Y ; \theta_{d}) \\
\end{aligned}
$$ (eq:conditional-independence-naive-bayes-1)

More precisely, after the simplification in {eq}`eq:conditional-independence-naive-bayes-1`,
the argmax expression in {eq}`eq:conditional-naive-bayes` can be written as

$$
\begin{aligned}
\mathbb{P}(Y = k \mid \mathbf{X} = \mathbf{x} ; \boldsymbol{\theta}) & = \dfrac{\mathbb{P}(Y = k ; \boldsymbol{\pi}) \mathbb{P}(\mathbf{X} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}})}{\mathbb{P}(\mathbf{X})} \\
&= \dfrac{\mathbb{P}(Y, X_1, X_2, \ldots X_D)}{\mathbb{P}(\mathbf{X})} \\
&= \dfrac{\mathbb{P}(Y = k ; \boldsymbol{\pi}) \prod_{d=1}^D \mathbb{P}(X_d = x_d \mid Y = k ; \theta_{kd})}{\mathbb{P}(\mathbf{X} = \mathbf{x})} \\
\end{aligned}
$$ (eq:naive-bayes-classifier-1)

Consequently, our argmax expression in {eq}`eq:argmax-naive-bayes-2` can be written as

$$
\arg \max_{k=1}^K \mathbb{P}(Y = k \mid \mathbf{X}) = \arg \max_{k=1}^K \mathbb{P}(Y = k ; \boldsymbol{\pi}) \prod_{d=1}^D \mathbb{P}(X_d = x_d \mid Y = k ; \theta_{kd})
$$ (eq:argmax-naive-bayes-6)

We also make some updates to the vector form {eq}`eq:argmax-naive-bayes-3` by updating $\mathbf{M_2}$ to:

$$
\begin{aligned}
  \mathbf{M_2} &= \begin{bmatrix}
  \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = 1) \\
  \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = 2) \\
  \vdots \\
  \mathbb{P}(\mathbf{X} = \mathbf{x} \mid Y = K)
  \end{bmatrix}_{K \times 1} \\
  &= \begin{bmatrix}
  \mathbb{P}(X_1 = x_1 \mid Y = 1 ; \theta_{11}) \mathbb{P}(X_2 = x_2 \mid Y = 1 ; \theta_{12}) \cdots \mathbb{P}(X_D = x_D \mid Y = 1 ; \theta_{1D}) \\
  \mathbb{P}(X_1 = x_1 \mid Y = 2 ; \theta_{21}) \mathbb{P}(X_2 = x_2 \mid Y = 2 ; \theta_{22}) \cdots \mathbb{P}(X_D = x_D \mid Y = 2 ; \theta_{2D}) \\
  \vdots \\
  \mathbb{P}(X_1 = x_1 \mid Y = K ; \theta_{K1}) \mathbb{P}(X_2 = x_2 \mid Y = K ; \theta_{K2}) \cdots \mathbb{P}(X_D = x_D \mid Y = K ; \theta_{KD})
  \end{bmatrix}_{K \times 1} \\
\end{aligned}
$$ (eq:naive-bayes-m2-updated)

To easily recover each row of $\mathbf{M_2}$, it is efficient to define a $K \times D$ matrix, denoted $\mathbf{M_3}$

$$
\begin{aligned}
  \mathbf{M_3} &= \begin{bmatrix}
  \mathbb{P}(X_1 = x_1 \mid Y = 1 ; \theta_{11}) & \mathbb{P}(X_2 = x_2 \mid Y = 1 ; \theta_{12}) & \cdots & \mathbb{P}(X_D = x_D \mid Y = 1 ; \theta_{1D}) \\
  \mathbb{P}(X_1 = x_1 \mid Y = 2 ; \theta_{21}) & \mathbb{P}(X_2 = x_2 \mid Y = 2 ; \theta_{22}) & \cdots & \mathbb{P}(X_D = x_D \mid Y = 2 ; \theta_{2D}) \\
  \vdots & \vdots & \ddots & \vdots \\
  \mathbb{P}(X_1 = x_1 \mid Y = K ; \theta_{K1}) & \mathbb{P}(X_2 = x_2 \mid Y = K ; \theta_{K2}) & \cdots & \mathbb{P}(X_D = x_D \mid Y = K ; \theta_{KD})
  \end{bmatrix}_{K \times D} \\
\end{aligned}
$$ (eq:naive-bayes-m3-explained)

where we can easily recover each row of $\mathbf{M_2}$ by taking the product of the corresponding row of $\mathbf{M_3}$.

## Parameter Vector

In the last section on {ref}`naive-bayes-conditional-independence`, we indicated parameters in the expressions.
Here we discuss a little on this newly introduced notation.

Each $\pi_k$ of $\boldsymbol{\pi}$ refers to the prior probability of class $k$, and $\theta_{kd}$ refers to the parameter of the
class conditional density for class $k$ and feature $d$[^kdparameters]. Furthermore,
the boldsymbol $\boldsymbol{\theta}$ is the parameter vector,

$$
\boldsymbol{\theta} = \left(\boldsymbol{\pi}, \{\theta_{kd}\}_{k=1, d=1}^{K, D} \right) = \left(\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right)
$$


```{prf:definition} The Parameter Vector
:label: parameter-vector

There is not much to say about the categorical component $\boldsymbol{\pi}$, since we are
just estimating the prior probabilities of the classes.

$$
\boldsymbol{\pi} = \begin{bmatrix} \pi_1 \\ \pi_2 \\ \vdots \\ \pi_K \end{bmatrix}_{K \times 1}
$$

The parameter vector (matrix) $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}=\{\theta_{kd}\}_{k=1, d=1}^{K, D}$ is a bit more complicated.
It resides in the $\mathbb{R}^{K \times D}$ space, where each element $\theta_{kd}$ is the parameter
associated with feature $d$ conditioned on class $k$.

$$
\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} = \begin{bmatrix}
\theta_{11} & \theta_{12} & \dots & \theta_{1D} \\
\theta_{21} & \theta_{22} & \dots & \theta_{2D} \\
\vdots & \vdots & \ddots & \vdots \\
\theta_{K1} & \theta_{K2} & \dots & \theta_{KD}
\end{bmatrix}_{K \times D}
$$

So if $K=3$ and $D=2$, then the parameter vector $\boldsymbol{\theta}$ is a $3 \times 2$ matrix, i.e.

$$
\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} = \begin{bmatrix}
\theta_{11} & \theta_{12} \\
\theta_{21} & \theta_{22} \\
\theta_{31} & \theta_{32}
\end{bmatrix}_{3 \times 2}
$$

This means we have effectively reduced our complexity from $\sim \mathcal{O}(2^D)$ to $\sim \mathcal{O}(KD + 1)$
assuming the same setup in {prf:ref}`2dparameters`.

One big misconception is that the elements in $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ are scalar values.
This is not true, for example, let's look at the first entry $\theta_{11}$, corresponding to
the parameter of class $K=1$ and feature $D=1$, i.e. $\theta_{11}$ is the parameter of the class conditional
density $\mathbb{P}(X_1 \mid Y = 1)$. Now $X_1$ can take on any value in $\mathbb{R}$, which is indeed a scalar,
we further assume that $X_1$ takes on a univariate Gaussian distribution, then $\theta_{11}$ is a vector of length 2, i.e.

$$
\theta_{11} = \begin{bmatrix} \mu_{11} & \sigma_{11} \end{bmatrix}
$$

where $\mu_{11}$ is the mean of the Gaussian distribution and $\sigma_{11}$ is the standard deviation of the Gaussian distribution.
This is something we need to take note of.

**We have also reduced the problem of estimating the joint distribution to just individual conditional distributions.**

Overall, before this assumption, you can think of estimating the joint distribution of $Y$ and $\mathbf{X}$,
and after this assumption, you can simply individually estimate each conditional distribution.
```

Notice that the shape of $\boldsymbol{\pi}$ is $K \times 1$, and the shape of $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ is $K \times D$.
This corresponds to the shape of the matrix $\mathbf{M_1}$ and $\mathbf{M_3}$ as defined in
{eq}`eq:naive-bayes-m1` and {eq}`eq:naive-bayes-m3`, respectively. This is expected since
$\mathbf{M_1}$ and $\mathbf{M_3}$ hold the PDFs while $\boldsymbol{\pi}$ and $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ hold the parameters
of these PDFs.

```{prf:remark} Empirical Parameters
:label: remark-empirical-parameters

It is worth noting that we are discussing the parameter vectors $\boldsymbol{\pi}$
and $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ which represents the true
underlying distribution. However, our ultimate goal is to estimate these parameters because
we do not have the underlying distributions at hand, otherwise there is no need to do
machine learning.

More concretely, our task is to find

$$
\hat{\boldsymbol{\pi}} \quad \text{ and } \quad \hat{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}}
$$
```

## Inductive Bias (Distribution Assumptions)

We still need to introduce some inductive bias into {eq}`eq:naive-bayes-classifier-1`, more concretely, we need to make some assumptions about the distribution
of $\mathbb{P}(Y)$ and $\mathbb{P}(X_d \mid Y)$.

For the target variable, we typically model it as a categorical distribution,

$$
\mathbb{P}(Y) \sim \mathrm{Categorical}(\boldsymbol{\pi})
$$

For the conditional distribution of the features, we typically model it according to what type of features we have.

For example, if we have binary features, then we can model it as a Bernoulli distribution,

$$
\mathbb{P}(X_d \mid Y) \sim \mathrm{Bernoulli}(\theta_{dk})
$$

If we have categorical features, then we can model it as a multinomial/catgorical distribution,

$$
\mathbb{P}(X_d \mid Y) \sim \mathrm{Multinomial}(\boldsymbol{\theta}_{dk})
$$

If we have continuous features, then we can model it as a Gaussian distribution,

$$
\mathbb{P}(X_d \mid Y) \sim \mathcal{N}(\mu_{dk}, \sigma_{dk}^2)
$$

To reiterate, we want to make some inductive bias assumptions of $\mathbf{X}$ conditional on $Y$,
as well as with $Y$. Note very carefully that we are not talking about the marginal distribution of
$\mathbf{X}$ here, instead, we are talking about the conditional distribution of $\mathbf{X}$ given $Y$. The distinction is subtle, but important.

### Targets (Categorical Distribution)

As mentioned earlier, both $Y^{(n)}$ and $\mathbf{X}^{(n)}$ are random variables/vectors.
This means we need to estimate both of them.

We first conveniently assume that $Y^{(n)}$ is a discrete random variable, and
follows the **[Category distribution](https://en.wikipedia.org/wiki/Categorical_distribution)**[^categorical-distribution],
an extension of the Bernoulli distribution to multiple classes. Instead of a single parameter $p$ (probability of success for Bernoulli),
the Category distribution has a vector $\boldsymbol{\pi}$ of $K$ parameters.

$$
\boldsymbol{\pi} = \begin{bmatrix} \pi_1 \\ \vdots \\ \pi_K \end{bmatrix}_{K \times 1}
$$

where $\pi_k$ is the probability of $Y^{(n)}$ taking on value $k$.

$$
Y^{(n)} \overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}) \quad \text{where } \boldsymbol{\pi} = \begin{bmatrix} \pi_1 \\ \vdots \\ \pi_K \end{bmatrix}_{K \times 1}
$$ (eq:target-category-distribution)

Equivalently,

$$
\mathbb{P}(Y^{(n)} = k) = \pi_k \quad \text{for } k = 1, 2, \cdots, K
$$ (eq:category-distribution)

Consequently, we just need to estimate $\boldsymbol{\pi}$ to recover $\mathbf{M_1}$ defined in {eq}`eq:naive-bayes-m1`.

Find $\hat{\boldsymbol{\pi}}$ such that $\hat{\boldsymbol{\pi}}$ maximizes the likelihood of the observed data.

$$
\hat{\boldsymbol{\pi}} = \arg\max_{\boldsymbol{\pi}} \mathcal{L}(\boldsymbol{\pi} \mid \mathcal{D})
$$


```{prf:definition} Categorical Distribution
:label: categorical-distribution

Let $Y$ be a discrete random variable with $K$ number of states.
Then $Y$ follows a categorical distribution with parameters $\boldsymbol{\pi}$ if

$$
\mathbb{P}(Y = k) = \pi_k \quad \text{for } k = 1, 2, \cdots, K
$$

Consequently, the PMF of the categorical distribution is defined more compactly as,

$$
\mathbb{P}(Y = k) = \prod_{k=1}^K \pi_k^{I\{Y = k\}}
$$

where $I\{Y = k\}$ is the indicator function that is equal to 1 if $Y = k$ and 0 otherwise.
```

```{prf:definition} Categorical (Multinomial) Distribution
:label: categorical-multinomial-distribution

This formulation is adopted by Bishop's{cite}`bishop2007`, the categorical distribution is defined as

$$
\mathbb{P}(\mathbf{Y} = \mathbf{y}; \boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{y_k}
$$ (eq:categorical-distribution-bishop)

where

$$
\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_K \end{bmatrix}
$$

is an one-hot encoded vector of size $K$,

The $y_k$ is the $k$-th element of $\mathbf{y}$, and is equal to 1 if $Y = k$ and 0 otherwise.
The $\pi_k$ is the $k$-th element of $\boldsymbol{\pi}$, and is the probability of $Y = k$.

This notation alongside with the indicator notation in the previous definition allows us to manipulate
the likelihood function easier.
```


```{prf:example} Categorical Distribution Example
:label: categorical-distribution-example

Consider rolling a fair six-sided die. Let $Y$ be the random variable that represents the outcome
of the dice roll. Then $Y$ follows a categorical distribution with parameters $\boldsymbol{\pi}$ where $\pi_k = \frac{1}{6}$ for $k = 1, 2, \cdots, 6$.

$$
\mathbb{P}(Y = k) = \frac{1}{6} \quad \text{for } k = 1, 2, \cdots, 6
$$

For example, if we roll a 3, then $\mathbb{P}(Y = 3) = \frac{1}{6}$.

With the more compact notation, the indicator function is $I\{Y = k\} = 1$ if $Y = 3$ and $0$ otherwise. Therefore, the PMF is

$$
\mathbb{P}(Y = k) = \prod_{k=1}^6 \frac{1}{6}^{I\{Y = k\}} = \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^1 \cdot \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^0 \cdot \left(\frac{1}{6}\right)^0 = \frac{1}{6}
$$

Using Bishop's notation, the PMF is still the same, only the realization $\mathbf{y}$ is not a scalar,
but instead a vector of size $6$. In the case where $Y = 3$, the vector $\mathbf{y}$ is

$$
\mathbf{y} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}
$$
```

### Discrete Features (Categorical Distribution)

Now, our next task is find parameters $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ to model the conditional distribution of $\mathbf{X}$ given $Y = k$,
and consequently, recovering the matrix $\mathbf{M_3}$ defined in {eq}`eq:naive-bayes-m3`.

In the case where (all) the features $X_d$ are categorical ($D$ number of features),
i.e. $X_d \in \{1, 2, \cdots, C\}$,
we can use the categorical distribution to model the ($D$-dimensional) conditional
distribution of $\mathbf{X} \in \mathbb{R}^{D}$ given $Y = k$.

$$
\begin{align*}
\mathbf{X} \mid Y = k &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}\left(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}}\right) \quad \text{for } k = 1, 2, \cdots, K
\end{align*}
$$ (eq:naive-bayes-categorical-feature-1)

where

$$
\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}} = \begin{bmatrix} \pi_{1, 1} & \dots & \pi_{1, D} \\ \vdots & \ddots & \vdots \\ \pi_{K, 1} & \dots & \pi_{K, D} \end{bmatrix}_{K \times D} \\
$$ (eq:naive-bayes-categorical-feature-2)

$\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}}$ is a matrix of size $K \times D$ where each
element $\pi_{k, d}$ is the parameter for the
probability distribution (PDF) of $X_d$ given $Y = k$.

$$
X_d \mid Y = k \overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\pi_{k, d})
$$

Furthermore,
each $\pi_{k, d}$ is **not a scalar** but a **vector of size $C$** holding the probability of $X_d = c$ given $Y = k$.

$$
\begin{align*}
\pi_{k, d} = \begin{bmatrix} \pi_{k, d, 1} & \dots & \pi_{k, d, C} \end{bmatrix}
\end{align*}
$$

Then the (chained) multi-dimensional conditional PDF of $\mathbf{X} = \begin{bmatrix} X_1 & \dots & X_D \end{bmatrix}$ given $Y = k$ is

$$
\begin{align*}
\mathbb{P}(\mathbf{X} = \mathbf{x} | Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}) &= \prod_{d=1}^D \text{Categorical}(X_d \mid Y = k; \pi_{k, d}) \\
&= \prod_{d=1}^D \prod_{c=1}^C \pi_{k, d, c}^{x_{c, d}} \quad \text{for } c = 1, 2, \cdots, C \text{ and } k = 1, 2, \cdots, K
\end{align*}
$$ (eq:naive-bayes-categorical-feature-3)

As an example, if $C=3$, $D=2$ and $K=4$, then the $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ is a $K \times D = 4 \times 2$ matrix, but for
each entry $\pi_{k, d}$, is a $1 \times C$ vector. If one really wants, we can also represent this as a
$4 \times 2 \times 3$ tensor, especially in the case of implementing it in code.

To be more verbose, when we find

$$
\mathbf{X} \mid Y = k \overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y\}})
$$

we are actually finding for all $k = 1, 2, \cdots, K$,

$$
\begin{align*}
\mathbf{X} \mid Y = 1 &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y=1\}}) \\
\mathbf{X} \mid Y = 2 &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y=2\}}) \\
\vdots \\
\mathbf{X} \mid Y = K &\overset{\small{\text{i.i.d.}}}{\sim} \text{Category}(\boldsymbol{\pi}_{\{\mathbf{X} \mid Y=K\}})
\end{align*}
$$

This is because we are also finding the argmax of the number of classes $K$ when we seek
the expression $\arg\max_{k=1, 2, \cdots, K} \mathbb{P}(Y = k | \mathbf{X} = \mathbf{x})$,
and therefore, we need to find the conditional PDF of $\mathbf{X}$ given $Y$ for each class $k$.

Each row above corresponds to each row of the matrix $\mathbf{M_2}$ defined in {eq}`eq:naive-bayes-m2`. We
can further decompose each $\mathbf{X} \mid Y = k$ into $D$ independent random variables, each of which
is modeled by a categorical distribution, thereby recovering each element of $\mathbf{M_3}$ {eq}`eq:naive-bayes-m3`.

See **Kevin Murphy's Probabilistic Machine Learning: An Introduction** pp 358 for more details.

(continuous-features-gaussian-distribution)=
### Continuous Features (Gaussian Distribution)

Here, the task is still the same, to find parameters $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ to model the conditional distribution of $\mathbf{X}$ given $Y = k$,
and consequently, recovering the matrix $\mathbf{M_3}$ defined in {eq}`eq:naive-bayes-m3`.

In the case where (all) the features $X_d$ are continuous ($D$ number of features),
we can use the Gaussian distribution to model the conditional distribution of $\mathbf{X}$ given $Y = k$.

$$
\begin{align*}
\mathbf{X} \mid Y = k \overset{\small{\text{i.i.d.}}}{\sim} \mathcal{N}(\theta_{\{\mathbf{X} \mid Y\}}) \quad \text{for } k = 1, 2, \cdots, K
\end{align*}
$$ (eq:naive-bayes-continuous-feature-1)

where

$$
\begin{align*}
\theta_{\{\mathbf{X} \mid Y\}} &= \begin{bmatrix} \theta_{1, 1} & \dots & \theta_{1, D} \\ \vdots & \ddots & \vdots \\ \theta_{K, 1} & \dots & \theta_{K, D} \end{bmatrix}_{K \times D} \\
&= \begin{bmatrix} (\mu_{1, 1}, \sigma_{1, 1}^2) & \dots & (\mu_{1, D}, \sigma_{1, D}^2) \\ \vdots & \ddots & \vdots \\ (\mu_{K, 1}, \sigma_{K, 1}^2) & \dots & (\mu_{K, D}, \sigma_{K, D}^2) \end{bmatrix}_{K \times D}
\end{align*}
$$ (eq:naive-bayes-continuous-feature-2)

$\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ is a $K \times D$ matrix, where each element
$\theta_{k, d}$ is a tuple of the mean and variance of the
Gaussian distribution modeling the conditional distribution of $X_d$ given $Y = k$.

To be more precise, each element in the matrix $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ is a tuple of the mean and variance of the
Gaussian distribution modeling the conditional distribution of $X_d$ given $Y = k$.

$$
\begin{align*}
X_d \mid Y = k &\overset{\small{\text{i.i.d.}}}{\sim} \mathcal{N}(\mu_{k, d}, \sigma_{k, d}^2) \quad \text{for } k = 1, 2, \cdots, K
\end{align*}
$$ (eq:naive-bayes-continuous-feature-3)

Then the (chained) multivariate Gaussian distribution of $\mathbf{X} = \begin{bmatrix} X_1 & \dots & X_D \end{bmatrix}$ given $Y = k$ is

$$
\begin{align*}
\mathbb{P}\left(\mathbf{X} = \mathbf{x} \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right) &= \prod_{d=1}^D \mathcal{N}(x_d \mid \mu_{k, d}, \sigma_{k, d}^2) \\
\end{align*}
$$ (eq:naive-bayes-continuous-feature-4)

where $\mu_{k, d}$ and $\sigma_{k, d}^2$ are the mean and variance of the
Gaussian distribution modeling the conditional distribution of $X_d$ given $Y = k$.

So in this case it amounts to estimating $\hat{\mu}_{k, d}$ and $\hat{\sigma}_{k, d}^2$ for each $k$ and $d$.

### Mixed Features (Discrete and Continuous)

So far we have assumed that each feature $X_d$ is either all discrete, or all continuous. This
need not be the case, and may not always be the case. In reality, we may have a mixture of both.

For example, if $X_1$ corresponds to the smoking status of a person (i.e. whether they smoke or not),
then this feature is binary, and can be modeled by a Bernoulli distribution.
On the other hand, if $X_2$ corresponds to the weight of a person, then this feature is continuous, and can be modeled by a Gaussian distribution.
The nice thing is since within each class $k$, the features $X_d$ are independent of each other, we can model each feature $X_d$ by its own distribution.

So, carrying over the example above, we have,

$$
\begin{align*}
X_1 \mid Y = k &\overset{\small{\text{i.i.d.}}}{\sim} \text{Bernoulli}(\pi_{\{X_1 \mid Y=k\}}) \\
X_2 \mid Y = k &\overset{\small{\text{i.i.d.}}}{\sim} \text{Gaussian}(\mu_{\{X_2 \mid Y=k\}}, \sigma_{\{X_2 \mid Y=k\}}^2)
\end{align*}
$$ (eq:naive-bayes-mixed-feature-1)

and subsequently, the chained PDF is

$$
\begin{align*}
\mathbb{P}\left(X_1 = x_1, X_2 = x_2 \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right) &= \prod_{d=1}^D \mathbb{P}\left(X_d = x_d \mid Y = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}\right) \\
&= \mathbb{P}\left(X_1 = x_1 \mid Y = k ; \boldsymbol{\pi}_{\{X_1 \mid Y\}}\right) \mathbb{P}\left(X_2 = x_2 \mid Y = k ; \boldsymbol{\theta}_{\{X_2 \mid Y\}}\right) \\
&= \pi_{\{X_1 \mid Y=k\}}^{x_1} (1 - \pi_{\{X_1 \mid Y=k\}})^{1 - x_1} \mathcal{N}(x_2 \mid \mu_{\{X_2 \mid Y=k\}}, \sigma_{\{X_2 \mid Y=k\}}^2)
\end{align*}
$$

See more details in [Machine Learning from Scratch](https://dafriedman97.github.io/mlbook/content/c4/concept.html).

## Model Fitting

We have so far laid out the model prediction process, the implicit and explicit assumptions, as well as
the model parameters.

Now, we need to figure out how to fit the model parameters to the data. After all, once we
find the model parameters that best fit the data, we can use the model to make predictions
using matrix $\mathbf{M_1}$ and $\mathbf{M_3}$ as defined in {prf:ref}`naive-bayes-inference-algorithm`.

### Fitting Algorithm

```{prf:algorithm} Naive Bayes Estimation Algorithm
:label: prf:naive-bayes-estimation-algorithm

For each entry in matrix $\mathbf{M_1}$, we seek to find its corresponding estimated parameter vector $\hat{\boldsymbol{\pi}}$:

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \begin{bmatrix} \hat{\pi}_1 \\ \vdots \\ \hat{\pi}_K \end{bmatrix}_{K \times 1} \\
\end{align*}
$$ (eq:naive-bayes-estimation-1)

where $\hat{\pi}_k$ is the estimated (empirical) probability of class $k$.

For each entry in matrix $\mathbf{M_3}$, we seek to find its corresponding estimated parameter matrix $\hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y\}}$:

$$
\begin{align*}
\hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y\}} &= \begin{bmatrix} \hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y=1\}} \\ \vdots \\ \hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y=K\}} \end{bmatrix}_{K \times D} \\
&= \begin{bmatrix} \hat{\theta}_{\{X_1 \mid Y=1\}} & \cdots & \hat{\theta}_{\{X_D \mid Y=1\}} \\ \vdots & \ddots & \vdots \\ \hat{\theta}_{\{X_1 \mid Y=K\}} & \cdots & \hat{\theta}_{\{X_D \mid Y=K\}} \end{bmatrix}_{K \times D} \\
&= \begin{bmatrix} \hat{\theta}_{11} & \cdots & \hat{\theta}_{1D} \\ \vdots & \ddots & \vdots \\ \hat{\theta}_{K1} & \cdots & \hat{\theta}_{KD} \end{bmatrix}_{K \times D} \\
\end{align*}
$$ (eq:naive-bayes-estimation-2)

where $\hat{\theta}_{kd}$ is the probability of feature $X_d$ given class $k$.

Both the underlying distribution $\boldsymbol{\pi}$ and $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ are estimated by maximizing the likelihood of the data,
using the Maximum Likelihood Estimation (MLE) method to obtain the maximum likelihood estimates (MLEs), which are denoted by $\hat{\boldsymbol{\pi}}$ and $\hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y\}}$ respectively.
```

### Maximum Likelihood Estimation

First, read chapter 8.1 of {cite}`chan_2021` for a refresher on MLE.

```{prf:remark} Univariate Maximum Likelihood Estimation
:label: remark-univariate-mle

In [LDA](https://dafriedman97.github.io/mlbook/content/c4/concept.html#linear-discriminative-analysis-lda),
 $\mathbf{X} \mid Y=k$, the distribution of the features $\mathbf{X}$ conditioned on $Y=k$, has no
assumption of conditional independence. Therefore, we need to estimate the parameters of
$\mathbf{X} = \{\mathbf{X}_1, \dots, \mathbf{X}_D\}$ jointly.

More concretely,

$$
\begin{align*}
\mathbf{X} \mid Y = k \overset{\text{i.i.d.}}{\sim} \mathcal{N}\left(\boldsymbol{\mu}_{\{X \mid Y=k\}}, \boldsymbol{\Sigma}_{\{X \mid Y=k\}}\right) \quad \forall k \in \{1, \dots, K\}
\end{align*}
$$

where $\boldsymbol{\mu}_{\{X \mid Y=k\}}$ is the mean vector of $\mathbf{X}$ given $Y=k$, and $\boldsymbol{\Sigma}_{\{X \mid Y=k\}}$ is the covariance matrix of $\mathbf{X}$ given $Y=k$.

However, in the case of Naive Bayes, the assumption of conditional independence allows us to estimate the parameters of $\mathbf{X} = \{\mathbf{X}_1, \dots, \mathbf{X}_D\}$ univariately,
conditional on $Y=k$.

Looking at expression {eq}`eq:naive-bayes-estimation-2`, we can see that each element
is indeed univariate, and we can estimate the parameters of each element univariately.
```

Everything we have talked about is just 1 single sample, and that won't work in the realm of
estimating the best parameters that fit the data. Since we are given a dataset $\mathcal{D}$
consisting of $N$ samples, we can estimate the parameters of the model by maximizing the likelihood of the data.

```{prf:definition} Likelihood Function of Naive Bayes
:label: def:naive-bayes-likelihood

Given **i.i.d.** random variables[^iid-tuple] $\left(\mathbf{X}^{(1)}, Y^{(1)}\right), \left(\mathbf{X}^{(2)}, Y^{(2)}\right), \dots, \left(\mathbf{X}^{(N)}, Y^{(N)}\right)$,
we can write the likelihood function (joint probability distribution)
as the product of the individual PDF of each sample[^iid-likelihood]:

$$
\begin{align*}
\mathcal{L}(\boldsymbol{\theta} \mid \mathcal{D}) \overset{\mathrm{def}}{=} \mathbb{P}(\mathcal{D} ; \boldsymbol{\theta}) &= \mathbb{P}\left(\mathcal{D} ; \left\{\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right\}\right) \\
&\overset{(a)}{=} \mathbb{P}\left(\left(\mathbf{X}^{(1)}, Y^{(1)}\right), \left(\mathbf{X}^{(2)}, Y^{(2)}\right), \dots, \left(\mathbf{X}^{(N)}, Y^{(N)}\right) ; \left\{\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right\}\right) \\
&\overset{(b)}{=} \prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right) \mathbb{P}\left(\mathrm{X}^{(n)} \mid Y^{(n)} = k ; \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right)  \\
&\overset{(c)}{=} \prod_{n=1}^N  \left\{\mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right) \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \mid Y^{(n)} = k ; \boldsymbol{\theta}_{k, d}\right) \right\} \\
\end{align*}
$$ (eq:naive-bayes-likelihood-1)

where each $\left(\mathbf{X}^{(n)}, Y^{(n)}\right)$ in equation $(a)$ is a sample from the dataset $\mathcal{D}$
and can be expressed more verbosely as a joint distribution $\left(\mathbf{X}^{(n)}, Y^{(n)}\right) = \left(\mathbf{X}_1^{(n)}, \dots, \mathbf{X}_D^{(n)}, Y^{(n)}\right)$
as in {eq}`eq:joint-distribution`.

Equation $(b)$ is the product of the individual PDF of each sample, where the multiplicand is as in {eq}`eq:joint-distribution`.

Equation $(c)$ is then a consequence of {eq}`eq:conditional-independence-naive-bayes-1`.
```

Then we can maximize

$$
\prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right)
$$ (eq:naive-bayes-likelihood-target)

and

$$
\prod_{n=1}^N  \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \middle \vert Y^{(n)} = k ; \boldsymbol{\theta}_{k, d}\right)
$$ (eq:naive-bayes-likelihood-feature)

individually since the above can be decomposed[^decomposed-likelihood].

```{prf:definition} Log Likelihood Function of Naive Bayes
:label: def:naive-bayes-log-likelihood

For numerical stability, we can take the log of the likelihood function:

$$
\begin{align*}
\log \mathcal{L}(\boldsymbol{\theta} \mid \mathcal{D}) &= \log \mathbb{P}\left(\mathcal{D} ; \left\{\boldsymbol{\pi}, \boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} \right\}\right) \\
\end{align*}
$$

where the log of the product of the individual PDF of each sample is the sum of the log of each PDF. We
will go into that later.
```

Stated formally,

```{prf:definition} Maximize Priors
:label: def:naive-bayes-max-priors

The notation for maximizing the prior probabilities is as follows:

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \arg \max_{\boldsymbol{\pi}} \mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) \\
&= \arg \max_{\boldsymbol{\pi}} \prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right) \\
\end{align*}
$$ (eq:naive-bayes-max-priors)

A reminder that the shape of $\hat{\boldsymbol{\pi}}$ is $K \times 1$.
```

Similarly, we can maximize the likelihood function of the feature parameters:

```{prf:definition} Maximize Feature Parameters
:label: def:naive-bayes-max-feature-params

The notation for maximizing the feature parameters is as follows:

$$
\begin{align*}
\hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y\}} &= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \mathcal{L}(\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} ; \mathcal{D}) \\
&= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \prod_{n=1}^N  \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \mid Y^{(n)} = k ; \boldsymbol{\theta}_{k, d}\right) \\
\end{align*}
$$ (eq:naive-bayes-max-feature-params)

A reminder that the shape of $\hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y\}}$ is $K \times D$.
```

### Estimating Priors

Before we start the formal estimation process, it is intuitive to think that the prior probabilities $\boldsymbol{\pi}_k$ should be proportional to the number of samples in each class. In other words, if we have $N_1$ samples in class 1, $N_2$ samples in class 2, and so on, then we should have

$$
\begin{align*}
\pi_1 &\propto N_1 \\
\pi_2 &\propto N_2 \\
\vdots & \quad \vdots \\
\pi_K &\propto N_K
\end{align*}
$$

For instance, if we have a dataset with $N=100$ samples with $K=3$ classes, and $N_1 = 10$, $N_2 = 30$ and $N_3 = 60$, then we should have $\pi_1 = \frac{10}{100} = 0.1$, $\pi_2 = \frac{30}{100} = 0.3$ and $\pi_3 = \frac{60}{100} = 0.6$. This is just the relative frequency of each class and
seems to be a sensible choice.

It turns out our intuition matches the formal estimation process derived from the maximum likelihood estimation (MLE) principle.

### Maximum Likelihood Estimation for Priors (Categorical Distribution)

We have seen earlier that we can maximize the priors and likelihood (target and feature parameters) separately.

Let's start with the priors. Let's state the expression from {eq}`eq:naive-bayes-max-priors` in definition
{prf:ref}`def:naive-bayes-max-priors` again:

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \arg \max_{\boldsymbol{\pi}} \mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) \\
&= \arg \max_{\boldsymbol{\pi}} \prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right) \\
\end{align*}
$$ (eq:naive-bayes-max-priors-repeated)

We need to write the multiplicand in {eq}`eq:naive-bayes-max-priors-repeated` in terms of
the PDF of the Category distribution, as decribed in {eq}`eq:categorical-distribution-bishop`.
Extending from {eq}`eq:naive-bayes-max-priors-repeated`, we have:

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \arg \max_{\boldsymbol{\pi}} \mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) \\
&= \arg \max_{\boldsymbol{\pi}} \prod_{n=1}^N  \mathbb{P}\left(Y^{(n)}=k ; \boldsymbol{\pi}\right) \\
&\overset{\mathrm{(a)}}{=} \arg \max_{\boldsymbol{\pi}} \prod_{n=1}^N  \left(\prod_{k=1}^K \pi_k^{y^{(n)}_k}\right) \\
\end{align*}
$$ (eq:naive-bayes-max-priors-2)

where $\left(\prod_{k=1}^K \pi_k^{y^{(n)}_k} \right)$ in equation $(a)$ is a consequence
of the definition of the Category distribution in {prf:ref}`categorical-multinomial-distribution`.

Subsequently, knowing maximizing the log likelihood is the same as maximizing the likelihood, we have:

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \arg \max_{\boldsymbol{\pi}} \mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) \\
&= \arg \max_{\boldsymbol{\pi}} \log \mathcal{L}(\boldsymbol{\pi} ; \mathcal{D}) \\
&\overset{\mathrm{(b)}}{=} \arg \max_{\boldsymbol{\pi}} \sum_{n=1}^N \log \left(\prod_{k=1}^K \pi_k^{y^{(n)}_k} \right) \\
&\overset{\mathrm{(c)}}{=} \arg \max_{\boldsymbol{\pi}} \sum_{n=1}^N \sum_{k=1}^K y^{(n)}_k \log \pi_k \\
&\overset{\mathrm{(d)}}{=} \arg \max_{\boldsymbol{\pi}} \sum_{k=1}^K N_k \log \pi_k \\
\end{align*}
$$ (eq:naive-bayes-max-priors-3)

where $N_k$ is the number of samples that belong to the $k$-th category.

```{prf:remark} Notation Overload
:label: notation-overload

We note to ourselves that we are reusing, and hence abusing the notation $\mathcal{L}$ for the log-likelihood function to be the same as the likelihood function, this is just for the ease of re-defining a new symbol for the log-likelihood function, $\log \mathcal{L}$.
```

Equation $(b)$ is derived because placing the logarithm outside the product is equivalent to summing the logarithms of the terms in the product.

Equation $(d)$ is derived by expanding equation $(c)$,

$$
\begin{align*}
\sum_{n=1}^N \sum_{k=1}^K y^{(n)}_k \log \pi_k &= \sum_{n=1}^N \left( \sum_{k=1}^K y^{(n)}_k \log \pi_k \right) \\
&= y^{(1)}_1 \log \pi_1 + y^{(1)}_2 \log \pi_2 + \dots + y^{(1)}_K \log \pi_K \\
&+ y^{(2)}_1 \log \pi_1 + y^{(2)}_2 \log \pi_2 + \dots + y^{(2)}_K \log \pi_K \\
&+ \qquad \vdots \qquad \\
&+ y^{(N)}_1 \log \pi_1 + y^{(N)}_2 \log \pi_2 + \dots + y^{(N)}_K \log \pi_K \\
&\overset{(e)}{=} \left( y^{(1)}_1 + y^{(2)}_1 + \dots + y^{(N)}_1 \right) \log \pi_1 \\
&+ \left( y^{(1)}_2 + y^{(2)}_2 + \dots + y^{(N)}_2 \right) \log \pi_2 \\
&+ \qquad \vdots \qquad \\
&+ \left( y^{(1)}_K + y^{(2)}_K + \dots + y^{(N)}_K \right) \log \pi_K \\
&\overset{(f)}{=} N_1 \log \pi_1 + N_2 \log \pi_2 + \dots + N_K \log \pi_K \\
&= \sum_{k=1}^K N_k \log \pi_k \\
\end{align*}
$$

where $(e)$ is derived by summing each column, and $N_k = y^{(1)}_k + y^{(2)}_k + \dots + y^{(N)}_k$
is nothing but the number of samples that belong to the $k$-th category. One just need to recall that
if we have say 6 samples of class $(0, 1, 2, 0, 1, 1)$ where $K=3$, then the one-hot encoded
representation of the samples will be

$$
\begin{align*}
\left[
\begin{array}{ccc}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 1 & 0 \\
\end{array}
\right]
\end{align*}
$$

and summing each column will give us $N_1 = 2$, $N_2 = 3$, and $N_3 = 1$.

Now we are finally ready to solve the estimation (optimization) problem for $\boldsymbol{\pi}$.

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \sum_{k=1}^K N_k \log \pi_k \\
\end{align*}
$$ (eq:naive-bayes-max-priors-4)

subject to the constraint that

$$
\sum_{k=1}^K \pi_k = 1
$$ (eq:naive-bayes-max-priors-constraint)

which is just saying the probabilities must sum up to 1.

We can also write the expression as

$$
\begin{aligned}
\max_{\boldsymbol{\pi}} &~~ \sum_{k=1}^K N_k \log \pi_k \\
\text{subject to} &~~ \sum_{k=1}^K \pi_k = 1
\end{aligned}
$$

This is a constrained optimization problem, and we can solve it using the Lagrangian method.

```{prf:definition} Lagrangian Method
:label: lagrangian-method

The Lagrangian method is a method to solve constrained optimization problems. The idea is to
convert the constrained optimization problem into an unconstrained optimization problem by
introducing a Lagrangian multiplier $\lambda$ and then solve the unconstrained optimization
problem.

Given a function $f(\mathrm{x})$ and a constraint $g(\mathrm{x}) = 0$, the Lagrangian function,
$\mathcal{L}(\mathrm{x}, \lambda)$ is defined as

$$
\begin{align*}
\mathcal{L}(\mathrm{x}, \lambda) &= f(\mathrm{x}) - \lambda g(\mathrm{x}) \\
\end{align*}
$$

where $\lambda$ is the Lagrangian multiplier and may be either positive or negative. Then,
the critical points of the Lagrangian function are the same as the critical points of the
original constrained optimization problem, i.e. setting the gradient vector of the Lagrangian
function $\nabla \mathcal{L}(\mathrm{x}, \lambda) = 0$ with respect to $\mathrm{x}$ and $\lambda$.
```

One note is that the notation of $\mathcal{L}$ seems to be overloaded again with the Lagrangian function,
we will have to change it to $\mathcal{L}_\lambda$ to avoid confusion. So, to reiterate, solving the Lagrangian function is equivalent to solving the constrained optimization problem.

In our problem, we can convert it to Lagrangian form as

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D}) \\
&= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \underbrace{\mathcal{L}(\boldsymbol{\pi} ; \mathcal{D})}_{f(\boldsymbol{\pi})} - \lambda \left(\underbrace{\sum_{k=1}^K \pi_k - 1}_{g(\boldsymbol{\pi})} \right) \\
&= \underset{\boldsymbol{\pi}}{\mathrm{argmax}} ~~ \sum_{k=1}^K N_k \log \pi_k - \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) \\
\end{align*}
$$

which is now an unconstrained optimization problem. Note that we used subtraction instead of addition form of the
Lagrangian function, so that we can frame it as a maximization problem (i.e. we want to reduce the "additional cost"
$\lambda$, which is a positive number, so if we add it, the expression will become a min-max problem,
we can just put a minus sign, so it become a max-max problem).

We can now solve it by setting the gradient vector of the Lagrangian function

$$
\nabla \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D}) = 0
$$ (eq:naive-bayes-max-priors-lagrangian-1)

with respect to $\boldsymbol{\pi}$ and $\lambda$, as follows,

$$
\begin{align*}
\nabla \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D}) &\overset{\mathrm{def}}{=} \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \boldsymbol{\pi}} = 0 \quad \text{and} \quad \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \lambda} = 0 \\
&\iff \frac{\partial}{\partial \boldsymbol{\pi}} \left( \sum_{k=1}^K N_k \log \pi_k - \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) \right) = 0 \quad \text{and} \quad \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \lambda} = 0 \\
\\
&\iff \begin{bmatrix} \frac{\partial \mathcal{L}_\lambda}{\partial \pi_1} \\ \vdots \\ \frac{\partial \mathcal{L}_\lambda}{\partial \pi_K} \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \end{bmatrix} \quad \text{and} \quad \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \lambda} = 0 \\
&\iff \begin{bmatrix} \frac{\partial}{\partial \pi_1} \left( N_1 \log \pi_1 - \lambda \left( \pi_1 - 1 \right) \right) \\ \vdots \\ \frac{\partial}{\partial \pi_K} \left( N_K \log \pi_K - \lambda \left( \pi_K - 1 \right) \right) \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \end{bmatrix} \quad \text{and} \quad \frac{\partial \mathcal{L}_\lambda(\boldsymbol{\pi}, \lambda ; \mathcal{D})}{\partial \lambda} = 0 \\
&\iff \begin{bmatrix} \frac{N_1}{\pi_1} - \lambda \\ \vdots \\ \frac{N_K}{\pi_K} - \lambda \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \end{bmatrix} \quad \text{and} \quad \sum_{k=1}^K \pi_k - 1 = 0 \\
\end{align*}
$$ (eq:naive-bayes-max-priors-lagrangian-2)

The reason we can unpack $\frac{\partial}{\partial \pi_k}\left( \sum_{k=1}^K N_k \log \pi_k - \lambda \left( \sum_{k=1}^K \pi_k - 1 \right) \right)$ as $\frac{\partial}{\partial \pi_k} \left( N_k \log \pi_k - \lambda \left( \pi_k - 1 \right) \right)$ is because we are dealing with partial derivatives, so other terms other than $\pi_k$ are constant.

Finally, we have a system of equations for each $\pi_k$ and if we can solve for $\pi_k$ for each $k$, we can then find the best estimate of $\boldsymbol{\pi}$. It turns out to solve for $\pi_k$, we have to find $\lambda$ first, and this can be solved by setting $\sum_{k=1}^K \pi_k - 1 = 0$ and solving for $\lambda$, which is the last equation in the system of equations above. We first express each $\pi_k$ in terms of $\lambda$,

$$
\begin{align*}
\frac{N_1}{\pi_1} - \lambda &= 0 \implies \pi_1 = \frac{N_1}{\lambda} \\
\frac{N_2}{\pi_2} - \lambda &= 0 \implies \pi_2 = \frac{N_2}{\lambda} \\
\vdots \\
\frac{N_K}{\pi_K} - \lambda &= 0 \implies \pi_K = \frac{N_K}{\lambda} \\
\end{align*}
$$

Then we substitute these expressions into the last equation in the system of equations above, and solve for $\lambda$,

$$
\begin{align*}
\sum_{k=1}^K \pi_k - 1 = 0 &\implies \sum_{k=1}^K \frac{N_k}{\lambda} - 1 = 0 \\
&\implies \sum_{k=1}^K \frac{N_k}{\lambda} = 1 \\
&\implies \sum_{k=1}^K N_k = \lambda \\
&\implies \lambda = \sum_{k=1}^K N_k \\
&\implies \lambda = N \\
\end{align*}
$$

and therefore, we can now solve for $\pi_k$,

$$
\boldsymbol{\hat{\pi}} = \begin{bmatrix}
\pi_1 = \frac{N_1}{N} \\ \pi_2 = \frac{N_2}{N} \\ \vdots \\ \pi_K = \frac{N_K}{N}
\end{bmatrix}_{K \times 1}
\implies \pi_k = \frac{N_k}{N} \quad \text{for} \quad k = 1, 2, \ldots, K
$$

We conclude that the maximum likelihood estimate of $\boldsymbol{\pi}$
is the same as the empirical relative frequency of each class in the training data. This coincides with our intuition.

For completeness of expression,

$$
\begin{align*}
\hat{\boldsymbol{\pi}} &= \arg \max_{\boldsymbol{\pi}} \mathcal{L}\left( \boldsymbol{\pi} ; \mathcal{D} \right) \\
&= \begin{bmatrix} \hat{\pi}_1 \\ \vdots \\ \hat{\pi}_K \end{bmatrix} \\
&= \begin{bmatrix} \frac{N_1}{N} \\ \vdots \\ \frac{N_K}{N} \end{bmatrix}
\end{align*}
$$ (eq:naive-bayes-max-priors-final)

### Estimating Likelihood (Gaussian Version)

Intuition: The likelihood parameters are the mean and variance of each feature for each class.

#### Maximum Likelihood Estimate for Likelihood (Continuous Feature Parameters)

Now that we have found the maximum likelihood estimate for the prior probabilities,
we now find the maximum likelihood estimate for the likelihood parameters.

Let's look at the expression {eq}`eq:naive-bayes-max-feature-params` from {prf:ref}`def:naive-bayes-max-feature-params` again:

$$
\begin{align*}
\hat{\boldsymbol{\theta}}_{\{\mathbf{X} \mid Y\}} &= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \mathcal{L}(\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} ; \mathcal{D}) \\
&= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \prod_{n=1}^N  \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \middle \vert Y^{(n)} = k ; \boldsymbol{\theta}_{k, d} \right) \\
\end{align*}
$$ (eq:naive-bayes-max-feature-params-repeated)

We will give a formulation for the case when all features $X_d$ are continuous. As mentioned
in {ref}`continuous-features-gaussian-distribution`, we will assume that the features $X_d$ given class $Y=k$
are distributed according to a Gaussian distribution.

```{admonition} Hand Wavy
:class: warning

This section will be a bit hand wavy as I did not derive it by hand, but one just need to remember we need
to find a total of $K \times D$ parameters. Of course, in the case of Gaussian distribution, that means
we need to find a total of $K \times D \times 2$ parameters, where the $2$ comes from the mean and variance.
```

Before we write the multiplicand in {eq}`eq:naive-bayes-max-feature-params-repeated` in terms of the PDF
of the Gaussian distribution, we will follow Kevin Murphy's method (pp 329) and represent

$$
\begin{align*}
\arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \prod_{n=1}^N  \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \middle \vert Y^{(n)} = k ; \boldsymbol{\theta}_{k, d} \right) &= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \prod_{n=1}^N  \prod_{k=1}^K \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \middle \vert Y^{(n)} = k ; \boldsymbol{\theta}_{k, d} \right)^{I\left\{ Y^{(n)} = k \right\}} \\
\end{align*}
$$ (eq:naive-bayes-max-feature-params-kevin-murphy-1)

Then he applied the log function to both sides of {eq}`eq:naive-bayes-max-feature-params-kevin-murphy-1`,

$$
\begin{align*}
\arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \log \left( \prod_{n=1}^N  \prod_{k=1}^K \prod_{d=1}^D \mathbb{P}\left(X_d^{(n)} \middle \vert Y^{(n)} = k ; \boldsymbol{\theta}_{k, d} \right)^{I\left\{ Y^{(n)} = k \right\}} \right) &= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \sum_{n=1}^N  \sum_{k=1}^K \sum_{d=1}^D I\left\{ Y^{(n)} = k \right\} \log \left( \mathbb{P}\left(X_d^{(n)} \middle \vert Y^{(n)} = k ; \boldsymbol{\theta}_{k, d} \right) \right) \\
&= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \sum_{k=1}^K \sum_{d=1}^D \left [\sum_{n=1: Y^{(n)} = k}^N \log \left(\mathbb{P}\left(X_d^{(n)} \middle \vert Y = k ; \boldsymbol{\theta}_{k, d} \right) \right)\right]\\
\end{align*}
$$ (eq:naive-bayes-max-feature-params-kevin-murphy-2)

where the notation $n=1: Y^{(n)} = k$ means that we are summing over all $n$ where $Y^{(n)} = k$. In other words,
we are looking at all the data points where the class label is $k$.

We can further simplify {eq}`eq:naive-bayes-max-feature-params-kevin-murphy-2` as:

$$
\begin{align*}
\arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \sum_{k=1}^K \sum_{d=1}^D \left [\sum_{n=1: Y^{(n)} = k}^N \log \left(\mathbb{P}\left(X_d^{(n)} \middle \vert Y = k ; \boldsymbol{\theta}_{k, d} \right) \right)\right] &= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \sum_{k=1}^K \sum_{d=1}^D \log \mathbb{P}\left(\mathcal{D}_{dk} ; \boldsymbol{\theta}_{k, d} \right) \\
&= \arg \max_{\boldsymbol{\theta}_{k, d}} \log \mathbb{P}\left(\mathcal{D}_{11} ; \boldsymbol{\theta}_{1, 1} \right) + \log \mathbb{P}\left(\mathcal{D}_{12} ; \boldsymbol{\theta}_{1, 2} \right) + \cdots + \log \mathbb{P}\left(\mathcal{D}_{1D} ; \boldsymbol{\theta}_{1, D} \right) + \log \mathbb{P}\left(\mathcal{D}_{21} ; \boldsymbol{\theta}_{2, 1} \right) + \log \mathbb{P}\left(\mathcal{D}_{22} ; \boldsymbol{\theta}_{2, 2} \right) + \cdots + \log \mathbb{P}\left(\mathcal{D}_{2D} ; \boldsymbol{\theta}_{2, D} \right) + \cdots + \log \mathbb{P}\left(\mathcal{D}_{K1} ; \boldsymbol{\theta}_{K, 1} \right) + \log \mathbb{P}\left(\mathcal{D}_{K2} ; \boldsymbol{\theta}_{K, 2} \right) + \cdots + \log \mathbb{P}\left(\mathcal{D}_{KD} ; \boldsymbol{\theta}_{K, D} \right) \\
\end{align*}
$$ (eq:naive-bayes-max-feature-params-kevin-murphy-3)

where $\mathcal{D}_{dk}$ is the data set of all the data points of feature $d$ and class $k$.

Now we can ***individually maximize*** the parameters of each feature and class pair in {eq}`eq:naive-bayes-max-feature-params-kevin-murphy-3`, i.e. estimate $\mathcal{D}_{dk}$ for each $d$ and $k$.

```{prf:example} Example on Feature 1 and Class 2
:label: example-naive-bayes-feature-1-class-2

For example, $\mathcal{D}_{12}$ refers to all the data points of feature $1$ and class $2$ and we can
maximize the parameters of this data set $\mathcal{D}_{12}$ in a similar vein from {prf:ref}`def:naive-bayes-max-feature-params`,
but now instead of
multiplying the probabilities, we are summing the log probabilities.

$$
\begin{align*}
\arg \max_{\theta_{1, 2}} \log \mathbb{P}\left(\mathcal{D}_{12} ; \theta_{1, 2} \right) &= \arg \max_{\theta_{1, 2}} \sum_{n=1: Y^{(n)} = 2}^N \log \left(\mathbb{P}\left(X_1^{(n)} \middle \vert Y = 2 ; \theta_{1, 2} \right) \right) \\
\end{align*}
$$

where we will attempt to find the best estimate $\theta_{1, 2} = \left(\mu_{2, 1}, \sigma_{2, 1} \right)$ for the parameters of the Gaussian distribution of feature $1$ and class $2$.

It turns out that the maximum likelihood estimate for the parameters of the Gaussian distribution is the sample mean and sample variance of the data set $\mathcal{D}_{12}$.

$$
\begin{align*}
\hat{\theta}_{1, 2} := \arg \max_{\theta_{1, 2}} \log \mathbb{P}\left(\mathcal{D}_{12} ; \theta_{1, 2} \right) &= \begin{bmatrix} \hat{\mu_{2, 1}} \\ \hat{\sigma}_{2, 1} \end{bmatrix} \\
&= \begin{bmatrix} \frac{1}{N_2} \sum_{n=1}^{N_2} x_1^{(n)} \\ \sqrt{\frac{1}{N_2} \sum_{n=1}^{N_2} \left( x_1^{(n)} - \hat{\mu}_{2, 1} \right)^2} \end{bmatrix} \\
&= \begin{bmatrix} \bar{x}_{2, 1} \\ s_{2, 1} \end{bmatrix} \\
\end{align*}
$$
```

Now for the general form

$$
\begin{align*}
\hat{\theta}_{k, d} := \arg \max_{\theta_{k, d}} \log \mathbb{P}\left(\mathcal{D}_{dk} ; \theta_{k, d} \right) &= \begin{bmatrix} \hat{\mu}_{k, d} \\ \hat{\sigma}_{k, d} \end{bmatrix} \\
&= \begin{bmatrix} \frac{1}{N_k} \sum_{n=1}^{N_k} x_d^{(n)} \\ \sqrt{\frac{1}{N_k} \sum_{n=1}^{N_k} \left( x_d^{(n)} - \hat{\mu}_{k, d} \right)^2} \end{bmatrix} \\
&= \begin{bmatrix} \bar{x}_{k, d} \\ s_{k, d} \end{bmatrix} \\
\end{align*}
$$

where $\bar{x}_{k, d}$ is the sample mean of the data set $\mathcal{D}_{dk}$ and $s_{k, d}$ is the sample standard deviation of the data set $\mathcal{D}_{dk}$.

For completeness, the parameter matrix $\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}$ defined in {eq}`eq:naive-bayes-estimation-2` becomes:

$$
\begin{align*}
\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}} &= \begin{bmatrix} \boldsymbol{\theta}_{11} & \boldsymbol{\theta}_{12} & \cdots & \boldsymbol{\theta}_{1D} \\ \boldsymbol{\theta}_{21} & \boldsymbol{\theta}_{22} & \cdots & \boldsymbol{\theta}_{2D} \\ \vdots & \vdots & \ddots & \vdots \\ \boldsymbol{\theta}_{K1} & \boldsymbol{\theta}_{K2} & \cdots & \boldsymbol{\theta}_{KD} \end{bmatrix} \\
&= \begin{bmatrix} \left(\bar{x}_{1, 1}, s_{1, 1} \right) & \left(\bar{x}_{1, 2}, s_{1, 2} \right) & \cdots & \left(\bar{x}_{1, D}, s_{1, D} \right) \\ \left(\bar{x}_{2, 1}, s_{2, 1} \right) & \left(\bar{x}_{2, 2}, s_{2, 2} \right) & \cdots & \left(\bar{x}_{2, D}, s_{2, D} \right) \\ \vdots & \vdots & \ddots & \vdots \\ \left(\bar{x}_{K, 1}, s_{K, 1} \right) & \left(\bar{x}_{K, 2}, s_{K, 2} \right) & \cdots & \left(\bar{x}_{K, D}, s_{K, D} \right) \end{bmatrix} \\
\end{align*}
$$

---

$$
\begin{align*}
&= \arg \max_{\boldsymbol{\theta}_{\{\mathbf{X} \mid Y\}}} \sum_{k=1}^K \sum_{d=1}^D \left [\sum_{n=1: Y^{(n)} = k}^N \log \left(\frac{1}{\sqrt{2 \pi \sigma_{k, d}^2}} \exp \left( -\frac{1}{2 \sigma_{k, d}^2} \left( X_d^{(n)} - \mu_{k, d} \right)^2 \right) \right)\right] \\
\end{align*}
$$

See derivations from section 4.2.5 and 4.2.6 of Probabilistic Machine Learning: An Introduction by Kevin Murphy
for the univariate and multivariate Gaussian case respectively.

## Decision Boundary

```{figure} ./assets/cs4780_lecture5_naive_bayes.png
---
name: fig_naive_bayes_linear
---
Naive Bayes leads to a linear decision boundary in many common cases. Illustrated here is the case where $P\left(x_\alpha \mid y\right)$ is Gaussian and where $\sigma_{\alpha, c}$ is identical for all $c$ (but can differ across dimensions $\alpha$ ). The boundary of the ellipsoids indicate regions of equal probabilities $P(\mathbf{x} \mid y)$. The red decision line indicates the decision boundary where $P(y=1 \mid \mathbf{x})=P(y=2 \mid \mathbf{x})$.. Image Credit: [CS 4780](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html)
```

Here $\alpha$ is just index $d$.

Suppose that $y_i \in\{-1,+1\}$ and features are multinomial
We can show that

$$
h(\mathbf{x})=\underset{y}{\operatorname{argmax}} P(y) \prod_{\alpha-1}^d P\left(x_\alpha \mid y\right)=\operatorname{sign}\left(\mathbf{w}^{\top} \mathbf{x}+b\right)
$$

That is,

$$
\mathbf{w}^{\top} \mathbf{x}+b>0 \Longleftrightarrow h(\mathbf{x})=+1 .
$$

As before, we define $P\left(x_\alpha \mid y=+1\right) \propto \theta_{\alpha+}^{x_\alpha}$ and $P(Y=+1)=\pi_{+}$:

$$
\begin{aligned}
{[\mathbf{w}]_\alpha } & =\log \left(\theta_{\alpha+}\right)-\log \left(\theta_{\alpha-}\right) \\
b & =\log \left(\pi_{+}\right)-\log \left(\pi_{-}\right)
\end{aligned}
$$

If we use the above to do classification, we can compute for $\mathbf{w}^{\top} \cdot \mathbf{x}+b$
Simplifying this further leads to

$$
\begin{aligned}
& \mathbf{w}^{\top} \mathbf{x}+b>0 \Longleftrightarrow \sum_{\alpha=1}^d[\mathbf{x}]_\alpha \overbrace{\left(\log \left(\theta_{\alpha+}\right)-\log \left(\theta_{\alpha-}\right)\right)}^{\left[\mathbf{w}]_\alpha\right.}+\overbrace{\log \left(\pi_{+}\right)-\log \left(\pi_{-}\right)}^b>0 \quad \text{ (Plugging in definition of w,b.) } \\
& \Longleftrightarrow \exp \left(\sum_{\alpha=1}^d[\mathbf{x}]_\alpha\left(\log \left(\theta_{\alpha+}\right)-\log \left(\theta_{\alpha-}\right)\right)+\log \left(\pi_{+}\right)-\log \left(\pi_{-}\right)\right)>1 \quad \text { (exponentiating both sides) } \\
& \Longleftrightarrow \prod_{\alpha=1}^d \frac{\exp \left(\log \theta_{\alpha+}^{[\mathbf{x}]_\alpha}+\log \left(\pi_{+}\right)\right)}{\exp \left(\log \theta_{\alpha-}^{[\mathbf{x}]_\alpha}+\log \left(\pi_{-}\right)\right)}>1 \quad \text{ (Because $a \log (b)=\log \left(b^a\right)$ and $\exp (a-b)=\frac{e^a}{e^b}$ operations) } \\
& \Longleftrightarrow \prod_{\alpha=1}^d \frac{\theta_{\alpha+}^{[\mathbf{x}]_\alpha} \pi_{+}}{\theta_{\alpha-}^{[\mathbf{x}]_\alpha} \pi_{-}}>1 \quad \text{ (Because $\exp (\log (a))=a$ and $e^{a+b}=e^a e^b$) } \\
& \Longleftrightarrow \frac{\prod_{\alpha=1}^d P\left([\mathbf{x}]_\alpha \mid Y=+1\right) \pi_{+}}{\prod_{\alpha=1}^d P\left([\mathbf{x}]_\alpha \mid Y=-1\right) \pi_{-}}>1 \quad \text{ (Because $P\left([\mathbf{x}]_\alpha \mid Y=-1\right)=\theta_{\alpha-}^{\mathbf{x}]_\alpha}$) } \\
& \Longleftrightarrow \frac{P(\mathbf{x} \mid Y=+1) \pi_{+}}{P(\mathbf{x} \mid Y=-1) \pi_{-}}>1 \quad \text{ (By the naive Bayes assumption.) } \\
& \Longleftrightarrow \frac{P(Y=+1 \mid \mathbf{x})}{P(Y=-1 \mid \mathbf{x})}>1 \quad \text{ (By Bayes rule (the denominator $P(\mathbf{x})$ cancels out, and $\pi_{+}=P(Y=+1)$.)) } \\
& \Longleftrightarrow P(Y=+1 \mid \mathbf{x})>P(Y=-1 \mid \mathbf{x}) \\
& \Longleftrightarrow \underset{y}{\operatorname{argmax}} P(Y=y \mid \mathbf{x})=+1 \quad \text{ (the point x lies on the positive side of the hyperplane iff Naive Bayes predicts +1) } \\
&
\end{aligned}
$$

### Connection with Logistic Regression

In the case of continuous features (Gaussian Naive Bayes), when the variance is independent of the class $\left(\sigma_{\alpha c}\right.$ is identical for all $c$ ), we can show that

$$
P(y \mid \mathbf{x})=\frac{1}{1+e^{-y\left(\mathbf{w}^{\top} \mathbf{x}+b\right)}}
$$

This model is also known as logistic regression. $N B$ and $L R$ produce asymptotically the same model if the Naive Bayes assumption holds.

See more proofs below:
- https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html.
- https://stats.stackexchange.com/questions/142215/how-is-naive-bayes-a-linear-classifier
- Section 9.3.4 of Probabilistic Machine Learning: An Introduction.

## Time and Space Complexity

Let $N$ be the number of training samples, $D$ be the number of features, and $K$ be the number of classes.

During training, the time complexity is $\mathcal{O}(NKD)$ if we are using a brute force approach.
In my [implementation](https://github.com/gao-hongnan/gaohn-probability-stats/blob/naive-bayes/src/generative/naive_bayes/naive_bayes.py),
the main training loop is in `_estimate_prior_parameters` and `_estimate_likelihood_parameters` methods.

In the former, we are looping through the classes $K$ times, but a hidden operation is calculating the
sum of the class counts, which is $\mathcal{O}(N)$, so the time complexity is $\mathcal{O}(NK)$, using the
vectorized operation `np.sum` helps speed up a bit.

In the latter, we are looping through the classes $K$ times, and for each class, we are looping through
the features $D$ times, and in each feature loop, we are calculating the mean and variance of the feature,
so the operation of calculating the mean and variance is $\mathcal{O}(N) + \mathcal{O}(N)$ respectively, bringing the time complexity
to $\mathcal{O}(2NKD) \approx \mathcal{O}(NKD)$. Even though we are using `np.mean` and `np.var` to speed up, the time complexity
for brute force approach is still $\mathcal{O}(NKD)$.

For the space complexity, we are storing the prior parameters and likelihood parameters, which are of size $K$ and $KD$ respectively,
in code, that corresponds to `self.pi` and `self.theta`, so the space complexity is $\mathcal{O}(K + KD) \approx \mathcal{O}(KD)$.

During inference/prediction, the time complexity for predicting one single sample
is $\mathcal{O}(KD)$, because the `predict_one_sample` method primarily calls the `_calculate_posterior` method, which in
turn invokes `_calculate_prior` and `_calculate_joint_likelihood` methods, and the time complexity of these two methods
is $\mathcal{O}(1)$ and $\mathcal{O}(KD)$ respectively. For `_calculate_prior`, it just involves us looking up the
`self.prior` parameter, which is a constant time operation. For `_calculate_joint_likelihood`, it involves us looping through
the class $K$ times and looping through the features $D$ times, so the time complexity is $\mathcal{O}(KD)$, the `mean`
and `var` parameters are now constant time since they are just looked up from `self.theta`. There is however a `np.prod` operation
towards the end, but the overall time complexity should still be in the order of $\mathcal{O}(KD)$.

For the space complexity, besides the stored (not counted) parameters, we are storing the posterior probabilities, which is of size $K$,
in code, that corresponds to `self.posterior`, so the space complexity is $\mathcal{O}(K)$, and if $K$ is small, then the space complexity
is $\mathcal{O}(1)$.

```{list-table} Time Complexity of Naive Bayes
:header-rows: 1
:name: time-complexity-naive-bayes

* - Train
  - Inference
* - $\mathcal{O}(NKD)$
  - $\mathcal{O}(KD)$
```

```{list-table} Space Complexity of Naive Bayes
:header-rows: 1
:name: space-complexity-naive-bayes

* - Train
  - Inference
* - $\mathcal{O}(KD)$
  - $\mathcal{O}(1)$
```

## References

- Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J. "Chapter 22.7 Maximum Likelihood." In Dive into Deep Learning, 2021.
- Chan, Stanley H. "Chapter 8.1. Maximum-Likelihood Estimation." In Introduction to Probability for Data Science, 172-180. Ann Arbor, Michigan: Michigan Publishing Services, 2021
- Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J. "Chapter 22.9 Naive Bayes." In Dive into Deep Learning, 2021.
- Hal Daumé III. "Chapter 9.3. Naive Bayes Models." In A Course in Machine Learning, January 2017.
- Murphy, Kevin P. "Chapter 9.3. Naive Bayes Models." In Probabilistic Machine Learning: An Introduction. Cambridge (Massachusetts): The MIT Press, 2022.
- James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani. "Chapter 4.4.4. Naive Bayes" In An Introduction to Statistical Learning: With Applications in R. Boston: Springer, 2022.
- Mitchell, Tom Michael. Machine Learning. New York: McGraw-Hill, 1997. (His new chapter on Generate and Discriminative Classifiers: Naive Bayes and Logistic Regression)
- Jurafsky, Dan, and James H. Martin. "Chapter 4. Naive Bayes and Sentiment Classification" In Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition. Noida: Pearson, 2022.
- Bishop, Christopher M. "Chapter 4.2. Probabilistic Generative Models." In Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2016

[^likelihood-1]: Not to be confused with the likelihood term $\mathbb{P}(\mathbf{X} \mid Y)$ in Bayes' terminology.
[^2dparameters]: Dive into Deep Learning, Section 22.9, this is only assuming that each feature $\mathbf{x}_d^{(n)}$ is binary, i.e. $\mathbf{x}_d^{(n)} \in \{0, 1\}$.
[^intractable]: Cite Dive into Deep Learning on this. Also, the joint probability is intractable because the number of parameters to estimate is exponential in the number of features. Use binary bits example, see my notes.
[^categorical-distribution]: [Category Distribution](https://en.wikipedia.org/wiki/Categorical_distribution)
[^joint-distribution]: [Joint Probability Distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution#Discrete_case)
[^chain-rule-of-probability]: [Chain Rule of Probability](https://en.wikipedia.org/wiki/Chain_rule_(probability))
[^conditional-independence]: [Conditional Independence](https://en.wikipedia.org/wiki/Conditional_independence)
[^kdparameters]: Probablistic Machine Learning: An Introduction, Section 9.3, pp 328
[^iid-likelihood]: Refer to page 470 of {cite}`chan_2021`. Note that we cannot write it as a product if the data is not independent and identically distributed.
[^iid-tuple]: $\left(\mathbf{X}^{(n)}, Y^{(1)}\right)$ is written as a tuple, when in fact they can be considered 1 single variable.
[^decomposed-likelihood]: Cite Kevin Murphy and Bishop.