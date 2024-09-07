# Concept

## Random Vectors

In earlier sections we see that if $X$ and $Y$ are two random variables, then their
joint distribution can be represented as $f_{X,Y}(x,y)$. Similarly, if $X_1, X_2, \ldots, X_N$ are
$N$ random variables, then their joint distribution can be represented as

$$
f_{X_1, X_2, \ldots, X_N}(x_1, x_2, \ldots, x_N)
$$

where $x_1, x_2, \ldots, x_N$ are the realizations of the random variables $X_1, X_2, \ldots, X_N$ respectively.

The notation is cumbersome, and when dealing with high-dimensional data, we often package them in
vectors/matrices.

```{prf:definition} Random Vectors
:label: def:random-vector

Let $X_1, X_2, \ldots, X_N$ be $N$ random variables. Then we denote the random vector $\boldsymbol{X}$
as follows:

$$
\boldsymbol{X} = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_N \end{bmatrix}_{N \times 1}
\quad \mapsto \quad
\boldsymbol{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \end{bmatrix}_{N \times 1}
$$

where $\boldsymbol{x}$ is the realization of $\boldsymbol{X}$.

The sample space of $\boldsymbol{X}$ is usually denoted $\mathcal{X} = \mathbb{R}^N$.

Note that the bolded symbol represents a vector, this is not to be confused with the design matrix
$\mathbf{X}$. To avoid confusion, we may redefine the design matrix to be $\mathbf{A}$ or $\mathbf{M}$.
```

```{prf:remark} Random Vectors
:label: rmk:random-vector

Linking back {ref}`imagenet` example in [From Single Variable to Joint Distributions](../from_single_variable_to_joint_distributions.md),
we can treat each image as a random vector $\boldsymbol{X}$, where the sample space is $\mathcal{X} = \mathbb{R}^{3 \times 224 \times 224}= \mathbb{R}^{150528}$.
We now have an image representing a Ferrari, we want to ask the question: what is the probability of drawing a Ferrari from
the sample space? This is equivalent to asking: what is the probability of drawing $x_1$, $x_2$, $x_3$, $\ldots$, $x_{150528}$
***simultaneously*** from the sample space $\mathcal{X}$?
```

## PDF of Random Vectors

```{prf:definition} PDF of Random Vectors
:label: def:pdf-random-vector

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}^{\intercal}_{N \times 1}$ be a random vector with sample space $\mathcal{X}$.

Let $\eventA$ be an event in $\mathcal{X}$. Then the probability of $\eventA$ is defined as

$$
\begin{aligned}
\P\left(\boldsymbol{X} \in \eventA\right) &= \int_{\eventA} f_{\boldsymbol{X}}(\boldsymbol{x}) \, d\boldsymbol{x} \\
&= \int \int \cdots \int f_{X_1, X_2, \ldots, X_N}(x_1, x_2, \ldots, x_N) \, dx_1 \, dx_2 \, \cdots \, dx_N
\end{aligned}
$$

where $f_{\boldsymbol{X}}(\boldsymbol{x})$ is the PDF of $\boldsymbol{X}$.
```

```{prf:definition} Marginal PDF of Random Vectors
:label: def:marginal-pdf-random-vector

Continuing from {prf:ref}`def:pdf-random-vector`, we can find the marginal PDF
of $X_n$ by integrating out the other random variables:

$$
f_{X_n} = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} f_{X_1, X_2, \ldots, X_N}(x_1, x_2, \ldots, x_N) \, dx_1 \, dx_2 \, \cdots \, dx_{n-1} \, dx_{n+1} \, \cdots \, dx_N
$$

where $f_{X_n}$ is the marginal PDF of $X_n$.
```

```{prf:definition} Joint CDF of Random Vectors
:label: def:joint-cdf-random-vector

Continuing from {prf:ref}`def:pdf-random-vector`, we can find the joint CDF as

$$
\begin{aligned}
F_{X_1, X_2, \ldots, X_N}(x_1, x_2, \ldots, x_N) &= \P \left( X_1 \leq x_1, X_2 \leq x_2, \ldots, X_N \leq x_N \right) \\
&= \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_N} f_{X_1, X_2, \ldots, X_N}(x_1, x_2, \ldots, x_N) \, dx_1 \, dx_2 \, \cdots \, dx_N
\end{aligned}
$$

where $F_{X_1, X_2, \ldots, X_N}(x_1, x_2, \ldots, x_N)$ is the joint CDF of $X_1, X_2, \ldots, X_N$.
```

## Independence

Integration gets difficult in high dimensions, one simplification is if the $N$ random variables are independent,
then the joint PDF can be written as the product of the marginal PDFs.

```{prf:definition} Independent Random Vectors
:label: def:independent-random-vector

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}^{\intercal}_{N \times 1}$
be a random vector with sample space $\mathcal{X} = \mathbb{R}^N$.

Then $\boldsymbol{X}$ is said to be independent if

$$
f_{\boldsymbol{X}}(\boldsymbol{x}) = f_{X_1}(x_1) \, f_{X_2}(x_2) \, \cdots \, f_{X_N}(x_N)
$$

where $f_{X_n}$ is the marginal PDF of $X_n$.
```

```{prf:definition} PDF of Independent Random Vectors
:label: def:pdf-independent-random-vector

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}^{\intercal}_{N \times 1}$ be an independent random vector with sample space $\mathcal{X} = \mathbb{R}^N$.

Then the PDF of $\boldsymbol{X}$ can be written as a product of $N$ individual marginal PDFs:

$$
\begin{aligned}
f_{\boldsymbol{X}}(\boldsymbol{x}) &= f_{X_1}(x_1) \, f_{X_2}(x_2) \, \cdots \, f_{X_N}(x_N)
\end{aligned}
$$

and therefore, given $\eventA$, the probability of $\eventA$ is

$$
\begin{aligned}
\P\left(\boldsymbol{X} \in \eventA\right) &= \int_{\eventA} f_{\boldsymbol{X}}(\boldsymbol{x}) \, d\boldsymbol{x} \\
&= \int \int \cdots \int_{\eventA} f_{X_1}(x_1) \, f_{X_2}(x_2) \, \cdots \, f_{X_N}(x_N) \, dx_1 \, dx_2 \, \cdots \, dx_N \\
&\overset{(a)}{=} \int_{\eventA_1} f_{X_1}(x_1) \, dx_1 \, \int_{\eventA_2} f_{X_2}(x_2) \, dx_2 \, \cdots \, \int_{\eventA_N} f_{X_N}(x_N) \, dx_N
\end{aligned}
$$

where $\eventA_n$ is the projection of $\eventA$ onto the $n$th axis.

The last equation $(a)$ only holds if we further assume $\eventA$ is [separable](https://en.wikipedia.org/wiki/Separable_space) {cite}`chan_2021`,
(i.e. $\eventA = \lsq a_1, b_1 \rsq \times \lsq a_2, b_2 \rsq \times \cdots \times \lsq a_N, b_N \rsq$),
then the probability of $\eventA$ is

$$
\begin{aligned}
\P\left(\boldsymbol{X} \in \eventA\right) &= \prod_{n=1}^N \int_{a_n}^{b_n} f_{X_n}(x_n) \, dx_n \\
\end{aligned}
$$
```

```{prf:definition} Joint Expectation of Independent Random Vectors
:label: def:joint-expectation-independent-random-vector

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}^{\intercal}_{N \times 1}$ be an independent random vector with sample space $\mathcal{X} = \mathbb{R}^N$.

Then the joint expectation of $\boldsymbol{X}$ is

$$
\begin{aligned}
\mathbb{E}\left[\boldsymbol{X}\right] &= \mathbb{E}\left[X_1\right] \, \mathbb{E}\left[X_2\right] \, \cdots \, \mathbb{E}\left[X_N\right] \\
&= \mathbb{E}\left[X_1\right] \, \mathbb{E}\left[X_2\right] \, \cdots \, \mathbb{E}\left[X_N\right] \\
\end{aligned}
$$
```

Due to the importance of $\iid$, we will restate it again here:

```{prf:definition} Independent and Identically Distributed (IID)
:label: def_iid_N

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}^{\intercal}_{N \times 1}$ be a random vector with sample space $\mathcal{X} = \mathbb{R}^N$.

Then $X_1, X_2, \ldots, X_N$ are said to be independent and identically distributed (i.i.d.) if the
following two conditions hold:

1. The random variables are **independent** of each other. That is, $\P \lsq X_i = x_i | X_j = x_j, j \neq i \rsq = \P \lsq X_i = x_i \rsq$ for all $i, j$.
2. The random variables have the **same distribution**. That is, $\P \lsq X_1 = x \rsq = \P \lsq X_2 = x \rsq = \ldots = \P \lsq X_N = x \rsq$ for all $x$.
```

## Expectation of Random Vectors

We now define the expectation of a random vector $\boldsymbol{X}$. Note that this is **not**
the same as {prf:ref}`def:joint-expectation-independent-random-vector`, where we are dealing with
the **joint** expectation of a random vector, which returns a scalar.

```{prf:definition} Expectation of Random Vectors
:label: def:expectation-random-vector

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}^{\intercal}_{N \times 1}$ be a random vector with sample space $\mathcal{X} = \mathbb{R}^N$.

Then the expectation of $\boldsymbol{X}$ is defined as

$$
\begin{aligned}
\boldsymbol{\mu} \overset{\text{def}}{=} \mathbb{E}\left[\boldsymbol{X}\right] &= \begin{bmatrix} \mathbb{E}\left[X_1\right] \\ \mathbb{E}\left[X_2\right] \\ \vdots \\ \mathbb{E}\left[X_N\right] \end{bmatrix}_{N \times 1} \\
&= \begin{bmatrix} \mu_1 \\ \mu_2 \\ \vdots \\ \mu_N \end{bmatrix}_{N \times 1}
\end{aligned}
$$

where $\mu_n$ is the expectation of $X_n$.

We also call this the **mean vector**.
```

The below are from {cite}`chan_2021`.

Since the mean vector is a vector of individual elements, we need to compute the marginal PDFs before computing the expectations:

$$
\mathbb{E}[\boldsymbol{X}]=\left[\begin{array}{c}
\mathbb{E}\left[X_1\right] \\
\vdots \\
\mathbb{E}\left[X_N\right]
\end{array}\right]=\left[\begin{array}{c}
\int_{\Omega} x_1 f_{X_1}\left(x_1\right) d x_1 \\
\vdots \\
\int_{\Omega} x_N f_{X_N}\left(x_N\right) d x_N
\end{array}\right],
$$

where the marginal $\mathrm{PDF}$ is determined by

$$
f_{X_n}\left(x_n\right)=\int_{\Omega} f_{\boldsymbol{X}_{\backslash n}}\left(\boldsymbol{x}_{\backslash n}\right) d \boldsymbol{x}_{\backslash n} .
$$

Note that $\boldsymbol{x}_{\backslash n}$ is a vector of all the elements without $x_n$.

In the equation above, $\boldsymbol{x}_{\backslash n}=\left[x_1, \ldots, x_{n-1}, x_{n+1}, \ldots, x_N\right]^T$ contains all the elements without $x_n$. For example, if the PDF is $f_{X_1, X_2, X_3}\left(x_1, x_2, x_3\right)$, then

$$
\mathbb{E}\left[X_1\right]=\int x_1 \underbrace{\int f_{X_1, X_2, X_3}\left(x_1, x_2, x_3\right) d x_2 d x_3}_{f_{X_1}\left(x_1\right)} d x_1 .
$$

Again, this will become tedious when there are many variables.

## Covariance of Random Vectors (Covariance Matrix)

We now define the covariance of a random vector $\boldsymbol{X}$.

```{prf:definition} Covariance Matrix
:label: def:covariance-matrix

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix}^{\intercal}_{N \times 1}$ be a random vector with sample space $\mathcal{X} = \mathbb{R}^N$.

Then the covariance matrix of $\boldsymbol{X}$ is defined as

$$
\boldsymbol{\Sigma} \stackrel{\text { def }}{=} \operatorname{Cov}(\boldsymbol{X})=\left[\begin{array}{cccc}
\operatorname{Var}\left[X_1\right] & \operatorname{Cov}\left(X_1, X_2\right) & \cdots & \operatorname{Cov}\left(X_1, X_N\right) \\
\operatorname{Cov}\left[X_2, X_1\right] & \operatorname{Var}\left[X_2\right] & \cdots & \operatorname{Cov}\left(X_2, X_N\right) \\
\vdots & \vdots & \ddots & \vdots \\
\operatorname{Cov}\left(X_N, X_1\right) & \operatorname{Cov}\left(X_N, X_2\right) & \cdots & \operatorname{Var}\left[X_N\right]
\end{array}\right] .
$$

Another compact way of defining it is

$$
\boldsymbol{\Sigma}=\operatorname{Cov}(\boldsymbol{X})=\mathbb{E}\left[(\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^T\right],
$$

where $\boldsymbol{\mu}=\mathbb{E}[\boldsymbol{X}]$ is the mean vector.
The notation $\boldsymbol{a} \boldsymbol{b}^T$ means the outer product {cite}`chan_2021`
defined as

$$
\boldsymbol{a \boldsymbol { b } ^ { T }}=\left[\begin{array}{c}
a_1 \\
\vdots \\
a_N
\end{array}\right]\left[\begin{array}{lll}
b_1 & \cdots & b_N
\end{array}\right]=\left[\begin{array}{cccc}
a_1 b_1 & a_1 b_2 & \cdots & a_1 b_N \\
\vdots & \vdots & \ddots & \vdots \\
a_N b_1 & a_N b_2 & \cdots & a_N b_N
\end{array}\right]
$$
```

```{prf:theorem} Covariance Matrix of Independent Random Variables
:label: thm:covariance-matrix-independent

If the random variables $X_1, \ldots, X_N$ of the random vector $\boldsymbol{X}$ are independent,
then the covariance matrix $\operatorname{Cov}(\boldsymbol{X})=\boldsymbol{\Sigma}$ is a **diagonal** matrix:

$$
\boldsymbol{\Sigma}=\operatorname{Cov}(\boldsymbol{X})=\left[\begin{array}{cccc}
\operatorname{Var}\left[X_1\right] & 0 & \cdots & 0 \\
0 & \operatorname{Var}\left[X_2\right] & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \operatorname{Var}\left[X_N\right]
\end{array}\right]_{N \times N}
$$

This is in line with the fact that the covariance of two independent
random variables is zero ({prf:ref}`prop:covariance`).
```

