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

## Joint Expectation

```{prf:definition} Joint Expectation
:label: def:joint_expectation

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

The joint expectation is

$$
\mathbb{E}[X Y]=\sum_{y \in \Omega_Y} \sum_{x \in \Omega_X} x y \cdot p_{X, Y}(x, y)
$$

if $X$ and $Y$ are discrete, or

$$
\mathbb{E}[X Y]=\int_{y \in \Omega_Y} \int_{x \in \Omega_X} x y \cdot f_{X, Y}(x, y) d x d y
$$

if $X$ and $Y$ are continuous. Joint expectation is also called correlation.
```

```{prf:definition} Cosine Dot Product
:label: def:cosine_dot_product

Let $\boldsymbol{x} \in \mathbb{R}^N$ and $\boldsymbol{y} \in \mathbb{R}^N$ be two vectors.

The cosine angle $\cos \theta$ can be defined as

$$
\cos \theta=\frac{\boldsymbol{x}^T \boldsymbol{y}}{\|\boldsymbol{x}\|\|\boldsymbol{y}\|},
$$

where $\|\boldsymbol{x}\|=\sqrt{\sum_{i=1}^N x_i^2}$ is the norm of the vector $\boldsymbol{x}$,
and $\|\boldsymbol{y}\|=\sqrt{\sum_{i=1}^N y_i^2}$ is the norm of the vector $\boldsymbol{y}$.

The inner product $\boldsymbol{x}^T \boldsymbol{y}$ defines the degree of similarity/correlation
between two vectors $\boldsymbol{x}$ and $\boldsymbol{y}$, where the cosine angle $\cos \theta$
is the cosine of the angle between the two vectors $\boldsymbol{x}$ and $\boldsymbol{y}$.
```

```{prf:theorem} Cauchy-Schwarz Inequality
:label: thm:cauchy_schwarz_inequality

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Then we have,

$$
\mathbb{E}[X Y] \leq \mathbb{E}[X^2] \mathbb{E}[Y^2]
$$
```

We can then view the joint expectation as the cosine dot product between the two
random variables. See {cite}`chan_2021` section 5.2.1, page 259-261.

## Covariance and Correlation Coefficient

```{prf:definition} Covariance
:label: def:covariance

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Then the **covariance** of $X$ and $Y$ is defined as,

$$
\operatorname{cov}(X, Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
$$

where $\mu_X=\mathbb{E}[X]$ and $\mu_Y=\mathbb{E}[Y]$ are the mean of $X$ and $Y$ respectively.

Note that if $X = Y$, then $\operatorname{cov}(X, Y)$ can be reduced to the variance of $X$.
Consequently, the covariance is a generalization of the variance.
```

```{prf:theorem} Covariance
:label: thm:covariance

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Then we have,

$$
\operatorname{Cov}(X, Y)=\mathbb{E}[X Y]-\mathbb{E}[X] \mathbb{E}[Y]
$$
```

```{prf:proof}
The proof is relative straightforward, we can just apply the definition of {prf:ref}`def:covariance`:

$$
\begin{aligned}
\operatorname{Cov}(X, Y) &=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)] \\
&=\mathbb{E}[XY - X \mu_Y - Y \mu_X + \mu_X \mu_Y] \\
&=\mathbb{E}[XY] - \mathbb{E}[X \mu_Y] - \mathbb{E}[Y \mu_X] + \mathbb{E}[\mu_X \mu_Y] \\
&=\mathbb{E}[XY] - \mathbb{E}[X] \mathbb{E}[Y] - \mathbb{E}[Y] \mathbb{E}[X] + \mathbb{E}[\mu_X] \mathbb{E}[\mu_Y] \\
&=\mathbb{E}[XY] - \mathbb{E}[X] \mathbb{E}[Y] - \mathbb{E}[X] \mathbb{E}[Y] + \mathbb{E}[X] \mathbb{E}[Y] \\
&=\mathbb{E}[XY] - \mathbb{E}[X] \mathbb{E}[Y]
\end{aligned}
$$
```

```{prf:theorem} Linearity of Covariance
:label: thm:linearity_of_covariance

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Then we have,

$$
\mathbb{E}\left[\alpha X + \beta Y\right]=\alpha \mathbb{E}[X] + \beta \mathbb{E}[Y]
$$

where $\alpha$ and $\beta$ are constants.

And,

$$
\operatorname{Var}(\alpha X + \beta Y)=\alpha^2 \operatorname{Var}(X) + \beta^2 \operatorname{Var}(Y) + 2 \alpha \beta \operatorname{Cov}(X, Y)
$$
```

```{prf:property} Covariance
:label: prop:covariance

For any two random variables $X$ and $Y$ with sample space $\Omega_X$ and $\Omega_Y$ respectively, we
have the following properties:

1. $\operatorname{Cov}(X, Y)=\operatorname{Cov}(Y, X)$
2. $\operatorname{Cov}(X, Y)=0$ if $X$ and $Y$ are independent
3. $\operatorname{Cov}(X, X)=\operatorname{Var}(X)$
```

After we have defined the covariance, we can define the **correlation
coefficient** of $X$ and $Y$ formally below. We can treat the **correlation
coefficient** $\rho$ as the cosine angle of the centralized random variables $X$
and $Y$ {cite}`chan_2021`. Note to fully appreciate why the correlation
coefficient is defined as the cosine angle, one can see the derivation in
{cite}`chan_2021` section 5.2.1, page 259-261.

```{prf:definition} Correlation Coefficient
:label: def:correlation_coefficient

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Then the **correlation coefficient** of $X$ and $Y$ is defined as,

$$
\begin{aligned}
\rho(X, Y) &= \cos \theta \\
&= \frac{\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]}{\sqrt{\mathbb{E}[(X-\mu_X)^2] \mathbb{E}[(Y-\mu_Y)^2]}} \\
&= \frac{\operatorname{Cov}(X, Y)}{\sqrt{\operatorname{Var}(X) \operatorname{Var}(Y)}}
\end{aligned}
$$
```

```{prf:property} Correlation Coefficient
:label: prop:correlation_coefficient

1. $-1 \leq \rho(X, Y) \leq 1$, an immediate consequence of the definition of cosine angle.
2. If $\rho(X, Y)=1$, then $X$ and $Y$ are perfectly positively correlated, in other words, $Y = \alpha X + \beta$ for some constants $\alpha$ and $\beta$, $\alpha > 0$.
3. If $\rho(X, Y)=-1$, then $X$ and $Y$ are perfectly negatively correlated, in other words, $Y = \alpha X + \beta$ for some constants $\alpha$ and $\beta$, $\alpha < 0$.
4. If $\rho(X, Y)=0$, then $X$ and $Y$ are uncorrelated, in other words, or in linear algebra lingo, $X$ and $Y$ are orthogonal and are linearly independent.
5. $\rho(\alpha X + \beta, \gamma Y + \delta) = \rho(X, Y)$, where $\alpha$, $\beta$, $\gamma$, and $\delta$ are constants.
```

## Independence and Correlation Coefficient

```{prf:theorem} Independence and Joint Expectation
:label: thm:independence_and_joint_expectation

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Then we have,

$$
\begin{aligned}
\mathbb{E}[XY] &= \mathbb{E}[X] \mathbb{E}[Y] \\
\end{aligned}
$$
```

```{prf:theorem} Independence and Covariance
:label: thm:independence_and_covariance

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Let the following two statements be:

1. $X$ and $Y$ are independent;
2. $\operatorname{Cov}(X, Y)=0$.

Then statement 1 implies statement 2, but statement 2 does not imply statement 1.
Independence is therefore a stronger condition than correlation {cite}`chan_2021`.

In other words:

- Independence $\implies$ Uncorrelated;
- Uncorrelated $\not\implies$ Independence.
```

## Empirical (Sample) Correlation Coefficient

Everything defined previously is for the population, but we can also define the
correlation coefficient for the sample via estimation.

```{prf:theorem} Empirical Correlation Coefficient
:label: thm:empirical_correlation_coefficient

Given a dataset of $N$ samples and $D=1$ features with a target variable $Y$,

$$
\mathcal{S}_{\{x, y\}} \overset{\mathbf{def}}{=} \left\{\left(x^{(1)}, y^{(1)}), \ldots, (x^{(N)}, y^{(N)}\right)\right\}
$$

where $x^{(n)}$ is the $n$-th sample and $y^{(n)}$ is the $n$-th target variable.

Then the **empirical correlation coefficient** of $X$ and $Y$ is defined as,

$$
\hat{\rho}\left(\mathcal{S}_{\{x, y\}}\right) = \frac{\sum_{n=1}^N (x^{(n)} - \bar{x})(y^{(n)} - \bar{y})}{\sqrt{\sum_{n=1}^N (x^{(n)} - \bar{x})^2 \sum_{n=1}^N (y^{(n)} - \bar{y})^2}}
$$

where $\bar{x} = \frac{1}{N} \sum_{n=1}^N x^{(n)}$ and $\bar{y} = \frac{1}{N} \sum_{n=1}^N y^{(n)}$
are the sample mean of $X$ and $Y$ respectively.

As $N \rightarrow \infty$, $\hat{\rho}\left(\mathcal{S}_{\{x, y\}}\right) \rightarrow \rho(X, Y)$.
```

In order to generate some plots of correlation, we introduce prematurely the
concept of **covariance matrix** in terms of a $2 \times 2$ matrix, this can be
scaled to higher dimensions as well, which we will learn later.

```{prf:definition} Covariance Matrix (2D)
:label: def:covariance_matrix_2d

Let $X$ and $Y$ be two random variables with sample space $\Omega_X$ and $\Omega_Y$ respectively.

Then the **covariance matrix** of $X$ and $Y$ is defined as,

$$
\begin{aligned}
\mathbf{Cov}(\mathbf{X}, \mathbf{Y}) &\overset{\mathbf{def}}{=} \operatorname{Cov}(\mathbf{X}, \mathbf{Y}) \\
&= \begin{bmatrix}
\operatorname{Cov}(X, X) & \operatorname{Cov}(X, Y) \\
\operatorname{Cov}(Y, X) & \operatorname{Cov}(Y, Y)
\end{bmatrix} \\
\end{aligned}
$$
```

In addition, we define a 2D Gaussian distribution categorized by its mean vector
$\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$.

```{prf:definition} Multivariate Gaussian Distribution (2D)
:label: def:multivariate_gaussian_distribution_2d

Let $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ be the mean vector and covariance matrix of a 2D Gaussian distribution respectively.

Then the **multivariate Gaussian distribution** is defined as,

$$
\begin{aligned}
\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2 \pi)^{D/2} \sqrt{\det{\boldsymbol{\Sigma}}}} \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) \\
\end{aligned}
$$
```

We then generate some data from a 2D Gaussian distribution with the following
parameters:

-   A bivariate Gaussian distribution with a correlation coefficient of
    $\rho = 0$ can be generated by mean vector $\boldsymbol{\mu} = [0, 0]^T$ and
    covariance matrix
    $\boldsymbol{\Sigma} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$.
-   A bivariate Gaussian distribution with a correlation coefficient of
    $\rho = 0.5$ can be generated by mean vector $\boldsymbol{\mu} = [0, 0]^T$
    and covariance matrix
    $\boldsymbol{\Sigma} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$.
-   A bivariate Gaussian distribution with a correlation coefficient of
    $\rho = 1$ can be generated by mean vector $\boldsymbol{\mu} = [0, 0]^T$ and
    covariance matrix
    $\boldsymbol{\Sigma} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}$.

```{code-cell} ipython3
:tags: [hide-input]

import sys
from pathlib import Path
parent_dir = str(Path().resolve().parents[3])
sys.path.append(parent_dir)

from omnivault.utils.visualization.style import use_svg_display

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

use_svg_display()

mean = [0, 0]
sigma_1 = [[2, 0], [0, 2]]
sigma_2 = [[2, 1], [1, 2]]
sigma_3 = [[2, 2], [2, 2]]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
x1 = stats.multivariate_normal.rvs(mean, sigma_1, 1000)
x2 = stats.multivariate_normal.rvs(mean, sigma_2, 1000)
x3 = stats.multivariate_normal.rvs(mean, sigma_3, 1000)

rho_1, _ = stats.pearsonr(x1[:, 0], x1[:, 1])
rho_2, _ = stats.pearsonr(x2[:, 0], x2[:, 1])
rho_3, _ = stats.pearsonr(x3[:, 0], x3[:, 1])

for ax, x, rho in zip(axes, [x1, x2, x3], [rho_1, rho_2, rho_3]):
    ax.scatter(x[:, 0], x[:, 1], color="blue", s=10)
    ax.set_title(f"Empirical ρ = {rho:.2f}")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

plt.show()
```

Notice that the empirical correlation coefficient
$\hat{\rho}(\mathcal{S}_{\{x, y\}})$ is close to the true correlation
coefficient $\rho(X, Y)$ with $N$ large.
