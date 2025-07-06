---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Cumulative Distribution Function

```{contents}
```

The PMF is one way to describe the distribution of a discrete random variable.
As we will see later on, PMF cannot be defined for continuous random variables.
The cumulative distribution function (CDF) of a random variable is another
method to describe the distribution of random variables. The advantage of the
CDF is that it can be defined for any kind of random variable (discrete,
continuous, and mixed) {cite}`pishro-nik_2014`.

The take away lesson here is that CDF is another way to describe the
distribution of a random variable. In particular, in continuous random
variables, we do not have an equivalent of PMF, so we use CDF instead.

## Definition

```{prf:definition} Cumulative Distribution Function
:label: def_cdf

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$ where
$\xi_i \in \R$ for all $i$. Note that $X(\xi_i) = x_i$ for all $i$ where $x_i$ is the state of $X$.

Then the **cumulative distribution function** $\cdf$ is defined as

$$
\cdf(x_k) \overset{\text{def}}{=} \P \lsq X \leq x_k \rsq = \sum_{\ell=1}^k \P \lsq X = x_{\ell} \rsq = \sum_{\ell=1}^k \pmf(x_{\ell})
$$ (eq:def_cdf)

Since $\P \lsq X = x_{\ell} \rsq$ is the probability mass function, we can also replace
the symbol with the $\pmf$ symbol.
```

```{prf:example} CDF
:label: example_cdf

Consider a random variable $X$ with the following probability mass function:

$$
\pmf(x) = \begin{cases}
    \frac{1}{4} & \text{if } x = 0 \\
    \frac{1}{2} & \text{if } x = 1 \\
    \frac{1}{4} & \text{if } x = 4 \\
\end{cases}
$$

Then by definition {prf:ref}`def_cdf`, we have the CDF of $X$ to be computed as:

$$
\begin{align}
    \cdf(0) & = \P \lsq X \leq 0 \rsq = \P \lsq X = 0 \rsq = \frac{1}{4}                                                                           \\
    \cdf(1) & = \P \lsq X \leq 1 \rsq = \P \lsq X = 0 \rsq + \P \lsq X = 1 \rsq = \frac{1}{4} + \frac{1}{2} = \frac{3}{4}                          \\
    \cdf(4) & = \P \lsq X \leq 4 \rsq = \P \lsq X = 0 \rsq + \P \lsq X = 1 \rsq + \P \lsq X = 4 \rsq = \frac{1}{4} + \frac{1}{2} + \frac{1}{4} = 1
\end{align}
$$

Thus, our CDF is given by:

$$
\cdf(x) = \begin{cases}
    \frac{1}{4} & \text{if } x \leq 0 \\
    \frac{3}{4} & \text{if } 0 < x \leq 1 \\
    1          & \text{if } x > 1
\end{cases}
$$
```

```{code-cell} ipython3
:tags: [hide-input]
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

p = np.array([0.25, 0.5, 0.25])
x = np.array([0, 1, 4])
F = np.cumsum(p)
# plot 2 diagrams in one figure
# y axis start from 0 to 1
fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(10, 5))
ax[0].set_ylim(0, 1)
ax[0].set_title("PMF")
ax[0].set_ylabel("Probability")
ax[0].set_xlabel("x")
ax[0].stem(x, p)
ax[0].grid(False)
ax[1].set_ylim(0, 1)
ax[1].set_title("CDF")
ax[1].set_ylabel("Probability")
ax[1].set_xlabel("x")
ax[1].step(x, F)
ax[1].grid(False)
plt.show()
```

## Properties

```{prf:theorem} Properties of CDF
:label: thm_cdf

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$
where $\xi_i \in \R$ for all $i$. Then, the CDF $\cdf$ of $X$ satisfies the following properties:

1. The CDF is a staircase function and is non-decreasing. That is, for any $\xi \in \S$, we have

    $$
    \cdf(x) \leq \cdf(x+1)
    $$

2. The CDF is a probability function.

    $$
    0 \leq \cdf(x) \leq 1
    $$

    In particular, we have the minimum of the CDF is 0 and the maximum is 1 for
    $x = -\infty$ and $x = \infty$ respectively.

3. The CDF is right continuous.
```

## PMF and CDF Conversion

```{prf:theorem} PMF and CDF Conversion
:label: thm_pmf_cdf

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$
where $\xi_i \in \R$ for all $i$. Note that $X(\xi_i) = x_i$ for all $i$ where $x_i$ is the state of $X$.
Then, the PMF of $X$ can be obtained from the CDF by

$$
\pmf(x_k) = \cdf(x_k) - \cdf(x_{k-1})
$$ (eq:pmf_cdf_1)

where $X$ has a countable set of states $\S$.
```
