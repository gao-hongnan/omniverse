---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Exponential Distribution

```{contents}
```

## Definition

```{prf:definition} Exponential Distribution (PDF)
:label: def_exponential_distribution_pdf

$X$ is a continuous random variable with an **exponential distribution** if the probability density function is given by:

$$
\pdf(x) = \begin{cases}
    \lambda e^{-\lambda x} & \text{if } x \geq 0 \\
    0 & \text{otherwise}
\end{cases}
$$ (eq:def_exponential_distribution)

where $\lambda > 0$ is the rate parameter (rate of decay).

Some conventions:
1. We write $X \sim \exponential(\lambda)$ to indicate that $X$ has an exponential distribution with rate parameter $\lambda$.
```

```{prf:remark} Exponential Distribution (PDF)
:label: rem_exponential_distribution_pdf

When $\lambda = 0$, we have,

$$
\pdf(x) = \pdf(0) = \lambda e^{-\lambda 0} = \lambda e^0 = \lambda
$$

This means that $\pdf(0)$ will be more than 1 if $\lambda > 0$ {cite}`chan_2021`.
```

```{prf:definition} Exponential Distribution (CDF)
:label: def_exponential_distribution_cdf

If $X$ is a continuous random variable with an exponential distribution with rate parameter $\lambda$, then the CDF is given by integrating the PDF defined in {prf:ref}`def_exponential_distribution_pdf`:

$$
\cdf(x) = \begin{cases}
    0 & \text{if } x < 0 \\
    1 - e^{-\lambda x} & \text{if } x \geq 0
\end{cases}
$$ (eq:def_exponential_distribution_cdf)
```

The PDF and CDF of two exponential distributions are shown below.

```{code-cell} ipython3
:tags: [hide-input]
import sys
from pathlib import Path
parent_dir = str(Path().resolve().parents[2])
sys.path.append(parent_dir)

import numpy as np
import scipy.stats as stats

from omnivault.utils.reproducibility.seed import seed_all
from omnivault.utils.probability_theory.plot import plot_continuous_pdf_and_cdf
seed_all()

# x = np.linspace(0, 10, 5000)
lambda_1, lambda_2 = 0.5, 2
scale_1, scale_2 = 1 / lambda_1, 1 / lambda_2
X1 = stats.expon(scale=scale_1)
X2 = stats.expon(scale=scale_2)

plot_continuous_pdf_and_cdf(X1, 0, 10, title="Exponential$(\lambda = 0.5)$")
plot_continuous_pdf_and_cdf(X2, 0, 10, title="Exponential$(\lambda = 2)$")
```

## Expectation and Variance

```{prf:theorem} Expectation and Variance of Exponential Distribution
:label: thm_exponential_distribution_expectation_variance

If $X$ is a continuous random variable with an exponential distribution with rate parameter $\lambda$, then the expectation and variance are given by:

$$
\expectation \lsq X \rsq = \frac{1}{\lambda} \qquad \text{and} \qquad \var \lsq X \rsq = \frac{1}{\lambda^2}
$$ (eq:thm_exponential_distribution_expectation_variance)
```

## References and Further Readings

Further readings is a must since Professor Chan give many intuition on how
Exponential distribution is used in real life. He also showed how Exponential
distribution is derived from the Poisson distribution.

-   Chan, Stanley H. "Chapter 4.5. Uniform and Exponential Random Variables." In
    Introduction to Probability for Data Science, 205-211. Ann Arbor, Michigan:
    Michigan Publishing Services, 2021.
-   Pishro-Nik, Hossein. "Chapter 4.2.2. Exponential Distribution" In
    Introduction to Probability, Statistics, and Random Processes, 249-252.
    Kappa Research, 2014.
