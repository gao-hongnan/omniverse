# Concept

```{contents}
:local:
```

## Definition

```{prf:definition} Geometric Distribution
:label: def:geo

Let $X$ be a **Geometric random variable**. Then the probability mass function (PMF) of $X$ is given by

$$
\P \lsq X = k \rsq = (1-p)^{k-1} p \qquad \text{for } k = 1, 2, \ldots
$$

where $0 \leq p \leq 1$ is called the geometric parameter.

We write

$$
X \sim \geometric(p)
$$

to say that $X$ is drawn from a geometric distribution with parameter $p$.
```

## Properties

```{prf:property} Expectation of Geometric Distribution
:label: prop:geo_exp

Let $X \sim \geometric(p)$ be a Geometric random variable with parameter $p$. Then the expectation of $X$ is given by

$$
\expectation \lsq X \rsq = \sum_{k=1}^{\infty} k \cdot \P \lsq X = k \rsq = \frac{1}{p}
$$
```

```{prf:property} Variance of Geometric Distribution
:label: prop:geo_var

Let $X \sim \geometric(p)$ be a Geometric random variable with parameter $p$. Then the variance of $X$ is given by

$$
\var \lsq X \rsq = \expectation \lsq X^2 \rsq - \expectation \lsq X \rsq^2 = \frac{1-p}{p^2}
$$
```

## Further Readings

-   Chan, Stanley H. "Chapter 3.5.3. Geometric random variable." In Introduction
    to Probability for Data Science, 149-152. Ann Arbor, Michigan: Michigan
    Publishing Services, 2021.
