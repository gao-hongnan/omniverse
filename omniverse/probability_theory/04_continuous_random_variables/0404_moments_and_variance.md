# Moments and Variance

The moment and variance of a continuous random variable $X$ are similar to the moment and
variance of a discrete random variable in [](../03_discrete_random_variables/0306_moments_and_variance.md),
but they are defined using integrals instead of sums.

```{admonition} Notation
:class: note

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a **continuous random variable** with sample space $\S$ and probability density function $\pdf$.
```

## Moments

```{prf:definition} Moments
:label: def_moments_continuous

The **$k$-th moment** of $X$ is defined as:

$$
\expectation \lsq X^k \rsq = \int_{-\infty}^{\infty} x^k \cdot \pdf \lsq x \rsq \, dx
$$

This definition follows from {prf:ref}`prop_expectation_function_continuous` in {doc}`0403_expectation`.

Essentially, this means that the $k$-th moment is the **expectation** of $X^k$.
```

## Variance

```{prf:definition} Variance
:label: def_variance_continuous

The **variance** of $X$ is defined as:

$$
\var \lsq X \rsq = \expectation \lsq \lpar X - \mu \rpar^2 \rsq = \int_{\S} \lpar x - \mu \rpar^2 \pdf \lpar x \rpar \, dx
$$ (eq_var_continuous_1)

where $\mu = \expectation \lsq X \rsq$ is the **expectation** of $X$.
```

We also have an equivalent definition of variance, which is more used in practice.

```{prf:definition} Variance (Alternative)
:label: def_variance_continuous_alt

The **variance** of $X$ is defined as:

$$
\var \lsq X \rsq = \expectation \lsq X^2 \rsq - \expectation \lsq X \rsq^2
$$ (eq_var_continuous_2)
```

## Concept

````{admonition} Concept
:class: important

- **Variance** is a measure of how spread out a distribution is. More concretely,
it is the expectation of the squared deviation from the expectation of the distribution.
One can think of for every data point in the distribution, how far is each data point from the
expectation (population mean). The variance is the average of these data points (squared to make it positive).

- **Variance** is **deterministic** and is synonymous with **Population Variance**.
- **Sample Variance** is the variance of a **random sample** from the true population, which is a random variable.
````

## References and Further Readings

- Chan, Stanley H. "Chapter 4.2.3. Momenets and variance." In Introduction to Probability for Data Science, 184-185. Ann Arbor, Michigan: Michigan Publishing Services, 2021.