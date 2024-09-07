# Moments and Variance

```{contents}
:local:
```

## Notation

```{admonition} Notation
:class: note

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$
unless otherwise stated.
```

## Moments

```{prf:definition} Moments
:label: def_moments

The **$k$-th moment** of $X$ is defined as:

$$
\expectation \lsq X^k \rsq = \sum_{x \in X(\S)} x^k \cdot \P \lsq X = x \rsq
$$

This definition follows from {prf:ref}`prop_expectation_function_discrete` in {doc}`0305_expectation`.

Essentially, this means that the $k$-th moment is the **expectation** of $X^k$.
```

## Variance

```{prf:definition} Variance
:label: def_variance

The **variance** of $X$ is defined as:

$$
\var \lsq X \rsq = \expectation \lsq \lpar X - \mu \rpar^2 \rsq
$$ (eq_var_1)

where $\mu = \expectation \lsq X \rsq$ is the **expectation** of $X$.

We denote $\sigma^2$ as $\var$ for short-hand notation.
```

We also have an equivalent definition of variance, which is more used in
practice.

```{prf:definition} Variance (Alternative)
:label: def_variance_alt

The **variance** of $X$ is defined as:

$$
\var \lsq X \rsq = \expectation \lsq X^2 \rsq - \expectation \lsq X \rsq^2
$$ (eq_var_2)
```

## Standard Deviation

```{prf:definition} Standard Deviation
:label: def_standard_deviation

In the definition of {prf:ref}`def_variance`, we have $\var \lsq X \rsq$ to have
a different unit than $X$. If $X$ is measured in meters, then $\var \lsq X \rsq$
is measured in meters squared. To solve this issue, we define a new measure
called the **standard deviation**, which is the square root of the variance {cite}`pishro-nik_2014`.

$$
\std \lsq X \rsq = \sqrt{\var \lsq X \rsq}
$$ (eq_std_1)
```

## Properties of Moments and Variance

The properties of moments and variance are as follows:

```{prf:property} Scaling
:label: prop_scaling

For any constant $c$, we have:

$$
\expectation \lsq c \cdot X \rsq = c^k \cdot \expectation \lsq X \rsq
$$ (eq_scaling_1)

where $k$ is the order of the moment.
```

```{prf:property} DC Shift
:label: prop_dc_shift

For any constant $c$, we have:

$$
\expectation \lsq (X + c) \rsq = \expectation \lsq X \rsq
$$ (eq_dc_shift_1)

The intuition is that shifting the random variable by a constant does not change
the spread of the random variable.
```

```{prf:property} Linearity
:label: prop_linearity

Combining {prf:ref}`prop_scaling` and {prf:ref}`prop_dc_shift`, we have:

$$
\expectation \lsq a \cdot X + b \rsq = a^k \cdot \expectation \lsq X \rsq
$$ (eq_linearity_1)

where $k$ is the order of the moment.
```

## Concept

```{admonition} Concept
:class: important

- **Variance** is a measure of how spread out a distribution is. More concretely,
it is the expectation of the squared deviation from the expectation of the distribution.
One can think of for every data point in the distribution, how far is each data point from the
expectation (population mean). The variance is the average of these data points (squared to make it positive).

- **Variance** is **deterministic** and is synonymous with **Population Variance**.
- **Sample Variance** is the variance of a **random sample** from the true population, which is a random variable.
```

## References and Further Readings

-   Pishro-Nik, Hossein. "Chapter 3.2.4. Variance." In Introduction to
    Probability, Statistics, and Random Processes, 202-206. Kappa
    Research, 2014.
-   Chan, Stanley H. "Chapter 3.4.4. Momenets and variance." In Introduction to
    Probability for Data Science, 133-136. Ann Arbor, Michigan: Michigan
    Publishing Services, 2021.
