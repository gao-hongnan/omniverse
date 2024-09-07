# Expectation

## Definition

```{prf:definition} Expectation
:label: def_continuous_expectation

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a continuous random variable with sample space $\S$
and let $\pdf$ be its probability density function.

Then the **expectation** of $X$ is defined as:

$$
\expectation \lsq X \rsq = \int_{\S} x \cdot \pdf(x) \, dx
$$
```

## Existence of Expectation

As seen in the discrete counterpart in [](../03_discrete_random_variables/0305_expectation.md),
we have a similar result for continuous random variables.

```{prf:theorem} Existence of Expectation
:label: thm_existence_of_expectation_continuous

A continuous random variable $X$ with sample space $\S$ has an **expectation** if
and only if it is [**absolutely integrable**](https://en.wikipedia.org/wiki/Absolutely_integrable_function).

That is,

$$
\expectation \lsq \lvert X \rvert \rsq \overset{\text{def}}{=} \int_{\S} \lvert x \rvert \cdot \pdf(x) \, dx < \infty
$$
```

```{prf:corollary}
:label: cor_continuous_expectation

For any continuous random variable $X$ with sample space $\S$ and probability density function $\pdf$,

$$
\lvert \expectation \lsq X \rsq \rvert \leq \expectation \lsq \lvert X \rvert \rsq
$$
```


## Properties of Expectation

Almost all properties of the discrete counterpart in [](../03_discrete_random_variables/0305_expectation.md)
holds here.

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a continous random variable with sample space $\S$ and probability density function $\pdf$.

Then the ***expectation*** of $X$ has the following properties:

```{prf:property} The Law of The Unconscious Statistician
:label: prop_expectation_function_continuous

For any function $g: \S \to \R$,

$$
\expectation \lsq g(X) \rsq = \int_{\S} g(x) \cdot \pdf(x) \, dx
$$
```

```{prf:property} Linearity
:label: prop_expectation_linearity_continuous

For any constants $a$ and $b$,

$$
\expectation \lsq aX + b \rsq = a \cdot \expectation(X) + b
$$
```

```{prf:property} Scaling
:label: prop_expectation_scaling_continuous

For any constant $c$,

$$
\expectation \lsq cX \rsq = c \cdot \expectation(X)
$$
```

```{prf:property} DC Shift
:label: prop_expectation_dc_shift_continuous

For any constant $c$,

$$
\expectation \lsq X + c \rsq = \expectation(X)
$$
```

```{prf:property} Stronger Linearity
:label: prop_expectation_stronger_linearity_continuous

It follows that for any random variables $X_1$, $X_2$, ..., $X_n$,

$$
\expectation \lsq \sum_{i=1}^n a_i X_i \rsq = \sum_{i=1}^n a_i \cdot \expectation \lsq X_i \rsq
$$
```

## Concept

````{admonition} Concept
:class: important

- **Expectation** is a measure of the mean value of a random variable and is **deterministic**. It
is also synonymous with the **population mean**.

- **Average** is a measure of the average value of a **random sample** from the true population
and is **random**.

- **Average** of a random sample is a random variable and as sample size increases, the **average** of a random sample converges to the **population mean**.
````

## References and Further Readings

- Chan, Stanley H. "Chapter 4.2. Expectation, Moment, and Variance." In Introduction to Probability for Data Science, 180–184. Ann Arbor, Michigan: Michigan Publishing Services, 2021.

