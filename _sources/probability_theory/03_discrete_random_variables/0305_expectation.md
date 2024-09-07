# Expectation

```{contents}
:local:
```

## Definition

```{prf:definition} Expectation
:label: def_discrete_expectation

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a discrete random variable with $\S = \lset \xi_1, \xi_2, \ldots \rset$.

Then the **expectation** of $X$ is defined as:

$$
\expectation(X) = \sum_{x \in X(\S)} x \cdot \P \lsq X = x \rsq
$$
```

## Existence of Expectation

```{prf:theorem} Existence of Expectation
:label: thm_existence_of_expectation_discrete

Let $\P$ be a probability function defined over the probability space $\pspace$.

A discrete random variable $X$ with $\S = \lset \xi_1, \xi_2, \ldots \rset$ has an **expectation** if
and only if it is **absolutely summable**.

That is,

$$
\expectation \lsq \lvert X \rvert \rsq \overset{\text{def}}{=} \sum_{x \in X(\S)} \lvert x \rvert \cdot \P \lsq X = x \rsq < \infty
$$
```

## Properties of Expectation

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a discrete random variable with
$\S = \lset \xi_1, \xi_2, \ldots \rset$.

Then the **_expectation_** of $X$ has the following properties:

```{prf:property} The Law of The Unconscious Statistician
:label: prop_expectation_function_discrete

For any function $g$,

$$
\expectation \lsq g(X) \rsq = \sum_{x \in X(\S)} g(x) \cdot \P \lsq X = x \rsq
$$

This is not a trivial result, [proof](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician)
can be found here.
```

```{prf:property} Linearity
:label: prop_expectation_linearity_discrete

For any constants $a$ and $b$,

$$
\expectation \lsq aX + b \rsq = a \cdot \expectation(X) + b
$$
```

```{prf:property} Scaling
:label: prop_expectation_scaling_discrete

For any constant $c$,

$$
\expectation \lsq cX \rsq = c \cdot \expectation(X)
$$
```

```{prf:property} DC Shift
:label: prop_expectation_dc_shift_discrete

For any constant $c$,

$$
\expectation \lsq X + c \rsq = \expectation(X)
$$
```

```{prf:property} Stronger Linearity
:label: prop_expectation_stronger_linearity_discrete

It follows that for any random variables $X_1$, $X_2$, ..., $X_n$,

$$
\expectation \lsq \sum_{i=1}^n a_i X_i \rsq = \sum_{i=1}^n a_i \cdot \expectation \lsq X_i \rsq
$$
```

## Concept

```{admonition} Concept
:class: important

- **Expectation** is a measure of the mean value of a random variable and is **deterministic**. It
is also synonymous with the **population mean**.

- **Average** is a measure of the average value of a **random sample** from the true population
and is **random**.

- **Average** of a random sample is a random variable and as sample size increases, the **average** of a random sample converges to the **population mean**.
```

## References and Further Readings

-   Pishro-Nik, Hossein. "Chapter 3.2.3. Functions of Random Variables." In
    Introduction to Probability, Statistics, and Random Processes, 199–201.
    Kappa Research, 2014.
-   Chan, Stanley H. "Chapter 3.4. Expectation." In Introduction to Probability
    for Data Science, 125-133. Ann Arbor, Michigan: Michigan Publishing
    Services, 2021.
