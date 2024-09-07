# Cumulative Distribution Function

## Definition

```{prf:definition} Cumulative Distribution Function
:label: def_continuous_cdf

Let $\P$ be a probability function defined over the probability space $\pspace$.

Let $X$ be a continuous random variable with sample space $\S = \R$ and let $\pdf$ be its probability density function.

Then the **cumulative distribution function (CDF)** of $X$ is defined as:

$$
\cdf \lpar x \rpar = \P \lsq X \leq x \rsq = \int_{-\infty}^x \pdf \lpar t \rpar \, dt
$$
```


## Properties of CDF

````{prf:proposition} Properties of CDF
:label: prop_continuous_discrete_cdf

Let $X$ be a random variable (either discrete or continuous), then the CDF $\cdf$ of $X$ satisfies the following properties:

1. The CDF is **non-decreasing**.

2. The CDF is a probability function.

    $$
    0 \leq \cdf(x) \leq 1
    $$

    In particular, we have the **minimum** of the CDF is 0 and the **maximum** is 1 for
    $x = -\infty$ and $x = \infty$ respectively.
````

````{prf:proposition} Probability of an Interval
:label: prop_continuous_cdf_interval

Let $X$ be a continuous random variable.

If the CDF $\cdf$ of $X$ is continuous at any $a \leq x \leq b$, then
the probability of an interval $[a, b]$ is given by:

$$
\P \lsq a \leq X \leq b \rsq = \cdf(b) - \cdf(a)
$$
````

````{prf:definition} Left and Right Continuity
:label: def_continuous_cdf_continuity

Let $X$ be a continuous random variable. Then its CDF $\cdf$ is said to be {cite}`chan_2021`:

- **left continuous** if $\cdf$ is continuous at $x=b$ if $\cdf(b) = \cdf(b^{-}) = \lim_{h \to 0} \cdf(b-h)$
- **right continuous** if $\cdf$ is continuous at $x=b$ if $\cdf(b) = \cdf(b^{+}) = \lim_{h \to 0} \cdf(b+h)$
- **continuous** if $\cdf$ is **left continuous** and **right continuous** at $x=b$. This means that

$$
\lim_{h \to 0} \cdf(b-h) = \cdf(b) = \lim_{h \to 0} \cdf(b+h)
$$
````

````{prf:theorem} CDF is Right Continuous
:label: thm_cdf_right_continuous

Let $X$ be a random variable (either discrete or continuous). Then its CDF $\cdf$ is always right continuous.

$$
\cdf(b) = \cdf(b^{+}) = \lim_{h \to 0} \cdf(b+h)
$$
````

````{prf:theorem} Define Probability at a Point
:label: thm_cdf_point

Let $X$ be a random variable (either discrete or continuous), then $\P \lsq X = b \rsq$ is given by

$$
\P \lsq X = b \rsq = \begin{cases}
    \cdf(b) - \cdf(b^{-}) & \text{if } \cdf \text{ is discontinuous at } b \\
    0 & \text{otherwise}
\end{cases}
$$
````

## PDF is Derivative of CDF

We have seen how we can convert a PDF $\pdf$ to a CDF $\cdf$ by integrating the PDF.
We now show how to convert a CDF $\cdf$ to a PDF $\pdf$ by taking the derivative of the CDF.

````{prf:theorem} PDF is Derivative of CDF
:label: thm_pdf_derivative_cdf

By the **Fundamental Theorem of Calculus** defined in {prf:ref}`fundamental_theorem_of_calculus`,
given a cumulative distribution function (CDF) $\cdf$ of a random variable $X$,
we can find its probability density function (PDF) $\pdf$ by taking the derivative of the CDF:

$$
\pdf \lpar x \rpar = \frac{d}{dx} \cdf \lpar x \rpar = \frac{d}{dx} \int_{-\infty}^x \pdf \lpar t \rpar \, dt
$$

**if** $\cdf$ is **differentiable** at $x$. If $\cdf$ is not differentiable at $x=b$, then

$$
\pdf(b) = \P \lsq X = b \rsq = \P \lsq X = b \rsq \delta(x-b)
$$
````

## References and Further Readings

- Chan, Stanley H. "Chapter 4.3. Cumulative Distribution Function." In Introduction to Probability for Data Science, 185-196. Ann Arbor, Michigan: Michigan Publishing Services, 2021.
