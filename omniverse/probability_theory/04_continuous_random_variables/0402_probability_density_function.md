# Probability Density Function

## Definition

As mentioned in {prf:ref}`def_continuous_random_variable`, a continuous random variable $X$ is defined by its probability density function (PDF) $f(x)$ or its cumulative distribution function (CDF) $F(x)$.

````{prf:definition} Probability Density Function
:label: def_probability_density_function

The **probability density function** (PDF) of a random variable $X$ is a mapping
$f_X(x)$

$$
\begin{align}
    \pdf: X(\S) &\to \R \\
    X(\S) \ni x &\mapsto \pdf(x)
\end{align}
$$

which satisfies the following properties {cite}`chan_2021`:

- **Non-negativity**: $\pdf(x) \geq 0$ for all $x \in \S$.
- **Unity**: $\int_{\S} \pdf(x) \ dx = 1$.
- **Measure of a set**: $\P \lsq \lset x \in A \rset \rsq = \int_A \pdf(x) \ dx$ for all $A \subseteq \S$.
````

````{prf:remark} Probability Density Function
:label: rem_probability_density_function

The probability density function $f(x)$ of a continuous random variable is similar to
the probability mass function $p(x)$ of a discrete random variable, but with two key differences:

1. Unlike the PMF $\pmf(x)$, the PDF $\pdf(x)$ ***is not a probability***.
   The PDF $\pdf(x)$ is a ***density***, which is a **measure** of the probability of a random variable $X$ taking on a value $x$. The higher the density, the more likely it is that $X$ takes on the value $x$ (or a value close to $x$).

2. This means that the PDF $\pdf(x)$ is ***not necessarily bounded*** and can be ***greater than 1***.
````

Notice that the definition of PDF above did not "explicitly" mention the probability of a random variable $X$,
instead it just mentions that the measure of a set to be $\P \lsq \lset x \in A \rset \rsq = \int_A \pdf(x) \ dx$.

The author further mentioned if we are dealing with 1-dimensional data (on the real line), then we can
then give a more intuitive definition of the PDF.

````{prf:definition} Probability Density Function (1-dimensional)
:label: def_probability_density_function_1d

Let $X$ be a continuous random variable on the real line $\R$. The **probability density function** (PDF) of $X$ is a mapping $f_X(x)$

$$
\begin{align}
    \pdf: X(\S) &\to \R \\
    X(\S) \ni x &\mapsto \pdf(x)
\end{align}
$$

that when ***when integrated over an interval $[a, b] \subseteq \R$***, yields
the probability of observing a value of $X$ in the interval $[a, b]$ (i.e $a \leq X \leq b$):

$$
\P \lsq a \leq X \leq b \rsq = \int_a^b \pdf(x) \ dx
$$

Notice that we have replaced $\int_{A}$ with $\int_a^b$ where $A = [a, b]$.
````

````{prf:definition} Zero Measure
:label: def_zero_measure

The probability of a continuous random variable $X$ taking on a value $x$ is zero:

$$
\P \lsq X = x \rsq = 0
$$
````

````{prf:remark} Open Equals Closed Interval
:label: rem_open_equals_closed_interval

By {prf:ref}`def_zero_measure`, all isolated points have zero measure
in the continuous space and therefore the probability of
an open interval $(a, b)$ is equivalent to the probability of a closed interval $[a, b]$.

More concretely, let $\P \lsq [a, b] \rsq = \P \lsq a \leq X \leq b \rsq$, then

$$
\begin{align}
    \P \lsq [a, b] \rsq = \P \lsq (a, b) \rsq = \P \lsq (a, b] \rsq = \P \lsq [a, b) \rsq
\end{align}
$$

This may not hold when the PDF of $\pdf(x)$ has a delta function at $a$ or $b$ {cite}`chan_2021`.
````


## References and Further Readings

- Chan, Stanley H. "Chapter 4.1. Probability Density Function." In Introduction to Probability for Data Science, 172-180. Ann Arbor, Michigan: Michigan Publishing Services, 2021.
