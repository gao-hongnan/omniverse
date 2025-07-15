# Mean, Median and Mode

```{contents}
```

## Median

```{prf:definition} Median
:label: probability-theory-median

Let $X$ be a continuous random variable with $P D F f_X$. The
median of $X$ is a point $c \in \mathbb{R}$ such that

$$
\int_{-\infty}^c f_X(x) d x=\int_c^{\infty} f_X(x) d x
$$
```

```{prf:theorem} Median from CDF
:label: probability-theory-median-from-cdf

The median of a random variable $X$ is the point $c$ such that

$$
F_X(c)=\frac{1}{2}
$$
```

## Mode

The mode is the peak of the PDF. We can see this from the definition below.

```{prf:definition} Mode
:label: probability-theory-mode

Let $X$ be a continuous random variable. The mode is the point
$c$ such that $f_X(x)$ attains the maximum:

$$
c=\underset{x \in \Omega}{\operatorname{argmax}} f_X(x)=\underset{x \in \Omega}{\operatorname{argmax}} \frac{d}{d x} F_X(x)
$$
```

Note that the mode of a random variable is not unique, e.g., a mixture of two
identical Gaussians with different means has two modes{cite}`chan_2021`.

## Mean

We have defined the mean as the expectation of $X$. Here, we show how to compute
the expectation from the CDF. To simplify the demonstration, let us first assume
that $X>0$.

```{prf:lemma} Mean from CDF (X > 0)
:label: probability-theory-mean-from-cdf-x-gt-0

Let $X>0$. Then $\mathbb{E}[X]$ can be computed from $F_X$ as

$$
\mathbb{E}[X]=\int_0^{\infty}\left(1-F_X(t)\right) d t
$$
```

```{prf:lemma} Mean from CDF (X < 0)
:label: probability-theory-mean-from-cdf-x-lt-0

Let $X<0$. Then $\mathbb{E}[X]$ can be computed from $F_X$ as

$$
\mathbb{E}[X]=\int_{-\infty}^0 F_X(t) d t
$$
```

```{prf:theorem} Mean from CDF
:label: probability-theory-mean-from-cdf

The mean of a random variable $X$ can be computed from the CDF as

$$
\mathbb{E}[X]=\int_0^{\infty}\left(1-F_X(t)\right) d t-\int_{-\infty}^0 F_X(t) d t .
$$
```

```{admonition} See Also
:class: seealso

See chapter 4.4. of _An Introduction to Probability for Data Science_ by
Chan{cite}`chan_2021` for a more rigorous treatment of the mean, median and
mode.
```

## References and Further Readings

-   Chan, Stanley H. "Chapter 4.4 Median, Mode, and Mean." In Introduction to
    Probability for Data Science, 196-201. Ann Arbor, Michigan: Michigan
    Publishing Services, 2021.
