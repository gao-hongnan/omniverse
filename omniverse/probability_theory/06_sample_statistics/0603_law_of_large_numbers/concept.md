# Concept

## Sample Average

```{prf:definition} Sample Average
:label: def-sample-average

The law of large numbers is a probabilistic statement about the sample average.
Suppose that we have a collection of i.i.d. random variables $X_1, \ldots, X_N$
the sample average of these $N$ random variables is defined as follows:

$$
\bar{X} = \frac{1}{N} \sum_{n=1}^N X_n
$$ (eq:sample-average)
```

```{prf:theorem} Expectation of Sample Average
:label: theorem-expectation-of-sample-average

In itself, the sample average $\bar{X}$ is a random variable. Therefore, we can compute its expectation.

If the random variables $X_1, \ldots, X_N$ are i.i.d. so that they have the same population mean
$\mathbb{E}\left[X_n\right]=\mu$ (for $n=1, \ldots, N$ ), then by the linearity of the expectation,

$$
\mathbb{E}\left[\bar{X}\right]=\frac{1}{N} \sum_{n=1}^N \mathbb{E}\left[X_n\right]=\mu
$$ (eq:expectation-of-sample-average)

Thus, the expectation of the sample average is the same as the expectation of the population
if the random variables are i.i.d.
```

```{prf:theorem} Variance of Sample Average
:label: theorem-variance-of-sample-average

As with any random variables, we can check the uncertainty of the sample average by computing its variance.

If $X_{1}, \ldots, X_{N}$ are i.i.d. random variables with the same variance $\operatorname{Var}\left[X_{n}\right]=\sigma^{2}($ for $n=1, \ldots, N)$, then

$$
\operatorname{Var}\left[\bar{X}\right]=\frac{1}{N^{2}} \sum_{n=1}^{N} \operatorname{Var}\left[X_{n}\right]=\frac{1}{N^{2}} \sum_{n=1}^{N} \sigma^{2}=\frac{\sigma^{2}}{N} .
$$

We easily see that as $N \rightarrow \infty$, the variance of the sample average goes to zero.
```

### Example of Convergence

In the previous section, we see that as $N$ grows, the variance of the sample average goes to zero. In other words, what this really means is as $N$ increases, there will be less deviation
of the sample average from the population mean. Let's see this in action in [the notebook here](convergence.ipynb).

## Weak Law of Large Numbers

```{prf:theorem} Weak Law of Large Numbers
:label: theorem-weak-law-of-large-numbers

Let $X_1, \ldots, X_N$ be $\iid$ random variables with **common** mean $\mu$ and variance $\sigma^2$. Each $X$ is distributed by the same probability distribution $\mathbb{P}$.

Let $\bar{X}$ be the sample average defined in {eq}`eq:sample-average` and $\mathbb{E}[X^2] < \infty$.

Then, for any $\epsilon > 0$, we have

$$
\lim_{N\to\infty}\mathbb{P}\left[\left|\underset{X \sim P}{\mathbb{E}}[X]-\frac{1}{N} \sum_{n=1}^N x_n\right|>\epsilon\right] := \lim_{N\to\infty} \mathbb{P}\left[\left|\mu - \bar{X}\right| > \epsilon\right] = 0
$$ (eq:weak-law-of-large-numbers-1)

This means that

$$
\bar{X} \xrightarrow{p} \mu \quad \text{as } N \to \infty
$$ (eq:weak-law-of-large-numbers-2)
```

In other words, as sample size $N$ grows, the probability that the sample average $\bar{X}$ differs from the population mean $\mu$ by more than $\epsilon$ approaches zero.
Note this is not saying that the *probability* of the difference between the sample average and the population mean is more than epsilon is zero, the expression is the probability that the difference is more than epsilon! So in laymen terms, as $N$ grows, then it is guaranteed
that the difference between the sample average and the population mean is no more than $\epsilon$. This seems strong since $\epsilon$ can be arbitrarily small, but it is still a probability bound.

## Strong Law of Large Numbers

```{prf:theorem} Strong Law of Large Numbers
:label: theorem-strong-law-of-large-numbers

Let $X_1, \ldots, X_N$ be $\iid$ random variables with **common** mean $\mu$ and variance $\sigma^2$.

Let $\bar{X}$ be the sample average defined in {eq}`eq:sample-average` and $\mathbb{E}[X^4] < \infty$.

Then, we have,

$$
\mathbb{P}\left[\lim_{N\to\infty} \bar{X} = \mu\right] = 1
$$ (eq:strong-law-of-large-numbers-1)

This means that

$$
\bar{X} \xrightarrow{\text{a.s.}} \mu \quad \text{as } N \to \infty
$$ (eq:strong-law-of-large-numbers-2)
```


## Further Readings

- [The Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers)
- Chan, Stanley H. "Chapter 6.3. Law of Large Numbers." In Introduction to Probability for Data Science. Ann Arbor, Michigan: Michigan Publishing Services, 2021.
- Pishro-Nik, Hossein. "Chapter 7.1.1. Law of Large Numbers." In Introduction to Probability, Statistics, and Random Processes. Kappa Research, 2014.