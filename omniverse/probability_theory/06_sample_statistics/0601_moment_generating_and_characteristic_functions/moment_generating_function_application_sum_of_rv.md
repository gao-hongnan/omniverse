# Application: Moment Generating Function and the Sum of Random Variables

This section is based on the work of {cite}`chan_2021`.

## Sum of Independent Random Variables

```{prf:theorem} Moment Generating Function of the Sum of Two Independent Random Variables
:label: thm:moment_generating_function_sum_of_2_rv

Let $X$ and $Y$ be independent random variables. Let $Z=X+Y$. Then

$$
M_{Z}(s)=M_{X}(s) M_{Y}(s) .
$$
```

```{prf:proof}
By the definition of MGF, we have that

$$
M_{Z}(s)=\mathbb{E}\left[e^{s(X+Y)}\right] \stackrel{(a)}{=} \mathbb{E}\left[e^{s X}\right] \mathbb{E}\left[e^{s Y}\right]=M_{X}(s) M_{Y}(s),
$$

where (a) is valid because $X$ and $Y$ are independent.
```

```{prf:corollary} Moment Generating Function of the Sum of $N$ Independent Random Variables
:label: cor:moment_generating_function_sum_of_N_rv

Consider independent random variables $X_{1}, \ldots, X_{N}$. Let $Z=\sum_{n=1}^{N} X_{n}$ be the sum of random variables. Then the MGF of $Z$ is

$$
M_{Z}(s)=\prod_{n=1}^{N} M_{X_{n}}(s) \text {. }
$$

If these random variables are further assumed to be identically distributed, the MGF is

$$
M_{Z}(s)=\left(M_{X_{1}}(s)\right)^{N} .
$$
```

```{prf:proof}
This follows immediately from the previous theorem:

$$
M_{Z}(s)=\mathbb{E}\left[e^{s\left(X_{1}+\cdots+X_{N}\right)}\right]=\mathbb{E}\left[e^{s X_{1}}\right] \mathbb{E}\left[e^{s X_{2}}\right] \cdots \mathbb{E}\left[e^{s X_{N}}\right]=\prod_{n=1}^{N} M_{X_{n}}(s) .
$$

If the random variables $X_{1}, \ldots, X_{N}$ are i.i.d., then the product simplifies to

$$
\prod_{n=1}^{N} M_{X_{n}}(s)=\prod_{n=1}^{N} M_{X_{1}}(s)=\left(M_{X_{1}}(s)\right)^{N}
$$
```

## Sum of Common Distributions via MGFs

We have seen earlier in {ref}`sum-of-common-distributions-via-convolutions` that the sum of two random variables with common distributions is also a random variable with a common distribution. In this section, we will use MGF to prove this fact as well.

```{prf:theorem} Sum of Poisson Random Variables is Poisson
:label: thm:moment_generating_function_sum_of_poisson_rv

Let $X_{1}, \ldots, X_{N}$ be a sequence of i.i.d. Poisson random variables with parameter $\lambda$. Let $Z=X_{1}+\cdots+X_{N}$ be the sum. Then $Z$ is a Poisson random variable with parameters $N \lambda$.
```

```{prf:proof}
The MGF of a Poisson random variable is

$$
\begin{aligned}
M_{X}(s)=\mathbb{E}\left[e^{s X}\right] & =\sum_{k=0}^{\infty} e^{s k} \frac{\lambda^{k}}{k !} e^{-\lambda} \\
& =e^{-\lambda} \sum_{k=0}^{\infty} \frac{\left(\lambda e^{s}\right)^{k}}{k !} \\
& =e^{-\lambda} e^{\lambda e^{s}}=e^{\lambda\left(e^{s}-1\right)} .
\end{aligned}
$$

Assume that we have a sum of $N$ i.i.d. Poisson random variables. Then, by the main theorem, we have that

$$
M_{Z}(s)=\left[M_{X}(s)\right]^{N}=e^{N \lambda\left(e^{s}-1\right)} .
$$

Therefore, the resulting random variable $Z$ is a Poisson with parameter $N \lambda$.
```

```{prf:theorem} Sum of Gaussian Random Variables is Gaussian
:label: thm:moment_generating_function_sum_of_gaussian_rv

Let $X_{1}, \ldots, X_{N}$ be a sequence of independent Gaussian random variables with parameters $\left(\mu_{1}, \sigma_{1}^{2}\right), \ldots,\left(\mu_{N}, \sigma_{N}^{2}\right)$. Let $Z=X_{1}+\cdots+X_{N}$ be the sum. Then $Z$ is a Gaussian random variable:

$$
Z=\operatorname{Gaussian}\left(\sum_{n=1}^{N} \mu_{n}, \sum_{n=1}^{N} \sigma_{n}^{2}\right) .
$$
```

```{prf:proof}
We skip the proof of the MGF of a Gaussian. It can be shown that

$$
M_{X}(s)=\exp \left\{\mu s+\frac{\sigma^{2} s^{2}}{2}\right\}
$$

When we have a sequence of Gaussian random variables, then

$$
\begin{aligned}
M_{Z}(s) & =\mathbb{E}\left[e^{s\left(X_{1}+\cdots+X_{N}\right)}\right] \\
& =M_{X_{1}}(s) \cdots M_{X_{N}}(s) \\
& =\left(\exp \left\{\mu_{1} s+\frac{\sigma_{1}^{2} s^{2}}{2}\right\}\right) \cdots\left(\exp \left\{\mu_{N} s+\frac{\sigma_{N}^{2} s^{2}}{2}\right\}\right) \\
& =\exp \left\{\left(\sum_{n=1}^{N} \mu_{n}\right) s+\left(\sum_{n=1}^{N} \sigma_{n}^{2}\right) \frac{s^{2}}{2}\right\}
\end{aligned}
$$

Therefore, the resulting random variable $Z$ is also a Gaussian. The mean and variance of $Z$ are $\sum_{n=1}^{N} \mu_{n}$ and $\sum_{n=1}^{N} \sigma_{n}^{2}$, respectively.
```

## Further Readings

- Chan, Stanley H. "Chapter 6.1.2. Sum of independent variables via MGF." In Introduction to Probability for Data Science. Ann Arbor, Michigan: Michigan Publishing Services, 2021.
- Pishro-Nik, Hossein. "Chapter 6.1.3. Moment Generating Functions." In Introduction to Probability, Statistics, and Random Processes. Kappa Research, 2014.