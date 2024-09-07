# Moment Generating Function

## Definition

In probability theory and statistics, a moment generating function (MGF) is a function that uniquely determines the probability distribution of a random variable. Specifically, the MGF of a random variable is the expected value of the exponential function raised to a certain power of the random variable.

```{prf:definition} Moment Generating Function
:label: def:moment_generating_function

For any random variable $X$, the moment-generating function (MGF) $M_{X}(s)$ is

$$
M_{X}(s)=\mathbb{E}\left[e^{s X}\right] .
$$

and it is well defined if there exists a positive constant $c$ such that
$M_{X}(s)$ is finite for all $s \in \mathbb{R}$ with $|s| \leq c$ (i.e. $s \in [-c, c]$]).

In other words, the MGF of $X$ is the expected value of the exponential function raised to a certain power of $X$.
```

By the [Law of the Unconscious Statistician](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician), and {prf:ref}`prop_expectation_function_discrete` for the discrete case and {prf:ref}`prop_expectation_function_continuous` for the continuous case, we can easily see the following:

For the discrete case, the MGF is

$$
M_{X}(s)=\sum_{x \in \Omega} e^{s x} p_{X}(x),
$$

whereas in the continuous case, the MGF is

$$
M_{X}(s)=\int_{-\infty}^{\infty} e^{s x} f_{X}(x) d x .
$$

The continuous case should remind us of the definition of a Laplace transform. For any function $f(t)$, the Laplace transform is

$$
\mathcal{L}[f](s)=\int_{-\infty}^{\infty} f(t) e^{s t} d t .
$$

From this perspective, we can interpret the MGF as the Laplace transform of the PDF. The argument $s$ of the output can be regarded as the coordinate in the Laplace space. If $s=-j \omega$, then $M_{X}(j \omega)$ becomes the Fourier transform of the PDF {cite}`chan_2021`.

```{prf:example} Moment Generating Function Example 1.
:label: ex:moment_generating_function_1

Consider a random variable $X$ with three states $0,1,2$ and with probability masses $\frac{2}{6}, \frac{3}{6}, \frac{1}{6}$ respectively. Find the MGF.

---

The moment-generating function is

$$
\begin{aligned}
M_{X}(s)=\mathbb{E}\left[e^{s X}\right] & =e^{s 0} \cdot \frac{2}{6}+e^{s 1} \cdot \frac{3}{6}+e^{s 2} \cdot \frac{1}{6} \\
& =\frac{1}{3}+\frac{e^{s}}{2}+\frac{e^{2 s}}{6} .
\end{aligned}
$$

*Question and solution from page 324 of {cite}`chan_2021`.*
```

```{prf:example} Moment Generating Function Example 2.
:label: ex:moment_generating_function_2

Find the MGF for an exponential random variable.

---

The MGF of an exponential random variable can be found as

$$
M_{X}(s)=\mathbb{E}\left[e^{s X}\right]=\int_{0}^{\infty} e^{s x} \lambda e^{-\lambda x} d x=\int_{0}^{\infty} \lambda e^{(s-\lambda) x} d x=\frac{\lambda}{\lambda-s}, \quad \text { if } \lambda>s
$$

*Question and solution from page 325 of {cite}`chan_2021`.*
```

## Properties

```{prf:theorem} Moment Generating Function Properties
:label: thm:moment_generating_function_properties

The $M G F$ has the properties that

- $M_{X}(0)=1$

- $\left.\frac{d^{k}}{d s^{k}} M_{X}(s)\right|_{s=0}=\mathbb{E}\left[X^{k}\right]$, for any positive integer $k$.
In other words, if $s$ is set to $0$, then the $k$-th derivative of the MGF is the $k$-th moment of $X$.

*Proof can be found in {cite}`chan_2021`, page 325.*
```


```{prf:example} Moments of Bernoulli Random Variable
:label: ex:moment_generating_function_bernoulli

Let $X$ be a Bernoulli random variable with parameter $p$. Find the first two moments using MGF.

---

The MGF of a Bernoulli random variable is

$$
\begin{aligned}
M_{X}(s) & =\mathbb{E}\left[e^{s X}\right] \\
& =e^{s 0} p_{X}(0)+e^{s 1} p_{X}(1) \\
& =(1)(1-p)+\left(e^{s}\right)(p) \\
& =1-p+p e^{s} .
\end{aligned}
$$

The first and the second moment, using the derivative approach, are

$$
\begin{gathered}
\mathbb{E}[X]=\left.\frac{d}{d s} M_{X}(s)\right|_{s=0}=\left.\frac{d}{d s}\left(1-p+p e^{s}\right)\right|_{s=0}=\left.p e^{s}\right|_{s=0}=p, \\
\mathbb{E}\left[X^{2}\right]=\left.\frac{d^{2}}{d s^{2}} M_{X}(s)\right|_{s=0}=\left.\frac{d^{2}}{d s^{2}}\left(1-p+p e^{s}\right)\right|_{s=0}=\left.p e^{s}\right|_{s=0}=p .
\end{gathered}
$$

*Question and solution from page 326 of {cite}`chan_2021`.*
```

The table below lists the MGF of some common discrete and continuous random variables.

$$
\begin{array}{lllll}
\text { Distribution } & \mathrm{PMF} / \mathrm{PDF} & \mathbb{E}[X] & \operatorname{Var}[X] & M_X(s) \\
\hline \text { Bernoulli } & p_X(1)=p \text { and } p_X(0)=1-p & p & p(1-p) & 1-p+p e^s \\
\text { Binomial } & p_X(k)=\left(\begin{array}{l}
n \\
k
\end{array}\right) p^k(1-p)^{n-k} & n p & n p(1-p) & \left(1-p+p e^s\right)^n \\
\text { Geometric } & p_X(k)=p(1-p)^{k-1} & \frac{1}{p} & \frac{1-p}{p^2} & \frac{p e^s}{1-(1-p) e^s} \\
\text { Poisson } & p_X(k)=\frac{\lambda^k e^{-\lambda}}{k !} & \lambda & \lambda & e^{\lambda\left(e^s-1\right)} \\
\text { Gaussian } & f_X(x)=\frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{(x-\mu)^2}{2 \sigma^2}\right\} & \mu & \sigma^2 & \exp \left\{\mu s+\frac{\sigma^2 s^2}{2}\right\} \\
\text { Exponential } & f_X(x)=\lambda \exp \{-\lambda x\} & \frac{1}{\lambda} & \frac{1}{\lambda^2} & \frac{\lambda}{\lambda-s} \\
\text { Uniform } & f_X(x)=\frac{1}{b-a} & \frac{a+b}{2} & \frac{(b-a)^2}{12} & \frac{e^{s b}-e^{s a}}{s(b-a)}
\end{array}
$$

## Why is it useful?

### Moments

The moment generating function of $\mathbf{X}$ provides us with all the moments of $\mathrm{X}$, which is why it is called the moment generating function.

### Uniquely determines the distribution

Moreover, the MGF uniquely determines the distribution, provided it exists. This means that if two random variables have the same MGF, they must have the same distribution. Therefore, finding the MGF of a random variable enables us to determine its distribution. This approach is particularly useful when working with sums of multiple independent random variables. The [proof can be found here](https://stats.stackexchange.com/questions/34956/proof-that-moment-generating-functions-uniquely-determine-probability-distributi#:~:text=If%20both%20moment%2Dgenerating%20functions,same%20theorem%20without%20a%20proof.).

## Further Readings

- Chan, Stanley H. "Chapter 6.1.1. Moment-generating function." In Introduction to Probability for Data Science. Ann Arbor, Michigan: Michigan Publishing Services, 2021.
- Pishro-Nik, Hossein. "Chapter 6.1.3. Moment Generating Functions." In Introduction to Probability, Statistics, and Random Processes. Kappa Research, 2014.