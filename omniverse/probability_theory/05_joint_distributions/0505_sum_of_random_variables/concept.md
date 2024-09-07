# Concept

## Intuition through Convolutions

{cite}`chan_2021` section 5.5.1.

## Convolutions of Random Variables

```{prf:theorem} Convolutions of Random Variables
:label: theorem:convolutions-of-random-variables

Let $X$ and $Y$ be two independent random variables with PDFs $f_X$ and $f_Y$, respectively.
Define $Z = X + Y$ where $Z$ is in itself a random variable.
Then, the PDF of $Z$ is given by

$$
f_Z(z) = \left(f_X \ast f_Y\right)(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) \, \mathrm{d}x
$$

where $\ast$ denotes convolution.
```

(sum-of-common-distributions-via-convolutions)=
## Sum of Common Distribution

The following proofs are from {cite}`chan_2021` section 5.5.3. Sum of common distributions.

```{prf:theorem} Sum of Poisson Random Variables
:label: theorem:sum-of-poisson-random-variables

Let $X_1 \sim \operatorname{Poisson}\left(\lambda_1\right)$ and $X_2 \sim \operatorname{Poisson}\left(\lambda_2\right)$. Then

$$
X_1+X_2 \sim \operatorname{Poisson}\left(\lambda_1+\lambda_2\right)
$$
```

```{prf:proof}
Let us apply the convolution principle.

$$
\begin{aligned}
& p_Y(k)=\mathbb{P}\left[X_1+X_2=k\right] \\
& =\mathbb{P}\left[X_1=\ell \cap X_2=k-\ell\right] \\
& =\sum_{\ell=0}^k \frac{\lambda_1^{\ell} e^{-\lambda_1}}{\ell !} \cdot \frac{\lambda_2^{k-\ell} e^{-\lambda_2}}{(k-\ell) !} \\
& =e^{-\left(\lambda_1+\lambda_2\right)} \sum_{\ell=0}^k \frac{\lambda_1^{\ell}}{\ell !} \cdot \frac{\lambda_2^{k-\ell}}{(k-\ell) !} \\
& =e^{-\left(\lambda_1+\lambda_2\right)} \cdot \frac{1}{k !} \underbrace{\sum_{\ell=0}^k \frac{k !}{\ell !(k-\ell) !} \lambda_1^{\ell} \lambda_2^{k-\ell}}_{=\sum_{\ell=0}^k\left(\begin{array}{l}
k \\
\ell
\end{array}\right) \lambda_1^{\ell} \lambda_2^{k-\ell}} \\
& =\frac{\left(\lambda_1+\lambda_2\right)^k}{k !} e^{-\left(\lambda_1+\lambda_2\right)} \text {, } \\
&
\end{aligned}
$$

where the last step is based on the binomial identity $\sum_{\ell=0}^k\left(\begin{array}{c}k \\ \ell\end{array}\right) a^{\ell} b^{k-\ell}=(a+b)^k$.
```

```{prf:theorem} Sum of Gaussian Random Variables
:label: theorem:sum-of-gaussian-random-variables

Let $X_1$ and $X_2$ be two Gaussian random variables such that

$$
X_1 \sim \operatorname{Gaussian}\left(\mu_1, \sigma_1^2\right) \quad \text { and } \quad X_2 \sim \operatorname{Gaussian}\left(\mu_2, \sigma_2^2\right) .
$$

Then

$$
X_1+X_2 \sim \operatorname{Gaussian}\left(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2\right) .
$$
```

```{prf:proof}
Let us apply the convolution principle.

$$
\begin{aligned}
f_Z(z) & =\int_{-\infty}^{\infty} f_X(t) f_Y(z-t) d t \\
& =\int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{\left(t-\mu_1\right)^2}{2 \sigma^2}\right\} \cdot \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{\left(z-t-\mu_2\right)^2}{2 \sigma^2}\right\} d t \\
& =\frac{1}{\sqrt{2 \pi \sigma^2}} \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{\left(t-\mu_1\right)^2+\left(z-t-\mu_2\right)^2}{2 \sigma^2}\right\} d t .
\end{aligned}
$$

We now complete the square:

$$
\begin{aligned}
\left(t-\mu_1\right)^2+(z & \left.-t-\mu_2\right)^2=\left[t^2-2 \mu_1 t+\mu_1^2\right]+\left[t^2+2 t\left(\mu_2-z\right)+\left(\mu_2-z\right)^2\right] \\
& =2 t^2-2 t\left(\mu_1-\mu_2+z\right)+\mu_1^2+\left(\mu_2-z\right)^2 \\
& =2\left[t^2-2 t \cdot \frac{\mu_1-\mu_2+z}{2}\right]+\mu_1^2+\left(\mu_2-z\right)^2 \\
& =2\left[t-\frac{\mu_1-\mu_2+z}{2}\right]^2-2\left[\frac{\mu_1-\mu_2+z}{2}\right]^2+\mu_1^2+\left(\mu_2-z\right)^2
\end{aligned}
$$

The last term can be simplified to

$$
\begin{aligned}
-2 & {\left[\frac{\mu_1-\mu_2+z}{2}\right]^2+\mu_1^2+\left(\mu_2-z\right)^2 } \\
& =-\frac{\mu_1^2-2 \mu_1\left(\mu_2-z\right)+\left(\mu_2-z\right)^2}{2}+\mu_1^2+\left(\mu_2-z\right)^2 \\
& =\frac{\mu_1^2+2 \mu_1\left(\mu_2-z\right)+\left(\mu_2-z\right)^2}{2}=\frac{\left(\mu_1+\mu_2-z\right)^2}{2} .
\end{aligned}
$$

Substituting these into the integral, we can show that

$$
\begin{aligned}
f_Z(z) & =\frac{1}{\sqrt{2 \pi \sigma^2}} \int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{2\left[t-\frac{\mu_1-\mu_2+z}{2}\right]^2+\frac{\left(\mu_1+\mu_2-z\right)^2}{2}}{2 \sigma^2}\right\} d t \\
& =\frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{\left(\mu_1+\mu_2-z\right)^2}{2\left(2 \sigma^2\right)}\right\} \underbrace{\int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left\{-\frac{\left[t-\frac{\mu_1-\mu_2+z}{2}\right]^2}{\sigma^2}\right\} d t}_{=\frac{1}{\sqrt{2}}} \\
& =\frac{1}{\sqrt{2 \pi(2 \sigma)^2}} \exp \left\{-\frac{\left(\mu_1+\mu_2-z\right)^2}{2\left(2 \sigma^2\right)}\right\} .
\end{aligned}
$$

Therefore, we have shown that the resulting distribution is a Gaussian with mean $\mu_1+\mu_2$ and variance $2 \sigma^2$.
```

````{prf:theorem} Sum of Common Distributions
:label: theorem:sum-of-common-distributions

Let $X_1$ and $X_2$ be two independent random variables that come from the same family of distributions.

Then, the PDF of $X_1 + X_2$ is given by

```{list-table} Sum of Common Distributions
:header-rows: 1
:name: table:sum-of-common-distributions

* - $X_1$
  - $X_2$
  - $X_1 + X_2$
* - $\bern(p)$
  - $\bern(p)$
  - $\binomial(n, p)$
* - $\binomial(n, p)$
  - $\binomial(m, p)$
  - $\binomial(m+n, p)$
* - $\poisson(\lambda_1)$
  - $\poisson(\lambda_2)$
  - $\poisson(\lambda_1 + \lambda_2)$
* - $\exponential(\lambda)$
  - $\exponential(\lambda)$
  - $\operatorname{Erlang}(2, \lambda)$
* - $\gaussian(\mu_1, \sigma_1)$
  - $\gaussian(\mu_2, \sigma_2)$
  - $\gaussian(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$
```

This holds for $N$ random variables as well.
````
