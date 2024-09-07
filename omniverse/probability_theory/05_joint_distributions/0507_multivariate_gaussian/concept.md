# Concept

## Multivariate Gaussian

```{prf:definition} Multivariate Gaussian Distribution
:label: def:multivariate_gaussian

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_D \end{bmatrix}^{\intercal}_{D \times 1}$ be a $D$-dimensional random vector with sample space $\mathcal{X} = \mathbb{R}^D$.

Then we say this random vector $\boldsymbol{X}$ is distributed according to the multivariate Gaussian distribution with mean vector

$$
\boldsymbol{\mu} = \begin{bmatrix} \mu_1 & \mu_2 & \cdots & \mu_D \end{bmatrix}^{\intercal}_{D \times 1}
$$

and covariance matrix

$$
\boldsymbol{\Sigma} = \begin{bmatrix} \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1D} \\ \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2D} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{D1} & \sigma_{D2} & \cdots & \sigma_{DD} \end{bmatrix}_{D \times D}
$$

if its probability density function (PDF) is given by

$$
f_{\boldsymbol{X}}(\boldsymbol{x}) = \frac{1}{(2 \pi)^{D/2} \lvert \boldsymbol{\Sigma} \rvert^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right)
$$ (eq:multivariate_gaussian_pdf)

where $\lvert \boldsymbol{\Sigma} \rvert$ is the determinant of $\boldsymbol{\Sigma}$.
```

```{prf:property} Expectation and Covariance of Multivariate Gaussian
:label: prop:mean_vector_covariance_matrix

By definition, the expectation of a multivariate Gaussian random vector $\boldsymbol{X}$ is given by

$$
\mathbb{E}[\boldsymbol{X}] = \boldsymbol{\mu}
$$

parameterized by the mean vector $\boldsymbol{\mu}$.

The covariance matrix $\boldsymbol{\Sigma}$ of a multivariate Gaussian random vector $\boldsymbol{X}$ is given by

$$
\operatorname{Cov}[\boldsymbol{X}] = \boldsymbol{\Sigma}
$$

parameterized by the covariance matrix $\boldsymbol{\Sigma}$.

Therefore, both are contained in the definition.
```

One can also easily see if $\boldsymbol{X}$ is a scalar (1-dimensional)
random variable $X$, then $D = 1$, $\boldsymbol{\mu} = \mu$, and
$\boldsymbol{\Sigma} = \sigma^2$, and by plugging into {eq}`eq:multivariate_gaussian_pdf`, the PDF of $X$ is merely

$$
f_{\boldsymbol{X}}(\boldsymbol{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( -\frac{1}{2 \sigma^2} (x - \mu)^2 \right)
$$ (eq:univariate_gaussian_pdf)

## Independence

If all the individual random variables $X_d$ in a multivariate Gaussian random vector $\boldsymbol{X}$ are independent, then the PDF can be greatly simplified
to a product of univariate Gaussian PDFs.

```{prf:definition} PDF of Independent Multivariate Gaussian Random Vectors
:label: def:independence

Let $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_D \end{bmatrix}^{\intercal}_{D \times 1}$ be a random vector following a multivariate Gaussian distribution with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$.

Suppose that all entries in $\boldsymbol{X} = \begin{bmatrix} X_1 & X_2 & \cdots & X_D \end{bmatrix}^{\intercal}_{D \times 1}$ are **independent** random variables (i.e. $X_i$ and $X_j$ are **independent** for all $i \neq j$).

Then the PDF of $\boldsymbol{X}$ is given by

$$
f_{\boldsymbol{X}}(\boldsymbol{x}) = \prod_{d=1}^D \frac{1}{\sqrt{(2 \pi) \sigma_{d}^2}} \exp \left( -\frac{(x_d - \mu_d)^2}{2 \sigma_{d}^2} \right)
$$ (eq:independent_multivariate_gaussian_pdf)

which is indeed a product of univariate Gaussian PDFs.
```

```{prf:proof}
Intuitively, if all the individual random variables $X_d$ in a multivariate Gaussian random vector $\boldsymbol{X}$ are independent, then when you draw such a random vector $\boldsymbol{X}$ from the sample space, the probability of drawing a particular value $\boldsymbol{x} = \begin{bmatrix} x_1 & x_2 & \cdots & x_D \end{bmatrix}^{\intercal}_{D \times 1}$ is akin to asking what is the probablity of drawing $X_1 = x_1$ **and** $X_2 = x_2$ **and** $\cdots$ **and** $X_D = x_D$ **simultaneously**. However, since they are
all independent, then drawing $X_i = x_i$ does not affect the probability of drawing $X_j = x_j$ for $i \neq j$. Therefore, the probability of drawing $\boldsymbol{x}$ is simply the product of the probabilities of drawing each individual $X_d$. This is in line with
what we understand of independence from earlier.

Formally, suppose $X_i$ and $X_j$ are independent for all $i \neq j$. Then, {prf:ref}`prop:covariance` states that $\operatorname{Cov}\left(X_i, X_j\right)=$ 0 . Consequently, the covariance matrix $\boldsymbol{\Sigma}$ is a diagonal matrix:

$$
\boldsymbol{\Sigma}=\left[\begin{array}{ccc}
\sigma_1^2 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \sigma_D^2
\end{array}\right]_{D \times D}
$$

where $\sigma_{d}^2=\operatorname{Var}\left[X_d\right]$. When this occurs, the exponential term in the Gaussian PDF is {cite}`chan_2021`

$$
(\boldsymbol{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})=\left[\begin{array}{c}
x_1-\mu_1 \\
\vdots \\
x_D-\mu_D
\end{array}\right]^T\left[\begin{array}{ccc}
\sigma_1^2 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \sigma_D^2
\end{array}\right]^{-1}\left[\begin{array}{c}
x_1-\mu_1 \\
\vdots \\
x_D-\mu_D
\end{array}\right]=\sum_{d=1}^D \frac{\left(x_d-\mu_d\right)^2}{\sigma_d^2} .
$$

Moreover, the determinant $|\boldsymbol{\Sigma}|$ is

$$
\left.|\boldsymbol{\Sigma}|=\mid \begin{array}{ccc}
\sigma_1^2 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \sigma_D
^2
\end{array}\right] \mid=\prod_{d=1}^D \sigma_d^2
$$

Substituting these results into the joint Gaussian PDF, we obtain

$$
f_{\boldsymbol{X}}(\boldsymbol{x}) = \prod_{d=1}^D \frac{1}{\sqrt{(2 \pi) \sigma_{d}^2}} \exp \left( -\frac{(x_d - \mu_d)^2}{2 \sigma_{d}^2} \right)
$$

which is a product of individual univariate Gaussian PDFs.
```

```{prf:remark} IID Assumption
:label: remark:iid_assumption

Do not confuse this with the $\iid$ assumption. In most Machine Learning context,
when we say $\iid$, it refers to the samples and not the individual random variables
in the random vector of each sample.

In supervised learning, implicitly or explicitly, one *always* assumes that the training set

$$
\begin{aligned}
\mathcal{S} &= \left\{\left(\mathbf{x}^{(1)}, y^{(1)}\right), \left(\mathbf{x}^{(2)}, y^{(2)}\right), \cdots, \left(\mathbf{x}^{(N)}, y^{(N)}\right)\right\} \\
\end{aligned}
$$

is composed of $N$ input/response tuples

$$
\left({\mathbf{X}}^{(n)} = \mathbf{x}^{(n)}, Y^{(n)} = y^{(n)}\right)
$$

that are ***independently drawn from the same (identical) joint distribution***

$$
\mathbb{P}_{\{\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\}}(\mathbf{x}, y)
$$

with

$$
\mathbb{P}(\mathbf{X} = \mathbf{x}, Y = y ; \boldsymbol{\theta}) = \mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x}) \mathbb{P}(\mathbf{X} = \mathbf{x})
$$

where $\mathbb{P}(Y = y \mid \mathbf{X} = \mathbf{x})$ is the conditional probability of $Y$ given $\mathbf{X}$,
the relationship that the learner algorithm/concept $c$ is trying to capture.

Then in this case, the i.i.d. assumption writes (also defined in {prf:ref}`def_iid`):

$$
\begin{aligned}
\left({\mathbf{X}}^{(n)}, Y^{(n)}\right) &\sim \mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y) \quad \text{and}\\
\left({\mathbf{X}}^{(n)}, Y^{(n)}\right) &\text{ independent of } \left({\mathbf{X}}^{(m)}, Y^{(m)}\right) \quad \forall n \neq m \in \{1, 2, \ldots, N\}
\end{aligned}
$$

and we sometimes denote

$$
\begin{aligned}
\left(\mathbf{x}^{(n)}, y^{(n)}\right) \overset{\text{i.i.d.}}{\sim} \mathbb{P}_{\{\mathcal{X}, \mathcal{Y}, \boldsymbol{\theta}\}}(\mathbf{x}, y)
\end{aligned}
$$

This **does not assume any independence within each sample** $\mathbf{X}^{(n)}$.
```


