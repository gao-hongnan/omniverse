# Covariance Matrix is Positive Semi-Definite

## The Properties of Covariance Matrix

The covariance matrix $\boldsymbol{\Sigma}$ is probably the heart and soul of
the multivariate normal distribution. There are extensive studies on this, we will
mention some important properties that will become more apparent later.

Some intuition, notice that the expression in {eq}`eq:multivariate_gaussian_pdf` has
the exponential term

$$
\exp \left( -\frac{1}{2} \left(\mathbf{x} - \boldsymbol{\mu}\right)^T \boldsymbol{\Sigma}^{-1} \left(\mathbf{x} - \boldsymbol{\mu}\right) \right)
$$

where $\boldsymbol{\mu}$ is the mean vector and $\boldsymbol{\Sigma}$ is the covariance matrix.

Note the $\boldsymbol{\Sigma}^{-1}$ term, which is a $D \times D$ matrix. This matrix is the
inverse of the covariance matrix $\boldsymbol{\Sigma}$, which is also a $D \times D$ matrix.
The implicit assumption is that $\boldsymbol{\Sigma}$ is invertible, how do we know for
sure that this must be true? What if $\boldsymbol{\Sigma}$ is not invertible?

The validity of $\boldsymbol{\Sigma}$ requires the concept of **positive semi-definite**.

### Covariance Matrix is Positive Semi-Definite

```{prf:definition} Positive Semi-Definite
:label: def:positive_semi_definite

A matrix $\boldsymbol{A} \in \mathbb{R}^{D \times D}$ is positive semi-definite if

$$
\boldsymbol{x}^T \boldsymbol{A} \boldsymbol{x} \geq 0 \quad \forall \boldsymbol{x} \in \mathbb{R}^D
$$
```

```{prf:definition} Positive Definite
:label: def:positive_definite

A matrix $\boldsymbol{A} \in \mathbb{R}^{D \times D}$ is positive definite if

$$
\boldsymbol{x}^T \boldsymbol{A} \boldsymbol{x} > 0 \quad \forall \boldsymbol{x} \in \mathbb{R}^D
$$
```

We will state without proof the following theorem:

```{prf:theorem} Positive Semi-Definite Matrix
:label: theorem:positive_semi_definite_matrix

A matrix $\boldsymbol{A} \in \mathbb{R}^{D \times D}$ is positive semi-definite if and only if
all its eigenvalues are non-negative.

$$
\lambda_d(\boldsymbol{A}) \geq 0 \quad \forall i \in \{1, 2, \ldots, D\}
$$

where $\lambda_d(\boldsymbol{A})$ is the $d$-th eigenvalue of $\boldsymbol{A}$.
```

```{prf:corollary} Positive Definite Matrix
:label: corollary:positive_definite_matrix

If a matrix $\boldsymbol{A} \in \mathbb{R}^{D \times D}$ is **positive definite** (but not necessarily positive semi-definite), then $\boldsymbol{A}$ is invertible.

This means there exists a matrix $\boldsymbol{A}^{-1} \in \mathbb{R}^{D \times D}$ such that

$$
\boldsymbol{A} \boldsymbol{A}^{-1} = \boldsymbol{A}^{-1} \boldsymbol{A} = \boldsymbol{I}
$$
```

Finally, we can show that the covariance matrix $\boldsymbol{\Sigma}$ is **always** symmetric positive semi-definite.

```{prf:theorem} Covariance Matrix is always Symmetric Positive Semi-Definite
:label: theorem:positive_semi_definite_covariance_matrix

Let $\boldsymbol{X} \in \mathbb{R}^D$ be a random vector with mean vector $\boldsymbol{\mu} \in \mathbb{R}^D$ and covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$.

Then the covariance matrix $\operatorname{Cov}(\boldsymbol{X}) = \boldsymbol{\Sigma}$ is always symmetric positive semi-definite.

$$
\begin{aligned}
\boldsymbol{\Sigma}^T &= \boldsymbol{\Sigma} \quad &\text{symmetric} \\
\boldsymbol{v}^T \boldsymbol{\Sigma} \boldsymbol{v} &\geq 0 \quad &\text{positive semi-definite} \\
\end{aligned}
$$

for any $\boldsymbol{v} \in \mathbb{R}^D$.
```

```{prf:proof}
Proof from {cite}`chan_2021`.

Symmetry follows immediately from the definition, because $\operatorname{Cov}\left(X_i, X_j\right)=\operatorname{Cov}\left(X_j, X_i\right)$. The positive semi-definiteness comes from the fact that

$$
\begin{aligned}
\boldsymbol{v}^T \boldsymbol{\Sigma} \boldsymbol{v} & =\boldsymbol{v}^T \mathbb{E}\left[(\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^T\right] \boldsymbol{v} \\
& =\mathbb{E}\left[\boldsymbol{v}^T(\boldsymbol{X}-\boldsymbol{\mu})(\boldsymbol{X}-\boldsymbol{\mu})^T \boldsymbol{v}\right] \\
& =\mathbb{E}\left[\boldsymbol{b}^T \boldsymbol{b}\right]=\mathbb{E}\left[\|\boldsymbol{b}\|^2\right] \geq 0
\end{aligned}
$$

where $\boldsymbol{b}=(\boldsymbol{X}-\boldsymbol{\mu})^T \boldsymbol{v}$.
```

### Importance

So we know if the covariance matrix $\boldsymbol{\Sigma}$ is positive semi-definite, then the multivariate Gaussian distribution is well-defined.

However, in empirical applications, we are often estimating the covariance matrix $\boldsymbol{\Sigma}$ from data. The estimated covariance matrix $\hat{\boldsymbol{\Sigma}}$ is not necessarily positive semi-definite. So a way to check the validity of the estimated covariance matrix $\hat{\boldsymbol{\Sigma}}$ is to check if it is positive semi-definite.

Furthermore, generally, if we solve an optimization problem involving the following:

$$
\begin{aligned}
\min_{\boldsymbol{\theta}} & \quad f(\boldsymbol{\theta}) = \boldsymbol{\theta}^T \boldsymbol{A} \boldsymbol{\theta} \\
\end{aligned}
$$

then $f$ will be **convex** if $\boldsymbol{A}$ is positive semi-definite. Now the
guarantee of convex is a big deal in optimization, because all local minima are global minima.