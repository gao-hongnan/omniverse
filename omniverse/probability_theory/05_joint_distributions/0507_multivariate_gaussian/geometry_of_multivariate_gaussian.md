# The Geometry of Multivariate Gaussians

By the eigendecomposition theorem, we can write the covariance matrix $\boldsymbol{\Sigma}$ as:

$$
\begin{aligned}
\boldsymbol{\Sigma} &= \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^{T} \\
&= \begin{bmatrix} \boldsymbol{u}_1 & \cdots & \boldsymbol{u}_D \end{bmatrix} \begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_D \end{bmatrix} \begin{bmatrix} \boldsymbol{u}_1 & \cdots & \boldsymbol{u}_D \end{bmatrix}^{T}
\end{aligned}
$$

where $\boldsymbol{u}_d$ is the $d$th eigenvector of $\boldsymbol{\Sigma}$ and $\lambda_d$ is the corresponding eigenvalue. The eigenvectors are orthogonal, so $\boldsymbol{U}^{T} \boldsymbol{U} = \boldsymbol{I}$.

The geometry of a multivariate Gaussian distribution is defined by its covariance matrix, which can be represented in terms of its eigenvalues and eigenvectors. The eigenvalues represent the magnitude of the variance along each eigenvector (principal axis) and determine the shape of the ellipsoid that characterizes the distribution. The eigenvectors define the orientation of the principal axes and hence the orientation of the ellipsoid. This information allows us to perform various operations such as transforming the coordinates of the data points, projecting the data onto a lower-dimensional subspace, or rotating the ellipsoid to align with the coordinate axes.

This is true, but it is not immediately obvious why. Let's see why.

First, we have established that the covariance matrix can be decomposed to a matrix of eigenvectors and a diagonal matrix of eigenvalues.
We claim that for each dimension $d$, the variance of the $d$th dimension is given by $\lambda_d$, and
the direction of the $d$th dimension is given by $\boldsymbol{u}_d$.

```{figure} ../assets/chan_fig5.18.png
---
name: fig_geometry_of_multivariate_gaussian
---
The shape of multivariate gaussian is defined by its mean and covariance. Image Credit: {cite}`chan_2021`.
```

## Why Eigenvalues and Eigenvectors defined the shape of Multivariate Gaussian?

Consider a random vector $\mathbf{X} \in \mathbb{R}^D$ with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$.
The probability density function of the multivariate Gaussian distribution is given by:

$$
\begin{equation}
f_{\boldsymbol{X}}(\boldsymbol{x}) = \frac{1}{(2 \pi)^{D/2} \lvert \boldsymbol{\Sigma} \rvert^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right)
\end{equation}
$$

where $|\boldsymbol{\Sigma}|$ is the determinant of $\boldsymbol{\Sigma} $.

The **level sets** (see [contours](../../01_mathematical_preliminaries/contours.ipynb))
of the Gaussian distribution correspond to the set of points in the $D$-dimensional space
that have the **same probability density**. The level set at a given level $\lambda$ is given by the equation:

$$
\begin{equation}
\frac{1}{(2 \pi)^{D/2} \lvert \boldsymbol{\Sigma} \rvert^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right) = \lambda
\end{equation}
$$ (eq:level_set_of_multivariate_gaussian)

where $\lambda$ is a constant. Rearranging, we get:

$$
\begin{align*}
&\quad
\frac{1}{(2 \pi)^{D/2} \lvert \boldsymbol{\Sigma} \rvert^{1/2}} \exp \left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right) = \lambda\\
\iff &\quad \exp \left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right) = \lambda \lvert \boldsymbol{\Sigma} \rvert^{1/2} (2 \pi)^{D/2}\\
\iff &\quad \left( -\frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu})^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) \right) = \log \lambda + \frac{D}{2} \log (2 \pi) + \frac{1}{2} \log \lvert \boldsymbol{\Sigma} \rvert\\
\iff &\quad \left(\boldsymbol{x} - \boldsymbol{\mu}\right)^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) = -2 \log \lambda - D \log (2 \pi) - \log \lvert \boldsymbol{\Sigma} \rvert\\
\iff &\quad \left(\boldsymbol{x} - \boldsymbol{\mu}\right)^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) = \lambda^{'} \\
\end{align*}
$$

So we have rearranged such that the right-hand side is a constant, and we represent it by $\lambda^{'}$. Why can we do this?
That is because in multivariate gaussian, you observed that only the exponential term is dependent on $\boldsymbol{x}$, and
therefore, we might as well use the simplified term above to represent the level set of the multivariate gaussian. It is
merely a change of variable.

We claim that the level set of the multivariate Gaussian distribution, defined by:

$$
\left(\boldsymbol{x} - \boldsymbol{\mu}\right)^{\intercal} \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}) = \lambda^{'}
$$ (eq:level_set_of_multivariate_gaussian_simplified)

is an ellipse in the $D$-dimensional space.

To see why the equation defines an ellipse, we can use the eigendecomposition of the covariance matrix. Let $\Sigma = Q \Lambda Q^T$ be the eigendecomposition, where $Q$ is an orthogonal matrix of eigenvectors and $\Lambda$ is a diagonal matrix of eigenvalues. Substituting this into the equation, we get:

$$
\begin{equation}
(X - \mu)^T Q \Lambda^{-1} Q^T (X - \mu) = c
\end{equation}
$$

Let $Y = Q^T (X - \mu)$. Then, the equation becomes:

$$
\begin{equation}
Y^T \Lambda^{-1} Y = c
\end{equation}
$$

This equation represents a set of ellipses in the Y space, where the eigenvectors of $\Lambda^{-1}$ are the principal axes of the ellipses and the eigenvalues of $\Lambda^{-1}$ are the reciprocals of the variances along the principal axes. The ellipses in the Y space can be transformed back into the X space by multiplying by $Q$.

Thus, we have shown that the level sets of a multivariate Gaussian distribution are ellipsoids centered at the mean vector and defined by the covariance matrix, which
in turn is defined by the principal axes and variances of the distribution (eigenvalues and eigenvectors of the covariance matrix).

---

```{prf:remark} Remark
:label: prf:remark:why_contours_of_multivariate_gaussian_are_elliptical

So indeed, the shape of the level set of the multivariate Gaussian distribution is an ellipse. In an
ellipse, there is the notion of a major axis and a minor axis. The major axis is the axis that is
longer than the minor axis. The major axis is the axis along which the distribution has the largest
variance. And by the reasoning we have just done, we see that the major axis of a multivariate Gaussian
is the eigenvector of the covariance matrix that corresponds to the largest eigenvalue.
```

## Further Readings

- Bishop, Christopher M. "Chapter 2.3. The Gaussian Distribution." In Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2016
- [Why probability contours for the multivariate Gaussian are elliptical](https://www.michaelchughes.com/blog/2013/01/why-contours-for-multivariate-gaussian-are-elliptical/#:~:text=Every%202D%20Gaussian%20concentrates%20its,a%20particular%20form%3A%20an%20ellipse.)
- [Why are contours of a multivariate Gaussian distribution elliptical?](https://stats.stackexchange.com/questions/326334/why-are-contours-of-a-multivariate-gaussian-distribution-elliptical)
- https://en.wikipedia.org/wiki/Ellipse