---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
mystnb:
  number_source_lines: true
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---



# Concept


```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-input, remove-output]

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt


def find_root_dir(current_path: Path | None = None, marker: str = '.git') -> Path | None:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path | None
        The starting path to search from. If None, the current working directory
        `Path.cwd()` is used.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path | None
        The path to the root directory. Returns None if the marker is not found.
    """
    if not current_path:
        current_path = Path.cwd()
    current_path = current_path.resolve()
    for parent in [current_path, *current_path.parents]:
        if (parent / marker).exists():
            return parent
    return None

root_dir = find_root_dir(marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.utils.visualization.style import use_svg_display
    from omnivault.machine_learning.estimator import BaseEstimator
    from omnivault.utils.reproducibility.seed import seed_all
    from omnivault.machine_learning.utils import make_meshgrid
else:
    raise ImportError("Root directory not found.")

import math
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich import print  # pylint: disable=redefined-builtin
from rich.pretty import pprint
from scipy.stats import bernoulli, binom, multivariate_normal, norm
from sklearn.mixture import GaussianMixture

use_svg_display()
seed_all(1992, False, False)
```

Gaussian Mixture Models are a statistical technique used for approximating the probability distribution of data, known as [**density estimation**](https://en.wikipedia.org/wiki/Density_estimation#:~:text=In%20statistics%2C%20probability%20density%20estimation,unobservable%20underlying%20probability%20density%20function.).

From the introduction in [the previous section](01_intro.md), one might ask the motivation of using a linear combination to approximate a probability distribution. The answer is that the linear combination of simple distributions is a flexible model that can approximate a wide variety of probability distributions.

Consider your dataset that exhibits a multimodal distribution (i.e. multiple modes), then a single Gaussian distribution will not be able to capture the distribution of the data.

## Intuition

### Simple Bi-Modal Distribution

The code below does the following:

1. `x_axis` is created as a NumPy array of evenly spaced values ranging from -15 to 15 with a step size of 0.001.

2. `gaussian_1` and `gaussian_2` are defined as dictionaries representing two normal distributions with given means and standard deviations:
   - Distribution 1 has a mean of -4 and a standard deviation of 2.
   - Distribution 2 has a mean of 4 and a standard deviation of 2.

3. The probability density functions (PDFs) for `gaussian_1` and `gaussian_2` are calculated using the `norm.pdf` function from the SciPy library. The PDFs are computed for each value in `x_axis`, with the respective mean and standard deviation for each distribution.

4. The `pdf_merged` variable is created by adding the PDFs of `gaussian_1` and `gaussian_2` element-wise, which represents the combined probability density function of both distributions.

```{code-cell} ipython3
:tags: [hide-input]

def get_gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    y = norm.pdf(x, mu, sigma)
    return y


def merge_gaussian_pdf(
    prior: List[float], pdf_1: np.ndarray, pdf_2: np.ndarray, *args: np.ndarray
) -> np.ndarray:
    pdfs = [pdf_1, pdf_2, *args]

    assert len(prior) == len(pdfs)

    # w1 * f1 + w2 * f2 + ...
    return np.sum([prior[i] * pdfs[i] for i in range(len(prior))], axis=0)
```

```{code-cell} ipython3
:tags: [hide-input]

x_axis = np.arange(-15, 15, 0.001)
gaussian_1 = {"mean": -4, "std": 2}
gaussian_2 = {"mean": 4, "std": 2}

pdf_1 = get_gaussian_pdf(x_axis, gaussian_1["mean"], gaussian_1["std"])
pdf_2 = get_gaussian_pdf(x_axis, gaussian_2["mean"], gaussian_2["std"])

weights_1, weights_2 = 0.5, 0.5

pdf_merged = merge_gaussian_pdf(
    [weights_1, weights_2], pdf_1, pdf_2
)  # weights_1 * pdf_1 + weights_2 * pdf_2
```

This code below will create a figure with three subplots, where the first subplot (`ax1`) contains Distribution 1 and Distribution 2, the second subplot (`ax2`) contains the Merged Distribution, and the third subplot (`ax3`) contains Distribution 1 and Distribution 2 as dotted lines and the Merged Distribution as a solid line. All subplots share the same x-axis.

```{code-cell} ipython3
:tags: [hide-input]

# Create a 3x1 grid of subplots with shared x-axis
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

# Plot Distribution 1 and Distribution 2 on the first subplot (ax1)
ax1.plot(x_axis, pdf_1, "r:", label="Gaussian 1")
ax1.plot(x_axis, pdf_2, "b:", label="Gaussian 2")
ax1.legend()

# Plot Merged Distribution on the second subplot (ax2)
ax2.plot(x_axis, pdf_merged, "g-", label="Gaussian Merged")
ax2.legend()

# Plot pdf_1 and pdf_2 as dotted lines, and pdf_merged as solid line on the third subplot (ax3)
ax3.plot(x_axis, pdf_1, "r:", label="Gaussian 1 (Dotted)")
ax3.plot(x_axis, pdf_2, "b:", label="Gaussian 2 (Dotted)")
ax3.plot(x_axis, pdf_merged, "g-", label="Gaussian Merged (Solid)")
ax3.legend()

# Show the plots
plt.show();
```

In this case, we say that

$$
\mathbb{P}\left[\mathbf{X} ; \boldsymbol{\theta} = \left(\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}\right)\right] = \textcolor{red}{0.5 \mathcal{N}\left(x \mid -4, 2\right)} + \textcolor{blue}{0.5 \mathcal{N}(x \mid 4, 2)}
$$

where this distribution is a mixture of two normal distributions with equal weights of 0.5.

The mixture components are:

$$
\mathcal{N}\left(x \mid -4, 2\right) \quad \mathcal{N}(x \mid 4, 2)
$$

parametrized by

$$
\boldsymbol{\mu} = \begin{bmatrix} \mu_1 & \mu_2 \end{bmatrix} = \begin{bmatrix} -4 & 4 \end{bmatrix} \quad \boldsymbol{\Sigma} = \begin{bmatrix} \Sigma_1 & \Sigma_2 \end{bmatrix} = \begin{bmatrix} 2 & 2 \end{bmatrix}
$$

The mixture weights are:

$$
\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix}
$$

### Generative Story

Now the generative story of such a mixture model is as follows.

If we have a mixture model with $K$ components, then we can sample from the mixture model by first sampling a component $k$ from the categorical distribution with parameters $\boldsymbol{\pi}$, and then sampling from the $k$-th component distribution with parameters $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$.

More concretely, if we know the following parameters:

- $\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 \end{bmatrix} = \begin{bmatrix} 0.5 & 0.5 \end{bmatrix}$
- $\boldsymbol{\mu} = \begin{bmatrix} \mu_1 & \mu_2 \end{bmatrix} = \begin{bmatrix} 2 & -2 \end{bmatrix}$
- $\boldsymbol{\Sigma} = \begin{bmatrix} \Sigma_1 & \Sigma_2 \end{bmatrix} = \begin{bmatrix} 3 & 1 \end{bmatrix}$

then we can sample from the mixture model by

- first sampling a component $k$ from the categorical distribution with parameters $\boldsymbol{\pi}$, which means
    either $k = 1$ or $k = 2$ with equal probability of 0.5.
- then once we know which component we sampled from, we can sample from the component distribution, which in this case is a normal distribution with mean $\mu_k$ and standard deviation $\Sigma_k$. For example, if we sampled $k = 1$, then we can sample from the first component distribution with mean $\mu_1 = 2$ and standard deviation $\Sigma_1 = 3$.

Note very carefully, this is the "generative" side, in machine learning we are interested in the "inference" side, which is to infer the parameters of the mixture model from the dataset $\mathcal{S}$!

Let's see in code how we can sample from a mixture model, and that if we sample enough
data points, the empirical distribution of the samples will converge to the true distribution of the mixture model.

```{code-cell} ipython3
:tags: [hide-input]

def generate_x(
    prior: List[float],
    mu: List[float],
    sigma: List[float],
    num_samples: int = 100,
    num_gaussians: int = 2,
) -> List[float]:
    X = []
    for _ in range(num_samples):
        # Select a Gaussian based on the prior
        selected_gaussian = np.random.choice(num_gaussians, p=prior)
        # Sample from the selected Gaussian
        x = norm.rvs(
            loc=mu[selected_gaussian], scale=sigma[selected_gaussian], size=None
        )
        X.append(x)
    return X
```

```{code-cell} ipython3
:tags: [hide-input]

x_axis = np.arange(-5, 15, 0.001)
prior = [0.4, 0.4, 0.2]  # weights
mu = [0, 5, 10]
sigma = [1, 1, 1]

gaussian_1 = norm.pdf(x_axis, loc=mu[0], scale=sigma[0])
gaussian_2 = norm.pdf(x_axis, loc=mu[1], scale=sigma[1])
gaussian_3 = norm.pdf(x_axis, loc=mu[2], scale=sigma[2])

mixture_pdf = merge_gaussian_pdf(
    prior, gaussian_1, gaussian_2, gaussian_3
)  # prior[0] * gaussian_1 + prior[1] * gaussian_2 + prior[2] * gaussian_3
```

```{code-cell} ipython3
:tags: [hide-input]

sample_sizes = [100, 500, 10000]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.ravel()

axes[0].plot(x_axis, mixture_pdf, label="Mixture PDF")
axes[0].set_title("Mixture PDF")

for i, n in enumerate(sample_sizes, start=1):
    samples = generate_x(prior, mu, sigma, num_samples=n, num_gaussians=len(prior))
    axes[i].hist(samples, bins=30, density=True, alpha=0.5, label=f"n = {n}")
    axes[i].plot(x_axis, mixture_pdf, label="Mixture PDF")
    axes[i].set_title(f"Generated Samples (n = {n})")

for ax in axes:
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.show()
```

In this code, we're using a Gaussian Mixture Model (GMM) to generate and visualize samples from a mixture of two Gaussian distributions. The purpose of the visualization is to demonstrate how the GMM can approximate the true underlying distribution as the number of samples increases.

Here's a step-by-step explanation of the code:

1. We define sample_sizes as a list containing the number of samples to generate for each subplot (100, 500, and 10000).

2. We create a 2x2 grid of subplots with shared x and y axes, with a figure size of 12x10.

3. We plot the mixture PDF on the first subplot (axes[0]). This represents the true underlying distribution that we are trying to approximate with our GMM.

4. We iterate over the sample_sizes list, and for each sample size, we use the generate_x function to generate samples from the GMM. The generate_x function takes the prior probabilities, the means and standard deviations of the Gaussians, the number of samples, and the number of Gaussians as input arguments.

5. For each sample size, we plot a histogram of the generated samples on the corresponding subplot. We normalize the histogram to match the density of the true underlying distribution. We also plot the mixture PDF on the same subplot to compare the generated samples with the true distribution.

6. We set the titles, x-axis labels, and y-axis labels for all subplots, and add a legend to each subplot.

7. We use plt.tight_layout() to adjust the spacing between subplots, and finally display the figure using plt.show().

In the context of GMM, this code demonstrates how the GMM can be used to generate samples from a mixture of Gaussian distributions. The generated samples are visualized as histograms, which are compared to the true underlying distribution (the mixture PDF) to show how well the GMM approximates the true distribution. As the number of samples increases, the histograms of the generated samples become closer to the mixture PDF, indicating that the GMM is effectively approximating the true distribution.

Note carefully again that this is under the assumption that we already know the parameters of the mixture model, which is not the case in machine learning. In machine learning, we are interested in the "inference" side, which is to infer the parameters of the mixture model from the dataset $\mathcal{S}$!

### Inference Story

Now let's flip the table and see how we can infer the parameters of the mixture model from the dataset $\mathcal{S}$.

The code above does the following:

1. Import necessary libraries: NumPy, Matplotlib, and GaussianMixture from scikit-learn.

2. Generate a synthetic dataset with three clusters:
   - Set a random seed to ensure reproducibility.
   - Define the number of samples (500).
   - Create a dataset by concatenating samples from three normal distributions with different means (0, 5, and 10) and the same standard deviation (1). The dataset is reshaped into a 2D array.

3. Fit a Gaussian Mixture Model (GMM) to the data:
   - Instantiate a GaussianMixture object with three components and a fixed random state.
   - Fit the GMM to the dataset `X`.

4. Plot the data and the Gaussian Mixture Model:
   - Create an array `x_plot` of 1000 linearly spaced values between -5 and 15.
   - Calculate the density of the GMM for each value in `x_plot` using the `score_samples` method.
   - Plot a histogram of the dataset with 30 bins, normalized by the total area.
   - Plot the GMM density estimation using a red line.
   - Add labels for the x-axis, y-axis, and a title for the plot.

5. Display the plot using `plt.show()`.


A reminder, we know the true distribution because we defined them ourselves. In reality,
we don't know the true distribution, and we want to infer the parameters of the mixture model from the dataset $\mathcal{S}$.
The purpose of defining the true distribution is for pedagogical purposes, so that we can compare the true distribution with the estimated distribution from the GMM.

```{code-cell} ipython3
:tags: [hide-input]

x_axis = np.arange(-5, 15, 0.001)
prior = [0.4, 0.4, 0.2]  # weights
mu = [0, 5, 10]
sigma = [1, 1, 1]

gaussian_1 = norm.pdf(x_axis, loc=mu[0], scale=sigma[0])
gaussian_2 = norm.pdf(x_axis, loc=mu[1], scale=sigma[1])
gaussian_3 = norm.pdf(x_axis, loc=mu[2], scale=sigma[2])

mixture_pdf = merge_gaussian_pdf(
    prior, gaussian_1, gaussian_2, gaussian_3
)  # prior[0] * gaussian_1 + prior[1] * gaussian_2 + prior[2] * gaussian_3
```

```{code-cell} ipython3
:tags: [hide-input]

num_samples = 10000
samples = generate_x(
    prior, mu, sigma, num_samples=num_samples, num_gaussians=len(prior)
)
X = np.array(samples).reshape(-1, 1)

# Fit a Gaussian Mixture Model to the generated samples
gmm = GaussianMixture(n_components=len(prior), random_state=1992)
gmm.fit(X)
```

Remarkable! The parameters inferred from our `GMM` model has the following:

```{code-cell} ipython3
:tags: [hide-input]

print("Prior:", gmm.weights_)
print("Mean:", gmm.means_.ravel())
print("Std:", np.sqrt(gmm.covariances_.ravel()))
```

When rounded to the nearest integer/decimal, the parameters inferred from our `GMM` model has the following:

- $\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \pi_3 \end{bmatrix} = \begin{bmatrix} 0.398, 0.4, 0.202 \end{bmatrix}$
- $\boldsymbol{\mu} = \begin{bmatrix} \mu_1 & \mu_2 & \mu_3 \end{bmatrix} = \begin{bmatrix} 0.0, 5.0, 10.0 \end{bmatrix}$
- $\boldsymbol{\Sigma} = \begin{bmatrix} \Sigma_1 & \Sigma_2 & \Sigma_3 \end{bmatrix} = \begin{bmatrix} 1.02, 0.99, 0.98 \end{bmatrix}$

Almost spot on with the true parameters!!!

The plot shows promising results, which is not surprising since our
estimated parameters are very close to the true parameters.

```{code-cell} ipython3
:tags: [hide-input]

# Plot the data and the Gaussian Mixture Model
x_plot = np.linspace(-5, 15, 1000).reshape(-1, 1)
density = np.exp(
    gmm.score_samples(x_plot)
)  # !!! gmm.score_samples(x_plot) returns the log-likelihood of the samples thus we need to take the exponential to get back raw probabilities.

plt.hist(samples, bins=30, density=True, alpha=0.5, label="Generated Samples")
plt.plot(x_plot, density, "-r", label="GMM Approximation")
plt.plot(x_axis, mixture_pdf, "-g", label="True Mixture PDF")
plt.xlabel("x")
plt.ylabel("density")
plt.title("Gaussian Mixture Model Approximation of Generated Samples")
plt.legend()
plt.show()
```

When you `predict` on the samples `X`, you get the cluster/component in which
each sample belongs to. In this case, the samples are generated from three
clusters, so the predicted labels are either 0, 1, or 2.

When you `predict_proba` on the samples `X`, you get the (log) probability
of each sample belonging to each cluster/component. In this case, the samples
are generated from three clusters, so the predicted probabilities are a 2D array
with shape `(n_samples, n_components)`.
Then the highest probability is the cluster/component in which the sample belongs to, which is the predicted label.

```{code-cell} ipython3
:tags: [hide-input]

gmm.predict(X), gmm.predict_proba(X)
```

#### Inference with 2D Data

We can also visualize the inferred parameters of the mixture model in 2D.

```{code-cell} ipython3
:tags: [hide-input]

# Generate a synthetic dataset with three clusters
np.random.seed(42)
n_samples = 500
X1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], int(0.4 * n_samples))
X2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], int(0.4 * n_samples))
X3 = np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]], int(0.2 * n_samples))
X = np.vstack([X1, X2, X3])
print(X.shape)
```

```{code-cell} ipython3
:tags: [hide-input]

# Fit a Gaussian Mixture Model to the data
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

print("Prior:")
pprint(gmm.weights_)

print("Mean:")
pprint(gmm.means_)

print("Covariance:")
pprint(gmm.covariances_)
```

```{code-cell} ipython3
:tags: [hide-input]

x = X[:, 0]
y = X[:, 1]
step = 0.01
X_plot, Y_plot = make_meshgrid(x, y, step=step)
print(X_plot.shape, Y_plot.shape)
# Plot the data points and the Gaussian Mixture Model contours
# x = np.linspace(-5, 15, 100)
# y = np.linspace(-5, 15, 100)
# X_plot, Y_plot = np.meshgrid(x, y)
pos = np.empty(X_plot.shape + (2,))
pos[:, :, 0] = X_plot
pos[:, :, 1] = Y_plot
print(pos.shape)
```

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.5)

# plot for each component a contour of the probability density function
for mean, cov in zip(gmm.means_, gmm.covariances_):
    rv = multivariate_normal(mean, cov)
    ax.contour(X_plot, Y_plot, rv.pdf(pos), levels=10)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gaussian Mixture Model Approximation of a Multimodal Distribution in 2D")
plt.show()
```

```{code-cell} ipython3
:tags: [hide-input]

# Generate a synthetic dataset with overlapping clusters
np.random.seed(42)
n_samples = 500
X1 = np.random.multivariate_normal([0, 0], [[2, 0.5], [0.5, 2]], int(0.4 * n_samples))
X2 = np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], int(0.4 * n_samples))
X3 = np.random.multivariate_normal(
    [0, 5], [[1.5, 0.5], [0.5, 1.5]], int(0.2 * n_samples)
)
X = np.vstack([X1, X2, X3])

# Fit a Gaussian Mixture Model to the data
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

print("Prior:")
pprint(gmm.weights_)

print("Mean:")
pprint(gmm.means_)

print("Covariance:")
pprint(gmm.covariances_)


# Plot the data points and the Gaussian Mixture Model contours
x = X[:, 0]
y = X[:, 1]
step = 0.01
X_plot, Y_plot = make_meshgrid(x, y, step=step)
print(X_plot.shape, Y_plot.shape)
# x = np.linspace(-6, 6, 100)
# y = np.linspace(-6, 10, 100)
# X_plot, Y_plot = np.meshgrid(x, y)

pos = np.empty(X_plot.shape + (2,))
pos[:, :, 0] = X_plot
pos[:, :, 1] = Y_plot
print(pos.shape)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.5)

for mean, cov in zip(gmm.means_, gmm.covariances_):
    rv = multivariate_normal(mean, cov)
    ax.contour(X_plot, Y_plot, rv.pdf(pos), levels=10)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gaussian Mixture Model Approximation of Overlapping Distributions in 2D")
plt.show()
```

### Prior is a Latent Variable

The weights we have defined for each gaussian component are called the **prior** of the mixture model.
This just means if we draw a sample $\mathbf{x}$ from the mixture model, the probability of it belonging to the $k$-th component is $\pi_k$.

In our case above, we have defined the prior as $\boldsymbol{\pi} = \begin{bmatrix} 0.4, 0.4, 0.2 \end{bmatrix}$, which means if we draw a sample $\mathbf{x}$ from the mixture model, the probability of it belonging to the
1st component is 0.4, the probability of it belonging to the 2nd component is 0.4, and the probability of it belonging to the 3rd component is 0.2.

The prior is a latent variable, which means it is not observed in the dataset $\mathcal{S}$, but it is inferred from the dataset $\mathcal{S}$. This may sound magical, but it actually is just the number of data points in each
component, divided by the total number of data points.

Recall our `samples` consist of 10000 data points with $3$ components. We defined the `prior = [0.4, 0.4, 0.2]`, which means the number of data points in each component is $4000$, $4000$, and $2000$ respectively.
This variable is unobserved, because we really do not know what it is when we were handed the dataset $\mathcal{S}$.

Let's see an example.


## Problem Formulation

```{prf:remark} Notation Reminder
:label: prf:remark-notation-gmm

Firstly, notation wise:

$$
\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}\right)
$$

means the probability density function (PDF) of a multivariate normal distribution with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ evaluated at $\boldsymbol{x}$. The $\mid$ symbol means "given" but
do not confuse with the conditional probability.
```

### A Primer

**Given** a set $\mathcal{S}$ containing $N$ data points:

$$
\mathcal{S} = \left\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(N)}\right\} \subset \mathbb{R}^{D}
$$

where the vector $\mathbf{x}^{(n)}$ is the $n$-th sample with $D$ number of features, given by:

$$
\mathbf{x}^{(n)} \in \mathbb{R}^{D} = \begin{bmatrix} x_1^{(n)} & x_2^{(n)} & \cdots & x_D^{(n)} \end{bmatrix}^{\mathrm{T}} \quad \text{where } n = 1, \ldots, N.
$$

We can further write $\mathcal{S}$ as a disjoint union of $K$ sets, as follows:

$$
\begin{aligned}
\mathcal{S} &:= \left\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(N)}\right\} \subset \mathbb{R}^{D} = C_1 \sqcup C_2 \sqcup \cdots \sqcup C_K \\
\end{aligned}
$$

where $C_k$ is the set of data points that belong to cluster $k$:

$$
C_k = \left\{\mathbf{x}^{(n)} \in \mathbb{R}^{D} \middle\vert z^{(n)} = k\right\} .
$$ (eq:cluster-def-gmm)

Furthermore, we denote $z^{(n)}$ as the ***true cluster assignment*** of the $n$-th data point $\mathbf{x}^{(n)}$.

However, in practice, we don't have access to the true cluster assignments $z^{(n)}$. Our goal is to estimate the cluster assignments $z^{(n)}$ from the data points $\mathbf{x}^{(n)}$.

In K-Means, we discussed Lloyd's algorithm, a hard clustering method that outputs estimated cluster assignments $\hat{y}^{(n)}$ for each data point $\mathbf{x}^{(n)}$.
Note do not be confused by the notation, here $z$ is the true cluster assignment, while $\hat{y}$ is the estimated cluster assignment. In K-Means however,
I conveniently used $y$ to denote the true cluster assignment, and $\hat{y}$ to denote the estimated cluster assignment. We have
an additional layer of complexity here, as we will be using $z$ to denote the true cluster assignment, and $y$ to denote the a-posteriori probability of the data point $\mathbf{x}^{(n)}$ belonging to the $k$-th cluster $C_{k}$.

As we have seen from the examples earlier, it is desirable to quantify the degree by which a data point belongs to a cluster. Soft clustering methods use a continues range, such as the closed interval $[0,1]$, of possible values for the degree of belonging. In contrast, hard clustering methods use only two possible values for the degree of belonging to a specific cluster, either "full belonging" or no "belonging at all". While hard clustering methods assign a given data point to precisely one cluster, soft clustering methods typically assign a data point to several different clusters with non-zero degree of belonging {cite}`jung_2023`.

Consequently, we can define for each data point $\mathbf{x}^{(n)} \in \mathcal{S}$, an associated cluster assignment vector $\widehat{\mathbf{y}}^{(n)}$ as follows:

$$
\widehat{\mathbf{y}}^{(n)} = \begin{bmatrix} \hat{y}_1^{(n)} & \hat{y}_2^{(n)} & \cdots & \hat{y}_K^{(n)} \end{bmatrix}^{\mathrm{T}} \quad \text{where } n = 1, \ldots, N.
$$

where $\hat{y}_k^{(n)}$ is the degree of belonging of the $n$-th data point $\mathbf{x}^{(n)}$ to the $k$-th cluster $C_k$. This is reminiscent of the your
usual classification problem, where we have a set of $K$ classes, and we want to assign a data point $\mathbf{x}^{(n)}$ to one of the $K$ classes.

In this case, we can think of $\widehat{\mathbf{y}}^{(n)}$ as the **posterior probability** of the $n$-th data point $\mathbf{x}^{(n)}$ belonging to the $k$-th cluster $C_k$,
or with our current setup, the **posterior probability** of the $n$-th data point $\mathbf{x}^{(n)}$ given the cluster assignment $z^{(n)} = k$.

$$
\begin{aligned}
\widehat{\mathbf{y}}^{(n)} &= \mathbb{P}\left(z^{(n)} = k \mid \mathbf{x}^{(n)}\right) \\
\end{aligned}
$$


```{prf:example} Example
:label: prf:example-gmm-1

Consider the following example:

$$
\begin{aligned}
\widehat{\mathbf{y}}^{(1)} &= \begin{bmatrix} 0.1 & 0.7 & 0.2 \end{bmatrix}^{\mathrm{T}} \\
\end{aligned}
$$

then it can be interpreted as the degree of belonging of the first data point $\mathbf{x}^{(1)}$ to each of the three clusters $C_1, C_2, C_3$ respectively.
In this example, the first data point $\mathbf{x}^{(1)}$ has a $10\%$ chance of belonging to cluster $C_1$, a $70\%$ chance of belonging to cluster $C_2$, and a $20\%$ chance of belonging to cluster $C_3$.
We can therefore say that the first data point $\mathbf{x}^{(1)}$ is more likely to belong to cluster $C_2$ than to cluster $C_1$ or cluster $C_3$.
```

However, to even find the posterior probability, we need to know what
the distribution of $\mathbb{P}$ is. In the next section, we will
discuss the distribution of the posterior probability and how we can
estimate it.

(gmm-problem-formulation-gaussian-mixture-model)=
### Gaussian Mixture Model

A widely used soft clustering method is the [**Gaussian Mixture Model (GMM)**](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model), which uses a probabilistic model for the data points $\mathcal{S}=\left\{\mathbf{x}^{(n)}\right\}_{n=1}^N$.

A Gaussian mixture model is a density model where we combine a finite number of $K$ Gaussian distributions $\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)$ so that

$$
\begin{aligned}
& p(\boldsymbol{x} ; \boldsymbol{\theta})=\sum_{k=1}^K \pi_k \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \mathbf{\Sigma}_k\right) && (\star) \\
& 0 \leqslant \pi_k \leqslant 1, \quad \sum_{k=1}^K \pi_k=1, && (\star\star) \\
\end{aligned}
$$ (eq:gmm-def-1)

where we defined $\boldsymbol{\theta}:=\left\{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k: k=1, \ldots, K\right\}$ as the collection of all parameters of the model. This convex combination of Gaussian distribution gives us significantly more flexibility for modeling complex densities than a simple Gaussian distribution (which we recover from $(\star \star)$ for $K=1$) {cite}`deisenroth_ong_faisal_2021`.

Overall, the main goal is to find the 3 parameters of the mixture model that best fit the data $\mathcal{S}$.

To elaborate further the setup defined in {eq}`eq:gmm-def-1`, we have:

- The probability distribution above means that the (*joint*) probability of observing a data point $\boldsymbol{x}^{(n)}$ is the sum of the probability of observing $\boldsymbol{x}^{(n)}$ from each of the $K$ clusters, parametrized by the mean and covariance vector/matrix, weighted by the probability of the cluster assignment $z^{(n)} = k$, parameterized by $\pi_k$. This is a mouthful, we will break it down further in the sections below.
  Notation wise, we can write this as:

    $$
    \mathbb{P}\left(\boldsymbol{X} ; \boldsymbol{\theta}\right) := \mathbb{P}_{\boldsymbol{\theta}}\left(\boldsymbol{X}\right) = \mathbb{P}_{\left\{\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}\right\}}\left(\boldsymbol{X}\right)
    $$

- The first constraint $(\star\star)$ ensures that the probability of observing a data point $\boldsymbol{x}^{(n)}$ from any of the $K$ clusters is $1$. This is a normalization constraint, and is necessary to ensure that the probability distribution is a valid probability distribution.

- The shape and dimensions for the parameters are given:
    - $\boldsymbol{\pi}$ is a vector of mixing coefficients (prior weights):

      $$
      \boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \cdots & \pi_k \end{bmatrix}^{\mathrm{T}} \in \mathbb{R}^K
      $$
    - $\boldsymbol{\mu}_k$ is a vector of means for the $k$-th cluster:

      $$
      \boldsymbol{\mu}_k = \begin{bmatrix} \mu_{k1} & \mu_{k2} & \cdots & \mu_{kD} \end{bmatrix}^{\mathrm{T}} \in \mathbb{R}^D
      $$
    - $\boldsymbol{\Sigma}_k$ is a covariance matrix for the $k$-th cluster:

      $$
      \boldsymbol{\Sigma}_k = \begin{bmatrix} \Sigma_{k11} & \Sigma_{k12} & \cdots & \Sigma_{k1D} \\ \Sigma_{k21} & \Sigma_{k22} & \cdots & \Sigma_{k2D} \\ \vdots & \vdots & \ddots & \vdots \\ \Sigma_{kD1} & \Sigma_{kD2} & \cdots & \Sigma_{kDD} \end{bmatrix} \in \mathbb{R}^{D \times D}
      $$

### The Perpectives

There are two interpretations of the Gaussian mixture model: the latent variable perspective and the data likelihood perspective.

#### The Mixture Model Perspective

Mixture model perspective: In this perspective, GMM is seen as a simple mixture of multiple Gaussian distributions. The goal is to model the probability density function (PDF) of the observed data as a weighted sum of the individual Gaussian PDFs. Each Gaussian component has its own mean and covariance matrix, and the model learns the weights, means, and covariances that best fit the data. This perspective focuses on the density estimation aspect of GMM and is less concerned with the underlying latent variables.

#### The Latent Variable Perspective

Latent variable perspective: In this perspective, GMM is viewed as a generative probabilistic model that assumes there are some hidden (latent) variables responsible for generating the observed data points. Each hidden variable corresponds to one of the Gaussian components in the mixture. The data points are assumed to be generated by first sampling the latent variable (component) from a categorical distribution and then sampling the data point from the corresponding Gaussian distribution. This perspective is closely related to the Expectation-Maximization (EM) algorithm, which alternates between estimating the component assignments (latent variables) and updating the Gaussian parameters (mean, covariance) to maximize the likelihood of the observed data.

#### Summary

Both perspectives ultimately lead to the same model, but they highlight different aspects of GMM and can be useful in different contexts. For example, the latent variable perspective is more suitable for clustering and classification tasks, while the mixture model perspective is more useful for density estimation and generating new samples from the modeled distribution.

In the next few sections, we will discuss the latent variable perspective, but note there
may be some mix of the two perspectives in the following sections. For example,
when we discuss about the posterior probability of the latent variables, we will
also mention that it is the "responsibilities" in the mixture model perspective.

## The Mixture Model Perspective

In [the previous section on](gmm-problem-formulation-gaussian-mixture-model),
we have actually already defined the Mixture Model Perspective of the Gaussian
Mixture Model.

### The Gaussian Mixture Model

To recap, a Gaussian mixture model is a density model where we combine a finite number of $K$ Gaussian distributions $\mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)$ so that

$$
\begin{aligned}
& p(\boldsymbol{x} ; \boldsymbol{\theta})=\sum_{k=1}^K \pi_k \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \mathbf{\Sigma}_k\right) && (\star) \\
& 0 \leqslant \pi_k \leqslant 1, \quad \sum_{k=1}^K \pi_k=1, && (\star\star) \\
\end{aligned}
$$ (eq:gmm-def-2)

where we defined $\boldsymbol{\theta}:=\left\{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k: k=1, \ldots, K\right\}$ as the collection of all parameters of the model. This convex combination of Gaussian distribution gives us significantly more flexibility for modeling complex densities than a simple Gaussian distribution (which we recover from $(\star \star)$ for $K=1$) {cite}`deisenroth_ong_faisal_2021`.

Overall, the main goal is to find the 3 parameters of the mixture model that best fit the data $\mathcal{S}$.

(gmm-responsibility)=
### The Responsibilities

Another quantity that will play an important role is the conditional probability of $\boldsymbol{z}$ given $\boldsymbol{x}$. We shall denote this quantity as the responsibility of the $k$-th component for generating the data point $\boldsymbol{x}$, and denote it as
$r^{(n)}_k$:

$$
\begin{aligned}
r^{(n)}_k \equiv p\left(z^{(n)}=k \mid \boldsymbol{x}\right) & = \frac{p\left(\boldsymbol{x} \mid z^{(n)}=k\right) p\left(z^{(n)}=k\right)}{p\left(\boldsymbol{x}\right)} \\
& =\frac{\pi_k \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)}{\sum_{k=1}^K \pi_k \mathcal{N}\left(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\right)}
\end{aligned}
$$ (eq:gmm-responsibility)

Therefore, mixture components have a high responsibility for a data point when the data point could be a plausible sample from that mixture component.

Note that $\boldsymbol{r}^{(n)}$ is a $K$ dimensional vector:

$$\boldsymbol{r}^{(n)} = \begin{bmatrix} r^{(n)}_1 \\ r^{(n)}_2 \\ \vdots \\ r^{(n)}_K \end{bmatrix} \in \mathbb{R}^K
$$

is a (normalized) probability vector, i.e., $\sum_{k} r^{(n)}_{k}=1$ with $r^{(n)}_{k} \geqslant 0$. This probability vector distributes probability mass among the $K$ mixture components, and we can think of $\boldsymbol{r}^{(n)}$ as a "soft assignment" of $\boldsymbol{x}^{(n)}$ to the $K$ mixture components. Therefore, the responsibility $r^{(n)}_{k}$ from {eq}`eq:gmm-responsibility` represents the probability that $\boldsymbol{x}^{(n)}$ has been generated by the $k$ th mixture component.

## The Latent Variable Perspective

Notice that if we approach the GMM from the latent variable perspective, we are more
interested in the probability of the latent variable
$\boldsymbol{z}^{(n)}$ given the data $\boldsymbol{x}^{(n)}$, as we will see later.

Within this model, we assume the following.

### The Generative Process

Consider a mental model that there are $K$ gaussian distributions, each representing a cluster. The data point $\boldsymbol{x}^{(n)}$ is generated by first sampling the latent variable $\boldsymbol{z}^{(n)}$ from a categorical distribution and then sampling the data point $\boldsymbol{x}^{(n)}$ from the corresponding Gaussian distribution. This is the generative process of the GMM.

More concretely, the sampling process can be described below.

```{prf:algorithm} GMM Sampling Process
:label: alg:gmm-sampling

The construction of this latent-variable model (see the corresponding graphical model in Figure 11.9) lends itself to a very simple sampling procedure (generative process) to generate data:

1. Sample $z^{(n)} \sim p(\boldsymbol{z})=\boldsymbol{\pi}$ where $z^{(n)}$ is a discrete random variable with $K$ possible values.

2. Sample $\boldsymbol{x}^{(n)} \sim p\left(\boldsymbol{x} \mid z^{(n)}\right)$.

In the first step, we select a mixture component $k$ at random according to $p(\boldsymbol{z})=\boldsymbol{\pi}$; in the second step we draw a sample from the corresponding mixture component $k$. When we discard the samples of the latent variable so that we are left with the $\boldsymbol{x}^{(n)}$, we have valid samples from the GMM. This kind of sampling, where samples of random variables depend on samples from the variable's parents in the graphical model, is called [ancestral sampling](https://en.wikipedia.org/wiki/Ancestral_sampling) {cite}`deisenroth_ong_faisal_2021`.
```

This generative process prompts a few questions:

- How do we define the categorical distribution $p(\boldsymbol{z})$ (i.e. the prior distribution of the latent variable $\boldsymbol{z}$)?
- How do we define the Gaussian distribution $p\left(\boldsymbol{x} \mid z^{(n)}\right)$ (i.e. the conditional distribution of the data $\boldsymbol{x}$ given the latent variable $\boldsymbol{z}$, also known as the likelihood)?
- How do we define the joint distribution $p\left(\boldsymbol{x}, \boldsymbol{z}\right)$ (i.e. the joint distribution of the data $\boldsymbol{x}$ and the latent variable $\boldsymbol{z}$)?
- How do we define the marginal distribution $p\left(\boldsymbol{x}\right)$ (i.e. the marginal distribution of the data $\boldsymbol{x}$)?
- How do we define the posterior distribution $p\left(\boldsymbol{z} \mid \boldsymbol{x}\right)$ (i.e. the posterior distribution of the latent variable $\boldsymbol{z}$ given the data $\boldsymbol{x}$)?

### Assumption 1: The Distribution of the Data Point $\boldsymbol{x}^{(n)}$ given the Latent Variable $\boldsymbol{z}^{(n)}$

Consider a mental model that there are $K$ gaussian distributions, each representing a cluster $C_k$. The data point $\boldsymbol{x}^{(n)}$ is generated by first sampling the latent variable $\boldsymbol{z}^{(n)}$ from a categorical distribution and then sampling the data point $\boldsymbol{x}^{(n)}$ from the corresponding Gaussian distribution. This is the generative process of the GMM.

We first start by defining the $K$ clusters, each represented by a different
(multivariate) gaussian distribution.

#### The Latent Clusters

Each cluster $C_k$ for $k=1, \ldots, K$ is represented by a multivariate gaussian distribution:

$$
\begin{aligned}
C_1 &:= \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \\
C_2 &:= \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) \\
&\vdots \\
C_K &:= \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \\
\end{aligned}
$$ (eq:cluster-def-gmm-2)

and the geometry of the clusters are completely determined by their
mean vectors $\boldsymbol{\mu}_k$ and covariance matrices $\boldsymbol{\Sigma}_k$.

This notation can be confusing because $C_k$ is not really a random variable, instead it is a probability distribution.
Consequently, the data points that $C_k$'s probability distribution generates are random variables, which we will define
next.

#### The Data Points $\boldsymbol{x}^{(n)}$ is the Likelihood

Any data point $\boldsymbol{x}$ can be generated by sampling from one of the $K$ clusters:

$$
\begin{aligned}
\boldsymbol{x}^{(n)} \in C_k &\sim \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \\
\end{aligned}
$$

which means the following:

$$
\begin{aligned}
\boldsymbol{x}^{(n)} \in C_k &:= \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \\
&= \frac{1}{\sqrt{(2\pi)^D \det{\boldsymbol{\Sigma}_k}}} \exp\left(-\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu}_k)^{\mathrm{T}} \boldsymbol{\Sigma}_k^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_k)\right) \quad \text{for } k = 1, \ldots, K.
\end{aligned}
$$ (eq:cluster-def-gmm-3)

where this distribution is parametrized by $\boldsymbol{\mu}_k$ is the mean vector of the $k$-th cluster, and $\boldsymbol{\Sigma}_k$ is the covariance matrix of the $k$-th cluster.

This formulation further allows us to interpret a specific data point $\boldsymbol{x}^{(n)}$ as a realization drawn from the probability distribution {eq}`eq:cluster-def-gmm-3` of a specific cluster $C_k$.

We can represent the distribution defined in {eq}`eq:cluster-def-gmm-3`
more concisely as:

$$
\begin{aligned}
\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} \mid Z^{(n)} = k &\sim \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
\begin{cases}
\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} \mid Z^{(n)} = 1 &\sim \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \\
\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} \mid Z^{(n)} = 2 &\sim \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) \\
&\vdots \\
\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} \mid Z^{(n)} = K &\sim \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \\
\end{cases}
\end{aligned}
$$ (eq:cluster-def-gmm-4)

Notice the similarity between the expression {eq}`eq:cluster-def-gmm-4` and the likelihood
expression in a Naive Bayes model? Yes, indeed this expression is none other than the
likelihood of observing the data point $\boldsymbol{x}^{(n)}$ given the latent variable $Z^{(n)} = k$.

#### The Likelihood of One Single Data Point $\boldsymbol{x}^{(n)}$

- Let $x^{(n)}$ denote the $n$-th data point, with $n = 1, \dots, N$.
- Let $z^{(n)}$ denote the latent variable corresponding to the $n$-th data point, representing the Gaussian component it belongs to. $z^{(n)}$ can take on values $1, \dots, K$, where $K$ is the number of Gaussian components.
- The likelihood of the $n$-th data point belonging to the $k$-th Gaussian component can be denoted as

  $$
  p(x^{(n)} | z^{(n)} = k ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \quad \text{for } k = 1, \dots, K
  $$

- Since each $p(x^{(n)} | z^{(n)} = k)$ is parametrized by the mean and covariance vector/matrix, we can write the below without ambiguity:

    $$
    p(x^{(n)} | z^{(n)} = k) = \mathcal{N}(x^{(n)} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    $$

    where $\mathcal{N}$ is the multivariate Gaussian distribution.

  Consequently, we obtain all $K$ components of the likelihood vector:

  $$
  \begin{aligned}
  \boldsymbol{L}^{(n)} &= \begin{bmatrix} p(x^{(n)} | z^{(n)} = 1) \\ p(x^{(n)} | z^{(n)} = 2) \\ \vdots \\ p(x^{(n)} | z^{(n)} = K) \end{bmatrix}_{K \times 1} \\
  &= \begin{bmatrix} \mathcal{N}(x^{(n)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \\ \mathcal{N}(x^{(n)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) \\ \vdots \\ \mathcal{N}(x^{(n)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \end{bmatrix}_{K \times 1} \\
  &= \begin{bmatrix} L_1^{(n)} \\ L_2^{(n)} \\ \vdots \\ L_K^{(n)} \end{bmatrix}_{K \times 1}
  \end{aligned}
  $$

  and all elements sum to 1, fully representing the likelihood of the $n$-th data point belonging to each of the $K$ Gaussian components.

#### The Likelihood of the Entire Dataset $\boldsymbol{X}$

We are only talking about the likelihood of a single data point $\boldsymbol{x}^{(n)}$ belonging to a specific cluster $C_k$ ($z^{(n)} = k$). We will now discuss how to compute the likelihood of the entire dataset $\boldsymbol{X}$ belonging to a specific cluster $C_k$.

Given $\mathcal{S} = \left\{ \boldsymbol{x}^{(1)}, \boldsymbol{x}^{(2)}, \ldots, \boldsymbol{x}^{(N)} \right\}$, the likelihood of the entire dataset $\boldsymbol{X}$ belonging to a specific cluster $C_k$ can be written as:

$$
\begin{aligned}
\boldsymbol{L} &= \begin{bmatrix} L_1^{(1)} & L_2^{(1)} & \cdots & L_K^{(1)} \\ L_1^{(2)} & L_2^{(2)} & \cdots & L_K^{(2)} \\ \vdots & \vdots & \ddots & \vdots \\ L_1^{(N)} & L_2^{(N)} & \cdots & L_K^{(N)} \end{bmatrix}_{N \times K} \\
&= \begin{bmatrix} p(\boldsymbol{x}^{(1)} | z^{(1)} = 1) & p(\boldsymbol{x}^{(1)} | z^{(1)} = 2) & \cdots & p(\boldsymbol{x}^{(1)} | z^{(1)} = K) \\ p(\boldsymbol{x}^{(2)} | z^{(2)} = 1) & p(\boldsymbol{x}^{(2)} | z^{(2)} = 2) & \cdots & p(\boldsymbol{x}^{(2)} | z^{(2)} = K) \\ \vdots & \vdots & \ddots & \vdots \\ p(\boldsymbol{x}^{(N)} | z^{(N)} = 1) & p(\boldsymbol{x}^{(N)} | z^{(N)} = 2) & \cdots & p(\boldsymbol{x}^{(N)} | z^{(N)} = K) \end{bmatrix}_{N \times K} \\
&= \begin{bmatrix} \left(\boldsymbol{L}^{(1)}\right)^T \\ \left(\boldsymbol{L}^{(2)}\right)^T \\ \vdots \\ \left(\boldsymbol{L}^{(N)}\right)^T \end{bmatrix}_{N \times K} \\
\end{aligned}
$$

### Assumption 2: The Latent Variable $\boldsymbol{z}$

We have discussed about the likelihood of a single data point $\boldsymbol{x}^{(n)}$ belonging to a specific cluster $C_k$ ($z^{(n)} = k$). The next logical question is to ask: what is the probability distribution of $\boldsymbol{z}$?

Similar to the feature vectors $\boldsymbol{x}^{(n)}$, the cluster assignment $z^{(n)}$ can also be
interpreted as realization drawn from a **latent** discrete random variable $Z$.

#### The Prior Distribution of $\boldsymbol{z}$

In contrast to the feature vectors $\boldsymbol{x}^{(n)}$, we do not observe (know) the true cluster indices $z^{(n)}$. After all, the goal of soft clustering is to estimate the cluster indices $z^{(n)}$. We obtain a soft clustering
method by estimating the cluster indices $z^{(n)}$ based solely on the data points in $\mathcal{S}$. To compute these estimates we assume that the (true) cluster indices $z^{(n)}$ are realizations of iid RVs with the common probability distribution (or probability mass function):

$$
\pi_k := \mathbb{P}\left(Z^{(n)} = k ; \boldsymbol{\pi}\right) \quad \text{for } k = 1, \ldots, K.
$$

where $\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \ldots & \pi_K \end{bmatrix}^{\mathrm{T}}$ is a $K$-dimensional vector of probabilities. It is also common to denote the prior distribution as a one-hot vector.

As mentioned in the previous step, one will soon realize that this is the **prior** in a Bayes model.
With this, we have answered the question of what the probability distribution of $Z$ is.

The (prior) probabilities are either assumed known or estimated from data. The choice for the probabilities $\pi_k$ could reflect some prior knowledge about different sizes of the clusters. For example, if cluster $C_1$ is known to be larger than cluster $C_2$, we might choose the prior probabilities such that $\pi_1 > \pi_2$ {cite}`jung_2023`.

#### The Categorical Distribution

Let's now discuss the probability distribution of the latent variable $Z$.

```{prf:definition} Categorical Distribution
:label: categorical-distribution-gmm

Let $Z$ be a discrete random variable with $K$ number of states.
Then $Z$ follows a categorical distribution with parameters $\boldsymbol{\pi}$ if

$$
\mathbb{P}(Z = k) = \pi_k \quad \text{for } k = 1, 2, \cdots, K
$$

Consequently, the PMF of the categorical distribution is defined more compactly as,

$$
\mathbb{P}(Z = k) = \prod_{k=1}^K \pi_k^{I\{Z = k\}}
$$

where $I\{Z = k\}$ is the indicator function that is equal to 1 if $Z = k$ and 0 otherwise.
```

More often, we use the [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) to represent the categorical distribution. The one-hot encoding is a vector of size $K$ where all elements are 0 except for the $k$-th element which is 1. For example, if $K = 3$, the one-hot encoding of $k = 2$ is $\mathbf{y} = \begin{bmatrix} 0 & 1 & 0 \end{bmatrix}^{\mathrm{T}}$.

```{prf:definition} Categorical (Multinomial) Distribution
:label: categorical-multinomial-distribution-gmm

This formulation is adopted by Bishop's{cite}`bishop2007`, the categorical distribution is defined as

$$
\mathbb{P}(\mathbf{Z} = \mathbf{z}; \boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_k}
$$ (eq:categorical-distribution-bishop-gmm)

where

$$
\mathbf{z} = \begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_K \end{bmatrix}
$$

is an one-hot encoded vector of size $K$,

The $z_k$ is the $k$-th element of $\mathbf{z}$, and is equal to 1 if $Y = k$ and 0 otherwise.
The $\pi_k$ is the $k$-th element of $\boldsymbol{\pi}$, and is the probability of $\mathbf{Z} = k$.

This notation alongside with the indicator notation in the previous definition allows us to manipulate
the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) in a more compact way.
```

#### Prior Distribution of the Entire Dataset $\mathcal{S}$

The prior distribution $\boldsymbol{\pi}$ is shared by all the data points in $\mathcal{S}$.

- Let $z^{(n)}$ denote the latent variable corresponding to the $n$-th data point, representing the Gaussian component it belongs to. $z^{(n)}$ can take on values $1, \dots, K$, where $K$ is the number of Gaussian components.
- The prior probability of the $k$-th Gaussian component can be denoted as $P(Z^{(n)} = k)$, for $k = 1, \dots, K$.
- These probabilities can be represented as a vector

    $$
    \boldsymbol{\pi} = \begin{bmatrix} p(z^{(n)} = 1) \\ p(z^{(n)} = 2) \\ \vdots \\ p(z^{(n)} = K) \end{bmatrix}_{K \times 1} = \begin{bmatrix} \pi_1 \\ \pi_2 \\ \vdots \\ \pi_K \end{bmatrix}_{K \times 1}
    $$

- The sum of all prior probabilities should be equal to 1, as they represent probabilities: $\sum_{k=1}^K p(z^{(n)} = k) = 1$.


In the context of GMM, the prior probabilities can be interpreted as the probability that a randomly chosen sample belongs to the $k$-th Gaussian component.

Note that this prior is a global one shared across all data points. In other words, the prior probability of a data point belonging to a Gaussian component is the same as the prior probability of any other data point belonging to the same Gaussian component.

### Assumption 3: The Joint Distribution of $\boldsymbol{x}^{(n)}$ and $\boldsymbol{z}^{(n)}$

So far, what have we gotten? We have defined two distributions, one is the likelihood of
observation $\boldsymbol{x}^{(n)}$ given the cluster assignment $z^{(n)}$ and the other is the prior
distribution of the cluster assignment $z^{(n)}$.

$$
\begin{aligned}
\boldsymbol{x}^{(n)} \mid z^{(n)}=k &\sim \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) && \text{for } k = 1, \ldots, K \\
z^{(n)}=k &\sim \text{Cat}(\boldsymbol{\pi}) && \text{for } k = 1, \ldots, K.
\end{aligned}
$$

Now, recall that when the likelihood and prior are
multiplied together, we obtain the **joint distribution** of the data and the cluster assignment,
as follows:

$$
\begin{aligned}
\left(z^{(n)}=k, \boldsymbol{x}^{(n)}\right) &\sim \text{Cat}(\boldsymbol{\pi}) \times \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \quad \text{for } k = 1, \ldots, K.
\end{aligned}
$$

which is equivalent to the following:

$$
\overbrace{\mathbb{P}\left(\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)}, Z^{(n)} = k ; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\pi}\right)}^{\text{joint distribution}} = \overbrace{\mathbb{P}\left(Z^{(n)} = k ; \boldsymbol{\pi}\right)}^{\text{prior}=\text{Cat}(\boldsymbol{\pi})}\overbrace{\mathbb{P}\left(\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} \mid Z^{(n)} = k ; \boldsymbol{\mu}, \boldsymbol{\Sigma}\right)}^{\text{likelihood}=\mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
$$

where

- $\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} \mid Z^{(n)} = k \sim \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) $ is the probability distribution of the data points $\boldsymbol{x}^{(n)}$ given the cluster assignment $z^{(n)}=k$. This is also known as the mixture component.
- $Z = k \sim \text{Cat}(\boldsymbol{\pi})$ is the probability distribution of the cluster assignment $z^{(n)}=k$, also known as the mixing coefficient $\pi_k$.

#### Why is the Joint Distribution the Product of the Likelihood and Prior?

One question that might come to mind is why the joint distribution is the product of the likelihood and prior. The answer is that the joint distribution is the product of the likelihood and prior because of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule_(probability)).

Let's just state the base case to see why.

By Bayes' rule, we have:

$$
P(A | B) = \frac{P(A \cap B)}{P(B)} \implies P(A \cap B) = P(A | B) P(B) \implies P(A ,B) = P(B | A) P(A)
$$

and if we set $A = x^{(n)}$ and $B = z^{(n)} = k$, then we have:

$$
P(x^{(n)} | z^{(n)} = k) = \frac{P(x^{(n)}, z^{(n)} = k)}{P(z^{(n)} = k)} \implies P(x^{(n)} ,z^{(n)} = k) = P(x^{(n)} | z^{(n)} = k) P(z^{(n)} = k)
$$

and $\cap$ is the intersection symbol, so we have the joint probability of the data point $x^{(n)}$ and the latent label $z^{(n)} = k$.

#### Weighted Likelihood

Recall that we defined the prior, $\boldsymbol{\pi}$, as the probability of a data point belonging to a Gaussian component, and the likelihood as the probability of a data point given the Gaussian component it belongs to, $\mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

Then we have just seen that the joint distribution of the data and the cluster assignment is the product of the prior and the likelihood. This is also known as the **weighted likelihood**.

The weighted likelihood is the joint probability of the data point $x^{(n)}$ and the latent label $z^{(n)} = k$. Basically it answers the question: "What is the probability of observing the data point $x^{(n)}$ and the latent label $z^{(n)} = k$?"

To see why, first consider how we define the **weighted likelihood**:

$$
P(x^{(n)}, z^{(n)} = k) = P(x^{(n)} | z^{(n)} = k) P(z^{(n)} = k)
$$

where $P(x^{(n)} | z^{(n)} = k)$ is the likelihood of the $n$-th data point belonging to the $k$-th Gaussian component, and $P(z^{(n)} = k)$ is the prior probability of the $k$-th Gaussian component.

First, some intuition, it is called weighted because the likelihood is weighted by the prior probability of the latent label $z^{(n)} = k$. In other words, if we have likelihoods $P(x^{(n)} | z^{(n)} = 2)$ for $k=2$
to be say $0.2$ and the prior probability of $z^{(n)} = 2$ to be $0.9$, then the weighted likelihood is $0.2 \times 0.9 = 0.18$ because we have super high confidence that the data point $x^{(n)}$ belongs to the $k=2$ Gaussian component. However, if the prior probability of $z^{(n)} = 2$ is $0.1$, then the weighted likelihood is $0.2 \times 0.1 = 0.02$ because we have low confidence that the data point $x^{(n)}$ belongs to the $k=2$ Gaussian component so the "likelihood" got weighed down by the low prior probability.


#### Weighted Likelihood of One Single Data Point $\boldsymbol{x}^{(n)}$

The weighted likelihood of a single data point $\boldsymbol{x}^{(n)}$ is obtained by multiplying the likelihood of the data point belonging to each Gaussian component by the corresponding mixing coefficient (weight) of that component. Let $\boldsymbol{\pi}$ be the vector of mixing coefficients, with $\pi_k$ representing the weight of the $k$-th Gaussian component. Then, the weighted likelihood of the data point $\boldsymbol{x}^{(n)}$ can be written as:

$$
\begin{aligned}
\boldsymbol{W}^{(n)} &= \begin{bmatrix} \pi_1 p(x^{(n)} | z^{(n)} = 1) \\ \pi_2 p(x^{(n)} | z^{(n)} = 2) \\ \vdots \\ \pi_K p(x^{(n)} | z^{(n)} = K) \end{bmatrix}_{K \times 1} \\
&= \begin{bmatrix} \pi_1 \mathcal{N}(x^{(n)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \\ \pi_2 \mathcal{N}(x^{(n)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) \\ \vdots \\ \pi_K \mathcal{N}(x^{(n)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \end{bmatrix}_{K \times 1} \\
&= \begin{bmatrix} W_1^{(n)} \\ W_2^{(n)} \\ \vdots \\ W_K^{(n)} \end{bmatrix}_{K \times 1}
\end{aligned}
$$

Here, $\boldsymbol{W}^{(n)}$ is the vector of weighted likelihoods of the data point $\boldsymbol{x}^{(n)}$ belonging to each of the $K$ Gaussian components, and $W_k^{(n)}$ represents the weighted likelihood of the data point $\boldsymbol{x}^{(n)}$ belonging to the $k$-th Gaussian component.

#### Weighted Likelihood of the Entire Dataset $\boldsymbol{X}$

To compute the weighted likelihood of the entire dataset $\boldsymbol{X}$, we need to calculate the weighted likelihood for each data point $\boldsymbol{x}^{(n)}$ and then combine them. For this purpose, we can represent the weighted likelihood of the entire dataset as a matrix $\boldsymbol{W}$ of size $N \times K$, where $N$ is the number of data points and $K$ is the number of Gaussian components:

$$
\boldsymbol{W} = \begin{bmatrix}
W_1^{(1)} & W_2^{(1)} & \cdots & W_K^{(1)} \\
W_1^{(2)} & W_2^{(2)} & \cdots & W_K^{(2)} \\
\vdots  & \vdots  & \ddots & \vdots  \\
W_1^{(N)} & W_2^{(N)} & \cdots & W_K^{(N)}
\end{bmatrix}_{N \times K}
$$

Each row of the matrix $\boldsymbol{W}$ corresponds to the weighted likelihood vector $\boldsymbol{W}^{(n)}$ for a data point $\boldsymbol{x}^{(n)}$. To obtain the weighted likelihood of the entire dataset, we can either sum or compute the product of all elements in the matrix $\boldsymbol{W}$, depending on the desired objective (e.g., maximizing the log-likelihood).


Now the returned is a matrix of shape $(N, K)$, where $N$ is the number of data points and $K$ is the number of Gaussian components. The $n$-th row and $k$-th column element is the weighted likelihood of the $n$-th data point belonging to the $k$-th Gaussian component.

$$
\begin{aligned}
\boldsymbol{W} &= \begin{bmatrix} p(z^{(1)} = 1) p(x^{(1)} \mid z^{(1)} = 1) & p(z^{(1)} = 2) p(x^{(1)} \mid z^{(1)} = 2) & \cdots & p(z^{(1)} = K) p(x^{(1)} \mid z^{(1)} = K) \\ p(z^{(2)} = 1) p(x^{(2)} \mid z^{(2)} = 1) & p(z^{(2)} = 2) p(x^{(2)} \mid z^{(2)} = 2) & \cdots & p(z^{(2)} = K) p(x^{(2)} \mid z^{(2)} = K) \\ \vdots & \vdots & \ddots & \vdots \\ p(z^{(N)} = 1) p(x^{(N)} \mid z^{(N)} = 1) & p(z^{(N)} = 2) p(x^{(N)} \mid z^{(N)} = 2) & \cdots & p(z^{(N)} = K) p(x^{(N)} \mid z^{(N)} = K) \end{bmatrix}_{N \times K} \\
&= \begin{bmatrix} \pi_1 \mathcal{N}(x^{(1)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) & \pi_2 \mathcal{N}(x^{(1)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) & \cdots & \pi_K \mathcal{N}(x^{(1)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \\ \pi_1 \mathcal{N}(x^{(2)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) & \pi_2 \mathcal{N}(x^{(2)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) & \cdots & \pi_K \mathcal{N}(x^{(2)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \\ \vdots & \vdots & \ddots & \vdots \\ \pi_1 \mathcal{N}(x^{(N)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) & \pi_2 \mathcal{N}(x^{(N)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) & \cdots & \pi_K \mathcal{N}(x^{(N)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \end{bmatrix}_{N \times K} \\
&= \begin{bmatrix} p(x^{(1)} ,z^{(1)} = 1) & p(x^{(1)} ,z^{(1)} = 2) & \cdots & p(x^{(1)} ,z^{(1)} = K) \\ p(x^{(2)} ,z^{(2)} = 1) & p(x^{(2)} ,z^{(2)} = 2) & \cdots & p(x^{(2)} ,z^{(2)} = K) \\ \vdots & \vdots & \ddots & \vdots \\ p(x^{(N)} ,z^{(N)} = 1) & p(x^{(N)} ,z^{(N)} = 2) & \cdots & p(x^{(N)} ,z^{(N)} = K) \end{bmatrix}_{N \times K}
\end{aligned}
$$

In code, we need to separate the weighted likelihood matrix $\boldsymbol{W}$ into two matrices,
as follows:

1. Mixing coefficients matrix, $\boldsymbol{\Pi}$, of shape $(N \times K)$, where each row contains the mixing coefficients $\boldsymbol{\pi}$ repeated for each data point:

$$
\boldsymbol{\Pi} = \begin{bmatrix} \pi_1 & \pi_2 & \cdots & \pi_K \\ \pi_1 & \pi_2 & \cdots & \pi_K \\ \vdots & \vdots & \ddots & \vdots \\ \pi_1 & \pi_2 & \cdots & \pi_K \end{bmatrix}_{N \times K}
$$

2. Likelihood matrix, $\boldsymbol{L}$, of shape $(N \times K)$, where each element $(i, j)$ represents the likelihood of the $i$-th data point belonging to the $j$-th Gaussian component:

$$
\boldsymbol{L} = \begin{bmatrix} \mathcal{N}(x^{(1)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) & \mathcal{N}(x^{(1)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) & \cdots & \mathcal{N}(x^{(1)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \\ \mathcal{N}(x^{(2)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) & \mathcal{N}(x^{(2)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) & \cdots & \mathcal{N}(x^{(2)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \\ \vdots & \vdots & \ddots & \vdots \\ \mathcal{N}(x^{(N)} | \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) & \mathcal{N}(x^{(N)} | \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2) & \cdots & \mathcal{N}(x^{(N)} | \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K) \end{bmatrix}_{N \times K}
$$

Now, you can obtain the weighted likelihood matrix $\boldsymbol{W}$ by performing element-wise multiplication (Hadamard product) of the mixing coefficients matrix $\boldsymbol{\Pi}$ and the likelihood matrix $\boldsymbol{L}$:

$$
\boldsymbol{W} = \boldsymbol{\Pi} \odot \boldsymbol{L}
$$

### Joint Distribution Fully Determines the Model

With the joint distribution defined, the model is fully determined. Why do we say so?
Because the joint distribution of the data point $\boldsymbol{x}^{(n)}$ and the latent variable $z^{(n)}$ is fully determined by the parameters $\boldsymbol{\theta}$, which are the model parameters.

Consequently, if we want to find the marginal, we need to integrate out the latent variable $z^{(n)}$ from the joint distribution. Then subsequently, we can also find the posterior distribution of the latent variable $z^{(n)}$ by using Bayes' rule. Therefore, when we say the joint distribution fully determines
the model, what is really means is that we have all the necessary tools to find anything related
to the random variables $z^{(n)}$ and $\boldsymbol{x}^{(n)}$.

### The Gaussian Mixture Model and the Marginal Distribution

#### The Gaussian Mixture Model

Recall that we defined our Gaussian Mixture Model as a linear combination of $K$ multivariate Gaussian distributions:

$$
\overbrace{\mathbb{P}\left(\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\pi}\right)}^{\text{marginal distribution}} = \sum_{k=1}^K \overbrace{\mathbb{P}\left(Z^{(n)} = k ; \boldsymbol{\pi}\right)}^{\text{prior}=\text{Cat}(\boldsymbol{\pi})}\overbrace{\mathbb{P}\left(\boldsymbol{X}^{(n)} = \boldsymbol{x}^{(n)} \mid Z^{(n)} = k ; \boldsymbol{\mu}, \boldsymbol{\Sigma}\right)}^{\text{likelihood}=\mathcal{N}(\boldsymbol{x} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
$$

We now claim that the mixture of $K$ multivariate Gaussian distributions is a valid distribution,
and it is none other than the marginal distribution of $\boldsymbol{X}^{(n)}$.

#### The Marginal Distribution

We go back to fundamentals and ask what is the marginal distribution of a random variable $X$?

In our setting, it is the probability of the data point $x^{(n)}$. Basically it
answers the question: "What is the probability of observing the data point $x^{(n)}$?"

Since $\boldsymbol{x}^{(n)}$ is a $D$-dimensional vector, we can think of it as a point in a $D$-dimensional space. The marginal distribution is the probability of observing this point in this space.
Since it is in high dimensions usually, $\boldsymbol{x}^{(n)}$ is usually a point in a high-dimensional space,
and hence follow a multi-variate distribution.

The **marginal distribution** of the data points $\mathbf{x}^{(n)}$:

$$
\begin{aligned}
\overbrace{p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\pi}\right)}^{\text{marginal distribution}} &= \sum_{k=1}^K p\left(z^{(n)} = k ; \boldsymbol{\pi}\right) p\left(\boldsymbol{x}^{(n)} \mid z^{(n)} = k ; \boldsymbol{\mu}, \boldsymbol{\Sigma}\right) \\
&= \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}^{(n)} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \\
\end{aligned}
$$ (eq-cluster-gmm-final)

where this is the actual **Gaussian Mixture Model** that we are trying to fit to the data.

One question is how do we get this marginal distribution? We can get it by marginalizing out the latent variable $z^{(n)}$ from the joint distribution. And what does it mean by "marginalizing out" the latent variable $z^{(n)}$? This concept is tied to the concept of conditional probability and the law of total probability.

#### Marginalizing Out the Latent Variable

Recall marginal distribution is none other than the denominator of the posterior distribution in Bayes' rule:

$$
\begin{aligned}
\mathbb{P}(Y \mid X) = \frac{\mathbb{P}(X \mid Y) \mathbb{P}(Y)}{\mathbb{P}(X)}.
\end{aligned}
$$

and the denominator is called the marginal distribution. The expansion of the denominator
as a summation of the numerator uses the [law of total probability](https://en.wikipedia.org/wiki/Law_of_total_probability).

In other words, to marginalize out the latent variable $Z$, we can simply sum over all possible values of $Z$.
As a result, we get a mixture of $K$ multivariate Gaussian distributions, where
each Gaussian distribution is defined by the parameters $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$
and its corresponding mixing coefficient $\pi_k$.

Thus, we have concluded in defining a systematic distribution in which our data points $\mathbf{x}^{(n)}$
come from. This is the **Gaussian Mixture Model**.

But more is to come, because estimating the parameters is not simple, there is no closed-form solution
to the problem. And if you look closely enough, the marginal distribution depends on
both the parameters $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$ and $\boldsymbol{\pi}$.
However, unlike our classification problem with true labels $y$, we do not have
access to the true labels $z$ in this case. We only have access to the data points $\mathbf{x}$.
This is a problem because we can no longer "estimate" the empirical distribution of
$z$ by simply counting the number of occurrences of each $z$ in the dataset. But hope is not lose
as we can make use of the **expectation-maximization (EM) algorithm** to solve this problem.

#### Marginal of One Single Data Point $\boldsymbol{x}^{(n)}$

The marginal of a single data point $\boldsymbol{x}^{(n)}$ is obtained by summing the weighted likelihoods of the data point belonging to each Gaussian component. Mathematically, it can be written as:

$$
\boldsymbol{M}^{(n)} = \begin{bmatrix} p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\pi}\right) \end{bmatrix} = \begin{bmatrix} \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}^{(n)} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \end{bmatrix}
$$

#### Marginal of the Entire Dataset $\boldsymbol{X}$

The marginal of the entire dataset $\boldsymbol{X}$ is collated as follows:

$$
\boldsymbol{M} = \begin{bmatrix} \boldsymbol{M}^{(1)} \\ \vdots \\ \boldsymbol{M}^{(N)} \end{bmatrix} = \begin{bmatrix} \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}^{(1)} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \\ \vdots \\ \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}^{(N)} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \end{bmatrix}
$$

### The Posterior Distribution

Now we can answer the posterior distribution of the cluster assignment $z^{(n)}$ given the data points $\mathbf{x}^{(n)}$.

$$
\begin{aligned}
\overbrace{p\left(z^{(n)}=k \mid \mathbf{x}^{(n)} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\pi}\right)}^{\text {posterior }} &= \frac{\overbrace{p\left(z^{(n)}=k ; \boldsymbol{\pi}\right)}^{\text {prior }} \cdot \overbrace{p\left(\mathbf{x}^{(n)} \mid z^{(n)}=k ; \boldsymbol{\mu}, \boldsymbol{\Sigma}\right)}^{\text {likelihood }}}{\underbrace{p\left(\mathbf{x}^{(n)} ; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\pi}\right)}_{\text {marginal }}} \\
&= \frac{\pi_k \mathcal{N}(\boldsymbol{x}^{(n)} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x}^{(n)} ; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)} \\
\end{aligned}
$$ (eq-cluster-gmm-posterior)

This is the degree of belonging is none other than the posterior distribution!


#### Posterior of One Single Data Point $\boldsymbol{x}^{(n)}$

Using the posterior distribution equation {eq}`eq-cluster-gmm-posterior`, we can calculate the posterior probability of a single data point $\boldsymbol{x}^{(n)}$ belonging to each of the $K$ Gaussian components. The result is a vector of size $K \times 1$, where the $k$-th element represents the posterior probability of the data point $\boldsymbol{x}^{(n)}$ belonging to the $k$-th Gaussian component:

$$
\boldsymbol{R}^{(n)}=\left[\begin{array}{c}
p\left(z^{(n)}=1 \mid \boldsymbol{x}^{(n)}\right) \\
p\left(z^{(n)}=2 \mid \boldsymbol{x}^{(n)}\right) \\
\vdots \\
p\left(z^{(n)}=K \mid \boldsymbol{x}^{(n)}\right)
\end{array}\right]_{K \times 1}
$$

Here, $\boldsymbol{R}^{(n)}$ is the posterior probability vector for the data point $\boldsymbol{x}^{(n)}$.

There is a reason we use $\boldsymbol{R}$ instead of say $\boldsymbol{P}$ for shorthand
representation of the posterior probability vector. The reason is that $\boldsymbol{R}$ is
also known as the **responsibility** of the $k$-th Gaussian component for the data point $\boldsymbol{x}^{(n)}$, which we will see later.

#### Posterior of the Entire Dataset $\boldsymbol{X}$

To compute the posterior probability of the entire dataset $\boldsymbol{X}$, we need to calculate the posterior probability for each data point $\boldsymbol{x}^{(n)}$ and then combine them. For this purpose, we can represent the posterior probability of the entire dataset as a matrix $\boldsymbol{R}$ of size $N \times K$, where $N$ is the number of data points and $K$ is the number of Gaussian components:

$$
\begin{aligned}
\boldsymbol{R} &= \begin{bmatrix} p\left(z^{(1)}=1 \mid \boldsymbol{x}^{(1)}\right) & p\left(z^{(1)}=2 \mid \boldsymbol{x}^{(1)}\right) & \cdots & p\left(z^{(1)}=K \mid \boldsymbol{x}^{(1)}\right) \\ p\left(z^{(2)}=1 \mid \boldsymbol{x}^{(2)}\right) & p\left(z^{(2)}=2 \mid \boldsymbol{x}^{(2)}\right) & \cdots & p\left(z^{(2)}=K \mid \boldsymbol{x}^{(2)}\right) \\ \vdots & \vdots & \ddots & \vdots \\ p\left(z^{(N)}=1 \mid \boldsymbol{x}^{(N)}\right) & p\left(z^{(N)}=2 \mid \boldsymbol{x}^{(N)}\right) & \cdots & p\left(z^{(N)}=K \mid \boldsymbol{x}^{(N)}\right) \end{bmatrix} \\
\end{aligned}
$$

Each row of the matrix $\boldsymbol{P}$ corresponds to the posterior probability vector $\boldsymbol{P}^{(n)}$ for a data point $\boldsymbol{x}^{(n)}$. The posterior probability of the entire dataset can be used to assess the overall clustering quality, assign data points to the most probable cluster, or update the model parameters in an iterative manner (e.g., using the Expectation-Maximization algorithm).


## Parameter Estimation (Mixture Model Perspective)

Assume we are given a dataset $\mathcal{S}$

$$
\mathcal{S}=\left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(N)}\right\}
$$

where each data point $\boldsymbol{x}_{n}$ are drawn i.i.d. from an unknown distribution $\mathcal{D}$ defined as:

$$
\begin{aligned}
\mathcal{D} &= \mathbb{P}\left(\mathcal{X}, \mathcal{Z} ; \boldsymbol{\theta} \right) \\
            &= \mathbb{P}_{\boldsymbol{\theta}}\left(\mathcal{X}, \mathcal{Z} \right) \\
            &= \mathbb{P}_{\left\{\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}\right\}}\left(\mathcal{X}, \mathcal{Z} \right) \\
\end{aligned}
$$

but since $\mathcal{Z}$ is treated as a latent variable, we only have information to:

$$
\begin{aligned}
\mathcal{D} &= \mathbb{P}\left(\mathcal{X} ; \boldsymbol{\theta} \right) \\
            &= \mathbb{P}_{\boldsymbol{\theta}}\left(\mathcal{X} \right) \\
            &= \mathbb{P}_{\left\{\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}\right\}}\left(\mathcal{X} \right) \\
\end{aligned}
$$

Our objective is to find a good approximation/representation of this unknown distribution $\mathbb{P}_{\left\{\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}\right\}}\left(\mathcal{X} \right)$ by means of a GMM with $K$ mixture components. The parameters of the GMM are the $K$ means $\boldsymbol{\mu}_{k}$, the covariances $\boldsymbol{\Sigma}_{k}$, and mixture weights $\pi_{k}$. We summarize all these free parameters in the symbol $\boldsymbol{\theta}$:

$$
\boldsymbol{\theta}:=\left\{\pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}: k=1, \ldots, K\right\}
$$

See [the section here](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#The_symbols) for more information.

### The Vectorized Parameters

However, to facilitate the notation, we will use the following vectorized representation of the parameters:

$$
\boldsymbol{\theta}:=\left\{\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}\right\}
$$

#### The Mixture Weights $\boldsymbol{\pi}$

$\boldsymbol{\pi}$ is a $K$-dimensional vector of mixture weights:

$$
\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \cdots & \pi_K \end{bmatrix}^{\mathrm{T}} \in \mathbb{R}^K
$$

and can be broadcasted to

$$
\boldsymbol{\pi} = \begin{bmatrix} \pi_1 & \pi_2 & \cdots & \pi_K \\ \pi_1 & \pi_2 & \cdots & \pi_K \\ \vdots & \vdots & \ddots & \vdots \\ \pi_1 & \pi_2 & \cdots & \pi_K \end{bmatrix} \in \mathbb{R}^{N \times K}
$$

for Hamadard product with $\boldsymbol{R}$.

#### The Means $\boldsymbol{\mu}$

These are the means of the Gaussian components in the Gaussian Mixture Model.

Let

$$
\boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \dots, \boldsymbol{\mu}_K
$$

be the mean vectors of the Gaussian components, with

$$
\boldsymbol{\mu}_k = \begin{bmatrix} \mu_{k1} \\ \mu_{k2} \\ \vdots \\ \mu_{kD} \end{bmatrix}_{D \times 1} \in \mathbb{R}^D
$$

being a column vector representing the mean of the $k$-th Gaussian component and $D$ being the number of features.

Thus collating all $K$ mean vectors $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \dots, \boldsymbol{\mu}_K$ into a matrix $M$ of dimensions $(K, D)$, we have

$$
\boldsymbol{M} = \begin{bmatrix} \boldsymbol{\mu}_1^T \\ \boldsymbol{\mu}_2^T \\ \vdots \\ \boldsymbol{\mu}_K^T \end{bmatrix}_{K \times D} = \begin{bmatrix} \mu_{11} & \mu_{12} & \cdots & \mu_{1D} \\ \mu_{21} & \mu_{22} & \cdots & \mu_{2D} \\ \vdots & \vdots & \ddots & \vdots \\ \mu_{K1} & \mu_{K2} & \cdots & \mu_{KD} \end{bmatrix}_{K \times D}.
$$

#### The Covariances $\boldsymbol{\Sigma}$

`self.covariances_`: These are the covariance matrices of the Gaussian components in the Gaussian Mixture Model. In the context of GMM, `self.covariances_` is a 3D array of shape `(num_components, num_features, num_features)`, where each "slice" along the first axis represents the covariance matrix of the corresponding Gaussian component. Let

$$
\boldsymbol{\Sigma}_1, \boldsymbol{\Sigma}_2, \dots, \boldsymbol{\Sigma}_K
$$

be the covariance matrices of the Gaussian components, with each $\boldsymbol{\Sigma}_k$ being a symmetric positive-definite matrix of dimensions $(D, D)$:

$$
\boldsymbol{\Sigma}_k = \begin{bmatrix} \sigma_{k11} & \sigma_{k12} & \cdots & \sigma_{k1D} \\ \sigma_{k21} & \sigma_{k22} & \cdots & \sigma_{k2D} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{kD1} & \sigma_{kD2} & \cdots & \sigma_{kDD} \end{bmatrix}.
$$

The `self.covariances_` array can be represented as a tensor $\boldsymbol{C}$ with dimensions $(K, D, D)$, where the $k$-th "slice" is the covariance matrix $\boldsymbol{\Sigma}_k$.

$$
\boldsymbol{C} = \begin{bmatrix} \boldsymbol{\Sigma}_1 \\ \boldsymbol{\Sigma}_2 \\ \vdots \\ \boldsymbol{\Sigma}_K \end{bmatrix}_{K \times D \times D}
$$

### Likelihood and Log-Likelihood of Marginal Distribution

As with any probabilistic model that requires parameter estimation, we need to define a likelihood function for the dataset $\mathcal{S}$.

In the following, we detail how to obtain a maximum likelihood estimate $\widehat{\boldsymbol{\theta}}$ of the model parameters $\boldsymbol{\theta}$. We start by writing down the likelihood of the **marginal likelihood of the observing the data**, i.e., the predictive distribution of the training data given the parameters. We exploit our i.i.d. assumption, which leads to the factorized likelihood

$$
\begin{aligned}
& \overbrace{p\left(\mathcal{S} = \left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(n)}\right\};\boldsymbol{\theta}\right)}^{\mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S} = \left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(n)}\right\}\right)} &&= \prod_{n=1}^{N} p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right) \\
& p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right) &&= \sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right),
\end{aligned}
$$ (eq-gmm-likelihood-1)

where every individual likelihood term $p\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\theta}\right)$ is a Gaussian mixture density.

Then we obtain the log-likelihood as

$$
\begin{aligned}
\log p\left(\mathcal{S} = \left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(n)}\right\};\boldsymbol{\theta}\right) &= \log \prod_{n=1}^{N} p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right) \\
&= \sum_{n=1}^{N} \log p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right) \\
&= \underbrace{\sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}_{\log\mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S} = \left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(n)}\right\}\right)} .
\end{aligned}
$$ (eq-gmm-log-likelihood-1)

We will abbreviate the log-likelihood as $\mathcal{L}\left(\boldsymbol{\theta} \mid \mathcal{S} = \left\{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(n)}\right\}\right)$ when the context is clear.

### No Closed-Form Solution

We aim to find parameters $\widehat{\boldsymbol{\theta}}^{*}$ that maximize the log-likelihood $\mathcal{L}$ defined in {eq}`eq-gmm-log-likelihood-1`. Our "normal" procedure would be to compute the gradient $\mathrm{d} \mathcal{L} / \mathrm{d} \boldsymbol{\theta}$ of the $\log$-likelihood with respect to the model parameters $\boldsymbol{\theta}$, set it to $\mathbf{0}$, and solve for $\boldsymbol{\theta}$. However, unlike our previous examples for maximum likelihood estimation (e.g., when we discussed [linear regression](../../influential/linear_regression/02_concept.md)), we cannot obtain a closed-form solution. However, we can exploit an iterative scheme to find good model parameters $\widehat{\boldsymbol{\theta}}$, which will turn out to be the EM algorithm for GMMs. The key idea is to update one model parameter at a time while keeping the others fixed {cite}`deisenroth_ong_faisal_2021`.

```{prf:remark} Remark: Closed-Form Solution for Single Gaussian
:label: prf:remark-gmm-closed-form

If we were to consider a single Gaussian as the desired density, the sum over $k$ in {eq}`eq-gmm-log-likelihood-1` vanishes, and the log can be applied directly to the Gaussian component, such that we get

$$
\log \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma})=-\frac{D}{2} \log (2 \pi)-\frac{1}{2} \log \operatorname{det}(\boldsymbol{\Sigma})-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})
$$

This simple form allows us to find closed-form maximum likelihood estimates of $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$, as discussed in [the chapter on maximum likelihood estimation](estimation-theory-mle-common-distributions). In {eq}`eq-gmm-log-likelihood-1`, we cannot move the log into the sum over $k$ so that we cannot obtain a simple closed-form maximum likelihood solution {cite}`deisenroth_ong_faisal_2021`.

Overall, what this means is that we can obtain a closed-form solution for the parameters of a single Gaussian, but not for a mixture of Gaussians.
```

To find out more why this constitutes a problem, one can read
section 9.2.1 in Bishop, Christopher M.'s book "Pattern Recognition and Machine Learning".

### Parameter Estimation (The Necessary Conditions)

Even though there is no closed form solution, we can still use iterative gradient-based optimization to find good model parameters $\widehat{\boldsymbol{\theta}}$. Consequently,
Any local optimum of a function exhibits the property that its gradient with respect to the parameters must vanish (necessary condition).

In our case, we obtain the following necessary conditions when we optimize the log-likelihood in {eq}`eq-gmm-log-likelihood-1` with respect to the GMM parameters $\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}$ :

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_{k}} & =\mathbf{0}^{\top} &&\Longleftrightarrow&& \sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}}=\mathbf{0}^{\top}, \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_{k}} & =\mathbf{0} &&\Longleftrightarrow&& \sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}=\mathbf{0}, \\
\frac{\partial \mathcal{L}}{\partial \pi_{k}} & =0 &&\Longleftrightarrow&& \sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\theta}\right)}{\partial \pi_{k}}=0 .
\end{aligned}
$$


In matrix/vector form, we have

the derivative of the log-likelihood with respect to the mean parameters $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \ldots, \boldsymbol{\mu}_K$ is

$$
\nabla_{\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \ldots, \boldsymbol{\mu}_K} \mathcal{L} = \mathbf{0}_{K \times D} \Longleftrightarrow \begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_1} \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_2} \\
\vdots \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_K}
\end{bmatrix} = \mathbf{0}_{K \times D}
$$

the derivative of the log-likelihood with respect to the covariance parameters $\boldsymbol{\Sigma}_1, \boldsymbol{\Sigma}_2 \ldots, \boldsymbol{\Sigma}_K$ is

$$
\nabla_{\boldsymbol{\Sigma}_1, \boldsymbol{\Sigma}_2 \ldots, \boldsymbol{\Sigma}_K} \mathcal{L} = \mathbf{0}_{K \times D \times D} \Longleftrightarrow \begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_1} \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_2} \\
\vdots \\
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_K}
\end{bmatrix} = \mathbf{0}_{K \times D \times D}
$$

the derivative of the log-likelihood with respect to the mixing coefficients $\pi_1, \pi_2 \ldots, \pi_K$ is

$$
\nabla_{\pi_1, \pi_2 \ldots, \pi_K} \mathcal{L} = \mathbf{0}_{K} \Longleftrightarrow \begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial \pi_1} \\
\frac{\partial \mathcal{L}}{\partial \pi_2} \\
\vdots \\
\frac{\partial \mathcal{L}}{\partial \pi_K}
\end{bmatrix} = \mathbf{0}_{K}
$$

### The Chain Rule (Matrix Calculus)

See section 5.2.2. Chain Rule of Mathematics for Machine Learning, written by Deisenroth, Marc Peter, Faisal, A. Aldo and Ong, Cheng Soon.

For all three necessary conditions, by applying the chain rule, we require partial derivatives of the form

$$
\frac{\partial \log p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\theta}}=\textcolor{orange}{\frac{1}{p\left(\boldsymbol{x}^{(n)} ; \theta\right)} } \textcolor{blue}{\frac{\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\theta}}} ,
$$ (eq-gmm-chain-rule-1)

where $\boldsymbol{\theta}=\left\{\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}, k=1, \ldots, K\right\}$ are the model parameters and

$$
\textcolor{orange}{\frac{1}{p\left(\boldsymbol{x}^{(n)} ; \theta\right)}=\frac{1}{\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} ; \mu_{k}, \Sigma_{k}\right)}} .
$$ (eq-gmm-chain-rule-2)

### Running Example

We will use the following running example to illustrate the GMM. This example
is from {cite}`deisenroth_ong_faisal_2021`.

```{prf:example} Running Example (Initialization)
:label: example-gmm-initialization

We consider a one-dimensional dataset $\mathcal{S}=\{-3,-2.5,-1,0,2,4,5\}$ consisting of seven data points and wish to find a GMM with $K=3$ components that models the density of the data. We initialize the mixture components as

$$
\begin{aligned}
& p_{1}(x)=\mathcal{N}(x \mid-4,1) \\
& p_{2}(x)=\mathcal{N}(x \mid 0,0.2) \\
& p_{3}(x)=\mathcal{N}(x \mid 8,3)
\end{aligned}
$$

and assign them equal weights $\pi_{1}=\pi_{2}=\pi_{3}=\frac{1}{3}$. The corresponding model (and the data points) are shown below.

Note we have not yet explained why are we "randomly" initializing the mixture components
and the mixture weights. This is part of the EM algorithm and will be explained in the next sections.
```

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def create_gmm(mus, sigmas, pis, x_range):
    pdfs = [pi * norm.pdf(x_range, mu, sigma) for pi, mu, sigma in zip(pis, mus, sigmas)]
    gmm_pdf = np.sum(pdfs, axis=0)
    return pdfs, gmm_pdf

def plot_gmm(data, mus, sigmas, pis, x_range, pdfs, gmm_pdf, title, ax=None):
    ax = ax or plt.gca()
    ax.scatter(data, np.zeros_like(data), marker='o', color='k', label='Data points')
    for k, pdf in enumerate(pdfs, start=1):
        ax.plot(x_range, pdf, label=f'$\mathcal{{N}}(x \mid {mus[k-1]}, {sigmas[k-1]})$')
    ax.plot(x_range, gmm_pdf, label='GMM', linestyle='--', color='black')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_title(title)

data = np.array([-3, -2.5, -1, 0, 2, 4, 5])
K = 3

# Initialize the mixture components
mus = np.array([-4, 0, 8])
sigmas = np.array([1, 0.2, 3])
pis = np.array([1/3, 1/3, 1/3])

x_range = np.linspace(-7.5, 15, 1000)
pdfs, gmm_pdf = create_gmm(mus, sigmas, pis, x_range)
plot_gmm(data, mus, sigmas, pis, x_range, pdfs, gmm_pdf, 'Initial Gaussian Mixture Model and Data Points')
```

Next, we can calculate the responsibilities $r^{(n)}_{k}$ for each data point $x^{(n)}$ and each mixture component $k$.

For our example from {prf:ref}`example-gmm-initialization`, we compute the responsibilities $r^{(n)}_{k}$

$$
\boldsymbol{R} = \left[\begin{array}{ccc}
1.0 & 0.0 & 0.0 \\
1.0 & 0.0 & 0.0 \\
0.057 & 0.943 & 0.0 \\
0.001 & 0.999 & 0.0 \\
0.0 & 0.066 & 0.934 \\
0.0 & 0.0 & 1.0 \\
0.0 & 0.0 & 1.0
\end{array}\right] \in \mathbb{R}^{N \times K} \text {. }
$$

Here the $n$th row tells us the responsibilities of all mixture components for $x^{(n)}$. The sum of all $K$ responsibilities for a data point (sum of every row) is 1 . The $k$ th column gives us an overview of the responsibility of the $k$ th mixture component. We can see that the third mixture component (third column) is not responsible for any of the first four data points, but takes much responsibility of the remaining data points. The sum of all entries of a column gives us the values $N_{k}$, i.e., the total responsibility of the $k$ th mixture component. In our example, we get $N_{1}=2.058, N_{2}=$ $2.008, N_{3}=2.934$ {cite}`deisenroth_ong_faisal_2021`.

### Estimating the Mean Parameters $\boldsymbol{\mu}_k$

```{prf:theorem} Update of the GMM Means
:label: theorem-gmm-update-means

The update of the mean parameters $\boldsymbol{\mu}_{k}, k=1, \ldots, K$, of the $G M M$ is given by

$$
\begin{aligned}
\boldsymbol{\mu}_{k}^{n e w} &= \frac{\sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{x}^{(n)}}{\sum_{n=1}^{N} r^{(n)}_{k}} \\
&= \frac{1}{N_{k}} \sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{x}^{(n)}
\end{aligned}
$$ (eq:gmm-update-means-1)

where

- the responsibilities $r^{(n)}_{k}$ are defined in {eq}`eq:gmm-responsibility`, which is the probability that the $k$ th mixture component generated the $n$-th data point.
- $N_{k}=\sum_{n=1}^{N} r^{(n)}_{k}$ can be interpreted the number of data points assigned to the $k$ th mixture component.
```

```{prf:remark} The update of the GMM Means depends on the responsibilities
:label: remark-gmm-update-means

The update of the means $\boldsymbol{\mu}_{k}$ of the individual mixture components in {eq}`eq:gmm-update-means-1` depends on all means, covariance matrices $\boldsymbol{\Sigma}_{k}$, and mixture weights $\pi_{k}$ via $r^{(n)}_{k}$ given in {eq}`eq:gmm-responsibility`. Therefore, we cannot obtain a closed-form solution for all $\boldsymbol{\mu}_{k}$ at once.

What this means is that in order to update the means, we need to first compute the responsibilities $r^{(n)}_{k}$, but computing the responsibilities requires us to know the means $\boldsymbol{\mu}_{k}$, which we want to update. This is a typical problem in iterative algorithms, which we will discuss in more detail in the next section.
```

```{prf:proof}
**The proof is taken from {cite}`deisenroth_ong_faisal_2021`.**

From {eq}`eq-gmm-chain-rule-1` we see that the gradient of the log-likelihood with respect to the mean parameters $\boldsymbol{\mu}_{k}, k=1, \ldots, K$, requires us to compute the partial derivative

$$
\begin{aligned}
\textcolor{blue}{\frac{\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}}} & =\sum_{j=1}^{K} \pi_{j} \frac{\partial \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}{\partial \boldsymbol{\mu}_{k}}=\pi_{k} \frac{\partial \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\partial \boldsymbol{\mu}_{k}} && (a) \\
& = \textcolor{blue}{\pi_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)} && (b) \\
\end{aligned}
$$

where we exploited that only the $k$ th mixture component depends on $\boldsymbol{\mu}_{k}$.

We use our result from (b) in {eq}`eq-gmm-chain-rule-1` and put everything together so that the desired partial derivative of $\mathcal{L}$ with respect to $\boldsymbol{\mu}_{k}$ is given as

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_{k}} & =\sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}}=\sum_{n=1}^{N} \textcolor{orange}{\frac{1}{p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}}  \textcolor{blue}{\frac{\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\mu}_{k}}} && (c) \\
& =\sum_{n=1}^{N} \textcolor{blue}{\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}} \underbrace{\boxed{\frac{\textcolor{blue}{\pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}}{\textcolor{orange}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \mu_{j}, \boldsymbol{\Sigma}_{j}\right)}}}}_{=r^{(n)}_{k}} && (d) \\
& =\sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1} && (e)  \\
\end{aligned}
$$ (eq:gmm-update-means-2)

Here we used the identity from {eq}`eq-gmm-chain-rule-2` and the result of the partial derivative in (b) to get to (d). The values $r^{(n)}_{k}$ are the responsibilities we defined in {eq}`eq:gmm-responsibility`.

We now solve (e) for $\boldsymbol{\mu}_{k}^{\text {new }}$ so that $\frac{\partial \mathcal{L}\left(\boldsymbol{\mu}_{k}^{\mathrm{new}}\right)}{\partial \boldsymbol{\mu}_{k}}=\mathbf{0}^{\top}$ and obtain

$$
\sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{x}^{(n)}=\sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{\mu}_{k}^{\mathrm{new}} \Longleftrightarrow \boldsymbol{\mu}_{k}^{\mathrm{new}}=\frac{\sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{x}^{(n)}}{\boxed{\sum_{n=1}^{N} r^{(n)}_{k}}}=\frac{1}{\boxed{N_{k}}} \sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{x}^{(n)},
$$ (eq:gmm-update-means-3)

where we defined

$$
N_{k}:=\sum_{n=1}^{N} r^{(n)}_{k}
$$ (eq:nk-1)

as the total responsibility of the $k$ th mixture component for the entire dataset. This concludes the proof of {prf:ref}`theorem-gmm-update-means`.
```


#### Some Intuition

Intuitively, {eq}`eq:gmm-update-means-1` can be interpreted as an importance-weighted Monte Carlo estimate of the mean, where the importance weights of data point $\boldsymbol{x}^{(n)}$ are the responsibilities $r^{(n)}_{k}$ of the $k$ th cluster for $\boldsymbol{x}^{(n)}, k=1, \ldots, K$. Therefore, the mean $\boldsymbol{\mu}_{k}$ is pulled toward a data point $\boldsymbol{x}^{(n)}$ with strength given by $r^{(n)}_{k}$. The means are pulled stronger toward data points for which the corresponding mixture component has a high responsibility, i.e., a high likelihood. Figure {numref}`fig-gmm-mean-updates` illustrates this.

```{figure} ./assets/mml-11.4.png
---
name: fig-gmm-mean-updates
---
Update of the mean parameter of mixture component in a GMM. The mean $\boldsymbol{\mu}$ is being pulled toward individual data points with the weights given by the corresponding responsibilities. Image Credit: {cite}`deisenroth_ong_faisal_2021`.
```

---

We can also interpret the mean update in {eq}`eq:gmm-update-means-1` as the expected value of all data points under the distribution given by

$$
\boldsymbol{r}_{k}:=\left[r_{1 k}, \ldots, r_{N k}\right]^{\top} / N_{k}
$$ (eq:responsibility-vector-1)

which is a normalized probability vector, i.e.,

$$
\boldsymbol{\mu}_{k} \leftarrow \mathbb{E}_{\boldsymbol{r}_{k}}[\mathcal{S}]
$$ (eq:responsibility-vector-2)

#### Update Mean of Running Example

```{prf:example} Running Example: Update Mean
:label: prf-gmm-update-means-example

In our example from {prf:ref}`example-gmm-initialization`, the mean values are updated as follows:

$$
\begin{aligned}
& \mu_{1}:-4 \rightarrow-2.7 \\
& \mu_{2}: 0 \rightarrow-0.4 \\
& \mu_{3}: 8 \rightarrow 3.7
\end{aligned}
$$

Here we see that the means of the first and third mixture component move toward the regime of the data, whereas the mean of the second component does not change so dramatically. Figure 11.3 illustrates this change, where Figure 11.3(a) shows the GMM density prior to updating the means and Figure 11.3(b) shows the GMM density after updating the mean values $\mu_{k}$.
```

```{code-cell} ipython3
:tags: [hide-input]

# Update the means
mus_new = np.array([-2.7, -0.4, 3.7])

# Create a new GMM with the updated means
pdfs_new, gmm_pdf_new = create_gmm(mus_new, sigmas, pis, x_range)

# Create the original GMM with the initial means
pdfs_original, gmm_pdf_original = create_gmm(mus, sigmas, pis, x_range)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot the original GMM
plot_gmm(data, mus, sigmas, pis, x_range, pdfs_original, gmm_pdf_original, 'Initial Gaussian Mixture Model and Data Points', ax1)

# Plot the updated GMM
plot_gmm(data, mus_new, sigmas, pis, x_range, pdfs_new, gmm_pdf_new, 'Updated Gaussian Mixture Model and Data Points with New Mean.', ax2)

plt.show()
```

The update of the mean parameters in {eq}`eq:gmm-update-means-1` look fairly straightforward. However, note that the responsibilities $r^{(n)}_{k}$ are a function of $\pi_{j}, \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}$ for all $j=1, \ldots, K$, such that the updates in {eq}`eq:gmm-update-means-1` depend on all parameters of the GMM, and a closed-form solution, which we obtained for linear regression, cannot be obtained.

Another important thing one needs to realize is that the update of the means's
right hand side's $N_k$ and $r^{(n)}_{k}$ are all based on the previous iteration's parameters (or current depending on how you term it). See code for concrete logical flow.

#### Estimating the Mean Parameters $\boldsymbol{\mu}_{k}$ in Python

We can estimate the mean parameters $\boldsymbol{\mu}_{k}$ in Python using the following code snippet:

```python
means = responsibilities.T @ X / nk[:, np.newaxis] # (K, D)
```

where

- `responsibilities` is a $N \times K$ matrix of responsibilities $r^{(n)}_{k}$, and
- `nk` is a $K$-dimensional vector of $N_{k}$ values.

Why? Because if you look at equation {eq}`eq:gmm-update-means-1`:

$$
\begin{aligned}
\boldsymbol{\mu}_{k}^{n e w} &= \frac{1}{N_{k}} \sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{x}^{(n)} \\
&= \frac{1}{N_{k}} \left[r^{(1)}_{k} \boldsymbol{x}^{(1)} + \ldots + r^{(N)}_{k} \boldsymbol{x}^{(N)}\right] \\
&= \frac{1}{N_{k}} \underbrace{\begin{bmatrix} r^{(1)}_{k} & \ldots & r^{(N)}_{k} \end{bmatrix}}_{\left(\boldsymbol{r}_k\right)^{T}} \underbrace{\begin{bmatrix} \boldsymbol{x}^{(1)} \\ \vdots \\ \boldsymbol{x}^{(N)} \end{bmatrix}}_{\boldsymbol{X}} \\
\end{aligned}
$$ (eq:gmm-update-means-1-repeated)

where the last equality leads is:

```python
responsibilities[:, k].T @ X
```

so in order to find all $K$ mean parameters $\boldsymbol{\mu}_{k}$, we just need to repeat the above code snippet for all $k=1, \ldots, K$:

$$
\begin{aligned}
\boldsymbol{M} = \begin{bmatrix} \boldsymbol{\mu}_{1}^{n e w} \\ \vdots \\ \boldsymbol{\mu}_{K}^{n e w} \end{bmatrix} &= \frac{1}{N_{k}} \begin{bmatrix} \boldsymbol{r}_{1}^{T} \boldsymbol{X} \\ \vdots \\ \boldsymbol{r}_{K}^{T} \boldsymbol{X} \end{bmatrix} \\
&= \frac{1}{N_k} \boldsymbol{R}^{T} \boldsymbol{X} \\
\end{aligned}
$$

where $\boldsymbol{M}$ is a $K \times D$ matrix of mean parameters $\boldsymbol{\mu}_{k}$
and $\boldsymbol{R}$ is a $N \times K$ matrix of responsibilities $r^{(n)}_{k}$.

So we update our code snippet to:

```python
responsibilities.T @ X
```

and to divide by $N_{k}$, we just need to broadcast the `nk` vector to the shape of the matrix obtained from the previous code snippet:

```python
responsibilities.T @ X / nk[:, np.newaxis]
```

and we are done finding the new mean parameters $\boldsymbol{\mu}_{k}$ in Python code (for all $k=1, \ldots, K$).


### Estimating the Covariance Parameters $\boldsymbol{\Sigma}_{k}$

```{prf:theorem} Update of the GMM Covariances
:label: theorem-gmm-update-covariance

**The proof is taken from {cite}`deisenroth_ong_faisal_2021`.**

The update of the covariance parameters $\boldsymbol{\Sigma}_{k}, k=1, \ldots, K$ of the $G M M$ is given by

$$
\boldsymbol{\Sigma}_{k}^{n e w}=\frac{1}{N_{k}} \sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top}
$$ (eq:gmm-update-covariance-1)

where $r^{(n)}_{k}$ and $N_{k}$ are defined in {eq}`eq:gmm-responsibility` and {eq}`eq:nk-1`, respectively.
```

```{prf:proof}
To prove {prf:ref}`theorem-gmm-update-covariance`, our approach is to compute the partial derivatives of the log-likelihood $\mathcal{L}$ with respect to the covariances $\boldsymbol{\Sigma}_{k}$, set them to $\mathbf{0}$, and solve for $\boldsymbol{\Sigma}_{k}$. We start with our general approach

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_{k}} = \sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}=\sum_{n=1}^{N} \textcolor{orange}{\frac{1}{p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}} \textcolor{blue}{\frac{\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}} .
\end{aligned}
$$ (eq:gmm-update-covariance-2)


We already know $1 / p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)$ from (11.16). To obtain the remaining partial derivative $\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right) / \partial \boldsymbol{\Sigma}_{k}$, we write down the definition of the Gaussian distribution $p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)$ (see (11.9)) and drop all terms but the $k$ th. We then obtain

$$
\begin{aligned}
& \textcolor{blue}{\frac{\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}} && (a) \\
& =\frac{\partial}{\partial \boldsymbol{\Sigma}_{k}}\left(\pi_{k}(2 \pi)^{-\frac{D}{2}} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\right)\right) && (b) \\
& =\pi_{k}(2 \pi)^{-\frac{D}{2}}\left[\textcolor{red}{\frac{\partial}{\partial \boldsymbol{\Sigma}_{k}} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\right)\right. && (c) \\
& \left.+\operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} \frac{\partial}{\partial \boldsymbol{\Sigma}_{k}} \exp \left(-\frac{1}{2}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\right)\right] \text {. } && (d)
\end{aligned}
$$ (eq:gmm-update-covariance-3)

We now use the identities

$$
\begin{aligned}
& \textcolor{red}{\frac{\partial}{\partial \boldsymbol{\Sigma}_{k}} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}}}  \stackrel{(5.101)}{=} \textcolor{red}{-\frac{1}{2} \operatorname{det}\left(\boldsymbol{\Sigma}_{k}\right)^{-\frac{1}{2}} \boldsymbol{\Sigma}_{k}^{-1}} \\
& \frac{\partial}{\partial \boldsymbol{\Sigma}_{k}}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right) \stackrel{(5.103)}{=}-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}
\end{aligned}
$$

and obtain (after some rearranging) the desired partial derivative required in {eq}`eq:gmm-update-covariance-2` as

$$
\begin{aligned}
\textcolor{blue}{\frac{\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}}= & \textcolor{blue}{\pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)  \cdot\left[-\frac{1}{2}\left(\boldsymbol{\Sigma}_{k}^{-1}-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\right)\right]}
\end{aligned}
$$ (eq:gmm-update-covariance-4)

Putting everything together, the partial derivative of the log-likelihood with respect to $\boldsymbol{\Sigma}_{k}$ is given by (do pay attention to color coding):

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\Sigma}_{k}} & =\sum_{n=1}^{N} \frac{\partial \log p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}} = \sum_{n=1}^{N} \textcolor{orange}{\frac{1}{p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}} \textcolor{blue}{\frac{\partial p\left(\boldsymbol{x}^{(n)} ; \boldsymbol{\theta}\right)}{\partial \boldsymbol{\Sigma}_{k}}} \\
& =\sum_{n=1}^{N} \underbrace{\frac{\textcolor{blue}{\pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}}{\textcolor{orange}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}, \boldsymbol{\Sigma}_{j}\right)}}}_{=r^{(n)}_{k}}  \cdot \textcolor{blue}{\left[-\frac{1}{2}\left(\boldsymbol{\Sigma}_{k}^{-1}-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\right)\right]} \\
& =-\frac{1}{2} \sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{\Sigma}_{k}^{-1}-\boldsymbol{\Sigma}_{k}^{-1}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top} \boldsymbol{\Sigma}_{k}^{-1}\right) \\
= & -\frac{1}{2} \boldsymbol{\Sigma}_{k}^{-1} \underbrace{\sum_{n=1}^{N} r^{(n)}_{k}}_{=N_{k}}+\frac{1}{2} \boldsymbol{\Sigma}_{k}^{-1}\left(\sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top}\right) \boldsymbol{\Sigma}_{k}^{-1} .
\end{aligned}
$$ (eq:gmm-update-covariance-5)

We see that the responsibilities $r^{(n)}_{k}$ also appear in this partial derivative. Setting this partial derivative to $\mathbf{0}$, we obtain the necessary optimality condition

$$
\begin{aligned}
& N_{k} \boldsymbol{\Sigma}_{k}^{-1}=\boldsymbol{\Sigma}_{k}^{-1}\left(\sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top}\right) \boldsymbol{\Sigma}_{k}^{-1} \\
& \Longleftrightarrow N_{k} \boldsymbol{I}=\left(\sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top}\right) \boldsymbol{\Sigma}_{k}^{-1}
\end{aligned}
$$ (eq:gmm-update-covariance-6)

where $\boldsymbol{I}$ is the identity matrix.

By solving for $\boldsymbol{\Sigma}_{k}$, we obtain

$$
\boldsymbol{\Sigma}_{k}^{\mathrm{new}}=\frac{1}{N_{k}} \sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top},
$$

where $\boldsymbol{r}_{k}$ is the probability vector defined in {eq}`eq:responsibility-vector-1`. This gives us a simple update rule for $\boldsymbol{\Sigma}_{k}$ for $k=1, \ldots, K$ and proves {prf:ref}`theorem-gmm-update-covariance`.

Similar to the update of $\boldsymbol{\mu}_{k}$ in {eq}`eq:gmm-update-means-1`, we can interpret the update of the covariance in {eq}`eq:gmm-update-covariance-1` as an importance-weighted expected value of the square of the centered data $\tilde{\mathcal{S}}_{k}:=\left\{\boldsymbol{x}^{(1)}-\boldsymbol{\mu}_{k}, \ldots, \boldsymbol{x}^{(N)}-\boldsymbol{\mu}_{k}\right\}$
```

#### Update Covariance Matrix of Running Example

```{prf:example} Running Example: Update Covariance Matrix
:label: prf-gmm-update-covariance-example

In our example from {prf:ref}`example-gmm-initialization`, the (co)variance values are updated as follows:

$$
\begin{aligned}
& \sigma_{1}^{2}: 1 \rightarrow 0.14 \\
& \sigma_{2}^{2}: 0.2 \rightarrow 0.44 \\
& \sigma_{3}^{2}: 3 \rightarrow 1.53
\end{aligned}
$$

Here we see that the means of the first and third mixture component move toward the regime of the data, whereas the mean of the second component does not change so dramatically.

The figure below illustrates the change in the (co)variance values. The figure on the left
shows the GMM density prior to updating the (co)variance values, whereas the figure on the right shows the GMM density after updating the (co)variance values.
```

```{code-cell} ipython3
:tags: [hide-input]

# Update the variances
sigmas_new = np.sqrt(np.array([0.14, 0.44, 1.53]))

# Create the updated GMM with new means and variances
pdfs_new2, gmm_pdf_new2 = create_gmm(mus_new, sigmas_new, pis, x_range)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot the GMM with updated means
plot_gmm(data, mus_new, sigmas, pis, x_range, pdfs_new, gmm_pdf_new, 'Updated Gaussian Mixture Model and Data Points with New Mean.', ax1)

# Plot the updated GMM with new means and variances
plot_gmm(data, mus_new, sigmas_new, pis, x_range, pdfs_new2, gmm_pdf_new2, 'Updated Gaussian Mixture Model and Data Points with New Variances', ax2)

plt.tight_layout()
plt.show()
```

#### Some Intuition

Similar to the update of the mean parameters, we can interpret {eq}`eq:gmm-update-means-1` as a Monte Carlo estimate of the weighted covariance of data points $\boldsymbol{x}^{(n)}$ associated with the $k$ th mixture component, where the weights are the responsibilities $r^{(n)}_{k}$. As with the updates of the mean parameters, this update depends on all $\pi_{j}, \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}, j=1, \ldots, K$, through the responsibilities $r^{(n)}_{k}$, which prohibits a closed-form solution {cite}`deisenroth_ong_faisal_2021`.

#### Estimating the Covariance Matrix $\boldsymbol{\Sigma}_{k}$ in Python

We can estimate the covariance matrix $\boldsymbol{\Sigma}_{k}$ in Python
using the following code snippet:

```python
covariances = np.zeros(                                                           # (K, D, D)
    (self.num_components, self.num_features, self.num_features)
)

for k in range(self.num_components):
    diff = X - means[k]                                                           # (N, D)
    weighted_diff = responsibilities[:, k].reshape(-1, 1) * diff                  # (N, D)
    cov_k = weighted_diff.T @ diff / nk[k]                                        # (D, D)
    covariances[k] = cov_k
```

where

- `responsibilities` is a $N \times K$ matrix of responsibilities $r^{(n)}_{k}$,
- `nk` is a $K$-dimensional vector of $N_{k}$ values, and
- `means` is a $K \times D$ matrix of mean parameters $\boldsymbol{\mu}_{k}$.

Why? Because if you look at the equation for updating the covariance matrices:

$$
\boldsymbol{\Sigma}_k^{n e w}=\frac{1}{N_k} \sum_{n=1}^N r_k^{(n)}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_k^{\text {new }}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_k^{\text {new }}\right)^T
$$

For each Gaussian component $k$, we compute the difference between the data points `X` and the updated mean `means[k]`. This results in a $(N, D)$ matrix `diff`, where $N$ is the number of data points.

To obtain the element-wise product of the responsibilities `responsibilities[:, k]` with the differences `diff`, we reshape the responsibilities to a column vector of shape $(N, 1)$ and multiply it element-wise with `diff`. This results in a $(N, D)$ matrix `weighted_diff`.

We then compute the covariance matrix for the $k$-th component by calculating the matrix product of the transpose of `weighted_diff` with `diff`, and then dividing the result by the $k$-th element of `nk`. This gives us a $(D, D)$ matrix `cov_k`.

Finally, we store the computed covariance matrix `cov_k` in the `covariances` array at the index `k`.

This Python code snippet computes the updated covariance matrices $\boldsymbol{\Sigma}_{k}$ for all Gaussian components $k=1, \ldots, K$.

---

We will derive the matrix justification for updating the covariance matrices $\boldsymbol{\Sigma}_{k}$:

$$
\begin{aligned}
\boldsymbol{\Sigma}_{k}^{new} &= \frac{1}{N_k}\sum_{n=1}^N r^{(n)}_{k}\left(\boldsymbol{x}^{(n)} - \boldsymbol{\mu}_{k}^{new}\right)\left(\boldsymbol{x}^{(n)} - \boldsymbol{\mu}_{k}^{new}\right)^T \\
\end{aligned}
$$

We will rewrite the summation as a matrix product. First, let's define the difference matrix $\boldsymbol{D}_k$:

$$
\boldsymbol{D}_k = \begin{bmatrix} \boldsymbol{x}^{(1)} - \boldsymbol{\mu}_{k}^{new} \\ \vdots \\ \boldsymbol{x}^{(N)} - \boldsymbol{\mu}_{k}^{new} \end{bmatrix} \in \mathbb{R}^{N \times D}
$$

Now, let's define a diagonal matrix $\boldsymbol{W}_k$ with the $r^{(n)}_{k}$ values on its diagonal:

$$
\boldsymbol{W}_k = \begin{bmatrix} r^{(1)}_{k} & 0 & \cdots & 0 \\ 0 & r^{(2)}_{k} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & r^{(N)}_{k} \end{bmatrix} \in \mathbb{R}^{N \times N}
$$

Now we can rewrite the covariance matrix update equation as:

$$
\begin{aligned}
\boldsymbol{\Sigma}_{k}^{new} &= \frac{1}{N_k} \boldsymbol{D}_k^T \boldsymbol{W}_k \boldsymbol{D}_k \\
\end{aligned}
$$

The Python code snippet computes the same equation as described above:

```python
covariances = np.zeros((K, D, D))

for k in range(K):
    diff = X - means[k]                                                           # (N, D)
    weighted_diff = responsibilities[:, k].reshape(-1, 1) * diff                  # (N, D)
    cov_k = weighted_diff.T @ diff / nk[k]                                        # (D, D)
    covariances[k] = cov_k
```

The only slighly difference is the code uses `*` which is the Hadamard product (element-wise product) instead of `@` which is the matrix product. They will result in the same result.

### Estimating the Mixing Coefficients (Prior/Weights) $\pi_{k}$

```{prf:theorem} Update of the GMM Mixture Weights
:label: thm:gmm-update-mixture-weights

The mixture weights of the GMM are updated as

$$
\pi_{k}^{\text {new }}=\frac{N_{k}}{N}, \quad k=1, \ldots, K
$$ (eq:gmm-update-mixture-weights-1)

where $N$ is the number of data points and $N_{k}$ is defined in {eq}`eq:nk-1`.
```

```{prf:proof}
To find the partial derivative of the log-likelihood with respect to the weight parameters $\pi_{k}, k=1, \ldots, K$, we account for the constraint $\sum_{k} \pi_{k}=1$ by using Lagrange multipliers (see Section 7.2 of Mathematics for Machine Learning {cite}`deisenroth_ong_faisal_2021`). The Lagrangian is

$$
\begin{aligned}
\mathfrak{L}&=\mathcal{L}+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right) \\
&=\sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right) \\
\end{aligned}
$$ (eq:gmm-update-mixture-weights-2)

where $\mathcal{L}$ is the log-likelihood from {eq}`eq-gmm-log-likelihood-1` and the second term encodes for the equality constraint that all the mixture weights need to sum up to 1. We obtain the partial derivative with respect to $\pi_{k}$ as

$$
\begin{aligned}
\frac{\partial \mathfrak{L}}{\partial \pi_{k}} & =\sum_{n=1}^{N} \frac{\mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}+\lambda \\
& =\frac{1}{\pi_{k}} \underbrace{\sum_{n=1}^{N} \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)}}_{=N_{k}}+\lambda=\frac{N_{k}}{\pi_{k}}+\lambda,
\end{aligned}
$$ (eq:gmm-update-mixture-weights-3)

and the partial derivative with respect to the Lagrange multiplier $\lambda$ as

$$
\frac{\partial \mathfrak{L}}{\partial \lambda}=\sum_{k=1}^{K} \pi_{k}-1
$$ (eq:gmm-update-mixture-weights-4)

Setting both partial derivatives to $\mathbf{0}$ (necessary condition for optimum) yields the system of equations

$$
\begin{aligned}
& \pi_{k}=-\frac{N_{k}}{\lambda} && (a)\\
& 1=\sum_{k=1}^{K} \pi_{k} && (b)
\end{aligned}
$$ (eq:gmm-update-mixture-weights-5)

Using (a) in (b) and solving for $\pi_{k}$, we obtain

$$
\sum_{k=1}^{K} \pi_{k}=1 \Longleftrightarrow-\sum_{k=1}^{K} \frac{N_{k}}{\lambda}=1 \Longleftrightarrow-\frac{N}{\lambda}=1 \Longleftrightarrow \lambda=-N .
$$ (eq:gmm-update-mixture-weights-6)

This allows us to substitute $-N$ for $\lambda$ in (a) to obtain

$$
\pi_{k}^{\mathrm{new}}=\frac{N_{k}}{N}
$$ (eq:gmm-update-mixture-weights-7)

which gives us the update for the weight parameters $\pi_{k}$ and proves {prf:ref}`thm:gmm-update-mixture-weights`.
```

#### Some Intuition

We can identify the mixture weight in {eq}`eq:gmm-update-mixture-weights-1` as the ratio of the total responsibility of the $k$ th cluster and the number of data points. Since $N=\sum_{k} N_{k}$, the number of data points can also be interpreted as the total responsibility of all mixture components together, such that $\pi_{k}$ is the relative importance of the $k$ th mixture component for the dataset.

```{prf:remark} Update of the GMM Mixture Weights depends on all Parameters
:label: rem:gmm-update-mixture-weights-depends-on-all-parameters

Since $N_{k}=\sum_{i=1}^{N} r^{(n)}_{k}$, the update equation (11.42) for the mixture weights $\pi_{k}$ also depends on all $\pi_{j}, \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}, j=1, \ldots, K$ via the responsibilities $r^{(n)}_{k}$.
```

#### Update Weight/Prior of Running Example

```{prf:example} Running Example: Update of the GMM Mixture Weights
:label: ex:gmm-update-mixture-weights

In our running example from Figure 11.1, the mixture weights are updated as follows:

$$
\begin{aligned}
& \pi_{1}: \frac{1}{3} \rightarrow 0.29 \\
& \pi_{2}: \frac{1}{3} \rightarrow 0.29 \\
& \pi_{3}: \frac{1}{3} \rightarrow 0.42
\end{aligned}
$$

Here we see that the third component gets more weight/importance, while the other components become slightly less important. The figure below illustrates the effect of updating the mixture weights. The left figure below shows the GMM density and its individual components prior to updating the mixture weights. The right figure shows the GMM density after updating the mixture weights.

Overall, having updated the means, the variances, and the weights once, we obtain the GMM shown in the figure below. Compared with the initialization shown in the very original, we can see that the parameter updates caused the GMM density to shift some of its mass toward the data points.

After updating the means, variances, and weights once, the GMM fit in the updated one is already remarkably better than its initialization from the original. This is also evidenced by the log-likelihood values, which increased from 28.3 (initialization) to 14.4 after one complete update cycle (you can verify this by hand or code).
```

```{code-cell} ipython3
:tags: [hide-input]

# Update the mixture weights
pis_new = np.array([0.29, 0.29, 0.42])

# Create a new GMM with the updated mixture weights
pdfs_updated_pis, gmm_pdf_updated_pis = create_gmm(mus_new, sigmas_new, pis_new, x_range)

# Create the original GMM with the initial means, sigmas, and pis
pdfs_original, gmm_pdf_original = create_gmm(mus, sigmas, pis, x_range)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot the updated GMM with new means and variances
plot_gmm(data, mus_new, sigmas_new, pis, x_range, pdfs_new2, gmm_pdf_new2, 'Updated Gaussian Mixture Model and Data Points with New Variances', ax1)

# Plot the updated GMM with new means, sigmas, and pis
plot_gmm(data, mus_new, sigmas_new, pis_new, x_range, pdfs_updated_pis, gmm_pdf_updated_pis, 'Updated Gaussian Mixture Model and Data Points with new Weights.', ax2)

plt.show()
```

We see the full cycle of updates below:

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
(ax1, ax2), (ax3, ax4) = axes

# Plot the original GMM
plot_gmm(data, mus, sigmas, pis, x_range, pdfs_original, gmm_pdf_original, 'Initial Gaussian Mixture Model and Data Points', ax1)

# Plot the updated GMM with new means
pdfs_updated_means, gmm_pdf_updated_means = create_gmm(mus_new, sigmas, pis, x_range)
plot_gmm(data, mus_new, sigmas, pis, x_range, pdfs_updated_means, gmm_pdf_updated_means, 'Updated Means', ax2)

# Plot the updated GMM with new sigmas
pdfs_updated_sigmas, gmm_pdf_updated_sigmas = create_gmm(mus_new, sigmas_new, pis, x_range)
plot_gmm(data, mus_new, sigmas_new, pis, x_range, pdfs_updated_sigmas, gmm_pdf_updated_sigmas, 'Updated Sigmas', ax3)

# Plot the updated GMM with new pis
pdfs_updated_pis, gmm_pdf_updated_pis = create_gmm(mus_new, sigmas_new, pis_new, x_range)
plot_gmm(data, mus_new, sigmas_new, pis_new, x_range, pdfs_updated_pis, gmm_pdf_updated_pis, 'Updated Mixture Weights', ax4)

plt.tight_layout()
plt.show()
```

## Why GMM has no Closed-Form Solution

We have emphasized along the way that the estimation of the parameters of a Gaussian Mixture Model is a difficult problem and has no closed-form solution. However, it may be
confusing to one why this is the case.

Since we seemingly have found estimates for the mean, covariance, and mixture weights of the GMM in the previous 3 sections, why is it that we cannot find a closed-form solution?

Well, there is some intricacy to this question. First, we clear the confusiong:

1. In the context of Gaussian Mixture Models, a closed-form solution refers to a single expression that can simultaneously provide the optimal values of all the parameters without any iterative steps or dependencies between the parameters. However, in the case of GMM, the responsibilities, which are crucial for updating the parameters, depend on the parameters themselves in a complex manner (more on this in Bishop's book).

2. Although we can optimally update each of the parameters **given** the other parameters, we cannot compute all the parameters at once because of their interdependence. What does this mean? This means when we update the means, we need to know the responsibilities, which depend on the means. However, when we update the responsibilities, we need to know the means, which depend on the responsibilities. This is a circular dependency, which means we cannot simultaneously update the means and the responsibilities (easily).

    Moreover, when we update the covariance matrices, we need to know **both** the means and the responsibilities! Again, ***simultaneous*** updates of the means, responsibilities, and covariance matrices are not possible.

    Finally, when we update the mixture weights, we need to know **all** the parameters, including the means, responsibilities, and covariance matrices. Again, ***simultaneous*** updates of all the parameters are not possible.

So the reason we can "find the estimates" just now is because we are not
simultaneously finding the optimal values of all the parameters. Instead, we are finding the optimal values of the parameters **given** the other parameters. This is a very important distinction.

This situation is similar to hard clustering in K-Means, where the ultimate goal is to jointly optimize cluster means and assignments, which is a NP-hard problem. However, we can optimize the cluster means **given** the cluster assignments, and then optimize the cluster assignments **given** the cluster means. This is a two-step process, which is not a closed-form solution, but gives local optima for each step. The same is true for GMM.

This interesting result gives rise to iterative methods like [Expectation-Maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm).


## The Expectation-Maximization (EM) Algorithm

We are now ready to introduce the Expectation-Maximization (EM) algorithm, which is a popular algorithm for estimating the parameters of a Gaussian Mixture Model.

As it stands, we will start by introducing the EM algorithm in the context of Gaussian Mixture Models. This is an slightly more *informal* treatment of the EM algorithm, which is meant to give you a general idea of how the algorithm works.

### Expectation-Maximization (EM) (Gaussion Mixture Model Perspective)

**This section below is from chapter 11.3. EM Algorithm, Mathematics for Machine Learning.**

Unfortunately, the updates in {eq}`eq:gmm-update-means-1`, {eq}`eq:gmm-update-covariance-1`, and {eq}`eq:gmm-update-mixture-weights-1` do not constitute a closed-form solution for the updates of the parameters $\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}$ of the mixture model because the responsibilities $r^{(n)}_{k}$ depend on those parameters in a complex way. However, the results suggest a simple iterative scheme for finding a solution to the parameters estimation problem via maximum likelihood. The expectation maximization algorithm was proposed by Dempster et al. (1977) and is a general iterative scheme for learning parameters (maximum likelihood or MAP) in mixture models and, more generally, latent-variable models.

In our example of the Gaussian mixture model, we choose initial values for $\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}$ and alternate until convergence between

- E-step: Evaluate the responsibilities $r^{(n)}_{k}$ (posterior probability of data point $n$ belonging to mixture component $k$ ).

- M-step: Use the updated responsibilities to reestimate the parameters $\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}$.

Every step in the EM algorithm increases the log-likelihood function (Neal and Hinton, 1999). For convergence, we can check the log-likelihood or the parameters directly. A concrete instantiation of the EM algorithm for estimating the parameters of a GMM is as follows.

```{prf:algorithm} EM Algorithm for GMM
:label: alg:em-gmm-1

1. Initialize $\boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}, \pi_{k}$.

2. E-step: Evaluate responsibilities $r^{(n)}_{k}$ for every data point $\boldsymbol{x}^{(n)}$ using current parameters $\pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}$ :

$$
r^{(n)}_{k}=\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)}{\sum_{j} \pi_{j} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j}\right)} .
$$

3. M-step: Reestimate parameters $\pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}$ using the current responsibilities $r^{(n)}_{k}$ (from E-step):

$$
\begin{aligned}
\boldsymbol{\mu}_{k} & =\frac{1}{N_{k}} \sum_{n=1}^{N} r^{(n)}_{k} \boldsymbol{x}^{(n)}, \\
\boldsymbol{\Sigma}_{k} & =\frac{1}{N_{k}} \sum_{n=1}^{N} r^{(n)}_{k}\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}^{(n)}-\boldsymbol{\mu}_{k}\right)^{\top}, \\
\pi_{k} & =\frac{N_{k}}{N} .
\end{aligned}
$$

4. **Convergence**: Check if the log-likelihood has increased. More concretely,
you can also check if the mean/sum of the marginal log-likelihood has increased.
See my code for this.


$$
\frac{1}{N} \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\boldsymbol{x}^{(n)} \mid \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}\right)
$$
```

### Expectation-Maximization (EM) (Latent Variable Perspective)

Given a joint distribution $p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})$ over observed variables $\mathbf{X}$ and latent variables $\mathbf{Z}$, governed by parameters $\boldsymbol{\theta}$, the goal is to maximize the likelihood function $p(\mathbf{X} \mid \boldsymbol{\theta})$ with respect to $\boldsymbol{\theta}$.

1. Choose an initial setting for the parameters $\theta^{\text {old }}$.
2. E step Evaluate $p\left(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{\text {old }}\right)$.
3. M step Evaluate $\boldsymbol{\theta}^{\text {new }}$ given by
4.
$$
\boldsymbol{\theta}^{\text {new }}=\underset{\boldsymbol{\theta}}{\arg \max } \mathcal{Q}\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text {old }}\right)
$$

where

$$
\mathcal{Q}\left(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text {old }}\right)=\sum_{\mathbf{Z}} p\left(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}^{\text {old }}\right) \ln p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) .
$$

and in a more general sense, we have:

$$
\begin{aligned}
\mathcal{Q}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{\text{old}}\right) & =\mathbb{E}_{\boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}}[\log p(\boldsymbol{x}, \boldsymbol{z} \mid \boldsymbol{\theta})] \\
& =\int \log p(\boldsymbol{x}, \boldsymbol{z} \mid \boldsymbol{\theta}) p\left(\boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta}^{\text{old}}\right) \mathrm{d} \boldsymbol{z}
\end{aligned}
$$

for the continuous case.

5. Check for convergence of either the log likelihood or the parameter values. If the convergence criterion is not satisfied, then let

$$
\boldsymbol{\theta}^{\text {old }} \leftarrow \boldsymbol{\theta}^{\text {new }}
$$

and return to step 2 .

### The Expectation Step (Posterior Inference and Responsibilities)

See Chapter 9.3. An Alternative View of EM, Pattern Recognition and Machine Learning.

### The Maximization Step (Parameter Estimation)

See Chapter 9.3. An Alternative View of EM, Pattern Recognition and Machine Learning.

## GMM and its Relation to K-Means

Gaussian mixture models are closely related to the $K$-means clustering algorithm. $K$-means also uses the EM algorithm to assign data points to clusters. If we treat the means in the GMM as cluster centers and ignore the covariances (or set them to $I$ ), we arrive at $K$-means. As also nicely described by MacKay (2003), $K$-means makes a "hard" assignment of data points to cluster centers $\boldsymbol{\mu}_{k}$, whereas a GMM makes a "soft" assignment via the responsibilities.

More information can be found in chapter 9.3.2 of Bishop's book, Pattern Recognition and Machine Learning.

## A Small Example

Consider a scenario where $N = 10$, $D = 2$, and $K = 3$ where the data points are as follows:

$$
\mathbf{X} = \begin{bmatrix}  1 &  3 \\ 2 &  4 \\ 3 &  5 \\ 4 &  6 \\ 5 &  7 \\ 6 &  8 \\ 7 &  9 \\ 8 & 10 \\ 9 & 11 \\ 10 & 12 \end{bmatrix}_{10 \times 2}
$$

### Random Initialization

To initialize the GMM, we randomly initialize some means $\boldsymbol{\mu}_{k}$, covariances $\boldsymbol{\Sigma}_{k}$, and mixing coefficients $\pi_{k}$. The means
can be the data points themselves, and the covariances can be set to $\sigma^{2} \boldsymbol{I}$ where $\sigma^{2}$ is the variance of the data points. The mixing coefficients can be set to $\frac{1}{K}$.

This is evident from the code in our implementation:

```python
self.weights_ = np.full(self.num_components, 1 / self.num_components)         # (K,)
self.means_ = X[np.random.choice(self.num_samples, self.num_components)]      # (K, D)
self.covariances_ = np.array(                                                 # (K, D, D)
    [np.cov(X.T) for _ in range(self.num_components)]
)
```

### K-Means and Hard Assignments

#### E-Step

Now in K-Means, the E-Step involves assigning each data point to the closest cluster center,
defined by the mean $\boldsymbol{\mu}_{k}$.

For example, let's assume a hypothetical where we have assigned the data points to the following clusters:

$$
\begin{aligned}
C_1 &= \{1, 2, 3, 4, 5\}, \\
C_2 &= \{6, 7, 8\}, \\
C_3 &= \{9, 10\}.
\end{aligned}
$$

where $C_k$ is the set of data points in cluster $k$ and $N_k$ is the number of data points in cluster $k$.

This is a hard assignment, where each data point is assigned to one and only one cluster.

We can define the assignments above as a matrix of posterior probabilities, where each row sums to 1 and each column $k$ sums to $N_k$.

What the posterior means is the probability that a data point $n$ belongs to cluster $k$ given the data $\mathbf{X}$ and the parameters $\boldsymbol{\theta}$. In this case, we have:

$$
\mathbf{R} = \begin{bmatrix} 1 & 0 & 0 \\ 1 & 0 & 0 \\ 1 & 0 & 0 \\ 1 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 1 \end{bmatrix}_{10 \times 3}
$$

where each row sums to 1 and each column $k$ sums to $N_k$.

So we have:

$$
\begin{aligned}
N_1 &= 5, \\
N_2 &= 3, \\
N_3 &= 2.
\end{aligned}
$$

Then the next step that follows is the M-Step, where we need to update the parameters of the K-Means model, which are the means $\boldsymbol{\mu}_{k}$.

The key here is we are updating the means of the clusters, not the means of the data points.
And it turns out our mean update is very similar, just the mean of the data points in each cluster.

#### M-Step

Therefore, computing the mean of each component/cluster is as follows:

$$
\begin{aligned}
\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n \in C_k} \boldsymbol{x}^{(n)}
\end{aligned}
$$

where $N_k$ is the number of data points in cluster $k$.


$$
\begin{aligned}
\boldsymbol{\mu}_1 &= \frac{1}{5} \sum_{n=1}^5 \boldsymbol{x}_n = \frac{1}{5} \begin{bmatrix} 1 + 2 + 3 + 4 + 5 \\ 3 + 4 + 5 + 6 + 7 \end{bmatrix} = \begin{bmatrix} 3 \\ 5 \end{bmatrix} \\
\boldsymbol{\mu}_2 &= \frac{1}{3} \sum_{n=6}^8 \boldsymbol{x}_n = \frac{1}{3} \begin{bmatrix} 6 + 7 + 8 \\ 8 + 9 + 10 \end{bmatrix} = \begin{bmatrix} 7 \\ 9 \end{bmatrix} \\
\boldsymbol{\mu}_3 &= \frac{1}{2} \sum_{n=9}^{10} \boldsymbol{x}_n = \frac{1}{2} \begin{bmatrix} 9 + 10 \\ 11 + 12 \end{bmatrix} = \begin{bmatrix} 9.5 \\ 11.5 \end{bmatrix}
\end{aligned}
$$

where each $\boldsymbol{\mu}_k$ is a $D \times 1$ vector.

This mean formula is intuitive because we are taking the average of the data points in each cluster. What is not so intuitive is the mean formula for the GMM, which is the weighted average of the data points in each cluster, where the weights are the posterior probabilities.

### GMM and Soft Assignments

Recall the formula for updating the means, covariances and weights of Gaussian mixture model,
as follows:

$$
\begin{aligned}
\boldsymbol{\mu}_k & =\frac{1}{N_k} \sum_{n=1}^N r^{(n)}_{k} \boldsymbol{x}_n, \\
\boldsymbol{\Sigma}_k & =\frac{1}{N_k} \sum_{n=1}^N r^{(n)}_{k}\left(\boldsymbol{x}_n-\boldsymbol{\mu}_k\right)\left(\boldsymbol{x}_n-\boldsymbol{\mu}_k\right)^{\top}, \\
\pi_k & =\frac{N_k}{N} .
\end{aligned}
$$

Now, we only talk about the mean update, to have a comparison with K-Means's mean update.
We notice there is an additional term $r^{(n)}_{k}$ in the mean update formula. We have
already known what they are, but let's view it as an extension of the posterior probabilities
matrix defined just now.

#### E-Step

Given some random initial parameters, we can compute the posterior probabilities matrix $\mathbf{R}$. Say

$$
\mathbf{R} = \begin{bmatrix} 0.9 & 0.1 & 0 \\ 0.8 & 0.2 & 0 \\ 0.7 & 0.3 & 0 \\ 0.75 & 0.25 & 0 \\ 0.85 & 0.15 & 0 \\ 0 & 0.9 & 0.1 \\ 0 & 0.8 & 0.2 \\ 0 & 0.85 & 0.15 \\ 0 & 0 & 1 \\ 0 & 0 & 1 \end{bmatrix}_{10 \times 3}
$$

where each row sums to 1 and each column $k$ sums to $N_k$ but this $N_k$ is not the number of data points in cluster $k$ but the sum of the soft assignments of all data points to cluster $k$. Note each row is just the $p(\boldsymbol{z}^{(n)}| \boldsymbol{x}^{(n)}, \boldsymbol{\pi})$ over all $K$ components.

In other words, we have the following equation for the posterior probability of the latent label $z^{(n)} = k$ given the data point $x^{(n)}$.

$$
\begin{aligned}
\boldsymbol{R} &= \begin{bmatrix} P(z^{(1)} = 1 \mid x^{(1)}) & P(z^{(1)} = 2 \mid x^{(1)}) & \cdots & P(z^{(1)} = K \mid x^{(1)}) \\ P(z^{(2)} = 1 \mid x^{(2)}) & P(z^{(2)} = 2 \mid x^{(2)}) & \cdots & P(z^{(2)} = K \mid x^{(2)}) \\ \vdots & \vdots & \ddots & \vdots \\ P(z^{(N)} = 1 \mid x^{(N)}) & P(z^{(N)} = 2 \mid x^{(N)}) & \cdots & P(z^{(N)} = K \mid x^{(N)}) \end{bmatrix}_{N \times K} \\
\end{aligned}
$$

and each row of $\boldsymbol{R}$ sums to 1. Why?

Because $P(z^{(n)} = k \mid x^{(n)})$ is a probability distribution over the $K$ components. Therefore, the sum of the probabilities over all $K$ components must be 1.

Consequently, when we sum up all elements in $\boldsymbol{R}$ we recover the total number of data points $N$.

The major difference between the matrix here versus the one we had in K-Means is that the matrix here is not binary, but it is continuous. This is because we are dealing with soft assignments, not hard assignments. In each row, the sum of the elements is still 1,
but it is no longer a scenario where each data point can only belong to one and only one cluster.

To this end, our $N_k$ is also different from the one we had in K-Means.

$N_k$ is not the number of data points in cluster $k$ but the sum of the soft assignments of all data points to cluster $k$. Let's see the $N_k$ for each cluster:

$$
\begin{aligned}
N_1 &= \sum_{n=1}^5 r^{(n)}_{1} = 0.9 + 0.8 + 0.7 + 0.75 + 0.85 + 0 + 0 + 0 + 0 + 0 = 4 \\
N_2 &= \sum_{n=6}^8 r^{(n)}_{2} = 0.1 + 0.2 + 0.3 + 0.25 + 0.15 + 0.9 + 0.8 + 0.85 + 0 + 0 = 3.55 \\
N_3 &= \sum_{n=9}^{10} r^{(n)}_{3} = 0 + 0 + 0 + 0 + 0 + 0.1 + 0.2 + 0.15 + 1 + 1 = 2.45
\end{aligned}
$$

Notice that $N_1 + N_2 + N_3 = 10$ but intuitively each $N_k$ is a soft representation of the number of data points in cluster $k$. So you can still interpret $N_k$ as the number of data points in cluster $k$ if
you want to for the sake of intuition.

#### M-Step

Now we can update the parameters $\boldsymbol{\mu}_k$, $\boldsymbol{\Sigma}_k$ and $\pi_k$.

We only focus on the mean update here. The other two are similar.

The mean update is given by the following formula:

$$
\boldsymbol{\mu}_k =\frac{1}{N_k} \sum_{n=1}^N r^{(n)}_{k} \boldsymbol{x}^{(n)}
$$


We have all the ingredients to compute the mean update. But we note to readers
that in K-Means, the mean formula is easy, the numerator is just the sum of all the data points in cluster $k$ and the denominator is just the total number of data points in cluster $k$.

Here the denominator is $N_k$, which we have talked about.

Now there is one more thing to note, the numerator is also not the sum of all the data
points $x^{(n)}$ in cluster $k$ but the sum of the soft assignments of **all** data points to cluster $k$.
It is now the weighted sum of all the data points $x^{(n)}$ in cluster $k$.

$$
\begin{aligned}
\boldsymbol{\mu}_1 &= \frac{1}{N_1} \sum_{n=1}^{10} r^{(n)}_{1} \boldsymbol{x}^{(n)} = \frac{1}{N_1} \begin{bmatrix} 0.9 \times 1 + 0.8 \times 2 + 0.7 \times 3 + 0.75 \times 4 + 0.85 \times 5 + 0 \times 6 + 0 \times 7 + 0 \times 8 + 0 \times 9 + 0 \times 10 \\ 0.9 \times 3 + 0.8 \times 4 + 0.7 \times 5 + 0.75 \times 6 + 0.85 \times 7 + 0 \times 8 + 0 \times 9 + 0 \times 10 + 0 \times 11 + 0 \times 12 \end{bmatrix} = \begin{bmatrix} 3.75 \\ 5.75 \end{bmatrix} \\
\boldsymbol{\mu}_2 &= \frac{1}{N_2} \sum_{n=1}^{10} r^{(n)}_{2} \boldsymbol{x}^{(n)} = \frac{1}{N_2} \begin{bmatrix} 0.1 \times 1 + 0.2 \times 2 + 0.3 \times 3 + 0.25 \times 4 + 0.15 \times 5 + 0.9 \times 6 + 0.8 \times 7 + 0.85 \times 8 + 0 \times 9 + 0 \times 10 \\ 0.1 \times 3 + 0.2 \times 4 + 0.3 \times 5 + 0.25 \times 6 + 0.15 \times 7 + 0.9 \times 8 + 0.8 \times 9 + 0.85 \times 10 + 0 \times 11 + 0 \times 12 \end{bmatrix} = \begin{bmatrix} 6.95 \\ 8.95 \end{bmatrix} \\
\boldsymbol{\mu}_3 &= \frac{1}{N_3} \sum_{n=1}^{10} r^{(n)}_{3} \boldsymbol{x}^{(n)} = \frac{1}{N_3} \begin{bmatrix} 0 \times 1 + 0 \times 2 + 0 \times 3 + 0 \times 4 + 0 \times 5 + 0.1 \times 6 + 0.2 \times 7 + 0.15 \times 8 + 1 \times 9 + 1 \times 10 \\ 0 \times 3 + 0 \times 4 + 0 \times 5 + 0 \times 6 + 0 \times 7 + 0.1 \times 8 + 0.2 \times 9 + 0.15 \times 10 + 1 \times 11 + 1 \times 12 \end{bmatrix} = \begin{bmatrix} 9.5 \\ 11.5 \end{bmatrix}
\end{aligned}
$$

so you can treat the numerator as the weighted sum of all the data points $x^{(n)}$ in cluster $k$.

And thus when you divide the weighted sum of all the data points $x^{(n)}$ in cluster $k$ by the sum of the soft assignments of **all** data points to cluster $k$, you get the weighted mean of all the data points $x^{(n)}$ in cluster $k$. This is the same as the weighted mean of all the data points $x^{(n)}$ in cluster $k$.

## Murphy's Plots

### GMM Demo

[Link here](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/03/gmm_plot_demo.ipynb).

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy import linalg
from scipy.stats import multivariate_normal

# try:
#     import probml_utils as pml
# except ModuleNotFoundError:
#     %pip install -qq git+https://github.com/probml/probml-utils.git
#     import probml_utils as pml

mu_1 = np.array([[0.22], [0.45]])
mu_2 = np.array([[0.5], [0.5]])
mu_3 = np.array([[0.77], [0.55]])
Mu = np.array([mu_1, mu_2, mu_3])

Sigma1 = np.array([[0.011, -0.01], [-0.01, 0.018]])
Sigma2 = np.array([[0.018, 0.01], [0.01, 0.011]])
Sigma3 = Sigma1
Sigma = np.array([Sigma1, Sigma2, Sigma3])
mixmat = np.array([[0.5], [0.3], [0.2]])


def sigmaEllipse2D(mu, Sigma, level=3, npoints=128):
    """
    SIGMAELLIPSE2D generates x,y-points which lie on the ellipse describing
    a sigma level in the Gaussian density defined by mean and covariance.

    Input:
        MU          [2 x 1] Mean of the Gaussian density
        SIGMA       [2 x 2] Covariance matrix of the Gaussian density
        LEVEL       Which sigma level curve to plot. Can take any positive value,
                    but common choices are 1, 2 or 3. Default = 3.
        NPOINTS     Number of points on the ellipse to generate. Default = 32.

    Output:
        XY          [2 x npoints] matrix. First row holds x-coordinates, second
                    row holds the y-coordinates. First and last columns should
                    be the same point, to create a closed curve.
    """
    phi = np.linspace(0, 2 * np.pi, npoints)
    x = np.cos(phi)
    y = np.sin(phi)
    z = level * np.vstack((x, y))
    xy = mu + linalg.sqrtm(Sigma).dot(z)
    return xy


def plot_sigma_levels(mu, P):
    xy_1 = sigmaEllipse2D(mu, P, 0.25)
    xy_2 = sigmaEllipse2D(mu, P, 0.5)
    xy_3 = sigmaEllipse2D(mu, P, 0.75)
    xy_4 = sigmaEllipse2D(mu, P, 1)
    xy_5 = sigmaEllipse2D(mu, P, 1.25)
    xy_6 = sigmaEllipse2D(mu, P, 1.5)

    plt.plot(xy_1[0], xy_1[1])
    plt.plot(xy_2[0], xy_2[1])
    plt.plot(xy_3[0], xy_3[1])
    plt.plot(xy_4[0], xy_4[1])
    plt.plot(xy_5[0], xy_5[1])
    plt.plot(xy_6[0], xy_6[1])
    plt.plot(mu[0], mu[1], "ro")


def plot_sigma_vector(Mu, Sigma):
    n = len(Mu)
    plt.figure(figsize=(12, 7))
    for i in range(n):
        plot_sigma_levels(Mu[i], Sigma[i])
    plt.tight_layout()
    # pml.savefig("mixgaussSurface.pdf")
    plt.show()


plot_sigma_vector(Mu, Sigma)


def plot_gaussian_mixture(Mu, Sigma, weights=None, x=None, y=None):
    if x == None:
        x = np.arange(0, 1, 0.01)
    if y == None:
        y = np.arange(-0.5, 1.2, 0.01)

    if len(Mu) == len(Sigma) == len(weights):
        pass
    else:
        print("Error: Mu, Sigma and weights must have the same dimension")
        return

    X, Y = np.meshgrid(x, y)
    Pos = np.dstack((X, Y))
    Z = 0

    for i in range(len(Mu)):
        Z = Z + weights[i] * multivariate_normal(Mu[i].ravel(), Sigma[i]).pdf(Pos)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="copper", lw=0.5, rstride=1, cstride=1)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.tight_layout()
    # pml.savefig("mixgaussSurface.pdf")
    plt.show()


weights = [0.5, 0.3, 0.2]
# plot_gaussian_mixture(Mu, Sigma, weights=weights)
```

### GMM 2D (sklearn)

[Link here](https://colab.research.google.com/github/probml/pyprobml/blob/master/notebooks/book1/03/gmm_2d.ipynb)

```{code-cell} ipython3
:tags: [hide-input]

# K-means clustering for semisupervised learning
# Code is from chapter 9 of
# https://github.com/ageron/handson-ml2


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

import itertools
from scipy import linalg
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from matplotlib.colors import LogNorm

import warnings
warnings.filterwarnings("ignore")

# color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'darkorange'])
color_iter = itertools.cycle(["r", "g", "b"])
prop_cycle = plt.rcParams["axes.prop_cycle"]
color_iter = prop_cycle.by_key()["color"]


if 0:
    K = 5
    blob_centers = np.array(
        [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
    )
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(
        n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7
    )

if 0:
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]
    y = np.r_[y1, y2]
    K = 3

if 1:
    # two off-diagonal blobs
    X1, _ = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    # three spherical blobs
    blob_centers = np.array([[-4, 1], [-4, 3], [-4, -2]])
    s = 0.5
    blob_std = np.array([s, s, s])
    X2, _ = make_blobs(
        n_samples=1000, centers=blob_centers, cluster_std=blob_std, random_state=7
    )

    X = np.r_[X1, X2]
    K = 5

plt.figure()
plt.scatter(X[:, 0], X[:, 1], 0.8)
plt.tight_layout()
plt.axis("equal")
# plt.savefig("figures/gmm_2d_data.pdf", dpi=300)
plt.show()

gm = GaussianMixture(n_components=K, n_init=10, random_state=42)
gm.fit(X)

w = gm.weights_
mu = gm.means_
Sigma = gm.covariances_

resolution = 100
grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

# score_samples is the log pdf
pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1 / resolution) ** 2
print("integral of pdf {}".format(pdf_probas.sum()))


# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html
def plot_gaussian_ellipse(gm, X):
    Y = gm.predict(X)
    means = gm.means_
    covariances = gm.covariances_
    K, D = means.shape
    if gm.covariance_type == "tied":
        covariances = np.tile(covariances, (K, 1, 1))
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        if gm.covariance_type == "spherical":
            covar = covar * np.eye(D)
        if gm.covariance_type == "diag":
            covar = np.diag(covar)
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(xy=mean, width=v[0], height=v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.3)
        splot.add_artist(ell)


plt.figure()
# plot_assignment(gm_full, X)
plot_gaussian_ellipse(gm, X)
plt.tight_layout()
plt.axis("equal")
# plt.savefig("figures/gmm_2d_clustering.pdf", dpi=300)
plt.show()


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], "k.", markersize=2)


def plot_centroids(centroids, weights=None, circle_color="w", cross_color="k"):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        s=30,
        linewidths=8,
        color=circle_color,
        zorder=10,
        alpha=0.9,
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=50,
        linewidths=50,
        color=cross_color,
        zorder=11,
        alpha=1,
    )


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution),
    )

    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(
        xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12)
    )
    plt.contour(
        xx,
        yy,
        Z,
        norm=LogNorm(vmin=1.0, vmax=30.0),
        levels=np.logspace(0, 2, 12),
        linewidths=1,
        colors="k",
    )

    # plot decision boundaries
    if 0:
        Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, linewidths=2, colors="r", linestyles="dashed")

    plt.plot(X[:, 0], X[:, 1], "k.", markersize=2)
    # plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def plot_assignment(gm, X):
    # plt.figure(figsize=(8, 4))
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    y_pred = gm.predict(X)
    K, D = gm.means_.shape
    for k in range(K):
        color = next(color_iter)
        plt.plot(X[y_pred == k, 0], X[y_pred == k, 1], "o", color=color)


gm_full = GaussianMixture(
    n_components=K, n_init=10, covariance_type="full", random_state=42
)
gm_tied = GaussianMixture(
    n_components=K, n_init=10, covariance_type="tied", random_state=42
)
gm_spherical = GaussianMixture(
    n_components=K, n_init=10, covariance_type="spherical", random_state=42
)
gm_diag = GaussianMixture(
    n_components=K, n_init=10, covariance_type="diag", random_state=42
)
gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)


def make_plot(gm, X, name):
    ttl = name
    # plt.figure(figsize=(8, 4))
    plt.figure()
    plot_gaussian_mixture(gm, X)
    fname = f"figures/gmm_2d_{name}_contours.pdf"
    plt.title(ttl)
    plt.tight_layout()
    plt.axis("equal")
    # plt.savefig(fname, dpi=300)
    plt.show()

    # plt.figure(figsize=(8, 4))
    plt.figure()
    # plot_assignment(gm, X)
    plot_gaussian_ellipse(gm, X)
    plt.title(ttl)
    fname = f"figures/gmm_2d_{name}_components.pdf"
    plt.tight_layout()
    plt.axis("equal")
    # plt.savefig(fname, dpi=300)
    plt.show()


if 1:
    make_plot(gm_full, X, "full")
    make_plot(gm_tied, X, "tied")
    make_plot(gm_spherical, X, "spherical")
    make_plot(gm_diag, X, "diag")


# Choosing K. Co,mpare to kmeans_silhouette
Ks = range(2, 9)
gms_per_k = [
    GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X) for k in Ks
]

bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]


plt.figure()
plt.plot(Ks, bics, "bo-", label="BIC")
# plt.plot(Ks, aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
# plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
if 0:
    plt.annotate(
        "Minimum",
        xy=(3, bics[2]),
        xytext=(0.35, 0.6),
        textcoords="figure fraction",
        fontsize=14,
        arrowprops=dict(facecolor="black", shrink=0.1),
    )
plt.legend()
plt.tight_layout()
# plt.savefig("figures/gmm_2d_bic_vs_k.pdf", dpi=300)
plt.show()
```

## References and Further Readings

I would say that the most important reference for this chapter is Bishop's book,
and the mathematics for machine learning book by Deisenroth et al. is also very
detailed for the derivations.

However, what is worth looking at is the code snippets from Murphy's book, he has a
wide array of examples in python code and it is very easy to follow.

- Bishop, Christopher M. "Chapter 2.3.9. Mixture of Gaussians." and "Chapter 9. Mixture Models and EM." In Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2016.
- Deisenroth, Marc Peter, Faisal, A. Aldo and Ong, Cheng Soon. "Chapter  11.1 Gaussian Mixture Models." In Mathematics for Machine Learning. : Cambridge University Press, 2020.
- Jung, Alexander. "Chapter 8.2. Soft Clustering with Gaussian Mixture Models." In Machine Learning: The Basics. Singapore: Springer Nature Singapore, 2023.
- Murphy, Kevin P. "Chapter 3.5 Mixture Models" and "Chapter 21.4 Clustering using mixture models." In Probabilistic Machine Learning: An Introduction. MIT Press, 2022.
- Vincent Tan, "Lecture 14-16." In Data Modelling and Computation (MA4270).