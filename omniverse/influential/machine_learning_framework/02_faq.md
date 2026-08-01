# FAQ: The Machine Learning Framework

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

-   {prf:ref}`remark-interpretation-true-risk`
-   Knowing the joint probability distribution of $\mathbf{X}$ and $Y$ means
    that the conditional probability of $Y$ given $\mathbf{X}$ is fully
    determined. This is evident from Bayes' theorem:

    $$
    \begin{align*}
    \mathbb{P}(Y=y\ |\ \mathbf{X} = \mathbf{x}) &= \frac{\mathbb{P}(\mathbf{X} = \mathbf{x}\ |\ Y=y)\ \mathbb{P}(Y=y)}{\mathbb{P}(\mathbf{X} = \mathbf{x})} \\
    \end{align*}
    $$

    since knowing the joint distribution
    $\mathbb{P}_{\mathcal{D}}(\mathcal{X}, \mathcal{Y}; \boldsymbol{\theta})$
    means the numerator
    $\mathbb{P}(\mathbf{X} = \mathbf{x}\ |\ Y=y)\ \mathbb{P}(Y=y)$ is fully
    determined since they are equal (recall the numerator is none other the
    probability of $\mathbf{X}$ and $Y$ happening at the same time). Then once
    the joint distribution is known, the denominator
    $\mathbb{P}(\mathbf{X} = \mathbf{x})$ is also fully determined since it is
    the marginal probability of $\mathbf{X}$. It follows that the conditional
    probability of $Y$ given $\mathbf{X}$ is fully determined.

    It is good to realize the ultimate goal in learning a supervised problem is
    generally specified as finding the conditional probability of $Y$ given
    $\mathbf{X}$, which is fully determined once the joint distribution is
    known.

## PCA

See
[the geometry of the multivariate Gaussian](../../probability_theory/05_joint_distributions/0507_multivariate_gaussian/geometry_of_multivariate_gaussian.md)
on treatment and intuition of eigenvectors and eigenvalues.

-   [Why probability contours for the multivariate Gaussian are elliptical](https://www.michaelchughes.com/blog/2013/01/why-contours-for-multivariate-gaussian-are-elliptical/#:~:text=Every%202D%20Gaussian%20concentrates%20its,a%20particular%20form%3A%20an%20ellipse.)
-   [Why are contours of a multivariate Gaussian distribution elliptical?](https://stats.stackexchange.com/questions/326334/why-are-contours-of-a-multivariate-gaussian-distribution-elliptical)

## Loss/Cost Functions (Optimization)

-   [How is it possible that the MSE used to train neural networks with gradient descent has multiple local minima?](https://ai.stackexchange.com/questions/11979/how-is-it-possible-that-the-mse-used-to-train-neural-networks-with-gradient-desc)
-   [Why Linear Regression Estimates the Conditional Mean](https://www.sophieheloisebennett.com/posts/linear-regression-conditional-mean/#:~:text=%E2%80%9CLinear%20regression%20estimates%20the%20conditional,of%20the%20response%20variable%20Y%20.)
-   [Formal Proof on Mean Minimizes SSE](https://math.stackexchange.com/questions/967138/formal-proof-that-mean-minimize-squared-error-function)
-   [Local Minima and Global Minima in Machine Learning](https://stats.stackexchange.com/questions/521786/what-are-global-minima-and-local-minima-in-machine-learning)
-   Error, Cost, Loss and Risk
    -   https://stats.stackexchange.com/questions/424170/error-cost-loss-risk-are-those-4-terms-the-same-in-the-context-of-machine-le
    -   https://stats.stackexchange.com/questions/179026/objective-function-cost-function-loss-function-are-they-the-same-thing

## K-Means

-   [Purity Score](https://stackoverflow.com/questions/34047540/python-clustering-purity-metric)
-   [Why doesn't k-means give the global minimum?](https://stats.stackexchange.com/questions/48757/why-doesnt-k-means-give-the-global-minimum)
