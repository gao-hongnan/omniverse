# Concept: Loss Functions

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

## Loss

```{prf:definition} Loss Function
:label: def-loss-function

Formally, a loss function is a map {cite}`jung_2023`

$$
\begin{aligned}
\mathcal{L}: \mathcal{X} \times \mathcal{Y} \times \mathcal{H} &\to \mathbb{R}_{+} \\
\left((\mathbf{x}, y), h\right) &\mapsto \mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)
\end{aligned}
$$

which maps a pair of data point $\mathbf{x}$ and label $y$ together with a hypothesis $h$ to a non-negative real number $\mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)$.

For example, if $h$ is an element from the linear map taking on the form $h(\mathbf{x})=\mathbf{w}^T\mathbf{x}$, then the loss is a function of the parameters $\mathbf{w}$ of the hypothesis $h$.
This means we seek to find $\hat{\mathbf{w}}$ that minimizes the loss function $\mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)$.

We sometimes abuse notation by writing $\mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)$ as $\mathcal{L}\left(y, \hat{y}\right)$, where $\hat{y}=h(\mathbf{x})$ is the predicted label of the hypothesis $h$ for the data point $\mathbf{x}$.
```

```{prf:definition} Loss Function as a Random Variable
:label: def-loss-function-rv

[In our random variables chapter](../../probability_theory/03_discrete_random_variables/0301_random_variables.md),
we defined a random variable as a map of sample space to real numbers.

The definition of a loss function satisfies the requirement to be a random variable.
Indeed, the loss function is usually expressed as a function of some other random variables.

Furthermore, each $\mathcal{L}(y, \hat{y})$ is a random variable associated with a specific data point $\mathbf{x}$ and label $y$.

Consequently, the sample space of the loss function is $\Omega_{\mathcal{X}} \times \Omega_{\mathcal{Y}} \times \Omega_{\mathcal{H}}$, where $\Omega_{\mathcal{X}}$, $\Omega_{\mathcal{Y}}$, and $\Omega_{\mathcal{H}}$ are the sample spaces of the data points, labels, and hypotheses, respectively.
Thus, by the definition of the random variable, the realization of the loss function is a non-negative real number $\mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)$,
this is also the **state** of the random variable.
```

## Further Readings

-   Jung, Alexander. "Chapter 2.3. The Loss." In Machine Learning: The Basics.
    Springer Nature Singapore, 2023.
-   [Wikipedia: Empirical Risk Minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization)
-   [Wikipedia: Loss Function](https://en.wikipedia.org/wiki/Loss_function)-
    [Empirical Risk Minimization](../empirical_risk_minimization/01_intro.md)
