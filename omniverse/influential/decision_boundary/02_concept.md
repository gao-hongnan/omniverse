# Concept: Decision Boundary

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

The interclass boundary, also known as the decision boundary, is derived from
the parameters estimated by the model. The decision boundary is found by setting
the predictions of class 0 and class 1 equal, and solving for the values of x
that produce equal predictions. In the case of logistic regression, the
parameters are the coefficients in the linear regression equation, and the
decision boundary is found by solving for x when the logistic function is equal
to 0.5. In the case of Gaussian naive Bayes, the parameters are the mean and
variance of each class, and the decision boundary is found by solving for x when
the posterior probabilities of the two classes are equal. In both cases, the
estimated parameters are used to derive the decision boundary, which is then
used to make predictions for new observations.

## Decision Boundary

We will assume that the dataset has $N$ samples,$K=2$ classes and $D$ features.

```{prf:definition} Decision Boundary
:label: def:decision_boundary

A decision boundary is a hyperplane that separates the two classes in a
dataset (sample) $\mathcal{S} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$.

More specifically, it is the set of points $\mathbf{x}$ such that the following
equation holds:

$$
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) = \mathbb{P}(\hat{Y} = 0 \mid \mathbf{X} ;  \boldsymbol{\hat{\theta}})
$$ (eq:decision_boundary_eq_1)

where $\hat{Y}$ is the predicted class (or logits), $\mathbf{X}$ is the feature vector, and
$\boldsymbol{\theta}$ is the estimated model parameters.

In other words, the decision regions are half-planes governed by the decision boundary, a hyperplane that separates the two classes.

Note that the equation above has an implicit constraint, that the threshold $t$ is set to 0.5. That means
that when the probability $\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}})$ is greater than 0.5, the predicted class is 1, and when it is less than 0.5, the predicted class is 0 (the converse for
the negative class since they are complement). As a result, if your threshold changes, your decision boundary will change as well. But where did this threshold slip in?

Consider the following equation:

$$
\begin{aligned}
&\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) = \mathbb{P}(\hat{Y} = 0 \mid \mathbf{X} ;  \boldsymbol{\hat{\theta}})  \\
&\textrm{s.t.} \quad  \mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) + \mathbb{P}(\hat{Y} = 0 \mid \mathbf{X} ;  \boldsymbol{\hat{\theta}}) = 1 \\
\end{aligned}
$$

where the constraint is merely the normalization of the probabilities. With this constraint, it follows
that $\mathbb{P}(\hat{Y}=0 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) = 1 - \mathbb{P}(\hat{Y}=1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}})$. Substituting this into the first equation, we get:

$$
\begin{aligned}
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) &= \mathbb{P}(\hat{Y} = 0 \mid \mathbf{X} ;  \boldsymbol{\hat{\theta}}) \\
&\iff \\
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) &= 1 - \mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ;  \boldsymbol{\hat{\theta}})  \\
&\iff \\
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) &+ \mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ;  \boldsymbol{\hat{\theta}}) = 1 \\
&\iff \\
2 \mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) &= 1 \\
&\iff \\
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) &= \frac{1}{2} \\
\end{aligned}
$$

So another way of writing {eq}`eq:decision_boundary_eq_1` is

$$
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) = \frac{1}{2}
$$ (eq:decision_boundary_eq_2)
```

## Binary Logistic Regression Decision Boundary

Consider the following logistic regression model with $K=2$ classes and $D=2$
features, and $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$:

$$
\begin{aligned}
Y &= \sigma(\boldsymbol{\theta}^T \mathbf{x}) \\
&= \sigma(\begin{bmatrix} \theta_1 & \theta_2 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}) \\
&= \sigma(\theta_1 x_1 + \theta_2 x_2) \\
&= \frac{1}{1 + e^{-(\theta_1 x_1 + \theta_2 x_2)}} \\
\end{aligned}
$$

where $\sigma(\cdot)$ is the sigmoid function.

If there is a bias, then we still have the same formula but with an additional
term $\theta_0$ and adding a column of 1s to the feature vector $\mathbf{X}$:

$$
\begin{aligned}
Y &= \sigma(\boldsymbol{\theta}^T \mathbf{x}) \\
&= \sigma(\begin{bmatrix} \theta_0 & \theta_1 & \theta_2 \end{bmatrix} \begin{bmatrix} 1 \\ x_1 \\ x_2 \end{bmatrix}) \\
&= \sigma(\theta_0 + \theta_1 x_1 + \theta_2 x_2) \\
&= \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2)}} \\
\end{aligned}
$$

In any case, we aim to model the probability of the positive class,
$\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}})$, where
$\hat{Y}$ is the predicted class (or logits), $\mathbf{X}$ is the feature
vector, and find the parameters $\boldsymbol{\hat{\theta}}$ that minimize the
loss function.

Now the decision boundary is given by:

$$
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) = \mathbb{P}(\hat{Y} = 0 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}})
$$ (eq:decision_boundary_logistic_eq_1)

where $\boldsymbol{\hat{\theta}}$ is the vector of parameters that minimizes the loss function.

Equivalently, you can also solve


$$

\begin{aligned} \mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ;
\boldsymbol{\hat{\theta}}) &= \frac{1}{2} \\ \end{aligned}

$$
(eq:decision_boundary_logistic_eq_2)

This means


$$

\begin{aligned} \frac{1}{1 + e^{-(\hat{\theta}\_1 x_1 + \hat{\theta}\_2 x_2 +
\hat{\theta}\_0)}} &= \frac{1}{2} \\ e^{-(\hat{\theta}\_1 x_1 + \hat{\theta}\_2
x_2 + \hat{\theta}\_0)} &= 1 \\ -(\hat{\theta}\_1 x_1 + \hat{\theta}\_2 x_2 +
\hat{\theta}\_0) &= \ln(1) \\ -(\hat{\theta}\_1 x_1 + \hat{\theta}\_2 x_2 +
\hat{\theta}\_0) &= 0 \\ \hat{\theta}\_1 x_1 + \hat{\theta}\_2 x_2 +
\hat{\theta}\_0 &= 0 \\ \boldsymbol{\hat{\theta}}^T \mathbf{x} &= 0 \\
\end{aligned}

$$

Thus the decision boundary is a line in the 2D space.

Note if you want to solve it by {eq}`eq:decision_boundary_logistic_eq_1`, then you just define


$$

\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) =
\frac{1}{1 + e^{-(\hat{\theta}\_1 x_1 + \hat{\theta}\_2 x_2 + \hat{\theta}\_0)}}

$$

and


$$

\mathbb{P}(\hat{Y} = 0 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) = 1 -
\mathbb{P}(\hat{Y} = 1 \mid \mathbf{X} ; \boldsymbol{\hat{\theta}}) =
\frac{1}{1 + e^{-(\hat{\theta}\_1 x_1 + \hat{\theta}\_2 x_2 + \hat{\theta}\_0)}}

$$

and equate them.

## Weighted Decision Boundary

See reference below.

So if there is more than 2 classes, then the decision boundary is a hyperplane that separates the classes. The hyperplane is defined by the equation:

If we want to weight the positive class ($y = 1$) more or less using $w$, here is the general decision boundary:
$$w{\Bbb P}(y=1|\boldsymbol{x}) = {\Bbb P}(y=0|\boldsymbol{x}) = \frac{w}{w+1}$$

For example, $w=2$ means point $\boldsymbol{x}$ will be assigned to positive class if ${\Bbb P}(y=1|\boldsymbol{x}) > 0.33$ (or equivalently if ${\Bbb P}(y=0|\boldsymbol{x}) < 0.66$), which implies favoring the positive class (increasing the true positive rate).

Here is the line for this general case:


$$

\begin{align*} &\frac{1}{1+e^{-\boldsymbol{\theta}^t\boldsymbol{x*+}}} =
\frac{1}{w+1} \\ &\Rightarrow e^{-\boldsymbol{\theta}^t\boldsymbol{x*+}} = w\\
&\Rightarrow \boldsymbol{\theta}^t\boldsymbol{x\_+} = -\text{ln}w\\ &\Rightarrow
\theta_0 + \theta_1 x_1+\cdots+\theta_d x_d = -\text{ln}w \end{align*}

$$


## Further Readings

- https://datascience.stackexchange.com/questions/49573/how-to-plot-logistic-regression-decision-boundary
$$
