# Concept: Empirical Risk Minimization

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
```

## Approximating True Risk with Empirical Risk

Our machine learning problem setup is as follows:

-   $\mathcal{D}$ is the fixed but unknown distribution of the data. Usually,
    this refers to the joint distribution of the input and the label,

    $$
    \begin{aligned}
    \mathcal{D} &= \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) \\
    &= \mathbb{P}_{\{\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\}}(\mathbf{x}, y)
    \end{aligned}
    $$

    where $\mathbf{x} \in \mathcal{X}$ and $y \in \mathcal{Y}$, and
    $\boldsymbol{\theta}$ is the parameter vector of the distribution
    $\mathcal{D}$.

-   A dataset $\mathcal{S}$

    $$
    \mathcal{S} \overset{\mathrm{iid}}{\sim} \mathcal{D} = \left \{ \left(\mathrm{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^N
    $$

    be the dataset with $N$ samples and $D$ predictors where each data point in
    $\mathcal{S}$ is characterized by features $\mathbf{x} \in \mathcal{X}$ and
    labels $y \in \mathcal{Y}$.

-   a hypothesis space $\mathcal{H}$ of computationally feasible maps
    $h: \mathcal{X} \rightarrow \mathcal{Y}$,
-   and a loss function $\mathcal{L}((\mathbf{x}, y), h)$ that measures the
    discrepancy between the predicted label $h(\mathbf{x})$ and the true label
    $y$.

Ideally we would like to learn a hypothesis $h \in \mathcal{H}$ such that
$\mathcal{L}((\mathbf{x}, y), h)$ is small for **any** data point
$(\mathbf{x}, y)$. However, to implement this informal goal we need to define
what is meant precisely by "any data point". Maybe the most widely used approach
to the define the concept of "any data point" is the $\textbf{i.i.d.}$
assumption.

```{prf:definition} True Risk/Generalization Error
:label: def-true-risk

Let $\left(\mathbf{x}, y\right)$ be any data point
drawn from an $\textbf{i.i.d.}$ joint distribution ({prf:ref}`def_iid`) $\mathbb{P}_{\mathcal{D}}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$,
then the (true) risk of a hypothesis $h \in \mathcal{H}$ is the expectation of the loss
incurred by $h$ on (the realizations of) a random data point.

$$
\mathcal{R}_{\mathcal{D}}\left\{\mathcal{L}((\mathbf{x}, y), h)\right\} := \mathbb{E}_{\mathcal{D}}\left[\mathcal{L}((\mathbf{x}, y), h)\right] = \int_{\mathcal{X}} \int_{\mathcal{Y}} \mathcal{L}((\mathbf{x}, y), h) \mathbb{P}_{\mathcal{D}}(\mathbf{x}, y) \mathrm{d} \mathbf{x} \mathrm{d} y
$$ (eq:true-risk)

Note that if $\mathrm{x}$ is $D$-dimensional, then the integral $\int_{\mathcal{X}}$ is actually
a $D$-dimensional integral.

We use $\mathcal{R}$ if $\mathcal{D}$ is clear from the context.
```

```{prf:remark} Interpretation of True Risk
:label: remark-interpretation-true-risk

We know in
[Joint Expection](../../probability_theory/05_joint_distributions/0502_joint_expectation_and_correlation/concept.md)
can be found by integrating over the joint distribution
$\mathbb{P}_{\mathcal{D}}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$,
where the integrand is usually the probability of the event happening multiplied
by the state of the event. In the case of the true risk, the event is the
realization of a data point, and the state of the event is the loss incurred by
the hypothesis $h$ on the data point.

One may wonder why $\mathcal{L}((\mathbf{x}, y), h)$ being the state of the
event has the associated probability $\mathbb{P}_{\mathcal{D}}(\mathbf{x}, y)$.
First, we recognize that $\mathcal{L}((\mathbf{x}, y), h)$ is a random variable,
and the probability of it **happening** is the probability of the event
$(\mathbf{x}, y)$ happening, which in turn is the probability of the joint
distribution
$\mathbb{P}_{\mathcal{D}}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$.

The main confusion can be caused by the lack of understanding of random
variables. For example, you can also define the joint expecation of the random
variables $\mathbf{x}$ and $y$ as

$$
\mathbb{E}_{\mathcal{D}}\left[\mathbf{x}, y\right] = \int_{\mathcal{X}} \int_{\mathcal{Y}} \left(x_1 \cdot x_2 \cdots x_D \cdot y\right) \cdot \mathbb{P}_{\mathcal{D}}(\mathbf{x}, y) \mathrm{d} \mathbf{x} \mathrm{d} y
$$

Then in that case, the state is the joint vector $\left(x_1 \cdot x_2 \cdots x_D, y\right)$, and the probability
of the event happening is the probability of the joint distribution $\mathbb{P}_{\mathcal{D}}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$.

Though the associated probability is the same, the state and the event are different.
```

```{prf:remark} $\textbf{i.i.d.}$ Assumption in ERM
:label: rem-iid-erm

Since the $\textbf{i.i.d.}$ assumption interprets data points as realizations of iid RVs with a
common probability distribution $\mathbb{P}_{\mathcal{D}}$, then the concept of
"any data point" can be made precise since any data point must be drawn independently
from the same distribution $\mathbb{P}_{\mathcal{D}}$.
```

```{prf:theorem} True Risk Minimization
:label: theorem-true-risk-minimization

If we do know the true distribution $\mathbb{P}_{\mathcal{D}}$, then we can minimize
the true risk directly by finding/learning a hypothesis $h \in \mathcal{H}$
such that its true risk {eq}`eq:true-risk` is minimized.

$$
h^{*} = \underset{h \in \mathcal{H}}{\mathrm{argmin}} \mathbb{E}_{\mathcal{D}}\left[\mathcal{L}((\mathbf{x}, y), h)\right]
$$ (eq: true-risk-minimization)
```

[](03_bayes_optimal_classifier.md)

```{prf:example} Bayes Optimal Classifier
:label: example-bayes-optimal-classifier

See [Bayes Optimal Classifer](03_bayes_optimal_classifier.md) for a concrete
example. The idea is that if we know the true distribution
$\mathbb{P}_{\mathcal{D}}$, then it is relatively easy to find the hypothesis
$h^{*}$ that minimizes the true risk.

We can simply construct such a hypothesis $h^{*}$ as

$$
h^{*}(\mathbf{x}) = \underset{y \in \mathcal{Y}}{\mathrm{argmax}} \mathbb{P}_{\mathcal{D}}\left(y \mid \mathbf{x}\right)
$$ (eq: bayes-optimal-classifier)

and declare that this hypothesis $h^{*}$ minimizes the true risk $R\left\{\mathcal{L}((\mathbf{x}, y), h)\right\}$.
```

Of course, in practice we do not know the true distribution
$\mathbb{P}_{\mathcal{D}}$, and therefore we cannot minimize the true risk
directly. Instead, the empirical risk minimization (ERM) principle states that
we can minimize the empirical risk. That is to say, we approximate the
expectation in {eq}`eq:true-risk` by the empirical average of the loss incurred
by $h$ on the training data $\mathcal{S}$.

More concretely, we state it as follows.

```{prf:definition} Empirical Risk
:label: def-empirical-risk

Let $\mathcal{S} = \left \{ \left(\mathbf{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^N$
be the dataset with $N$ samples and $D$ predictors where each data point in $\mathcal{S}$
follows an $\textbf{i.i.d.}$ joint distribution $\mathbb{P}_{\mathcal{D}}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$,
then the empirical risk of a hypothesis $h \in \mathcal{H}$ is the average of the loss
incurred by $h$ on the training data.

$$
\hat{\mathcal{R}}_{\mathcal{S}} \left\{\mathcal{L}((\mathbf{x}, y), h)\right\} := \frac{1}{N} \sum_{n=1}^N \mathcal{L}\left(\left(\mathbf{x}^{(n)}, y^{(n)}\right), h\right)
$$ (eq:empirical-risk)

To ease notation, we use $\hat{\mathcal{R}}$ if $\mathcal{S}$ is clear from the context.
```

```{prf:example} Empirical Risk
:label: example-empirical-risk

Let's cite the example from {cite}`a_course_in_machine_learning`.

We write $\mathbb{E}_{(x, y) \sim \mathcal{D}}[\mathcal{L}(y, f(\boldsymbol{x}))]$ for the expected loss. Expectation means "average." This is saying "if you drew a bunch of $(x, y)$ pairs independently at random from the underlying distribution $\mathcal{D}$, what would your average loss be?
More formally, if $\mathcal{D}$ is a discrete probability distribution, then this expectation can be expanded as:

$$
\mathbb{E}_{(\boldsymbol{x}, y) \sim \mathcal{D}}[\mathcal{L}(y, f(\boldsymbol{x}))]=\sum_{(x, y) \in \mathcal{D}}[\mathcal{D}(\boldsymbol{x}, y) \mathcal{L}(y, f(\boldsymbol{x}))]
$$

This is exactly the weighted average loss over the all $(x, y)$ pairs in $\mathcal{D}$, weighted by their probability, $\mathcal{D}(\boldsymbol{x}, y)$. If $\mathcal{D}$ is a finite discrete distribution, for instance defined by a finite data set $\mathcal{S} = \left\{\left(x_1, y_1\right), \ldots,\left(x_N, y_N\right)\right.$ that puts equal weight on each example (probability $1 / N$ ), then we get:

$$
\begin{aligned}
\mathbb{E}_{(x, y) \sim \mathcal{D}}[\mathcal{L}(y, f(x))] & =\sum_{(x, y) \in \mathcal{D}}[\mathcal{D}(x, y) \mathcal{L}(y, f(x))] & \text{definition of expectation} \\
& =\sum_{n=1}^N\left[\mathcal{D}\left(x_n, y_n\right) \mathcal{L}\left(y_n, f\left(x_n\right)\right)\right] & \mathcal{D}_{\text{train}} \text{is discrete and finite} \\
& =\sum_{n=1}^N\left[\frac{1}{N} \mathcal{L}\left(y_n, f\left(x_n\right)\right)\right] & \text{definition of } \mathcal{D}_{\text{train}} \\
& =\frac{1}{N} \sum_{n=1}^N\left[\mathcal{L}\left(y_n, f\left(x_n\right)\right)\right] & \text{rearranging terms}
\end{aligned}
$$


This is exactly the average loss on that dataset.

The most important thing to remember is that there are two equivalent ways to think about expections: (1) The expectation of some function $g$ is the weighted average value of $g$, where the weights are given by the underlying probability distribution. (2) The expectation of some function $g$ is your best guess of the value of $g$ if you were to draw a single item from the underlying probability distribution.
```

```{prf:theorem} Empirical Risk Minimization
:label: theorem-empirical-risk-minimization

The empirical risk minimization (ERM) principle states that we can minimize the empirical risk
as follows:

$$
\begin{aligned}
\hat{h} \in \mathcal{H} &= \underset{h \in \mathcal{H}}{\mathrm{argmin}} \hat{\mathcal{R}}\left\{\mathcal{L}((\mathbf{x}, y), h)\right\} \\
&= \underset{h \in \mathcal{H}}{\mathrm{argmin}} \frac{1}{N} \sum_{n=1}^N \mathcal{L}\left(\left(\mathbf{x}^{(n)}, y^{(n)}\right), h\right)
\end{aligned}
$$ (eq: empirical-risk-minimization)

We see that $\hat{h}$ may not be unique, so it can mean any hypothesis that minimizes the empirical risk.
```

```{prf:remark} Risk vs Loss
:label: rem-risk-vs-loss

Note that the empirical risk $\mathcal{R}$ is the expectation of the loss $\mathcal{L}$.
They are not the same thing!

Sometimes we also call the risk the cost function or the objective function.
```

```{prf:remark} ERM is "learning from data using trial and error"
:label: rem-erm-trial-and-error

As we can see now, there is no closed-form solution to the empirical risk minimization problem and therefore
we may try to find a good hypothesis $h$ by trial and error. And a good hypothesis $h$ is one that has
the least empirical risk among all hypotheses in $\mathcal{H}$.
```

(emprical-risk-approximates-true-risk)=

## Empirical Risk Minimization approximates True Risk Minimization

```{prf:theorem} Empirical Risk Minimization approximates True Risk Minimization when $N \rightarrow \infty$
:label: theorem-erm-approximates-trm

If we have a large enough dataset $\mathcal{S}$ with $N \rightarrow \infty$,
then the empirical risk minimization (ERM) principle approximates the true risk minimization (TRM) principle.

$$
\mathcal{R}_{\mathcal{D}}\left\{\mathcal{L}((\mathbf{x}, y), h)\right\} \approx \hat{\mathcal{R}}\left\{\mathcal{L}((\mathbf{x}^{(n)}, y^{(n)}), h)\right\} \quad \text{as} \quad N \rightarrow \infty
$$ (eq: empirical-risk-approximates-true-risk)

This is just a corollary of The Convergence Theorem of The Law of Large Numbers
({prf:ref}`theorem-weak-law-of-large-numbers`). Indeed, we can treat
each $\mathcal{L}((\mathbf{x}, y), h)$ as a $\textbf{i.i.d.}$ random variable such that
$\mathcal{L}^{(n)}((\mathbf{x}^{(n)}, y^{(n)}), h)$ is the loss incurred by $h$ on the $n$-th sample.
Then $\mathcal{L}^{(1)}, \mathcal{L}^{(2)}, \ldots, \mathcal{L}^{(N)}$ are $\textbf{i.i.d.}$ random variables,
then the sample mean

$$
\hat{\mathcal{R}}\left\{\mathcal{L}\left(\left(\mathbf{x}^{(n)}, y^{(n)}\right), h\right)\right\} = \frac{1}{N} \sum_{n=1}^N \mathcal{L}^{(n)}\left(\left(\mathbf{x}^{(n)}, y^{(n)}\right), h\right)
$$

converges to the true risk expectation $\mathcal{R}_{\mathcal{D}}\left\{\mathcal{L}((\mathbf{x}, y), h)\right\}=\mathbb{E}_{\mathcal{D}}\left[\mathcal{L}((\mathbf{x}, y), h)\right]$ as $N \rightarrow \infty$.
```

Therefore, ERM is motivated by the law of large numbers, and in turn, the law of
large numbers is motivated by the samples being $\textbf{i.i.d.}$
{cite}`jung2022machine`.

Note that the choice of the loss function $\mathcal{L}$ is crucial to the
performance of the hypothesis $h$. And for different tasks such as Linear
Regression, Logistic Regression, and other learning algorithms, we can design
different loss functions, and in turn different empirical risk minimization
formulations.
