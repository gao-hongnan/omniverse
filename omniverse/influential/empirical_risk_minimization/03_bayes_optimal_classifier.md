# Bayes Optimal Classifier

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
```

## Definition

For a classification problem on $\mathcal X\times\mathcal Y$ with feature space
$\mathcal X = \mathbb R^D$ and $K$ classes $\mathcal Y = \{1,\ldots,K\}$ (i.e.
an element in $\mathcal{X} \times \mathcal{Y}$ is a pair
$(\boldsymbol{x}, y) \in \mathbb{R}^D \times \{1, \ldots, K\}$).

**Assume** that the conditional distribution of $X$ given the target $Y$ is
given by:

$$
\begin{align*}
\mathbf{X} \mid Y &= 1 &\overset{\small{\text{i.i.d.}}}{\sim} \mathbb{P}_{\{\mathbf{X} \mid Y=1\ ; \boldsymbol{\theta}\}} \\
\mathbf{X} \mid Y &= 2 &\overset{\small{\text{i.i.d.}}}{\sim} \mathbb{P}_{\{\mathbf{X} \mid Y=2\ ; \boldsymbol{\theta}\}} \\
\vdots \\
\mathbf{X} \mid Y &= K &\overset{\small{\text{i.i.d.}}}{\sim} \mathbb{P}_{\{\mathbf{X} \mid Y=K\ ; \boldsymbol{\theta}\}}
\end{align*}
$$

Then we can define a classifier $h$ as a rule that maps an observation $X=x$ to
estimate the true underlying class $Y$ as $h(x) := \hat{y}$. In this case, the
classifer is just a hypothesis $h: \mathbb{R}^D \to \{1, \ldots, K\}$ where $h$
classifies an observation $\boldsymbol{x}$ to the class $h(\boldsymbol{x})$.

The probability of misclassification or true risk, of a classifier $h$ can be
defined as:

$$
\mathcal{R}_{\mathcal{D}}(h) = \mathbb{P}_{\mathcal{D}}(h(\mathbf{x}) \neq y) = \mathbb{E}_{\mathcal{D}}[1_{h(\mathbf{x}) \neq y}]
$$

where $1_{h(\mathbf{x}) \neq y}$ is an indicator function that is 1 if
$h(\mathbf{x}) \neq y$ and 0 otherwise. Note that $h(\mathbf{x}) \neq y$ is
equivalent to the loss function $\mathcal{L}$ defined over a single pair
$(\mathbf{x}, y)$ as
$\mathcal{L}(h(\mathbf{x}), y) = 1_{h(\mathbf{x}) \neq y}$[^0-1-loss]. This loss
function just means the probability in which the classifier $h$ makes a mistake.
This form is exactly the same form as the generalization error in
{prf:ref}`def-true-risk`.

We can construct a classifier $h^{*}$ that minimizes the true risk if we know
the true distribution $\mathcal{P}_{\mathcal{D}}$. Such a classifier is called
the Bayes classifier. The Bayes classifier is defined as:

$$
h^{*} := \underset{k \in \mathcal{Y}}{\mathrm{argmax}} \quad \mathbb{P}_{\mathcal{D}}(Y=k\ |\ \boldsymbol{X} = \boldsymbol{x})
$$

It can be [shown](https://en.wikipedia.org/wiki/Bayes_classifier) that $h^{*}$
is the classifier that minimizes the true risk $\mathcal{R}_{\mathcal{D}}$ for
such a construction of the classifier, along with the 0-1 loss function.

This means for classifier $h \in \mathcal{H}$, you **always** have

$$
\mathcal{R}_{\mathcal{D}}(h) \geq \mathcal{R}_{\mathcal{D}}(h^{*})
$$

```{prf:remark} Bayes Optimal Classifier and Naive Bayes
:label: remark-bayes-optimal-classifier-naive-bayes

If the components of $\mathbf{X}$ are conditionally independent given $Y$, then the Bayes classifier is
the same as the Naive Bayes classifier.
```

```{prf:remark} Some remarks
:label: remark-bayes-optimal-classifier

Note that this definition merely says that the Bayes classifier achieves minimal zero-one loss over any other
deterministic classifier, it does not say anything about it achieving zero error.

Also note that this is of course not true for the general case of a classifier $h \in \mathcal{H}$, where
the underlying distribution $\mathcal{P}_{\mathcal{D}}$ is unknown.
```

## Further Readings

-   Hal Daumé III. "Chapter 2.1. Data Generating Distributions." In A Course in
    Machine Learning, January 2017.
-   [What does it mean for the Bayes Classifier to be optimal?](https://stats.stackexchange.com/questions/567299/what-does-it-mean-for-the-bayes-classifier-to-be-optimal)
-   [Wikipedia: Bayes classifier](https://en.wikipedia.org/wiki/Bayes_classifier)

[^0-1-loss]: This is basically the 0-1 loss function.
