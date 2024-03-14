# Machine Learning Notations

## Conventions

We largely follow the
[Machine Learning: The Basics](https://link.springer.com/book/10.1007/978-981-16-8193-6)
book in terms of notations. Some of them are also taken from
[Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlbook/).

## Learning Problem (Conditional Maximum Likelihood Estimation)

As an example, we refer to this important paragraph:

We consider a multi-class classification problem with $c$ classes, $c \geq 1$.
Let $y=$ $\{1, \ldots, c\}$ denote the output space and $\mathcal{D}$ a
distribution over $\mathcal{X} \times \mathcal{y}$. The learner receives a
labeled training sample
$S=\left(\left(x_1, y_1\right), \ldots,\left(x_m, y_m\right)\right) \in(\mathcal{X} \times \mathcal{y})^m$
drawn i.i.d. according to $\mathcal{D}$. As in Chapter 12, we assume that,
additionally, the learner has access to a feature mapping
$\Phi: X \times y \rightarrow \mathbb{R}^N$ with $\mathbb{R}^N$ a normed vector
space and with $\|\Phi\|_{\infty} \leq r$. We will denote by $\mathcal{H}$ a
family of real-valued functions containing the component feature functions
$\Phi_j$ with $j \in[N]$. Note that in the most general case, we may have
$N=+\infty$. The problem consists of using the training sample $S$ to learn an
accurate conditional probability $\mathrm{p}[\cdot \mid x]$, for any $x \in X$.

This is extracted from section 13.1. Learning Problem in
[Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlbook/). This
treatment is important for you to appreciate the following notations.

---

## Mathematical Notations

We often abuse notation and use $\mathcal{D}$ to denote the dataset
$\mathcal{S}$ or $\mathcal{D}_{\mathrm{train}}$ and the probability distribution
$\mathcal{D} = \mathbb{P}_{\mathcal{D}}$. The context should be clear. However,
as of now, we will try to be consistent in the usage of $\mathcal{D}$ where it
is a probability distribution and not a dataset.

This source of abuse notation is because $\mathcal{D}$ is written as shorthand
for the word _dataset_. We try to avoid it here. As the true dataset can be
infinite and is defined by $\mathcal{X} \times \mathcal{Y}$ for learning
problems with labels.

Abuse notation: $\mathcal{S}$ or $\mathcal{D}_{\mathrm{train}}$ or
$\mathcal{D}_{\{\mathbf{x}, y\}}$

````{div} full-width
```{list-table} Machine Learning Notations
:header-rows: 1
:name: machine-learning-notations
:widths: 10 40 50

* - Notation
  - Description
  - Example
* - $\mathbf{z}$
  - A data point which is characterized by several properties that we divide $\mathbf{z}$ into low-level properties (features) and high-level properties (labels). More concretely,
  $\mathbf{z}=\left(\mathbf{x}, y\right)$, where $\mathbf{x}$ is a feature vector and $y$ is a label.
  -
* - $n$
  - They usually represent index $n=1,2, \ldots$, that enumerates the data points within a dataset.
  -
* - $N$
  - The number of data points in (the size of) a dataset.
  -
* - $d$
  - They usually represent index $d=1,2, \ldots$, that enumerates the features within a feature vector.
  -
* - $D$
  - The dimensionality of a feature vector $\mathbf{x}$.
  -
* - $\mathcal{X}$
  - The feature space of a ML method consists of all potential feature values that a data point can have. However, we typically use feature spaces that $\mathcal{X}$ are much larger than the set of different feature values arising in finite datasets. The majority of the methods discussed in this book uses the feature space $\mathcal{X}=\mathbb{R}^D$ consisting of all Euclidean vectors of length $D$.
  - Let's consider a simple example of a machine learning problem where we want to predict the price of a house based on its features. In this case, the feature space $\mathcal{X}$ consists of all possible combinations of feature values for a house.

    Let's assume we have the following features:

    1. Area (in square feet)
    2. Number of bedrooms
    3. Number of bathrooms
    4. Age of the house (in years)

    The feature vector for each house is then a 4-dimensional vector, with each dimension corresponding to one of these features. For example, a specific house might have the following feature vector:

    $$
    \mathbf{x} = \begin{bmatrix} 2000 \\ 3 \\ 2 \\ 10 \end{bmatrix}
    $$

    In this example, the feature vector represents a house with 2,000 square feet of area, 3 bedrooms, 2 bathrooms, and an age of 10 years. The feature space $\mathcal{X}$ in this case is $\mathbb{R}^4$, as there are 4 dimensions for the feature vector. This means that the feature space consists of all possible 4-dimensional real-valued vectors. Note that in practice, some values might not make sense (e.g., negative area or negative number of bedrooms), but in general, $\mathbb{R}^4$ represents the entire space of possible feature vectors.

* - $\mathcal{Y}$
  - The label space $\mathcal{Y}$ of a ML method consists of all potential label values that a data point can have. We often use label spaces that are larger than the set of different label values arising in a give dataset (e.g., a training set). We refer to a ML problems (methods) using a numeric label space, such as $\mathcal{Y}=\mathbb{R}$ or $\mathcal{Y}=\mathbb{R}^3$, as regression problems (methods). ML problems (methods) that use a discrete label space, such as $\mathcal{Y}=\{0,1\}$ or $\mathcal{Y}=\{$ "cat", "dog", "mouse" $\}$ are referred to as classification problems (methods).
  -
* - $\mathcal{D}$
  - The fixed (true) but unknown distribution of the data over the input and label space $\mathcal{X} \times \mathcal{Y}$. Usually, this refers
to the joint distribution of the input and the label,

    $$
    \begin{aligned}
    \mathcal{D} &= \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) \\
    &= \mathbb{P}_{\{\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}\}}(\mathbf{x}, y)
    \end{aligned}
    $$

    where $\mathbf{x} \in \mathcal{X}$ and $y \in \mathcal{Y}$, and $\boldsymbol{\theta}$ is the
    parameter vector of the distribution $\mathcal{D}$.'
  -
* - $\boldsymbol{\theta}$
  - The parameter vector of a distribution $\mathcal{D}$.
  -
* - $\hat{\boldsymbol{\theta}}$
  - The parameter vector of a distribution $\mathcal{D}$ that is estimated from a dataset $\mathcal{S}$.
  -
* - $\boldsymbol{\Theta}$
  - The parameter space of a distribution $\mathcal{D}$. All $\boldsymbol{\theta} \in \boldsymbol{\Theta}$ are valid parameters of $\mathcal{D}$.
  -
* - $\underset{(\mathbf{x}, y) \sim \mathcal{D}}{\mathbb{P}} = \mathbb{P}_{\mathcal{D}} = \mathbb{P}_{\mathcal{D}}(\mathbf{x}, y)$
  - The fixed (true) but unknown distribution of the data. Usually samples are
  drawn $\textbf{i.i.d.}$ from $\underset{(\mathbf{x}, y) \sim \mathcal{D}}{\mathbb{P}}$.

    Sometimes, we also denote it by $\mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$
    where $\boldsymbol{\theta}$ is a set of parameters that define the distribution $\mathbb{P}$.
    **This notation is very similar to $\mathcal{D}$, and may be used interchangeably.**
  -
* - $\mathcal{S}$ or $\mathcal{S}_{\mathrm{train}}$ or $\mathcal{S}_{\{\mathbf{x}, y\}}$
  - A (sample) dataset $\mathcal{S}=\left\{\mathbf{z}^{(1)}, \ldots, \mathbf{z}^{(N)}\right\}$ is a list of individual data points $\mathbf{z}^{(n)}$, for $n=1, \ldots, N$.

    $$
    \mathcal{S} \overset{\mathrm{iid}}{\sim} \mathcal{D} = \left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \}_{n=1}^N = \left \{ \left(\mathrm{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^N \in \left(\mathcal{X}, \mathcal{Y} \right)^N
    $$

    This means that $\mathcal{S}$ is a collection of samples drawn $\textbf{i.i.d.}$ from an unknown distribution $\mathcal{D}$ over the joint distribution of the input and the label space
    $\mathcal{X} \times \mathcal{Y}$.
  -
* - $\mathcal{S}_{\mathrm{test}}$
  - A test dataset $\mathcal{S}_{\mathrm{test}}=\left\{\mathbf{z}^{(1)}, \ldots, \mathbf{z}^{(N_{\mathrm{test}})}\right\}$ is a list of individual data points $\mathbf{z}^{(n)}$, for $n=1, \ldots, N_{\mathrm{test}}$.

    $$
    \mathcal{S}_{\mathrm{test}} \overset{\mathrm{iid}}{\sim} \mathcal{D} = \left \{ \left(\mathrm{X}^{(n)}, Y^{(n)} \right) \right \}_{n=1}^{N_{\mathrm{test}}} = \left \{ \left(\mathrm{x}^{(n)}, y^{(n)} \right) \right \}_{n=1}^{N_{\mathrm{test}}} \in \left(\mathcal{X}, \mathcal{Y} \right)^{N_{\mathrm{test}}}
    $$

    This means that $\mathcal{S}_{\mathrm{test}}$ is a collection of samples drawn $\textbf{i.i.d.}$ from an unknown distribution $\mathcal{D}$ over the joint distribution of the input and the label space
    $\mathcal{X} \times \mathcal{Y}$.
  -
* - $\hat{\mathcal{D}}$
  - This denotes the empirical distribution defined by the sample dataset (train) $\mathcal{S}$. To be
  even more pedantic, $\hat{\mathcal{D}}^{(n)}$ is the empirical distribution defined by the $n$-th data point in the sample dataset $\mathcal{S}$
  (see [Foundations of Machine Learning](https://cs.nyu.edu/~mohri/mlbook/))
  -
* - $x_{d}^{(n)}$
  - The $d$-th feature of the $n$-th data point in the dataset $\mathcal{S}$.
  -
* - $x_{d}$
  - The $d$-th feature of a data point $\mathbf{z}$.
  -
* - $\mathbf{x}$
  - The feature vector of a data point $\mathbf{z}$. In vector form we write
  $\mathbf{x} = \begin{bmatrix} x_{1} & x_{2} & \cdots & x_{D} \end{bmatrix}^{\mathrm{T}}$
  where each $x_{d}$ is the $d$-th feature of $\mathbf{z}$.
  -
* - $\mathbf{x}^{(n)}$
  - The feature vector of the $n$-th data point in a dataset $\mathcal{S}$. This is
  $\mathbf{x}^{(n)}$ with superscript $^{(n)}$.
  -
* - $y$
  - The label of a data point $\mathbf{z}$.
  -
* - $y^{(n)}$
  - The label of the $n$-th data point in a dataset $\mathcal{S}$.
  -
* - $\left(\mathbf{x}^{(n)}, y^{(n)}\right)$
  - The $n$-th data point in a dataset $\mathcal{S}$.
  -
* - $\mathbf{X} \in \mathbb{R}^{N \times D}$
  - The design/feature matrix of a dataset $\mathcal{S}$ is a matrix $\mathbf{X}$ with $N$ rows and $D$ columns. Each row $\mathbf{x}^{(n)}$ of $\mathbf{X}$ is the feature vector of the $n$-th data point in $\mathcal{S}$.
  -
* - $h(\cdot)$
  - A hypothesis map that reads in features $\mathbf{x}$ of a data point and delivers a prediction $\hat{y}=h(\mathbf{x})$ for its label $y$.

    $$
    h: \mathcal{X} \to \mathcal{Y}
    $$
  -
* - $\mathcal{H}$
  - A hypothesis space or model used by a ML method. The hypothesis space consists of different hypothesis maps $h: \mathcal{X} \to \mathcal{Y}$ between which the ML method has to choose.

    For example, the hypothesis space of a linear regression method is

    $$
    \mathcal{H}=\{h: \mathbb{R}^n \to \mathbb{R} \text{ where } h(\mathbf{x})=\mathbf{w}^T\mathbf{x} \mid \mathbf{w} \in \mathbb{R}^n\}
    $$
  -
* - $c$
  - A concept is the true relationship between the features $\mathbf{x}$ and the labels $y$ of the true input-label space $\mathcal{X} \times \mathcal{Y}$.

    $$
    c: \mathcal{X} \to \mathcal{Y}
    $$
  -
* - $\mathcal{C}$
  - A concept space is the set of all possible concepts that can be learned by a ML method.

    $$
    \mathcal{C}=\{c: \mathcal{X} \to \mathcal{Y} \mid c \text{ is a concept}\}
    $$

    More concretely, the [difference](https://datascience.stackexchange.com/questions/23901/what-is-the-difference-between-concept-class-and-hypothesis)
    between a concept and a hypothesis is that a concept is the true relationship between the features and the labels of a dataset, while a hypothesis is the learned relationship between the features and the labels of a dataset.
  -
* - $R_k$
  - The decision region of a classifier $h(\cdot)$ is the set of all data points $\mathbf{x}$ for which $h(\mathbf{x})=k$.

    $$
    R_k := \left\{\mathbf{x} \in \mathbb{R}^n \mid h(\mathbf{x})=k\right\} \subseteq \mathcal{X}
    $$

    Each label $y \in \mathcal{Y}$ corresponds to a decision region $R_y$.

    More concretely, a hypothesis $h$ partitions the feature space into decision regions.
    A specific decision region $R_{\hat{y}}$ consists of all feature vectors
    $\mathbf{x}$ that are mapped to the same predicted label $h(\mathbf{x})=\hat{y}$ for all $\mathbf{x} \in R_{\hat{y}}$.
  -
* - $\mathcal{L}$ or $\mathcal{L}(\boldsymbol{\theta})$
  - Formally, a loss function is a map {cite}`jung2022machine`

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

    Usually, $\mathcal{L}$ is applied to one single training sample $\left(\mathbf{x}, y\right)$ (i.e. the
    loss of a single data point).

    Sometimes we subscript $\mathcal{L}$ with the training set $\mathcal{S}$ to indicate that the loss is applied to all training samples $\left(\mathbf{x}^{(n)}, y^{(n)}\right)$ in the training set $\mathcal{S}$.

    **Important is that the loss function in machine learning is a function of the parameters $\boldsymbol{\theta}$ of the hypothesis $h$ and not of the data points $\mathbf{x}$ and labels $y$!**
  -
* - $\widehat{\mathcal{L}}$ or $\widehat{\mathcal{L}}\left(h \mid \mathcal{S}\right)$
  - This is just the empirical loss, as a shorthand, we often abbreviate as $\widehat{\mathcal{L}}\left(h\right)$ when
      context of the training set $\mathcal{S}$ is clear.
  -
* - - $\mathcal{R}$
    - $\mathcal{R}_{\mathcal{D}}$
  - The true risk function $\mathcal{R}$ is the expected loss $\mathcal{L}$ over all (any) data points drawn $\textbf{i.i.d.}$ from the distribution $\mathcal{D}$.

    $$
    \begin{aligned}
    \mathcal{R}_{\mathcal{D}}\left\{\mathcal{L}((\mathbf{x}, y), h)\right\} &:= \mathbb{E}_{\left(\mathbf{x}, y\right) \sim \mathcal{D}}\left[\mathcal{L}((\mathbf{x}, y), h)\right] \\
    &:= \mathbb{E}_{\left(\mathbf{x}, y\right) \sim \mathbb{P}[\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}]} \left[\mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)\right]
    \end{aligned}
    $$
  -

* - - $\underset{{\left(\mathbf{x}, \mathbf{y}\right) \sim \mathcal{D}}}{\mathbb{E}} \left[\mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)\right]$
    - $\mathbb{E}_{\mathcal{D}}\left[\mathcal{L}((\mathbf{x}, y), h)\right]$
    - $\mathbb{E}_{\left(\mathbf{x}, y\right) \sim \mathbb{P}[\mathbf{X}, Y]} \left[\mathcal{L}\left(\left(\mathbf{x}, y\right), h\right)\right]$
  - Almost same definition as $\mathcal{R}_{\mathcal{D}}$ but with a different notation
    and may have different interpretation.

    Note a possible confusion can arise especially in the chapter of Bias-Variance Tradeoff,
    where we used $\mathbb{E}_{\mathcal{S}}$ to mean the expected loss over all possible training sets
    $\mathcal{S}$, but here we use $\mathbb{E}_{\left(\mathbf{x}, y\right) \sim \mathcal{D}}$ to mean the expected loss over all data points drawn $\textbf{i.i.d.}$ from the distribution $\mathcal{D}$.
  - The notation $\underset{{\left(\mathbf{x}, \mathbf{y}\right) \sim \mathcal{D}}}{\mathbb{E}}$ denotes that the expectation is taken over the joint distribution $\mathcal{D}$ of the input-output pairs $(\mathbf{x}, \mathbf{y})$. In other words, it represents the average value of a function over all possible input-output pairs drawn from the distribution $\mathcal{D}$.

    To understand it better, consider a (loss) function $\ell(\mathbf{x}, \mathbf{y})$. The expectation of this function with respect to the distribution $\mathcal{D}$ can be calculated as follows:

    $$
    \underset{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}}{\mathbb{E}}[\ell(\mathbf{x}, \mathbf{y})]=\int_{\mathcal{X}} \int_{\mathcal{Y}} \ell(\mathbf{x}, \mathbf{y}) \cdot \mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}) d \mathcal{Y} d \mathcal{X}
    $$

    In this equation, $\mathbb{P}(\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta})$ represents the probability density function of the joint distribution $\mathcal{D}$, and the integral is taken over the entire input space $\mathcal{X}$ and output space $\mathcal{Y}$. This operation computes the average value of the function $\ell(\mathbf{x}, \mathbf{y})$ over all possible input-output pairs, weighted by their likelihood under the distribution $\mathcal{D}$.

    In the context of learning curves and risk minimization, the expectation over the distribution $\mathcal{D}$ is used to compute the average performance of a hypothesis on new, unseen input-output pairs drawn from the same distribution as the training data. This helps in understanding the generalization ability of the hypothesis and in choosing the best hypothesis from the hypothesis set $\mathcal{H}$, which is expected to perform well on new data points.

* - $\underset{\mathcal{S} \sim \mathcal{D}^N}{\mathbb{E}}$
  - The notation $\underset{\mathcal{S} \sim \mathcal{D}^N}{\mathbb{E}}$ represents the expectation taken over multiple training sets $\mathcal{S}$ of size $n$ sampled from the underlying data-generating distribution $\mathcal{D}$. In other words, it is an average over various realizations of the training set $\mathcal{S}$.
  - In machine learning, we aim to learn a hypothesis $h_S: \mathcal{X} \to \mathcal{Y}$ from a training set $\mathcal{S}$ that generalizes well to unseen data points. However, we only have access to a single training set, which is a finite sample from the true data-generating distribution $\mathcal{D}$. Ideally, we want our learning algorithm to perform well on average over many different training sets of size $n$ sampled from $\mathcal{D}$, not just the one we happen to have.

    When we consider the expectation $\underset{\mathcal{S} \sim \mathcal{D}^N}{\mathbb{E}}$, we are accounting for this variability in the possible training sets that could have been observed. By taking the expectation over different realizations of $\mathcal{S}$, we can study the expected behavior of our learning algorithm and the hypotheses it produces.

    For example, we might be interested in the expected out-of-sample error (true risk) of a hypothesis learned from a random training set $\mathcal{S}$ of size $n$. In this case, we would calculate the expectation as:

    $$
    \underset{\mathcal{S} \sim \mathcal{D}^N}{\mathbb{E}}\left[\mathcal{R}_{\mathcal{D}}\left(h_S\right)\right] := \underset{\mathcal{S} \sim \mathcal{D}^N}{\mathbb{E}}\left[\underset{{\left(\mathbf{x}, \mathbf{y}\right) \sim \mathcal{D}}}{\mathbb{E}}\left[\ell\left(\mathbf{y}, h_S\left(\mathbf{x}\right)\right)\right]\right]
    $$

    This quantity represents the average out-of-sample error of the learned hypothesis $h_S$ across multiple possible training sets $\mathcal{S}$, each sampled from the true data-generating distribution $\mathcal{D}$. By evaluating this expectation, we can better understand the generalization properties of our learning algorithm.

* - $\hat{\mathcal{R}}$ or $\hat{\mathcal{R}}_{\mathcal{S}}$ or $\widehat{\mathcal{R}}\left(h \mid \mathcal{S}\right)$
  - The empirical risk function (in-sample error) $\hat{\mathcal{R}}_{\mathcal{S}}$ is the average loss $\mathcal{L}$ over all data points in the training set $\mathcal{S}$. This is used in practice because we do not know the true underlying
  joint distribution of the dataset $\mathcal{S}$ generated by $\mathcal{D}$ $\left(\mathbb{P}[\mathcal{X}, \mathcal{Y} ; \boldsymbol{\theta}]\right)$.

    $$
    \hat{\mathcal{R}}_{\mathcal{S}} := \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}\left(\left(\mathbf{x}^{(n)}, y^{(n)}\right), h\right)
    $$

    where $N$ is the number of training samples in the training set $\mathcal{S}$, i.e. $N=|\mathcal{S}|$.
  -

* - $\mathcal{J}$ or $\mathcal{J}_{\mathcal{D}}$
  - Note that this term is similar to the risk function $\mathcal{R}$, but sometimes the cost function can be more general in the sense
    it does not necessarily have to be the expected loss $\mathcal{L}$ over all data points drawn $\textbf{i.i.d.}$ from the distribution $\mathcal{D}$.
    See K-Means for example, the cost function is the sum of squared distances between the data points and their cluster centers and is not the expected loss.
  -
* - $\hat{\mathcal{J}}$ or $\hat{\mathcal{J}}_{\mathcal{S}}$ or $\widehat{\mathcal{J}}\left(h \mid \mathcal{S}\right)$
  - The (empirical/predicted) cost function $\hat{\mathcal{J}}$ is usually the loss function $\mathcal{L}$ averaged over all training samples $\left(\mathbf{x}^{(n)}, y^{(n)}\right)$
    in the training set $\mathcal{S}$.

    $$
    \hat{\mathcal{J}} := \frac{1}{N_{\mathrm{train}}} \sum_{n=1}^{N_{\mathrm{train}}} \mathcal{L}\left(\left(\mathbf{x}^{(n)}, y^{(n)}\right), h\right)
    $$
    To this end, it is quite similar to the empirical risk function $\hat{\mathcal{R}}_{\mathcal{S}}$ and is
    used quite commonly (i.e. Andrew Ng's [Machine Learning course](https://www.coursera.org/learn/machine-learning)).

    Of course, as mentioned, it can be also be used as the sum of squared distances between the data points and their cluster centers in the case of K-Means,
    and hence not restricted to the expected loss $\mathcal{L}$.
  -
* - $\mathcal{A}$
  - The learning algorithm $\mathcal{A}$ is the algorithm that is used to learn the parameters $\boldsymbol{\theta}$ of the hypothesis $h$.
  -
* - $VC$ (Vapnik-Chervonenkis) dimension
  - The VC dimension of a hypothesis space $\mathcal{H}$ is a measure of its capacity or complexity. It is the largest number of points that can be shattered (perfectly classified) by $\mathcal{H}$. Formally, the VC dimension is defined as:

    $$
    VC(\mathcal{H}) = \max\{ n \mid \exists \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n \in \mathcal{X} \text{ such that } \mathcal{H} \text{ shatters } \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}\}
    $$

    where "shattering" means that for every possible labeling of the points, there exists a hypothesis in $\mathcal{H}$ that classifies them correctly. The VC dimension is a fundamental concept in statistical learning theory and provides insights into the generalization ability of a learning algorithm.
  -
```
````

## Common Terminologies

```{list-table} Common Machine Learning Terminologies
:header-rows: 1
:name: common-ml-terminologies

* - Terminology
  - Definition
  - Comments
* - Intraclass Variance
  - The variance of the data points within a class $k \in \{1, \ldots, K\}$.
  - For example, embeddings learnt from softmax loss has higher intraclass variance than embeddings learnt from arcface loss.

    This is because softmax loss just need to push the correct class to the top of the softmax distribution,
    while arcface loss needs to push the correct class to the top of the softmax distribution and
    push all other classes away from the correct class. So the embeddings learnt from arcface loss
    is more "tightly clustered" together.
* - Interclass Variance
  - The variance across all $K$ classes. Essentially the variance of the entire dataset.
  - In Machine Learning, we want to maximize the interclass variance and minimize the intraclass variance.

    Why? We want those predicted embeddings in each class to be as similar as possible,
    and those in different groups must be as different as possible.
```

## Further Readings

- Jung, Alexander. Machine Learning: The Basics. Springer Nature
  Singapore, 2023.
- Mohri, Mehryar, Rostamizadeh Afshi and Talwalkar Ameet. Foundations of Machine
  Learning. The MIT Press, 2018.
