# The Loss Landscape

```{contents}
:local:
```

## Convergence and Generalization

I think it really depends on our loss function $\mathcal{L}$ or equivalently the
cost function $\mathcal{J}$.

-   **Case 1**: $\mathcal{L}(\boldsymbol{\theta})$ is convex over $\Theta$ (note
    the emphasis that the loss is a function of the parameters, not the data),
    where $\boldsymbol{\theta} \in \Theta \subseteq \mathbb{R}^D$.

    -   $\mathcal{L}$ has a unique global minimum $\boldsymbol{\theta}^*$ in
        $\Theta$:

        $$
        \exists \boldsymbol{\theta}^*\in \Theta, \forall \boldsymbol{\theta} \in \Theta, \mathcal{L}(\boldsymbol{\theta}^*) \leq \mathcal{L}(\boldsymbol{\theta})
        $$

        where $D$ is the dimension of the parameter space (we are being slighly
        less pedantic here as we are not specifying the topology of the
        parameter space, but let's assume this parameter is a flattened vector
        of all the parameters of the model).

    -   Optimization algorithms such as gradient descent can be employed to find
        the global minimum $\boldsymbol{\theta}^*$ in $\Theta$ that minimizes
        $\mathcal{L}$.

    -   Any local minimum $\boldsymbol{\theta}^*$ in $\Theta$ is also the global
        minimum $\boldsymbol{\theta}^*$ in $\Theta$.

    -   Given an appropriate learning rate $\eta$, the negative gradient of
        $\mathcal{L}$ always points in the direction of the steepest descent in
        $\Theta$. Hence, gradient-based algorithms are guaranteed to converge to
        the global minimum $\boldsymbol{\theta}^*$ when $\mathcal{L}$ is convex
        over $\Theta$.

-   **Case 2**: $\mathcal{L}(\boldsymbol{\theta})$ for deep neural networks over
    $\Theta$, where $\boldsymbol{\theta} \in \Theta \subseteq \mathbb{R}^D$.

    -   **Non-convexity**: Unlike simple models where the loss might be convex,
        the loss landscape of deep neural networks is typically non-convex. This
        non-convexity can lead to multiple minima (all eigenvalues of the loss
        function’s Hessian at zero gradient > 0) and saddle points (where some
        eigenvalues of the Hessian are positive and some are negative).

    -   **Local Minima and Saddle Points**: While there may be many local
        minima, recent research suggests that in high-dimensional spaces (like
        those of deep nets), saddle points are more prevalent. At a saddle
        point, the gradient is zero, but it's neither a minimum nor a maximum.

    -   **Optimization Algorithms**: Gradient-based methods, like gradient
        descent and its variants (e.g., SGD, Adam), are commonly used. While
        these methods are not guaranteed to find the global minimum due to the
        non-convex nature of the loss, they are often effective at finding "good
        enough" local minima.

So I think that is why researchers often empirically observe that deep neural
networks "converge" when the loss curves start to flatten out. But one thing to
distinguish is that convergence is not the same as generalization. The former
refers to the process by which a model's loss decreases to a stable value,
indicating that the model has effectively learned from the training data.
However, generalization refers to the model's ability to apply what it has
learned to new, unseen data. This distinction is crucial because a model can
converge and still perform poorly on new data if it has overfit to the training
data.

## Theoretical Bounds on Loss

One critical consideration is the determination of theoretical bounds for the
loss function $\mathcal{L}$. Understanding these bounds enables researchers and
practitioners to gauge the lowest possible loss value that is unattainable in
practice, thereby setting benchmarks for model performance. However, this topic
is too theoretical and often not useful in practice. Nevertheless, I penned down
some thoughts on this matter.

### Theoretical Bounds Based on Loss Function

The lower bound seems to be tied to the loss function being used. For example,
both the mean squared error (MSE) and the cross-entropy loss can range from 0 to
$\infty$. Does that mean the lower bound is 0? Is our problem answered?

No. The loss function is a function of the model's parameters with data as
input. So the lower bound of a particular combination of model parameters and
data is much more complex than just the lower bound of the loss function.

### Convergence of Generative Pre-trained Transformer

It can be shown that the given the Markov assumption and a token context window
size of $\tau$, the loss function $\mathcal{L}$ is a
[consistent estimator](https://en.wikipedia.org/wiki/Consistent_estimator) of
the true distribution $\mathcal{D}$, and the the objective
$\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$
converges to the true conditional probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \boldsymbol{\Theta})$ over $\mathcal{D}$ as the
size of the corpus $\mathcal{S}$ goes to infinity, if the model has sufficient
capacity and the optimization algorithm is appropriate {cite}`math11112451`.

Furthermore, the proposition that the conditional entropy
$H\left(X_t \mid X_{<t}\right)$ of the true data-generating process is upper
bounded by the by the logarithm of the size of the vocabulary $\mathcal{V}$,
i.e., $H\left(X_t \mid X_{<t}\right) \leq \log |\mathcal{V}|$
{cite}`math11112451`.

The proposition that the conditional entropy has an upper limit, carries
significant implications for optimizing autoregressive self-supervised learning
models. Specifically, because the conditional entropy cannot exceed the
logarithm of the vocabulary size $\mathcal{V}$, we infer a similar upper limit
on perplexity. This cap on perplexity offers a valuable benchmark for evaluating
and comparing different models, establishing a theoretical maximum for model
performance based on the size of the vocabulary {cite}`math11112451`.

You can find more details
[here](https://www.gaohongnan.com/transformer/decoder/concept.html#convergence).

### Theoretical Bounds Based on Data and Model Capacity

In an idealized setting, if your data were noise-free and the neural network had
the capacity to represent the underlying function perfectly, then the training
loss could, in theory, be zero. However, in real-world scenarios with noisy data
or inherent ambiguities, the lower bound on the loss might be greater than zero.
This is especially true for regression tasks where the noise in the data sets a
floor on how low the loss can go.

### Empirical Bounds

In practice, the best way to determine a realistic lower bound is empirically,
by training various models on your data and observing the lowest loss achieved.
Over time, as you experiment with different architectures, regularization
methods, and training strategies, you can get a sense of what a good lower bound
for your specific problem and dataset might be.

### Generalization Gap

It's worth noting that even if the training loss is very low, the validation or
test loss might be higher due to overfitting. The difference between training
and validation loss is referred to as the "generalization gap." A model that has
a very low training loss but a significantly higher validation loss may not be
as useful as one with a slightly higher training loss but a smaller
generalization gap.

### Loss Landscape

Deep neural networks have a highly non-convex loss landscape. While there might
be many local minima, recent research suggests that many of these minima are
surrounded by flat regions (often referred to as "plateaus") and that these
different minima might have very similar loss values. This makes determining a
strict lower bound challenging.

In summary, while there isn't a universal lower bound for the loss value in deep
neural networks that applies across all scenarios, understanding the specifics
of a given problem, dataset, and model can provide insights into what a
reasonable lower bound might be.

## Some Causes on Poor Convergence

-   **Learning Rate**: The choice of learning rate is crucial. If it's too
    large, gradient descent can oscillate around the minimum or even diverge. If
    it's too small, convergence can be very slow. For convex problems, there are
    theoretical bounds on the learning rate to ensure convergence.
-   **Convergence to Global Minimum**: While gradient descent is guaranteed to
    converge to the global minimum for convex functions, the convergence might
    be slow, especially if the function is poorly conditioned or if the learning
    rate is not well-tuned.
-   **Noise and Stochasticity**: In the context of machine learning, we often
    use Stochastic Gradient Descent (SGD) or its variants, which estimate the
    gradient using a subset of the data (batch). This introduces noise into the
    gradient updates, which can cause oscillations. However, on average, the
    method still moves towards the global minimum for convex functions.

## References and Further Readings

-   [On Loss Functions for Deep Neural Networks in Classification](https://arxiv.org/abs/1702.05659)
-   [Cross-Entropy Loss Functions: Theoretical Analysis and Applications](https://arxiv.org/abs/2304.07288)
-   [Generalization Bound of Gradient Descent for Non-Convex Metric Learning](https://proceedings.neurips.cc/paper/2020/hash/6f5e4e86a87220e5d361ad82f1ebc335-Abstract.html)
-   [Convergence Analysis of Deep Residual Networks](https://arxiv.org/abs/2205.06571)
-   [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
