# Conceptual Questions

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

## Monotonicity of $k$-means Updates

We state two criterions without proof. Proofs can be found
[here](https://www.gaohongnan.com/influential/kmeans_clustering/02_concept.html#the-necessary-conditions-to-minimize-the-objective-function).

### Criterion 1: The Optimal Assignment

Fix the cluster centers
$\boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_K$, we seek the
optimal assignment $\mathcal{A}^*(\cdot)$ that minimizes the cost function
$\widehat{\mathcal{J}}_{\mathcal{S}}(\cdot)$.

We claim that the optimal assignment $\mathcal{A}^*(\cdot)$ follows the _nearest
neighbor_ rule, which means that,

$$
\begin{aligned}
\mathcal{A}^*(n) = \underset{k \in \{1, 2, \ldots, K\}}{\operatorname{argmin}} \left\|\mathbf{x}^{(n)} - \boldsymbol{v}_k \right\|^2 .
\end{aligned}
$$

Then the assignment $\mathcal{A}^*$ is the optimal assignment that minimizes the
cost function $\widehat{\mathcal{J}}_{\mathcal{S}}$.

This is quite intuitive as we are merely assigning each data point
$\mathbf{x}^{(n)}$ to cluster $k$ whose center $\boldsymbol{v}_k$ is closest to
$\mathbf{x}^{(n)}$.

We rephrase the claim by saying that for any assignment $\mathcal{A}$, we have

$$
\begin{aligned}
\widehat{\mathcal{J}}_{\mathcal{S}}(\mathcal{A}, \boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_K) &\geq \widehat{\mathcal{J}}_{\mathcal{S}}(\mathcal{A}^*, \boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_K) \\
\end{aligned}
$$

### Criterion 2: The Optimal Cluster Centers

Fix the assignment $\mathcal{A}^*(\cdot)$, we seek the optimal cluster centers
$\boldsymbol{v}_1^*, \boldsymbol{v}_2^*, \ldots, \boldsymbol{v}_K^*$ that
minimize the cost function $\widehat{\mathcal{J}}_{\mathcal{S}}$.

We claim that the optimal cluster centers is the mean of the data points
assigned to each cluster.

$$
\begin{aligned}
\boldsymbol{v}_k^* = \frac{1}{\left|\hat{C}_k^*\right|} \sum_{\mathbf{x}^{(n)} \in \hat{C}_k^*} \mathbf{x}^{(n)}
\end{aligned}
$$

where $\left|\hat{C}_k^*\right|$ is the number of data points assigned to
cluster $k$. We can denote it as $N_k$ for convenience.

We can also rephrase this claim by saying that for any cluster centers
$\boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_K$, fixing the
assignment $\mathcal{A}^*$, we have

$$
\begin{aligned}
\widehat{\mathcal{J}}_{\mathcal{S}}(\mathcal{A}^*, \boldsymbol{v}_1, \boldsymbol{v}_2, \ldots, \boldsymbol{v}_K) &\geq \widehat{\mathcal{J}}_{\mathcal{S}}(\mathcal{A}^*, \boldsymbol{v}_1^*, \boldsymbol{v}_2^*, \ldots, \boldsymbol{v}_K^*) \\
\end{aligned}
$$

### Cost Function of K-Means Monotonically Decreases

The cost function $\widehat{\mathcal{J}}$ of K-Means monotonically decreases.
This means

$$
\begin{aligned}
\widehat{\mathcal{J}}_{\mathcal{S}}^{[t+1]}\left(\hat{C}_1^{[t+1]}, \ldots, \hat{C}_K^{[t+1]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]} \right) \leq \widehat{\mathcal{J}}_{\mathcal{S}}^{[t]}\left(\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t]}, \ldots, \boldsymbol{\mu}_K^{[t]} \right)
\end{aligned}
$$

for each iteration $t$.

**Proof:**

This is a consequence of {prf:ref}`criterion:kmeans-optimal-assignment` and
{prf:ref}`criterion:kmeans-optimal-cluster-centers`.

In particular, the objective function $\widehat{\mathcal{J}}$ is made up of two
steps, the assignment step and the update step. We minimize the assignment step
by finding the optimal assignment $\mathcal{A}^{*}(\cdot)$, and we minimize the
update step by finding the optimal cluster centers $\boldsymbol{\mu}_k^{*}$
based on the optimal assignment $\mathcal{A}^{*}(\cdot)$ at each iteration.

Consequently, if we can show that the following:

$$
\begin{aligned}
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]} \right) \leq \widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t]}, \ldots, \boldsymbol{\mu}_K^{[t]} \right)
\end{aligned}
$$

and then,

$$
\begin{aligned}
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t+1]}, \ldots, \hat{C}_K^{[t+1]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]} \right) \leq \widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]} \right)
\end{aligned}
$$

then we can easily show that the cost function $\widehat{\mathcal{J}}$
monotonically decreases.

**First**, once all the samples $\left\{\mathbf{x}^{(n)}\right\}_{n=1}^N$ are
assigned to the clusters as per the **assignment step** in
{eq}`eq:kmeans-classify`, we will recover the cost at the $t$-th iteration,
defined as:

```{math}
:label: eq:kmeans-convergence-repeat-1

\widehat{\mathcal{J}}_{\mathcal{S}}^{[t]}\left(\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t]}, \ldots, \boldsymbol{\mu}_K^{[t]} \right) = \sum_{k=1}^K \sum_{n \in \hat{C}_k^{[t]}} \left\|\mathbf{x}^{(n)} - \boldsymbol{\mu}_k^{[t]}\right\|^2
```

Note in particular that the base case is we initialized the cluster centers
$\boldsymbol{\mu}_k^{[0]}$ for the first iteration $t=0$ and this induces the
clusters $\hat{C}_1^{[0]}, \ldots, \hat{C}_K^{[0]}$ for which we assigned each
data point $\mathbf{x}^{(n)}$ to the closest cluster center
$\boldsymbol{\mu}_k^{[0]}$. If we just look at the base case, the mean is
randomly initialized, and so there may be room of improvement, which is why we
need the **update step**.

**Next**, we recalculate the cluster centers $\boldsymbol{\mu}_k^{[t+1]}$ based
on the clusters for the $t$-th iteration. In other words, we find the cluster
centers for the $t+1$-th iteration based on the cluster assignments for the
$t$-th iteration. We claim that this new cluster centers
$\boldsymbol{\mu}_k^{[t+1]}$ will minimize the cost function
$\widehat{\mathcal{J}}$.

We **fix** the assignment $\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}$, and then
show that the cluster centers
$\boldsymbol{\mu}_k^{[t+1]} = \dfrac{1}{\left|\hat{C}_k^{[t]}\right|} \sum_{n \in \hat{C}_k^{[t]}} \mathbf{x}^{(n)}$
minimizes the cost function $\widehat{\mathcal{J}}$, which means:

$$
\begin{aligned} \boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]}
= \underset{\boldsymbol{\mu}_1, \ldots,
\boldsymbol{\mu}_K}{\operatorname{argmin}}
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots,
\hat{C}_K^{[t]}, \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K \right)
\end{aligned}
$$

and subsequently,

```{math}
:label: eq:kmeans-convergence-repeat-2

\begin{aligned} \widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]},
\ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots,
\boldsymbol{\mu}_K^{[t+1]} \right) \leq
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots,
\hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t]}, \ldots, \boldsymbol{\mu}_K^{[t]}
\right) \end{aligned}
```

because this step cannot increase the cost
$\widehat{\mathcal{J}}\left(\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t]}, \ldots, \boldsymbol{\mu}_K^{[t]} \right)$
as the new cluster centers minimizes the cost function $\widehat{\mathcal{J}}$
when we replace the cluster centers
$\boldsymbol{\mu}_1^{[t]}, \ldots, \boldsymbol{\mu}_K^{[t]}$ by the new cluster
centers $\boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]}$.

**Next**, as we have finished one cycle in the $t$-th iteration, now we turn our
attention to the $t+1$-th iteration. As usual, we look at the first step, which
is the **assignment step**. We **fix** the cluster centers
$\boldsymbol{\mu}_k^{[t+1]}$ found in the previous step, and then show that the
assignment $\hat{C}_1^{[t+1]}, \ldots, \hat{C}_K^{[t+1]}$ will minimize the cost
function $\widehat{\mathcal{J}}$, which means:

$$
\begin{aligned} \hat{C}_1^{[t+1]}, \ldots, \hat{C}_K^{[t+1]} =
\underset{\hat{C}_1, \ldots, \hat{C}_K}{\operatorname{argmin}}
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1, \ldots, \hat{C}_K,
\boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]} \right)
\end{aligned}
$$

and subsequently,

```{math}
:label: eq:kmeans-convergence-repeat-3

\begin{aligned} \widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t+1]},
\ldots, \hat{C}_K^{[t+1]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots,
\boldsymbol{\mu}_K^{[t+1]} \right) \leq
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots,
\hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots,
\boldsymbol{\mu}_K^{[t+1]} \right) \end{aligned}
```

because this step cannot increase the cost
$\widehat{\mathcal{J}}\left(\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots, \boldsymbol{\mu}_K^{[t+1]} \right)$
as the new assignments minimizes the cost function $\widehat{\mathcal{J}}$ when
we replace the cluster assignments $\hat{C}_1^{[t]}, \ldots, \hat{C}_K^{[t]}$ by
the new assignments $\hat{C}_1^{[t+1]}, \ldots, \hat{C}_K^{[t+1]}$.

**Finally**, we can show that the cost function $\widehat{\mathcal{J}}$ is
**decreasing** in each iteration.

Combining {eq}`eq:kmeans-convergence-repeat-2` and
{eq}`eq:kmeans-convergence-repeat-3`, we have the following inequality:

```{math}
:label: eq:kmeans-convergence-repeat-4

\begin{aligned} \widehat{\mathcal{J}}_{\mathcal{S}}^{[t+1]} &=
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t+1]}, \ldots,
\hat{C}_K^{[t+1]}, \boldsymbol{\mu}_1^{[t+1]}, \ldots,
\boldsymbol{\mu}_K^{[t+1]} \right) \\ &\leq
\widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots,
\hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t + 1]}, \ldots, \boldsymbol{\mu}_K^{[t
-   1]} \right) \\ &\leq
    \widehat{\mathcal{J}}_{\mathcal{S}}\left(\hat{C}_1^{[t]}, \ldots,
    \hat{C}_K^{[t]}, \boldsymbol{\mu}_1^{[t]}, \ldots,
    \boldsymbol{\mu}_K^{[t]} \right) =
    \widehat{\mathcal{J}}_{\mathcal{S}}^{[t]} \end{aligned}
```

Refer to Chip's ML Interview book and references like
[ML Stack Cafe](https://www.mlstack.cafe/blog/k-means-clustering-interview-questions)
for more questions.
