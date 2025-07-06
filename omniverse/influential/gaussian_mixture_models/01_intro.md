# Mixture Models

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
```

```{tableofcontents}

```

This section talks about mixture models.

## Introduction

One way to create more complex probability models is to take a convex combination of simple distributions. This is called a mixture model. This has the form

$$
\begin{alignat}{3}
\mathbb{P}(\boldsymbol{X} = \boldsymbol{x} ~;~ \boldsymbol{\theta}) &=\sum_{k=1}^K \pi_k p_k(\boldsymbol{x})  \\
\text{subject to} &\quad  0 \leq \pi_k \leq 1 \quad \forall k \\
&\quad \sum_{k=1}^K \pi_k=1
\end{alignat}
$$ (eq-mixture-model-1)

where

- $p_k$ is the $k$-th **mixture component**, each $p_k$ can be a member of a family of distributions, e.g. $\mathcal{N}(\boldsymbol{x} ~;~ \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$, $\operatorname{Bernoulli}(\boldsymbol{x} ~;~ \boldsymbol{\theta}_k)$, $\operatorname{Cat}(\boldsymbol{x} ~;~ \boldsymbol{\theta}_k)$, etc.
-  $\pi_k$ are the **mixture weights** which satisfy $0 \leq \pi_k \leq 1$ and $\sum_{k=1}^K \pi_k=1$

---

We can re-express this model as a hierarchical model, in which we introduce the discrete latent variable $z \in\{1, \ldots, K\}$, which specifies which distribution to use for generating the output $\boldsymbol{x}$. The prior on this latent variable is $p(z=k ~;~ \boldsymbol{\theta})=\pi_k$, and the conditional is $p(\boldsymbol{x} ~;~ z=k, \boldsymbol{\theta})=p_k(\boldsymbol{x})=p\left(\boldsymbol{x} ~;~ \boldsymbol{\theta}_k\right)$. That is, we define the following joint model:

$$
\begin{aligned}
p(z ~;~ \boldsymbol{\theta}) & =\operatorname{Cat}(z ~;~ \boldsymbol{\pi}) \\
p(\boldsymbol{x} \mid z=k, \boldsymbol{\theta}) & =p\left(\boldsymbol{x} ~;~ \boldsymbol{\theta}_k\right)
\end{aligned}
$$

where $\boldsymbol{\theta}=\left(\pi_1, \ldots, \pi_K, \boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_K\right)$ are all the model parameters. The "generative story" for the data is that we first sample a specific component $z$, and then we generate the observations $\boldsymbol{x}$ using the parameters chosen according to the value of $z$. By marginalizing out $z$, we recover equation {eq}`eq-mixture-model-1`:

$$
p(\boldsymbol{x} ~;~ \boldsymbol{\theta})=\sum_{k=1}^K p(z=k ~;~ \boldsymbol{\theta}) p(\boldsymbol{x} \mid z=k, \boldsymbol{\theta})=\sum_{k=1}^K \pi_k p\left(\boldsymbol{x} ~;~ \boldsymbol{\theta}_k\right)
$$

## References and Further Readings

- Bishop, Christopher M. "Chapter 2.3.9. Mixture of Gaussians." and "Chapter 9. Mixture Models and EM." In Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2016.
- Deisenroth, Marc Peter, Faisal, A. Aldo and Ong, Cheng Soon. "Chapter  11.1 Gaussian Mixture Models." In Mathematics for Machine Learning. : Cambridge University Press, 2020.
- Jung, Alexander. "Chapter 8.2. Soft Clustering with Gaussian Mixture Models." In Machine Learning: The Basics. Singapore: Springer Nature Singapore, 2023.
- Murphy, Kevin P. "Chapter 3.5 Mixture Models" and "Chapter 21.4 Clustering using mixture models." In Probabilistic Machine Learning: An Introduction. MIT Press, 2022.
- Vincent Tan, "Lecture 14-16." In Data Modelling and Computation (MA4270).
- [Nathaniel Dake: Gaussian Mixture Models](https://www.nathanieldake.com/Machine_Learning/04-Unsupervised_Learning_Cluster_Analysis-04-Cluster-Analysis-Gaussian-Mixture-Models.html)