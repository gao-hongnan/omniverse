# Probability Mass Function

```{contents}
```

## Definition

```{prf:definition} State Space
:label: def_state_space

The set of all possible states of $X$ is called the **state space** of $X$ and is denoted as $X(\S)$.

In particular, the state space of a discrete random variable $X$ is a countable set
as per {prf:ref}`def_discrete_random_variables`.
```

```{prf:example} State Space of Coin Toss
:label: example_state_space_coin_toss

Let us revisit the example in {prf:ref}`example_random_variable_coin_toss` and examine the state space of $X$.

The state space of $X$ is the set of all possible values that $X$ can take. As enumerated in the example,
we see that the state space of $X$ is $\{0, 1, 2\}$ (i.e. $X$ takes 3 states 0, 1 and 2).
```

```{prf:example} State Space of Dice Roll
:label: example_state_space_dice_roll

Let us revisit the example in {prf:ref}`example_random_variable_dice_roll` and examine the state space of $X$.

The state space of $X$ is the set of all possible values that $X$ can take. As enumerated in the example,
we see that the state space of $X$ is $\{1, 2, 3, 4, 5, 6\}$ (i.e. $X$ takes 6 states 1, 2, 3, 4, 5 and 6).

For two dice rolls, the state space is $\{(1, 1), (1, 2), \ldots, (6, 6)\}$ where each state is a tuple of two dice rolls.
```

```{prf:definition} Probability Mass Function
:label: def_pmf

The **probability mass function** (PMF) of a random variable $X$ is a function that maps each state
$x$ in the state space $X(\S)$ to its probability $\pmf(x) = \P\lsq X = x \rsq$.

We denoted the PMF as

$$
\begin{align}
    \pmf: X(\S) &\to [0, 1] \\
    X(\S) \ni x &\mapsto \pmf(x)
\end{align}
$$
```

```{prf:example} PMF of Coin Toss
:label: example_pmf_coin_toss

Let us revisit {prf:ref}`example_random_variable_coin_toss` on coin toss and compute the PMF of $X$.

Recall the sample space is given by $\S = \{(HH), (HT), (TH), (TT)\}$ and the state space is given by
$X(\S) = \{0, 1, 2\}$ as enumerated in {prf:ref}`example_state_space_coin_toss`.

Thus, our domain of $\pmf$ is $X(\S) = \{0, 1, 2\}$ we have 3 mappings to compute:

$$
\begin{align}
    \pmf(0) &= \P\lsq X = 0 \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = 0 \rset \rsq = \P\lsq \{(TT)\} \rsq = \dfrac{1}{4} \\
    \pmf(1) &= \P\lsq X = 1 \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = 1 \rset \rsq = \P\lsq \{(HT), (TH)\} \rsq = \dfrac{2}{4} \\
    \pmf(2) &= \P\lsq X = 2 \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = 2 \rset \rsq = \P\lsq \{(HH)\} \rsq = \dfrac{1}{4}
\end{align}
$$

Here we have enumerated all the possible states of $X$ and computed the probability of each state.
Thus, the PMF of $X$ is completely determined by the 3 mappings above.
```

```{prf:example} PMF of Two Dice Rolls
:label: example_pmf_two_dice_rolls

Let us revisit {prf:ref}`example_random_variable_dice_roll` on two dice rolls and compute the PMF of $X$.

Recall the sample space is given by $\S = \lset (1, 1), (1, 2), \ldots, (6, 6) \rset$ and the state space is given by
$X(\S) = \lset (1, 1), (1, 2), \ldots, (6, 6) \rset$ as enumerated in {prf:ref}`example_state_space_dice_roll`.

Thus, our domain of $\pmf$ is $X(\S) = \lset (1, 1), (1, 2), \ldots, (6, 6) \rset$ we have 36 states to compute:

$$
\begin{align}
    \pmf((1, 1)) &= \P\lsq X = (1, 1) \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = (1, 1) \rset \rsq = \P\lsq \{(1, 1)\} \rsq = \dfrac{1}{36} \\
    \pmf((1, 2)) &= \P\lsq X = (1, 2) \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = (1, 2) \rset \rsq = \P\lsq \{(1, 2)\} \rsq = \dfrac{1}{36} \\
    \vdots \\
    \pmf((6, 6)) &= \P\lsq X = (6, 6) \rsq = \P\lsq \lset \xi \in \S \st X(\xi) = (6, 6) \rset \rsq = \P\lsq \{(6, 6)\} \rsq = \dfrac{1}{36}
\end{align}
$$
```

## Normalization

```{prf:theorem} Normalization Property of PMF
:label: thm_pmf_normalization

A PMF should satisfy the following normalization property:

$$
\sum_{x \in X(\S)} \pmf(x) = 1
$$ (eq:pmf_normalization)
```

```{prf:proof}
**TODO**
```

## Sturges' Rule and Cross Validation

See
[Introduction to Probability for Data Science](https://probability4datascience.com/index.html)
section 3.2.5.
