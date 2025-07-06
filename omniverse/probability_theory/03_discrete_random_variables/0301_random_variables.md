# Random Variables

```{contents}
```

## Definition

```{prf:definition} Random Variables
:label: random_variables

A **random variable** $X$ is a function defined by the mapping

$$
\begin{align}
X: \S &\to \R \\
\xi &\mapsto X(\xi)
\end{align}
$$

which maps an *outcome* $\xi \in \S$ to a real number $X(\xi) \in \R$.

We denote the range of $X$ to be $x$ and shorthand the notation of $X(\xi) = x$ to be $X = x$.
```

```{prf:definition} Pre-image of a Random Variable
:label: pre_image

Given a random variable $X: \S \to \R$, define a singleton set $\{x\} \subset \R$, then by
the pre-image definition of a function, we have

$$
X^{-1}(\{x\}) = \lset \xi \in \S \st X(\xi) \in \{x\} \rset
$$

which is equivalent to

$$
X^{-1}(x) \defeq \lset \xi \in \S \st X(\xi) = x \rset
$$
```

## Examples

```{prf:example} Coin Toss
:label: example_random_variable_coin_toss

Consider a fair coin and define an **experiment** of throwing the coin twice.

Define the **random variable** $X$ to be the total number of heads in an experiment.
(i.e if you throw 1 head and 1 tail the total number of heads in this experiment is 1).

What is the probability of getting 1 head in an experiment, i.e. $\P \lsq X = 1 \rsq$?

**Solution**

We define the **sample space** $\S$ of this experiment to be $\{(HH), (HT), (TH), (TT)\}$.

We enumerate each **outcome** $\xi_i$ in the **sample space** as

- $\xi_1 = HH$
- $\xi_2 = HT$
- $\xi_3 = TH$
- $\xi_4 = TT$

$\color{red} {\textbf{First}}$, recall that $X$ is a ***function*** that map an outcome $\xi$ from the **sample space** $\S$
to a number $x$ in the real space $\R$. In this context it means that $X$ maps one of the four outcomes
$\xi_i$ to the total number of heads in the experiment
(i.e $X(\cdot) = \textbf{number of heads}$).

It is important to note that the codomain of $X$ is not any arbitrary number. We can only map our 4
outcomes $\xi_i$ in the domain to 3 distinct numbers $0$, $1$ or $2$, which we will see by manually
enumerating each case below.

$$
X(\xi_1) = 2, \quad X(\xi_2) = 1, \quad X(\xi_3) = 1, \quad X(\xi_4) = 0
$$ (eq:outcome_to_number_of_heads)


With that, this random variable $X$ is **completely determined**.

$\color{red} {\textbf{Secondly}}$, we need to examine carefully what is meant by $\P[X(\xi) = 1]$
since this will answer the question on what is the probability of getting 1 head.

However, $X(\xi) = 1$ is an expression and not an event that the probability measure $\P$ expects.
Here we should recall that the probability law $\P(\cdot)$ is always applied to an **event**
$E \in \E$ where $E$ is a set.

So we need to map this expression to an event $E \in \E$. So you can ask yourself how to establish
this "mapping" of $X(\xi) = 1$ to an event in our event space $\E$. This seems pretty easy since
we already know that $X(\xi) = 1$ has two cases matched in {eq}`eq:outcome_to_number_of_heads`, namely
$X(\xi_2) = 1$ and $X(\xi_3) = 1$. So we can simply define the event $E$ to be
$\lset \xi_2, \xi_3 \rset = \{(HT), (TH)\}$.

We verify that $E = \lset \xi_2, \xi_3 \rset$ is indeed an event in $\E$:

$$
\E = \{\emptyset, \{\xi_1\}, \{\xi_2\}, \{\xi_3\}, \{\xi_4\}, \{\xi_1\, \xi_2\}, \{\xi_1\, \xi_3\}, \{\xi_1\, \xi_4\}, \{\xi_2\, \xi_3\}, \{\xi_2\, \xi_4\}, \{\xi_3\, \xi_4\}, \S \}
$$

More concretely, given an expression $X(\xi) = x$, we construct the event set $E$ by enumerating all the outcomes $\xi_i$ in the sample space $\S$ that satisfy $X(\xi) = x$.

$$
E = \lset \xi \in \S \st X(\xi) = x \rset
$$

and this coincides with the pre-image of $x$ in the random variable $X$ as defined in {prf:ref}`pre_image`.

$\color{red} {\textbf{Consequently}}$, we have

$$
\begin{align}
    \P[X(\xi) = 1] &= \P[\{(HT), (TH)\}] \\
                   &= \dfrac{2}{4} \\
                   &= 0.5
\end{align}
$$
```

```{prf:example} Dice Roll
:label: example_random_variable_dice_roll

Consider a fair dice and define an **experiment** of rolling the dice once.

The sample space is then

$$
\S = \lset 1, 2, 3, 4, 5, 6 \rset
$$

If we roll two fair dice, then the sample space is

$$
\S = \lset (1, 1), (1, 2), \ldots, (6, 6) \rset
$$

So the probability of rolling say $(1, 2)$ is,

$$
\P[\{(1, 2)\}] = \dfrac{1}{36}
$$
```

## Probability Measure $\P$

In the chapter on [](../02_probability/0202_probability_space.md), we have
defined a probability measure $\P$ on a sample space $\S$ as

$$
\begin{align}
    \P: \mathcal{F} &\to [0, 1] \\
    E &\mapsto \P(E)
\end{align}
$$

as per Probability Law ({prf:ref}`probability_law`).

The question initially is that $\P \lsq X = x \rsq$ does not seem to take in an
event $E$ in $\E$ but an expression $X(\xi) = x$ instead. This needs to be
emphasized that they are the same, as the expression $X = x$ is just an event
$E$ in $\E$ (i.e. $X = x$ evaluates to $\lset \xi \in \S \st X(\xi) = x \rset$)
where $\lset \xi \in \S \st X(\xi) = x \rset \in \E$.

## Variable vs Random Variable

```{prf:example} Variable vs Random Variable
:label: example_variable_vs_random_variable

Professor Stanley Chan gave a good example of the difference between a variable and a random variable.

The main difference is that a variable is **deterministic** while a random variable is **non-deterministic**.

Consider solving the following equation:

$$
2X = x
$$

Then, if $x$ is a fixed constant, then $X = \dfrac{x}{2}$ is a variable.

However, if $x$ is not fixed, meaning that it can have multiple states, then $X$ is a random variable
since it is not deterministic.

Tie back to the example in {prf:ref}`example_random_variable_coin_toss`, we note that $X$ is a random variable since the
total number of heads $x$ in an experiment is not fixed. It can be 0, 1 or 2 depending on your toss.
```

## Summary

1. A random variable $X$ is a function that has the sample space $\S$ as its
   domain and the real space $\R$ as its codomain.

2. $X(\S)$ is the set of all possible values that $X$ can take and the mapping
   is not necessarily a bijective function since $X(\xi)$ can take on the same
   value for different outcomes $\xi$.

3. The elements in $X(\S)$ are denoted as $x$ (i.e. $x \in X(\S)$). They are
   often called the **states** of $X$.

4. It is important to not confused $X$ and $x$. $X$ is a function while $x$ are
   the states of $X$.

5. When we write $\P\lsq X = x \rsq$, we are describing the probability of the
   random variable $X$ taking on a **_particular_** state $x$. This is
   equivalent to $\P \lsq \lset \xi \in \S \st X(\xi) = x \rset \rsq$.

6. Random variables need not be **bijective**, see
   [here](https://math.stackexchange.com/questions/202540/is-a-random-variable-bijective).
