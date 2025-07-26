# Probability Axioms

```{contents}
```

## The Three Axioms of Probability (Kolmogorov Axioms)

A probability function $\P$ defined over the probability space $\pspace$ must
satisfy the three axioms below. Recall that the **probability function** in a
well defined **experiment** is a function $\P: \E \to [0, 1]$. Informally, for
any event $A$, $\P(A)$ is defined as the probability of event $A$ happening.

This probability function/law $\P(A)$ must satisfy the following three axioms:

```{prf:axiom} Non-Negativity
:label: axiom:non-negativity

$\P(A) \geq 0$ for any event $A \subseteq \S$.
```

```{prf:axiom} Normalization
:label: axiom:normalization

$\sum_{i=1}^{n}\P(A_i) = 1$
    where $A_i$ are all possible outcomes for $i = 1, 2,..., n$.
```

```{prf:axiom} Additivity
:label: axiom:additivity

Given a countable sequence of
    **disjoint events** $A_1, A_2, ..., A_n,... \subset \S$, we have

$$
\P\left(\bigsqcup_{i=1}^{\infty} A_i \right) = \sum_{i=1}^{\infty}\P[A_i]
$$
```

## Corollaries of the Axioms

### Probability of the Empty Set

```{prf:corollary} Probability of Empty Set
:label: corollary:probability-of-empty-set

$\P(\emptyset) = 0$
```

### Probability of Complements

```{prf:corollary} Probability of Complements
:label: corollary:probability-of-complements

$\P(A) = 1 - \P(A^{c})$
```

### Inclusion-Exclusion - Unions of Two Non-Disjoint Sets

```{prf:corollary} Inclusion-Exclusion
:label: corollary:inclusion-exclusion

$$
\P(A \cup B) = \P(A) + \P(B) - \P(A \cap B)
$$
```

### Inequality Bounds

```{prf:corollary} Inequality Bounds
:label: corollary:inequality-bounds

a)  Union Bound: $\P(A \cup B) \leq \P(A) + P(B)$

b)  Monotonicity: If $A \subset B$, then $\P(A) \leq \P(B)$

c)  Numeric Bound: It immediately follows that for any event $A$,
    $\P(A) \leq 1$ by Monotonicity and Axiom 2. Because an event $A$ can
    at most be summed to 1, so any events must be a subset of its
    summation, so the probability of the summation is 1, hence
    $\P(A) \leq 1$ by Monotonicity.
```

## Mutually Exclusive Events

In logic and probability theory, two events (or propositions) are mutually
exclusive or disjoint if they cannot both occur at the same time.

A clear example is the set of outcomes of a single coin toss, which can result
in either heads or tails, but not both. To be concise, let the event of coin
landing on head be $A$, the event of coin landing on tails be $B$, event $A$ and
$B$ can never occur at the same time, it is the case that one cannot be true if
the other one is true, or at least one of them cannot be true.

One can draw a venn diagram with event $A$ and $B$ being disjoint to illustrate
the idea.

Also, by **Unions of Two Non-Disjoint Sets**, it is easy to see that
$\P(A \cup B) = \P(A) + \P(B)$ since $\P(A \cap B) = 0$. For the coin toss
example, since $A \cup B$ spans the entire sample space, it is intuitive that
$P(A \cup B)$ (the probability of throwing a head or a tail) is 1.
