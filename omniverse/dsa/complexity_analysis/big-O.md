# Big-O

## Important First Read

- Examples: https://en.wikipedia.org/wiki/Big_O_notation#Example
- Properties: https://en.wikipedia.org/wiki/Big_O_notation#Properties
  - Notably there should be no confusion on adding or multiplying two big O.
- https://cs.stackexchange.com/questions/366/what-goes-wrong-with-sums-of-landau-terms/389#389
- https://utkarsh1504.github.io/DSA-Java/intro-complexity

## Prove $f \cdot \mathcal{O}(g) = \mathcal{O}(f \cdot g)$

To prove $f \cdot \mathcal{O}(g) = \mathcal{O}(f \cdot g)$ rigorously, we need
to show that if a function $h(n)$ belongs to $f \cdot \mathcal{O}(g)$, then
$h(n)$ also belongs to $\mathcal{O}(f \cdot g)$, and vice versa.

### Proof:

1. **Part 1: $f \cdot \mathcal{O}(g) \subseteq \mathcal{O}(f \cdot g)$**

   Suppose $h(n)$ is a function such that $h(n) \in f \cdot \mathcal{O}(g)$. By
   definition of Big O, there exist constants $C > 0$ and $n_0 \in \mathbb{N}$
   such that for all $n \geq n_0$, the inequality
   $h(n) \leq C \cdot f(n) \cdot g(n)$ holds.

   This directly implies that $h(n) \in \mathcal{O}(f \cdot g)$, because $h(n)$
   is bounded above by a constant multiple of $f(n) \cdot g(n)$ for all
   $n \geq n_0$.

2. **Part 2: $\mathcal{O}(f \cdot g) \subseteq f \cdot \mathcal{O}(g)$**

   Conversely, let $h(n) \in \mathcal{O}(f \cdot g)$. This means there exist
   constants $C' > 0$ and $n_0' \in \mathbb{N}$ such that for all $n \geq n_0'$,
   we have $h(n) \leq C' \cdot f(n) \cdot g(n)$.

   We can express $h(n)$ as $h(n) = f(n) \cdot k(n)$, where
   $k(n) = C' \cdot g(n)$ for all $n \geq n_0'$. Clearly,
   $k(n) \in \mathcal{O}(g)$, because it is bounded above by a constant multiple
   of $g(n)$ for all $n \geq n_0'$.

   Therefore, $h(n) = f(n) \cdot k(n)$ belongs to $f \cdot \mathcal{O}(g)$,
   since $k(n) \in \mathcal{O}(g)$ and $h(n)$ is simply $f(n)$ multiplied by a
   function from $\mathcal{O}(g)$.

### Conclusion:

Since both $f \cdot \mathcal{O}(g) \subseteq \mathcal{O}(f \cdot g)$ and
$\mathcal{O}(f \cdot g) \subseteq f \cdot \mathcal{O}(g)$ have been shown, we
can conclude that $f \cdot \mathcal{O}(g) = \mathcal{O}(f \cdot g)$, completing
the proof.

## f and g

In Big O notation, the functions $f(n)$ and $g(n)$ represent the following:

- $f(n)$ typically represents the actual complexity function of an algorithm,
  that is, how the running time or space requirement of an algorithm grows with
  the size of the input $n$.
- $g(n)$ represents a function that provides an upper bound on $f(n)$. This
  function is used to categorize the growth rate of $f(n)$.

The relationship between $f$ and $g$ in the definition of Big O is such that
$f(n)$ is bounded above by some constant multiple of $g(n)$ for all $n$ larger
than some $n_0$. Mathematically, this is expressed as:

$$
0 \leq f(n) \leq Mg(n) \quad \text{for all} \quad n \geq n_0
$$

Here, $M$ is a positive constant, and $n_0$ is the point beyond which the
inequality holds. This definition means that for sufficiently large $n$, $f(n)$
does not grow faster than a constant multiple of $g(n)$. The Big O notation
$O(g(n))$ represents the set of all functions $f(n)$ that satisfy this
condition.

In the context of binary search, for instance, if we let $f(n) = \log(n)$ (the
actual time complexity of binary search), then we can say that $f(n)$ is in
$O(\log(n))$ because $\log(n)$ is bounded above by a constant multiple of itself
for all $n$ larger than 1. Therefore, $g(n) = \log(n)$ in this case.
