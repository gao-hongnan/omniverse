# Big-O

## Important First Read

- Examples: https://en.wikipedia.org/wiki/Big_O_notation#Example
- Properties: https://en.wikipedia.org/wiki/Big_O_notation#Properties

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
