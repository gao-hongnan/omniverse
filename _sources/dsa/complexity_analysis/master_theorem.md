# Master Theorem [^master-theorem]

In the
[analysis of algorithms](https://en.wikipedia.org/wiki/Analysis_of_algorithms),
the master theorem for divide-and-conquer recurrences provides an
[asymptotic analysis](https://en.wikipedia.org/wiki/Asymptotic_analysis) (using
[Big O notation](https://en.wikipedia.org/wiki/Big_O_notation)) for
[recurrence relations](https://en.wikipedia.org/wiki/Recurrence_relation) of
types that occur in the analysis of many
[divide and conquer algorithms](https://en.wikipedia.org/wiki/Divide_and_conquer_algorithm).
The approach was first presented by
[Jon Bentley](https://en.wikipedia.org/wiki/Jon_Louis_Bentley),
[Dorothea Blostein](https://en.wikipedia.org/wiki/Dorothea_Blostein) (née
Haken), and [James B. Saxe](https://en.wikipedia.org/wiki/James_B._Saxe) in
1980, where it was described as a "unifying method" for solving such
recurrences. The name "master theorem" was popularized by the widely-used
algorithms textbook
[Introduction to Algorithms](https://en.wikipedia.org/wiki/Introduction_to_Algorithms)
by [Cormen](https://en.wikipedia.org/wiki/Thomas_H._Cormen),
[Leiserson](https://en.wikipedia.org/wiki/Charles_E._Leiserson),
[Rivest](https://en.wikipedia.org/wiki/Ronald_L._Rivest), and
[Stein](https://en.wikipedia.org/wiki/Clifford_Stein).

Not all recurrence relations can be solved with the use of this theorem; its
generalizations include the
[Akra–Bazzi method](https://en.wikipedia.org/wiki/Akra%E2%80%93Bazzi_method).

## Introduction

Consider a problem that can be solved using a
[recursive algorithm](https://en.wikipedia.org/wiki/Recursive_algorithm) such as
the following:

````{prf:algorithm} Generic Divide and Conquer Algorithm
:label: master-theorem-generic-divide-and-conquer-algorithm

```python
procedure p(input x of size n):
    if n < some constant k:
        Solve x directly without recursion
    else:
        Create a subproblems of x, each having size n/b
        Call procedure p recursively on each subproblem
        Combine the results from the subproblems
```
````

The above algorithm divides the problem into a number of subproblems
recursively, each subproblem being of size $\frac{n}{b}$. Its
[solution tree](https://en.wikipedia.org/wiki/Tree_structure) in
{numref}`master-theorem-wikipedia-master-theorem-1` has a node for each
recursive call, with the children of that node being the other calls made from
that call. The leaves of the tree are the base cases of the recursion, the
subproblems (of size less than $k$) that do not recurse. The above example would
have a child nodes at each non-leaf node. Each node does an amount of work that
corresponds to the size of the subproblem n passed to that instance of the
recursive call and given by $f(n)$. The total amount of work done by the entire
algorithm is the sum of the work performed by all the nodes in the tree.

The runtime of an algorithm such as the $p$ above on an input of size $n$,
usually denoted $\mathcal{T}(n)$, can be expressed by the
[recurrence relation](https://en.wikipedia.org/wiki/Recurrence_relation)

```{math}
:label: master-theorem-recurrence-relation

\mathcal{T}(n)=a\;\mathcal{T}\left({\frac {n}{b}}\right)+f(n),
```

where $f(n)$ is the time to create the subproblems and combine their results in
the above procedure. This equation can be successively substituted into itself
and expanded to obtain an expression for the total amount of work done. The
master theorem allows many recurrence relations of this form to be converted to
[$\Theta$-notation](https://en.wikipedia.org/wiki/Big_O_notation) directly,
without doing an expansion of the recursive relation.

```{figure} ./assets/wikipedia-master-theorem-1.svg
---
name: master-theorem-wikipedia-master-theorem-1
---

Solution Tree.

**Image Credit:**
[Wikipedia](<https://en.wikipedia.org/wiki/Master_theorem_(analysis_of_algorithms)#/media/File:Recursive_problem_solving.svg>)
```

## Generic Form

The master theorem always yields
[asymptotically tight bounds](https://en.wikipedia.org/wiki/Asymptotically_tight_bound)
to recurrences from
[divide and conquer algorithms](https://en.wikipedia.org/wiki/Divide_and_conquer_algorithm)
that partition an input into smaller subproblems of equal sizes, solve the
subproblems recursively, and then combine the subproblem solutions to give a
solution to the original problem. The time for such an algorithm can be
expressed by adding the work that they perform at the top level of their
recursion (to divide the problems into subproblems and then combine the
subproblem solutions) together with the time made in the recursive calls of the
algorithm. If $\mathcal{T}(n)$ denotes the total time for the algorithm on an
input of size $n$, and $f(n)$ denotes the amount of time taken at the top level
of the recurrence then the time can be expressed by a
[recurrence relation](https://en.wikipedia.org/wiki/Recurrence_relation) that
takes the form:

```{math}
:label: master-theorem-recurrence-relation-generic-form

\mathcal{T}(n) = a\;\mathcal{T} \left({\frac{n}{b}}\right) + f(n),
```

Here $n$ is the size of an input problem, $a$ is the number of subproblems in
the recursion, and $b$ is the factor by which the subproblem size is reduced in
each recursive call $(b > 1)$. Crucially, $a$ and $b$ must not depend on $n$.
The theorem below also assumes that, as a base case for the recurrence,
$\mathcal{T}(n)=\Theta(1)$ when $n$ is less than some bound $\kappa>0$, the
smallest input size that will lead to a recursive call.

Recurrences of this form often satisfy one of the three following regimes, based
on how the work to split/recombine the problem $f(n)$ relates to the critical
exponent $c_{\text {crit }}=\log _b a$. (The table below uses standard
[big $\mathcal{O}$ notation](https://en.wikipedia.org/wiki/Big_O_notation)).

$$
c_{\text {crit }}=\log_b a=\log (\text{# subproblems} ) / \log (\text{relative subproblem size})
$$

```{list-table} Master Theorem Cases
:header-rows: 1
:name: master-theorem-cases

*   - Case
    - Description
    - Condition on $f(n)$ in relation to $c_{\text{crit}}$, i.e. $\log_b a$
    - Master Theorem bound
    - Notational examples
*   - 1
    -   Work to split/recombine a problem is dwarfed by subproblems, i.e.,
        the recursion tree is leaf-heavy.
    -   When $f(n) = \mathcal{O}(n^c)$ where $c < c_{\text{crit}}$ (upper-bounded by a
        lesser exponent polynomial)
    -   $\ldots$ then $\mathcal{T}(n) = \Theta(n^{c_{\text{crit}}})$ (The splitting
        term does not appear; the recursive tree structure dominates.)
    -   If $b = a^2$ and $f(n) = \mathcal{O}(n^{1/2 - \epsilon})$, then
        $\mathcal{T}(n) = \Theta(n^{1/2})$.
*   - 2
    -   Work to split/recombine a problem is comparable to subproblems.
    -   When $f(n) = \Theta(n^{c_{\text{crit}}} \log^k n)$ for a $k \geq 0$
        (rangebound by the critical-exponent polynomial, times zero or more
        optional logs)
    -   $\ldots$ then $\mathcal{T}(n) = \Theta(n^{c_{\text{crit}}} \log^{k+1} n)$ (The
        bound is the splitting term, where the log is augmented by a single
        power.)
    -   If $b = a^2$ and $f(n) = \Theta(n^{1/2})$, then
        $\mathcal{T}(n) = \Theta(n^{1/2} \log n)$. If $b = a^2$ and
        $f(n) = \Theta(n^{1/2} \log n)$, then
        $\mathcal{T}(n) = \Theta(n^{1/2} \log^2 n)$.
*   - 3
    -   Work to split/recombine a problem dominates subproblems, i.e., the
        recursion tree is root-heavy.
    -   When $f(n) = \Omega(n^c)$ where $c > c_{\text{crit}}$ (lower-bounded
        by a greater-exponent polynomial)
    -   $\ldots$ this doesn't necessarily yield anything. Furthermore, if
        $a f(\frac{n}{b}) \leq k f(n)$ for some constant $k < 1$ and
        sufficiently large $n$ (often called the regularity condition), then
        the total is dominated by the splitting term $f(n)$:
        $\mathcal{T}(n) = \Theta(f(n))$
    -   If $b = a^2$ and $f(n) = \Omega(n^{1/2 + \epsilon})$ and the
        regularity condition holds, then $\mathcal{T}(n) = \Theta(f(n))$.
```

```{list-table} Master Theorem Case 2 Extensions
:header-rows: 1
:name: master-theorem-case-2-extensions

*   -   Case
    -   Condition on $f(n)$ in relation to $c\_{\text{crit}}$, i.e.
        $\log_b a$
    -   Master Theorem bound
    -   Notational examples
*   -   2a
    -   When $f(n) = \Theta(n^{c\_{\text{crit}}} \log^k n)$ for any
        $
        k > -1$
    -   $\ldots$ then
        $\mathcal{T}(n) = \Theta(n^{c\_{\text{crit}}} \log^{k+1} n)
        $ (The
        bound is the splitting term, where the log is augmented by a single
        power.)
    -   If $b = a^2$ and $f(n) = \Theta(n^{1/2} / \log^{1/2} n)$, then
        $\mathcal{T}(n) = \Theta(n^{1/2} \log^{1/2} n)$.
*   -   2b
    -   When $f(n) = \Theta(n^{c\_{\text{crit}}} \log^k n)$ for
        $k =
        -1$
    -   $\ldots$ then
        $\mathcal{T}(n) = \Theta(n^{c\_{\text{crit}}} \log \log n)
        $ (The
        bound is the splitting term, where the log reciprocal is replaced by
        an iterated log.)
    -   If $b = a^2$ and $f(n) = \Theta(n^{1/2} / \log n)$, then
        $
        \mathcal{T}(n) = \Theta(n^{1/2} \log \log n)$
*   -   2c
    -   When $f(n) = \Theta(n^{c\_{\text{crit}}} \log^k n)$ for any
        $k
        < -1$
    -   $\ldots$ then $\mathcal{T}(n) = \Theta(n^{c\_{\text{crit}}})$ (The bound is
        the splitting term, where the log disappears.)
    -   If $b = a^2$ and $f(n) = \Theta(n^{1/2} / \log^2 n)$, then
        $\mathcal{T}(n) = \Theta(n^{1/2})$
```

## Examples

### Case 1

$$
\mathcal{T}(n)=8 T\left(\frac{n}{2}\right)+1000 n^2
$$

As one can see from the formula above:

$$
\begin{aligned}
& a=8, b=2, f(n)=1000 n^2, \text { so } \\
& f(n)=O\left(n^c\right), \text { where } c=2
\end{aligned}
$$

Next, we see if we satisfy the case 1 condition:

$$
c_{\text{crit}} = \log _b a=\log _2 8=3>c .
$$

It follows from the first case of the master theorem that

$$
\mathcal{T}(n)=\Theta\left(n^{\log _b a}\right)=\Theta\left(n^3\right)
$$

(This result is confirmed by the exact solution of the recurrence relation,
which is $\mathcal{T}(n)=1001 n^3-1000 n^2$, assuming $T(1)=1$ ).

### Case 2

$$
\mathcal{T}(n)=2 T\left(\frac{n}{2}\right)+10 n
$$

As we can see in the formula above the variables get the following values:

$$
\begin{aligned}
& a=2, b=2, c=1, f(n)=10 n \\
& f(n)=\Theta\left(n^c \log ^k n\right) \text { where } c=1, k=0
\end{aligned}
$$

Next, we see if we satisfy the case 2 condition:
$c_{\text{crit}} = \log _b a=\log _2 2=1$, and therefore, c and $\log _b a$ are
equal

So it follows from the second case of the master theorem:

$$
\mathcal{T}(n)=\Theta\left(n^{\log _b a} \log ^{k+1} n\right)=\Theta\left(n^1 \log ^1 n\right)=\Theta(n \log n)
$$

Thus the given recurrence relation $\mathcal{T}(n)$ was in $\Theta(n \log n)$.
(This result is confirmed by the exact solution of the recurrence relation,
which is $\mathcal{T}(n)=n+10 n \log _2 n$, assuming $T(1)=1$).

### Case 3

$$
\mathcal{T}(n)=2 T\left(\frac{n}{2}\right)+n^2
$$

As we can see in the formula above the variables get the following values:

$$
\begin{aligned}
& a=2, b=2, f(n)=n^2 \\
& f(n)=\Omega\left(n^c\right), \text { where } c=2
\end{aligned}
$$

Next, we see if we satisfy the case 3 condition:

$$
c_{\text{crit}} = \log _b a=\log _2 2=1 \text {, and therefore, yes, } c>\log _b a
$$

The regularity condition also holds:

$$
2\left(\frac{n^2}{4}\right) \leq k n^2, \text { choosing } k=1 / 2
$$

So it follows from the third case of the master theorem:

$$
\mathcal{T}(n)=\Theta(f(n))=\Theta\left(n^2\right) .
$$

Thus the given recurrence relation $\mathcal{T}(n)$ was in
$\Theta\left(n^2\right)$, that complies with the $f(n)$ of the original formula.
(This result is confirmed by the exact solution of the recurrence relation,
which is $\mathcal{T}(n)=2 n^2-n$, assuming $T(1)=1$.)

## Inadmissble Equations

The following equations cannot be solved using the master theorem: ${ }^{[4]}$

-   $\mathcal{T}(n)=2^n \mathcal{T}\left(\frac{n}{2}\right)+n^n$ $a$ is not a
    constant; the number of subproblems should be fixed
-   $\mathcal{T}(n)=2 \mathcal{T}\left(\frac{n}{2}\right)+\frac{n}{\log n}$
    non-polynomial difference between $f(n)$ and $n^{\log _b a}$ (see below;
    extended version applies)
-   $\mathcal{T}(n)=0.5 \mathcal{T}\left(\frac{n}{2}\right)+n$ $a<1$ cannot have
    less than one sub problem
-   $\mathcal{T}(n)=64 \mathcal{T}\left(\frac{n}{8}\right)-n^2 \log n$ $f(n)$,
    which is the combination time, is not positive
-   $\mathcal{T}(n)=\mathcal{T}\left(\frac{n}{2}\right)+n(2-\cos n)$ case 3 but
    regularity violation.

In the second inadmissible example above, the difference between $f(n)$ and
$n^{\log _b a}$ can be expressed with the ratio
$\frac{f(n)}{n^{\log _b a}}=\frac{n / \log n}{n^{\log _2 2}}=\frac{n}{n \log n}=\frac{1}{\log n}$.
It is clear that $\frac{1}{\log n}<n^\epsilon$ for any constant $\epsilon>0$.
Therefore, the difference is not polynomial and the basic form of the Master
Theorem does not apply. The extended form (case $2 b$ ) does apply, giving the
solution $\mathcal{T}(n)=\Theta(n \log \log n)$.

## Application to Common Algorithms

```{list-table} Common Algorithms
:header-rows: 1
:name: master-theorem-application-to-common-algorithms

*   -   Algorithm
    -   Recurrence relationship
    -   Run time
    -   Comment
*   -   Binary search
    -   $\mathcal{T}(n) = \mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(1)$
    -   $\mathcal{O}(\log n)$
    -   Apply Master theorem case $c = \log_b a$, where
            $a = 1, b = 2,
            c = 0, k = 0^{[5]}$
*   -   Binary Tree traversal
    -   $\mathcal{T}(n) = 2\mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(1)$
    -   $\mathcal{O}(n)$
    -   Apply Master theorem case $c < \log_b a$ where
            $a = 2, b = 2,
            c = 0^{[5]}$
*   -   Optimal sorted matrix search
    -   $\mathcal{T}(n) = 2\mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(\log n)$
    -   $\mathcal{O}(n)$
    -   Apply the
        [Akra-Bazzi](https://en.wikipedia.org/wiki/Akra%E2%80%93Bazzi_method)
        theorem for $p = 1$ and
        $g(u) = \log(u)
            $ to get $\Theta(2n - \log n)$
*   -   Merge sort
    -   $\mathcal{T}(n) = 2\mathcal{T}\left(\frac{n}{2}\right) + \mathcal{O}(n)$
    -   $\mathcal{O}(n \log n)$
    -   Apply Master theorem case $c = \log_b a$, where
            $a = 2, b = 2,
            c = 1, k = 0$
```

## References and Further Readings

-   [Master Theorem](<https://en.wikipedia.org/wiki/Master_theorem_(analysis_of_algorithms)>)
-   [Akra-Bazzi Method](https://en.wikipedia.org/wiki/Akra%E2%80%93Bazzi_method)
-   [Asymptotic Complexity](https://en.wikipedia.org/wiki/Asymptotic_complexity)

[^master-theorem]:
    This is almost a verbatin copy of the
    [Master Theorem](<https://en.wikipedia.org/wiki/Master_theorem_(analysis_of_algorithms)>)
    Wikipedia page with modified formatting and notations.
