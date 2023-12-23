---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Concept

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from IPython.display import display
from typing import Sequence, TypeVar, Optional


import sys
from pathlib import Path

def find_root_dir(current_path: Path | None = None, marker: str = '.git') -> Path | None:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path | None
        The starting path to search from. If None, the current working directory
        `Path.cwd()` is used.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path | None
        The path to the root directory. Returns None if the marker is not found.
    """
    if not current_path:
        current_path = Path.cwd()
    current_path = current_path.resolve()
    for parent in [current_path, *current_path.parents]:
        if (parent / marker).exists():
            return parent
    return None

current_file_path = Path("__file__")
root_dir          = find_root_dir(current_file_path, marker='omnivault')

if root_dir is not None:
    sys.path.append(str(root_dir))
    from omnivault.utils.testing.test_framework import TestFramework
else:
    raise ImportError("Root directory not found.")
```

## Introduction

[**Linear search**](https://en.wikipedia.org/wiki/Linear_search), also known as
sequential search, is a fundamental algorithm in the field of computer science,
serving as a bedrock for understanding more complex algorithms. It represents
the most _intuitive_ method of searching for an element within a list. The
simplicity of linear search lies in its straightforward approach: examining each
element in the list **_sequentially_** until the desired element is found or the
end of the list is reached.

This algorithm, in its essence, begins with the assumption of an **unordered**
list. Without any prior knowledge of the arrangement of elements, linear search
treats each item in the list with **equal importance**, methodically checking
each until a match is found. This aspect of linear search, while simple, is
crucial in understanding how more complex searching algorithms evolve and
optimize under different conditions.

### Extensions

As we progress from the basic unordered scenario, the concept of an ordered list
introduces a subtle yet significant shift in the algorithm’s operation. In an
ordered list, the elements are arranged in a known sequence, either ascending or
descending. This arrangement allows for certain optimizations in the search
process. For instance, the search can be terminated early if the current element
in the sequence is greater (or lesser, depending on the order) than the element
being searched for, as it confirms the absence of the target element in the
remaining part of the list.

Furthermore, exploring the concept of a probabilistic array adds another
dimension to our understanding of linear search. In a probabilistic setting,
each element's likelihood of being the target is different and known. This
knowledge can be leveraged to modify the traditional linear search into a more
efficient version, where the order of search is determined based on the
probability of finding the target element, thus potentially reducing the average
number of comparisons needed.

### Intuition

The **linear search** algorithm operates on a fundamental principle of
_sequential access_ in data structures. This intuition is best understood by
considering the data structure it commonly operates on, such as a
[list](<https://en.wikipedia.org/wiki/List_(abstract_data_type)>). In computer
science, the representation of a list in memory is typically implemented as a
linear data structure. This means that the elements of the list are stored in
[**contiguous**](<https://en.wikipedia.org/wiki/Fragmentation_(computing)>)
memory locations. Each element in the list is stored at a memory address that
sequentially follows the previous element, adhering to a defined order. Although
not strictly relevant to the algorithm, having spatial locality between elements
in the list enables efficient access and traversal operations.

To elucidate the mechanism of linear search, let's consider the task of locating
a specific target element $\tau$ within a list $\mathcal{A}$. The process
unfolds in a _sequential manner_: beginning with the first element, the
algorithm examines each subsequent element in turn. This _sequential traversal_
is the hallmark of linear search, distinguishing it from more complex search
algorithms that may jump or divide the list in their search process.

```{figure} ./assets/linear-search-1.svg
---
name: linear-search-1
---
Linear Search Algorithm.
```

````{admonition} Tikz Code
:class: dropdown
```latex
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning}

\begin{document}
\begin{tikzpicture}[
    box/.style={draw, rectangle, minimum size=1cm, thick, outer sep=0pt},
    index/.style={above, font=\footnotesize},
    arrow/.style={thick, -Stealth, bend left=60}
]

% Draw boxes and indices
\foreach \x [count=\i] in {10,50,30,70,80,60,20,90,40} {
    \node[box] (box\i) at (\i-1,0) {\x};
    \node[index] at (box\i.north) {\number\numexpr\i-1\relax}; % Place index
}

% Draw arrows
\foreach \i [evaluate=\i as \nexti using int(\i+1)] in {1,...,6} {
    \draw[arrow] ([yshift=5mm]box\i.north) to ([yshift=5mm]box\nexti.north);
}

% Draw the "Find '20'" label
\node[above=of box1, align=center, yshift=15mm] (find) {Find '20'};

% Connect the "Find '20'" label with the starting arrow
\draw[arrow] (find.south) to ([yshift=5mm]box1.north);

% Highlight the found box
\node[box, fill=green!30] at (box7) {20};

\end{tikzpicture}
\end{document}
```
````

As the algorithm progresses through list $\mathcal{A}$, searching for the target
element $\tau$, there are two primary outcomes to consider:

1. **Element Discovery**: If the element $\tau$ is encountered at any point
   during the traversal, the search is immediately successful. The algorithm can
   then return a positive result, such as `True`, or, more informatively, the
   index $i$ at which $\tau$ was found in $\mathcal{A}$. Returning the index $i$
   provides not only confirmation of the element's presence but also its exact
   _location_ within the list. This outcome signifies that
   $\tau = \mathcal{A}[i]$, where $\mathcal{A}[i]$ denotes the element at the
   $i^{th}$ position in list $\mathcal{A}$.

2. **Exhaustive Search without Discovery**: Conversely, if the traversal reaches
   the end of the list without encountering $\tau$, it implies that the element
   is not present in $\mathcal{A}$. In this case, the algorithm concludes with a
   negative result, typically returning `False`. This denotes that for all
   indices $i$ in $\mathcal{A}$, $\tau \neq \mathcal{A}[i]$.

The _simplicity_ of this approach is its most defining characteristic (read:
brute force). Unlike algorithms that rely on specific data structure properties
(such as [binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm),
which requires a **sorted** array), linear search _makes no such assumptions_.
It treats each element in $\mathcal{A}$ with equal priority and does not benefit
from any particular arrangement of the data. We will see later on some
extensions of linear search that leverage the ordering of elements to optimize
the search process or if the list is probabilistic.

## Unordered Sequential Search

We start with the simplest scenario: searching for an element in an unordered
list $\mathcal{A}$.

We make some explicit assumptions:

1. **All Integers**: The set/array $\mathcal{A}$ is a set/list of integers.
2. **Unsorted**: The set/array $\mathcal{A}$ need not be ordered/sorted.
3. **Uniform Probability of Each Element Being the Target**: This assumption
   implies that before the search begins, every element in the set/array
   $\mathcal{A}$ is equally likely to be the target element $\tau$. In other
   words, there is no prior information or distribution pattern that indicates
   some elements are more likely to be $\tau$ than others. This assumption is
   crucial for the linear search algorithm, as it underpins the decision to
   search each element in $\mathcal{A}$ sequentially without prioritizing any
   specific elements.

### Algorithm (Iterative)

#### Pseudocode

````{prf:algorithm} Unordered Linear Search Pseudocode (Iterative)
:label: unordered-linear-search-pseudocode-iterative

```
Algorithm : unordered_linear_search_iterative(A, t)

    Input : A = [a_0, a_1, ..., a_{N-1}] (list of elements),
            t (target value)
    Output: Index of t in A or -1 if not found

    1: Initialize n = 0

    2: while n < N do
        3:    if A[n] == t then
        4:        return n
        5:    end if
        6:    n = n + 1
       7: end while

    8: return -1
```
````

#### Mathematical Representation

```{prf:algorithm} Unordered Linear Search Mathematical Representation (Iterative)
:label: unordered-linear-search-mathematical-representation-iterative

Given a list $\mathcal{A}$ of $N$ elements with values
$\mathcal{A}_0, \mathcal{A}_1, \ldots, \mathcal{A}_{N-1}$, and a target value
$\tau$, the goal is to find the index of the target $\tau$ in $\mathcal{A}$
using an _iterative linear search_ procedure.

The search space $\mathcal{S}$ for linear search can be conceptualized in two
ways:

1. **As the Set of Indices in $\mathcal{A}$**: The search space is defined as
   the set of all possible indices within the list $\mathcal{A}$, which can be
   mathematically represented as:

   $$
   \mathcal{S} = \{ n \in \mathbb{N} \,|\, 0 \leq n < N \}
   $$

   This interpretation focuses on the positions (indices) within the array
   during the search process.

2. **As the Array Itself**: Alternatively, the search space can be directly
   equated to the array $\mathcal{A}$ itself, denoted as:

   $$
   \mathcal{S} := \mathcal{A}
   $$

   This view considers the elements (values) of the array as the domain of the
   search.

However, unlike binary search, this search space $\mathcal{S}$ is static and
encompasses the entire array throughout the search process. The linear search
algorithm inspects each element in $\mathcal{S}$ sequentially until it finds the
target $\tau$ or exhausts the search space. Consequently, we will just
interchangeably use the array $\mathcal{A}$ as the search space $\mathcal{S}$.

The algorithm performs the following steps:

1. **Initialization**:

   1. Set the initial index $n = 0$.

2. **Iterative Procedure**: While $n < N$:

   1. Check the current element: If $\mathcal{A}_n = \tau$, the search
      terminates successfully. Return $n$ as the index where $\tau$ was found in
      $\mathcal{A}$.
   2. If $\mathcal{A}_n \neq \tau$, increment the index: $n = n + 1$.

3. **Termination**: The algorithm terminates when $n \geq N$, indicating the end
   of the list has been reached without finding $\tau$. In this case, return
   $-1$ to indicate that $\tau$ is not found in $\mathcal{A}$.
```

#### Loop Invariant

After detailing the iterative approach to the **Unordered Sequential Search**,
it is imperative to delve into the **correctness** of the algorithm. This aspect
is crucial in
[algorithmic design](https://en.wikipedia.org/wiki/Algorithm_design) as it
assures that the method not only works in theory but also functions correctly in
practical scenarios. The correctness of an algorithm can typically be
established by proving two key properties: its **invariance** and
**termination**, often termed as the **loop invariant** and **loop
termination**.

-   **Invariance** ensures that the algorithm always maintains certain
    conditions that lead to a correct solution,
-   **Termination** guarantees that the algorithm will eventually complete its
    execution.

We state the loop invariant theorem below, with the core logic extended from
[Introduction to Algorithms](https://en.wikipedia.org/wiki/Introduction_to_Algorithms)
{cite}`cormen_2022_introduction_to_algorithms`.

```{prf:theorem} Loop Invariant Theorem
:label: linear-search-loop-invariant-theorem

Consider a loop within an algorithm intended to perform a specific computation
or reach a particular state. Let $\mathcal{P}(n)$ be a property (loop invariant)
related to the loop's index $n$ or the state of the computation at the start of
each iteration. The loop invariant must satisfy the following conditions to
prove the correctness of the algorithm:

1. **Initialization**: Before the first iteration of the loop, $\mathcal{P}(n)$
   holds true. Specifically, when the loop index $n$ is at its initial value
   (often $0$ or $1$ depending on the context), the property $\mathcal{P}(n)$ is
   satisfied.

2. **Maintenance**: If $\mathcal{P}(n)$ is true before an iteration of the loop,
   it remains true before the start of the next iteration. Formally, if
   $\mathcal{P}(k)$ holds at the start of the $k^{th}$ iteration, then after
   executing the loop body and updating the loop index to $k+1$,
   $\mathcal{P}(k+1)$ should also hold. This ensures that the correctness of
   $\mathcal{P}(n)$ is maintained throughout the execution of the loop.

3. **Termination**: When the loop terminates, the invariant $\mathcal{P}(n)$
   combined with the reason for termination gives a useful property that helps
   to establish the correctness of the algorithm. The loop's termination
   condition must be such that when it is met, the goals of the algorithm are
   achieved, and the loop invariant provides insight into why these goals are
   met.
```

Proving that the use of a loop invariant in algorithms is fundamentally a form
of mathematical induction (specifically, simple induction) involves drawing
parallels between the steps of the loop invariant method and the steps of simple
mathematical induction. In what follows, we will examine the similarities
between the two methods and how they relate to each other.

##### Mathematical Induction

Mathematical induction typically includes two steps:

1. **Base Case**: Show that a statement or property is true for the initial
   value, usually for $n = 0$ or $n = 1$.

2. **Inductive Step**: Assume the statement is true for some arbitrary natural
   number $n=k$ (this is the induction hypothesis), and then prove that it's
   true for $n=k + 1$.

3. **Conclusion**: Since the base case and inductive step are proven, the
   statement is true for all natural numbers starting from the base case.

##### Loop Invariant and Induction

Now, let's align this with the loop invariant steps:

1. **Initialization (Base Case in Induction)**:

    - In loop invariants: You show that $\mathcal{P}(n)$ is true before the
      first iteration of the loop, where $n$ is at its initial value.
    - In induction: This is analogous to proving the base case, where you show
      that the property holds for the initial value.

2. **Maintenance (Inductive Step in Induction)**:

    - In loop invariants: Assuming $\mathcal{P}(k)$ is true at the start of the
      $k^{th}$ iteration, you must show that $\mathcal{P}(k+1)$ is true at the
      end of the $k^{th}$ iteration. This is the same as showing that
      $\mathcal{P}(k+1)$ is true at the start of the $(k+1)^{th}$ iteration.
      This is done by examining the changes made in the loop body and how they
      affect $\mathcal{P}(n)$.
    - In induction: This directly corresponds to the inductive step where you
      assume the statement is true for $k$ and prove it for $k+1$.

3. **Termination**:
    - In loop invariants: When the loop terminates, the invariant
      $\mathcal{P}(n)$ along with the termination condition helps in
      establishing the correctness of the algorithm. This corresponds to
      concluding in induction that since the base case and inductive step are
      proven, the property holds for all natural numbers starting from the base
      case.

Therefore, the process of using a loop invariant to prove the correctness of an
algorithm can be viewed as an application of simple mathematical induction. The
initialization step corresponds to the base case in induction, and the
maintenance step corresponds to the inductive step. This relationship highlights
the fundamental role that mathematical induction plays in formal reasoning about
algorithms.

(omniverse-dsa-searching-algorithms-linear-search-iterative-algorithm-correctness)=

#### Correctness

Let $\mathcal{A}$ be an array of $N$ elements,
$\mathcal{A}_{0}, \mathcal{A}_{1}, \ldots, \mathcal{A}_{N-1}$, and let $\tau$ be
a target value. Consider a loop in an algorithm that iterates over $\mathcal{A}$
with the intention of finding the index of $\tau$ in $\mathcal{A}$. The loop
invariant for this algorithm can be stated as follows:

```{prf:proof}
We first define the **Loop Invariant** as follows:

Define the invariant as: At the start of each iteration, the target element
$\tau$ is not in the subset of the list $\mathcal{A}[0..n-1]$, where $n$ is the
current index. This implies that if $\tau$ is in the list, it must be in the
remaining part $\mathcal{A}[n..N-1]$.

This loop invariant must satisfy three conditions:

1. **Initialization**:

   - Before the first iteration (when $n = 0$), the subarray
     $\mathcal{A}[0..n-1]$ is empty. Therefore, the statement that $\tau$ is not
     in $\mathcal{A}[0..n-1]$ is trivially true, satisfying the initialization
     condition of the loop invariant. This is the
     [**vacuous truth**](https://en.wikipedia.org/wiki/Vacuous_truth) case.

2. **Maintenance**:

   - Assume that the loop invariant holds true at the beginning of the
     $n=k^{th}$ iteration, this implies that $\tau$ is not in
     $\mathcal{A}[0..k-1]$.
   - If $\mathcal{A}[k] \neq \tau$, then $n$ increments to $n=k+1$, and now the
     subarray $\mathcal{A}[0..(k+1)-1] = \mathcal{A}[0..k]$ does not contain
     $\tau$. Thus, the invariant that $\tau$ is not in $\mathcal{A}[0..n-1]$ is
     maintained for the next iteration (note now $n=k+1$).

3. **Termination**:
   - The algorithm terminates under two conditions:
     - If $\mathcal{A}[n] = \tau$ at any iteration, the algorithm successfully
       terminates, returning the index $n$. This indicates $\tau$ was found in
       the remaining search space.
     - If the algorithm exhausts the search space $\mathcal{S}$ without finding
       $\tau$ (i.e., $n \geq N$), it concludes that $\tau$ is not present in
       $\mathcal{A}$ and terminates, returning $-1$. This is consistent with the
       loop invariant, as the invariant implies $\tau$ is not in the searched
       portion of $\mathcal{A}$.
```

The proof focuses on the maintenance of the invariant and the termination
conditions, ensuring that the algorithm is both correct (it will find $\tau$ if
it is present) and finite (it will terminate after a finite number of steps
regardless of the outcome).

The intuition is quite simple, every time before we start the next iteration, we
need to ensure that the target element $\tau$ is not in the subarray we have
searched so far. If $\tau$ is not in the subarray, then we can safely move on to
the next iteration. If we cannot assume such invariance, then we cannot be sure
that we have exhaustively and correctly searched the portion of the list already
examined.

This invariance is crucial for the correctness of the algorithm. Without this
assurance, there is a possibility that $\tau$ was already present in the part of
$\mathcal{A}$ that has been searched, and moving to the next iteration would
mean potentially overlooking the target. Therefore, maintaining this invariant
at each step ensures that if $\tau$ is indeed in the list, it will be found in
the remaining unsearched part of $\mathcal{A}$.

### Implementation

```{code-cell} ipython3
Real = TypeVar("Real", int, float, covariant=False, contravariant=False)

def unordered_linear_search_iterative_for_loop(
    container: Sequence[Real], target: Real
) -> Tuple[bool, int]:
    """Perform a linear search on an unordered sequence using for loop."""
    index = 0
    for item in container:
        if item == target:
            return True, index
        index += 1
    return False, -1


def unordered_linear_search_iterative_while_loop(
    container: Sequence[Real], target: Real
) -> Tuple[bool, int]:
    """Perform a linear search on an unordered sequence using while loop."""
    # fmt: off
    index  = 0
    length = len(container)
    # fmt: on

    while index < length:
        if container[index] == target:
            return True, index
        index += 1
    return False, -1
```

### Tests

```{code-cell} ipython3
tf = TestFramework()

unordered_list = [1, 2, 32, 8, 17, 19, 42, 13, 0]

@tf.describe("Testing unordered_linear_search_iterative_for_loop function")
def test_unordered_linear_search_iterative_for_loop():
    @tf.individual_test("Target not present in the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_for_loop(unordered_list, -1),
            (False, -1),
            "Should return (False, -1)",
        )

    @tf.individual_test("Target at the beginning of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_for_loop(unordered_list, 1),
            (True, 0),
            "Should return (True, 0)",
        )

    @tf.individual_test("Target at the end of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_for_loop(unordered_list, 0),
            (True, 8),
            "Should return (True, 8)",
        )

    @tf.individual_test("Target in the middle of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_for_loop(unordered_list, 17),
            (True, 4),
            "Should return (True, 4)",
        )

    @tf.individual_test("Empty list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_for_loop([], 1),
            (False, -1),
            "Should return (False, -1)",
        )

    @tf.individual_test("List with duplicate elements")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_for_loop([1, 1, 1], 1),
            (True, 0),
            "Should return (True, 0)",
        )

@tf.describe("Testing unordered_linear_search_iterative_while_loop function")
def test_unordered_linear_search_iterative_while_loop():
    @tf.individual_test("Target not present in the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_while_loop(unordered_list, -1),
            (False, -1),
            "Should return (False, -1)",
        )

    @tf.individual_test("Target at the beginning of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_while_loop(unordered_list, 1),
            (True, 0),
            "Should return (True, 0)",
        )

    @tf.individual_test("Target at the end of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_while_loop(unordered_list, 0),
            (True, 8),
            "Should return (True, 8)",
        )

    @tf.individual_test("Target in the middle of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_while_loop(unordered_list, 17),
            (True, 4),
            "Should return (True, 4)",
        )

    @tf.individual_test("Empty list")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_while_loop([], 1),
            (False, -1),
            "Should return (False, -1)",
        )

    @tf.individual_test("List with duplicate elements")
    def _():
        tf.assert_equals(
            unordered_linear_search_iterative_while_loop([1, 1, 1], 1),
            (True, 0),
            "Should return (True, 0)",
        )
```

### Time Complexity

We need to split the time complexity into a few cases, this is because the time
complexity **_heavily_** depends on the position of the target element we are
searching for.

We first establish the time complexity function $\mathcal{T}(N)$ to represent
the exact number of operations (comparisons) an algorithm performs for a given
input size $N$.

In the case of linear search, the time complexity function $\mathcal{T}(N)$ is
defined as the exact number of comparisons needed to find the target element
$\tau$ in a list $\mathcal{A}$ of size $N$.

#### Best Case

If the element we are searching for is at the beginning of the list, then $\tau$
is at the first index of $\mathcal{A}$, and the algorithm terminates after
performing only one comparison. Therefore, $\mathcal{T}(N) = 1$. Consequently,
the time complexity of the best case is in the order of $\mathcal{O}(1)$.

#### Worst Case

If the element is at the end of the list, then the time complexity is in the
order of $\mathcal{O}(N)$, because we need to check every element in the list
($\mathcal{T}(N) = N$).

#### Average Case

On average, the time complexity is $\mathcal{O}\left(\frac{N}{2}\right)$. This
average means that for a list with $N$ elements, there is an equal chance that
the element we are searching for is at the beginning, middle, or end of the
list. In short, it is a uniform distribution. And therefore the **expected**
time complexity is $\mathcal{O}\left(\frac{N}{2}\right)$. Let's define it more
rigorously below.

##### Defining the Random Variable

-   In the context of linear search, the random variable $X$ can be defined as
    the index position at which the target element $\tau$ is located in the
    array $\mathcal{A}$.
-   This means $X$ takes on values in the set $\{0, 1, 2, \ldots, N-1\}$, where
    $N$ is the size of the array and indexed by $n=0, 1, 2, \ldots, N-1$.
-   We denote $X = n$ as the realization that the target element $\tau$ is at
    the $n$-th position in the array and $\mathbb{P}[X = n]$ as the probability
    of this event occurring. In other words, if $X = 2$, then
    $\mathbb{P}[X = 2]$ is the probability that the target element $\tau$ is at
    the $2$-nd position in the array.

##### Uniform Distribution of $X$

-   When we say that $X$ is uniformly distributed, we mean that the probability
    of the target element $\tau$ being at any specific index $N$ in the array is
    equal for all indices.
-   Mathematically, this is expressed as $\mathbb{P}[X = n] = \frac{1}{N}$ for
    all $i$ in $\{0, 1, 2, \ldots, N-1\}$.
-   This is based on the assumption that the target element $\tau$ is equally
    likely to be at any position in the array, reflecting a scenario where there
    is no prior knowledge about the position of $\tau$.

##### Calculating Expected Value $\mathbb{E}[\mathcal{T}(N)]$

1. **Calculating Expected Value of $X$**:

    - The expected value (average) of $X$, denoted as $\mathbb{E}[X]$,
      represents the average index position where $\tau$ is expected to be
      found.
    - It is computed as the sum of each possible value of $X$ multiplied by its
      probability:

        $$
        \begin{aligned}
        \mathbb{E}[X]  &= \sum_{n=1}^{N} n \cdot P(X = n) \\
                       &= \sum_{n=1}^{N} n \cdot \frac{1}{N} \\
                       &= \frac{1}{N} \sum_{n=1}^{N} n \\
                       &= \frac{1}{N} \cdot \frac{N(N+1)}{2} \\
                       &= \frac{N+1}{2}
        \end{aligned}
        $$

    - In other words, the expected value of $X$ is $\frac{N+1}{2}$, which means
      that on average, the target element $\tau$ is expected to be found around
      the middle of the array. This translates to
      $\mathcal{T}(N) = \frac{N+1}{2}$.

        Why?

        Because the average number of comparisons is $\frac{N+1}{2}$, and each
        comparison takes constant time, the average time complexity is also
        $\frac{N+1}{2}$.

2. **Relating $\mathbb{E}[X]$ to Time Complexity $\mathcal{T}(N)$**:

    - The time complexity function $\mathcal{T}(N)$ represents the number of
      comparisons required to find $\tau$ in the array $\mathcal{A}$.
    - When the linear search algorithm finds $\tau$ at index $0 < n <= N$, it
      performs $n + 1$ comparisons (since the first comparison is at index 0).
    - Thus, the function $\mathcal{T}(n) = n + 1$, where $n$ is the index at
      which $\tau$ is found.
    - To determine the expected time complexity $\mathbb{E}[\mathcal{T}(N)]$, we
      need to consider the expected number of comparisons, which depends on the
      average position $\mathbb{E}[X]$ where $\tau$ is located:

        $$
        \begin{aligned}
        \mathbb{E}[\mathcal{T}(N)] &= \mathbb{E}[X + 1] \\
                                   &= \mathbb{E}[X] + 1 \\
                                   &= \frac{N+1}{2} + 1 \\
                                   &= \frac{N+3}{2}
        \end{aligned}
        $$

    - This result shows that on average, the linear search algorithm will
      perform $\frac{N+3}{2}$ comparisons to find $\tau$ in an array of size $N$
      under the assumption of uniform distribution. Note that the $+1$ in the
      expression $\frac{N+1}{2} + 1$ accounts for the fact that the comparison
      starts at index $0$.

##### Time Complexity Using Big O Notation $\mathcal{O}(g(N))$

1. **Definition of Big O Notation**:

    - Big O notation provides an upper bound on the growth rate of a function.
      In the context of time complexity, it is used to describe the worst-case
      scenario for the growth rate of the number of operations an algorithm
      performs.
    - Formally, a function $f(N)$ is in $\mathcal{O}(g(N))$ if there exist
      constants $C > 0$ and $N_0$ such that for all $N \geq N_0$,
      $0 \leq f(N) \leq C \cdot g(N)$.

2. **Applying to Linear Search**:

    - From the earlier analysis, we have that the average number of comparisons
      needed to find $\tau$ is $\frac{N+3}{2}$.
    - To express this in Big O notation, we need to find a function $g(N)$ and
      constants $C$ and $N_0$ that satisfy the formal definition.

3. **Choosing $g(N)$ and Constants**:

    - Let's choose $g(N) = N$. This choice reflects the linear nature of the
      time complexity function $\mathbb{E}[\mathcal{T}(N)]$.
    - Now, we need to find $C$ and $N_0$ such that
      $\mathbb{E}[\mathcal{T}(N)] \leq C \cdot g(N)$ for all $N \geq N_0$.

4. **Proof**:

    - We note that $\mathbb{E}[\mathcal{T}(N)] = \frac{N+3}{2}$.
    - Choose $C = 1$ and $N_0 = 1$. For all $N \geq N_0$:

        $$
        \begin{aligned}
        0 \leq \mathbb{E}[\mathcal{T}(N)]  &= \frac{N+3}{2} \\
                                           &\leq N \quad (\text{since } N+3 \leq 2N \text{ for } N \geq 1) \\
                                           &= C \cdot g(N)
        \end{aligned}
        $$

    - This demonstrates that $\mathbb{E}[\mathcal{T}(N)] = \frac{N+3}{2}$ is
      bounded above by $C \cdot g(N)$, confirming that the average case time
      complexity of the linear search algorithm is $\mathcal{O}(N)$.
    - This shows that the average time complexity of linear search grows
      linearly with the size of the array $N$ and is bounded by a linear
      function of $N$.

#### Time Complexity Table

However, so far we assumed that the element we are searching for is in the list.
If the element is not in the list, then the time complexity is $\mathcal{O}(N)$
for all cases, because we need to check every element in the list.

```{list-table} Time Complexity of Linear Search
:header-rows: 1
:name: linear-search-time-complexity-linear-search-iterative

* - Case
  - Worst Case
  - Average Case
  - Best Case
* - Element is in the list
  - $\mathcal{O}(N)$
  - $\mathcal{O}\left(\frac{N}{2}\right)$
  - $\mathcal{O}(1)$
* - Element is not in the list
  - $\mathcal{O}(N)$
  - $\mathcal{O}(N)$
  - $\mathcal{O}(N)$
```

### Space Complexity

#### Input Space Complexity

-   **Definition**: Input space complexity refers to the space used by the
    inputs to the algorithm.
-   **Linear Search Context**: For linear search, the primary input is the list
    $\mathcal{A}$ of size $N$.
-   **Analysis**: Since the list $\mathcal{A}$ is an essential input to the
    algorithm and occupies space proportional to its size, the input space
    complexity is directly related to the length of this list.
-   **Space Complexity**: Thus, the input space complexity is $\mathcal{O}(N)$,
    where $N$ is the number of elements in $\mathcal{A}$.

#### Auxiliary Space Complexity

-   **Definition**: Auxiliary space complexity accounts for the extra or
    temporary space used by an algorithm, excluding the space taken by the
    inputs.
-   **Linear Search Context**: In linear search, the only additional space used
    is for a few variables, such as the index variable (and possibly a boolean
    flag).
-   **Analysis**: The space required for these variables does not scale with the
    size of the input $N$; rather, it remains constant.
-   **Space Complexity**: Therefore, the auxiliary space complexity for linear
    search is $\mathcal{O}(1)$.

#### Total Space Complexity

-   **Definition**: Total space complexity combines both the input and auxiliary
    space complexities.
-   **Linear Search Context**: The total space used by the algorithm includes
    the space for the list $\mathcal{A}$ and the constant extra space for the
    variables is just $\mathcal{O}(N)$.

### The Recursive Counterpart

After discussing the iterative approach, let's consider the recursive
implementation of the unordered linear search. Recursion offers an alternative
way to perform the search by breaking down the problem into smaller subproblems.

#### Recursive Method Overview

The `unordered_linear_search_recursive` function implements linear search on an
unordered sequence using recursion. It checks each element in sequence, starting
from the beginning of the list, and returns the index of the target element if
found. If the target is not found, it returns `-1`. This is achieved through the
following steps:

1. **Base Case - Empty Container**: Checks if the container is empty. If so,
   returns `-1`, indicating the target is not found.
2. **Base Case - Target Found**: Checks if the first element in the container is
   the target. If so, returns the current index.
3. **Recursive Case**: Calls itself with the rest of the container (excluding
   the first element) and increments the index.

#### Implementation

```{code-cell} ipython3
def unordered_linear_search_recursive(
    container: Sequence[Real], target: Real, index: int = 0
) -> int:
    """Perform a linear search on an unordered sequence using recursion."""
    # Base case: if container is empty
    if not container:
        return -1     # not found

    # Base case: if the target is found
    if container[0] == target:
        return index  # found

    # notice we increment index by 1 to mean index += 1 in the iterative case
    return unordered_linear_search_recursive(container[1:], target, index + 1)
```

Using Python Tutor to visualize recursive calls
[here](https://pythontutor.com/render.html#code=def%20f%28container,%20target,%20index%3D0%29%3A%0A%20%20%20%20if%20len%28container%29%20%3D%3D%200%3A%20%20%23%20if%20not%20container%20is%20also%20fine%0A%20%20%20%20%20%20%20%20return%20-1%20%20%23%20not%20found%0A%0A%20%20%20%20if%20container%5B0%5D%20%3D%3D%20target%3A%20%20%23%20this%20is%20base%20case%0A%20%20%20%20%20%20%20%20return%20index%20%20%23%20found%0A%0A%20%20%20%20%23%20notice%20we%20increment%20index%20by%201%20to%20mean%20index%20%2B%3D%201%20in%20the%20iterative%20case%0A%20%20%20%20return%20f%28container%5B1%3A%5D,%20target,%20index%20%2B%201%29%20%20%23%20recursive%20case%0A%20%20%20%20%0Aunordered_list%20%3D%20%5B1,%202,%2032,%208,%2017,%2019,%2042,%2013,%200%5D%0Aprint%28f%28unordered_list,%2013%29%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false).

Embedded:

<iframe src="https://pythontutor.com/render.html#code=def%20f%28container,%20target,%20index%3D0%29%3A%0A%20%20%20%20if%20len%28container%29%20%3D%3D%200%3A%20%20%23%20if%20not%20container%20is%20also%20fine%0A%20%20%20%20%20%20%20%20return%20-1%20%20%23%20not%20found%0A%0A%20%20%20%20if%20container%5B0%5D%20%3D%3D%20target%3A%20%20%23%20this%20is%20base%20case%0A%20%20%20%20%20%20%20%20return%20index%20%20%23%20found%0A%0A%20%20%20%20%23%20notice%20we%20increment%20index%20by%201%20to%20mean%20index%20%2B%3D%201%20in%20the%20iterative%20case%0A%20%20%20%20return%20f%28container%5B1%3A%5D,%20target,%20index%20%2B%201%29%20%20%23%20recursive%20case%0A%20%20%20%20%0Aunordered_list%20%3D%20%5B1,%202,%2032,%208,%2017,%2019,%2042,%2013,%200%5D%0Aprint%28f%28unordered_list,%2013%29%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false" width="800" height="600">
</iframe>

#### Tests

```{code-cell} ipython3
@tf.describe("Testing unordered_linear_search_recursive function")
def test_unordered_sequential_search_recursive():
    @tf.individual_test("Target not present in the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_recursive(unordered_list, -1),
            -1,
            "Should return -1",
        )

    @tf.individual_test("Target at the beginning of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_recursive(unordered_list, 1),
            0,
            "Should return 0",
        )

    @tf.individual_test("Target at the end of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_recursive(unordered_list, 0),
            8,
            "Should return 8",
        )

    @tf.individual_test("Target in the middle of the list")
    def _():
        tf.assert_equals(
            unordered_linear_search_recursive(unordered_list, 17),
            4,
            "Should return 4",
        )

    @tf.individual_test("Empty list")
    def _():
        tf.assert_equals(
            unordered_linear_search_recursive([], 1),
            -1,
            "Should return -1",
        )

    @tf.individual_test("List with duplicate elements")
    def _():
        tf.assert_equals(
            unordered_linear_search_recursive([1, 1, 1], 1),
            0,
            "Should return 0",
        )
```

#### Complexity Analysis

-   **Time Complexity**: The time complexity of the recursive implementation is
    similar to the iterative approach. In the worst case (target not present or
    at the end), it checks each element once, leading to $\mathcal{O}(N)$
    complexity. The average and best cases are also similar to the iterative
    approach.

-   **Space Complexity**: The space complexity of the recursive implementation
    is a bit different from the iterative one. Each recursive call adds a new
    layer to the call stack. In the worst case, there will be $N$ recursive
    calls, leading to a space complexity of $\mathcal{O}(N)$.

We can further optimize the recursive implementation by using tail recursion to
reduce the space complexity to $\mathcal{O}(1)$.

```{code-cell} ipython3
def unordered_linear_search_tail_recursive(
    container: Sequence[Real], target: Real, index: int = 0
) -> int:
    """Perform a linear search on an unordered sequence using tail recursion."""
    # Check if we have reached the end of the container
    if index == len(container):
        return -1  # Target not found

    # Check if the target is at the current index
    if container[index] == target:
        return index  # Target found

    # Recurse with the next index
    return unordered_linear_search_tail_recursive(container, target, index + 1)
```

#### Time Complexity Table

It's beneficial to include a time complexity table for the recursive
implementation as well. However, note that while the time complexities are
similar to the iterative version, the space complexity differs due to the nature
of recursive calls.

```{list-table} Time Complexity of Linear Search (Recursive)
:header-rows: 1
:name: linear-search-time-complexity-recursive

* - Case
  - Worst Case
  - Average Case
  - Best Case
* - Element is in the list
  - $\mathcal{O}(N)$
  - $\mathcal{O}\left(\frac{N}{2}\right)$
  - $\mathcal{O}(1)$
* - Element is not in the list
  - $\mathcal{O}(N)$
  - $\mathcal{O}(N)$
  - $\mathcal{O}(N)$
```

## Ordered and Probabilistic Linear Search

Here we briefly touch upon two variations of linear search, ordered linear
search and probabilistic linear search for the sake of completeness. For a much
more detailed explanation, please refer to the
[blog post here](https://ozaner.github.io/sequential-search/).

### Ordered Linear Search: Efficiency in Sorted Arrays

**Overview**: Ordered linear search is a powerful variation of the standard
linear search algorithm, specifically tailored for use with sorted arrays. This
method leverages the inherent order within the array to optimize the search
process, allowing for early termination and thus potentially reducing the number
of comparisons needed to find a target element.

**How It Works**: In an ordered linear search, the algorithm sequentially checks
each element, but with a key advantage – if it encounters an element greater (or
lesser, depending on the sort order) than the target, it can immediately
conclude that the target is not in the list. This early termination is a
game-changer in terms of efficiency, particularly when the target is located
near the beginning of the array or is not present at all.

**Why It's Effective**: This approach is most effective in scenarios where the
data is inherently ordered or can be sorted prior to the search. Common examples
include searching through chronological records, numerical data, or
alphabetically sorted lists.

**Learn More**: To dive deeper into ordered linear search, understand its
specific use cases, and explore its implementation,
[click here for a detailed guide and examples](https://ozaner.github.io/sequential-search/).

### Probabilistic Search: Harnessing Data to Optimize Searches

**Overview**: Probabilistic search represents a significant advancement in
search algorithms by incorporating the likelihood of each element being the
target into the search process. This method is particularly useful in situations
where certain elements are more frequently searched for than others, a common
scenario in many real-world applications.

**The Power of Probability**: In probabilistic search, each element in the array
is assigned a probability based on its likelihood of being the target, derived
from historical data or specific characteristics. The search algorithm then
prioritizes elements with higher probabilities, leading to a more efficient
search process. This approach is ideal for datasets where user behavior or
element popularity can be predicted or measured.

**Applications and Benefits**: From online shopping platforms optimizing product
searches based on buying patterns to search engines prioritizing frequently
queried terms, probabilistic search has wide-ranging applications. Its adoption
can significantly enhance user experience by reducing search times and improving
overall efficiency.

**Learn More**: To gain a comprehensive understanding of probabilistic search,
explore its methodologies, and see how it's revolutionizing data search in
various fields,
[click here for an in-depth exploration](https://ozaner.github.io/sequential-search/).

#### Problem Scenario: Optimized Product Search in an Online Store

##### Background

An online store, "TechGear", sells a variety of electronic products. Based on
historical sales data, some products are more frequently searched for by
customers than others. TechGear wants to optimize their product search algorithm
to quickly locate products that customers are more likely to search for.

##### Sales Data and Probabilities

TechGear has the following catalog of products with associated probabilities
based on their sales data:

| Product ID | Product Name     | Probability (Sales-based) |
| ---------- | ---------------- | ------------------------- |
| 1          | Smartphone X     | 0.30                      |
| 2          | Laptop Pro       | 0.25                      |
| 3          | Headphones Beat  | 0.15                      |
| 4          | Smartwatch 2     | 0.10                      |
| 5          | Camera Zoom      | 0.05                      |
| 6          | Portable Charger | 0.05                      |
| 7          | Drone Fly        | 0.05                      |
| 8          | Gaming Console Z | 0.05                      |

The probabilities reflect the likelihood of each product being searched for,
based on past customer behavior. For instance, 'Smartphone X' has the highest
search probability at 0.30.

##### Objective

TechGear wants to implement a linear search algorithm that utilizes this
probability data to improve the efficiency of product searches in their catalog.

##### Search Algorithm Implementation

1. **Reorder the Catalog**: The first step is to reorder the catalog of products
   based on the probabilities, placing higher probability items towards the
   beginning.

2. **Search Process**: Implement a linear search that goes through the reordered
   list. If a customer searches for 'Smartwatch 2', the algorithm will check
   products in the order of their likelihood (starting with 'Smartphone X', then
   'Laptop Pro', and so on) until it finds 'Smartwatch 2'.

3. **Evaluate Performance**: The performance of this probabilistic search
   algorithm can be compared with a regular linear search to demonstrate the
   efficiency gained by prioritizing products based on their search probability.

##### Expected Outcome

The expectation is that, on average, the probabilistic search algorithm will
locate products faster than a regular linear search, especially for items with
high search probabilities. This approach can significantly improve customer
experience by reducing search times. Intuitively, it makes sense, because
customers are more likely to search for products with higher probabilities, and
the algorithm prioritizes these items.

## Summary

In conclusion, while linear search is one of the simplest search algorithms, its
variations like ordered and probabilistic searches demonstrate its adaptability
and effectiveness in diverse scenarios. From basic implementations in unordered
lists to more sophisticated applications in sorted and probabilistic lists,
linear search remains a crucial tool in the algorithmic toolkit. It is also a
good "naive/brute-force" approach to solving problems.

## References and Further Readings

-   [Runestone Academy Sequential Search](https://runestone.academy/ns/books/published/pythonds/SortSearch/TheSequentialSearch.html)
-   [Wikipedia Linear Search](https://en.wikipedia.org/wiki/Linear_search)
-   [Stack Overflow Recursive Linear Search](https://stackoverflow.com/questions/4295608/recursive-linear-search-returns-list-index)
-   [Ozaner Sequential Search](https://ozaner.github.io/sequential-search/)
-   [Expected Number of Iterations of an Exhaustive Search - Mathematics Stack Exchange](https://math.stackexchange.com/questions/2048236/expected-number-of-iterations-of-an-exhaustive-search)
-   [Average Time Complexity of Linear Search - Computer Science Stack Exchange](https://cs.stackexchange.com/questions/140716/average-time-complexity-of-linear-search)
-   [Loop Invariant of Linear Search - Stack Overflow](https://stackoverflow.com/questions/5585020/loop-invariant-of-linear-search)
-   [Analysis of Linear Search - CLRS Chapter 2, Exercise 2.1-3](https://atekihcan.github.io/CLRS/02/E02.01-03/)
-   [Proof of Linear Search - Computer Science Stack Exchange](https://cs.stackexchange.com/questions/6597/proof-of-linear-search)
-   [Proving the Correctness of Linear Search Algorithm (CLRS Exercise 2.1-3) - Quora](https://www.quora.com/How-do-you-prove-the-correctness-of-a-linear-search-algorithm-for-exercise-2-1-3-in-CLRS)
-   [Heap Invariant and Its Proof - Columbia University](https://www.columbia.edu/~cs2035/courses/csor4231.F05/heap-invariant.pdf)
