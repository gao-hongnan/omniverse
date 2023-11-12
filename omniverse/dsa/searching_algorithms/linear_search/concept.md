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

def find_root_dir(current_path: Path, marker: str) -> Optional[Path]:
    """
    Find the root directory by searching for a directory or file that serves as a
    marker.

    Parameters
    ----------
    current_path : Path
        The starting path to search from.
    marker : str
        The name of the file or directory that signifies the root.

    Returns
    -------
    Path or None
        The path to the root directory. Returns None if the marker is not found.
    """
    current_path = current_path.resolve()
    for parent in current_path.parents:
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

## Contiguous

- **Contiguous Memory Allocation**: In most programming languages, when a list
  (or an array, which is a fixed-size list) is created, a contiguous block of
  memory is allocated. This contiguous allocation ensures that each element in
  the list can be accessed by computing its memory address based on the starting
  memory address of the list and the element's position (or index) in the list.

- **Index-Based Access**: The position of each element in the list is determined
  by its index, a numerical representation starting from 0 (in zero-indexed
  languages like Python, C, and Java) or 1 (in one-indexed languages like Lua
  and MATLAB). The memory address of an element at index $n$ can be calculated
  using the formula: `Base Address + (n * Size of Element)`. This formula
  leverages the fact that elements are stored at evenly spaced intervals in
  memory.

- **Relative Positioning**: The sequential storage implies that the memory
  location of an element is relative to its immediate neighbors. For example, in
  a list of integers (assuming each integer occupies 4 bytes and the list starts
  at memory address 100), the element at index 1 would be at memory address 104,
  index 2 at 108, and so on. This relative positioning enables efficient access
  and traversal operations, as moving to the next or previous element involves a
  consistent step in memory.

Understanding this memory model is crucial for appreciating how basic operations
on lists, like indexing and iterating, are performed and why they have the time
complexities they do. For instance, accessing any element in a list via its
index is an $\mathcal{O}(1)$ operation because it involves a direct calculation
to find the element's memory address, regardless of the list's size.

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

Given a list $\mathcal{A}$ of $N$ elements with values or records
$\mathcal{A}_0, \mathcal{A}_1, \ldots, \mathcal{A}_{N-1}$, and a target value
$\tau$, the goal is to find the index of the target $\tau$ in $\mathcal{A}$
using an iterative linear search procedure.

The search space $\mathcal{S}$ for the linear search is defined as the set of
indices in $\mathcal{A}$:

$$
\mathcal{S} = \{ n \in \mathbb{N} \,|\, 0 \leq n < N \}
$$

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

- **Invariance** ensures that the algorithm always maintains certain conditions
  that lead to a correct solution,
- **Termination** guarantees that the algorithm will eventually complete its
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

   - In loop invariants: You show that $\mathcal{P}(n)$ is true before the first
     iteration of the loop, where $n$ is at its initial value.
   - In induction: This is analogous to proving the base case, where you show
     that the property holds for the initial value.

2. **Maintenance (Inductive Step in Induction)**:

   - In loop invariants: Assuming $\mathcal{P}(k)$ is true at the start of the
     $k^{th}$ iteration, you must show that $\mathcal{P}(k+1)$ is true at the
     end of the $k^{th}$ iteration. This is the same as showing that
     $\mathcal{P}(k+1)$ is true at the start of the $(k+1)^{th}$ iteration. This
     is done by examining the changes made in the loop body and how they affect
     $\mathcal{P}(n)$.
   - In induction: This directly corresponds to the inductive step where you
     assume the statement is true for $k$ and prove it for $k+1$.

3. **Termination**:
   - In loop invariants: When the loop terminates, the invariant
     $\mathcal{P}(n)$ along with the termination condition helps in establishing
     the correctness of the algorithm. This corresponds to concluding in
     induction that since the base case and inductive step are proven, the
     property holds for all natural numbers starting from the base case.

Therefore, the process of using a loop invariant to prove the correctness of an
algorithm can be viewed as an application of simple mathematical induction. The
initialization step corresponds to the base case in induction, and the
maintenance step corresponds to the inductive step. This relationship highlights
the fundamental role that mathematical induction plays in formal reasoning about
algorithms.

#### Correctness

Let $\mathcal{A}$ be an array of $N$ elements,
$\mathcal{A}[0], \mathcal{A}[1], \ldots, \mathcal{A}[N-1]$, and let $\tau$ be a
target value. Consider a loop in an algorithm that iterates over $\mathcal{A}$
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

On average, the time complexity is $\mathcal{O}(\frac{N}{2})$. This average
means that for a list with $N$ elements, there is an equal chance that the
element we are searching for is at the beginning, middle, or end of the list. In
short, it is a uniform distribution. And therefore the **expected** time
complexity is $\mathcal{O}(\frac{N}{2})$. Let's define it more rigorously below.

##### Defining the Random Variable

- In the context of linear search, the random variable $X$ can be defined as the
  index position at which the target element $\tau$ is located in the array
  $\mathcal{A}$.
- This means $X$ takes on values in the set $\{0, 1, 2, \ldots, N-1\}$, where
  $N$ is the size of the array and indexed by $n=0, 1, 2, \ldots, N-1$.
- We denote $X = n$ as the realization that the target element $\tau$ is at the
  $n$-th position in the array and $\mathbb{P}[X = n]$ as the probability of
  this event occurring. In other words, if $X = 2$, then $\mathbb{P}[X = 2]$ is
  the probability that the target element $\tau$ is at the $2$-nd position in
  the array.

##### Uniform Distribution of $X$

- When we say that $X$ is uniformly distributed, we mean that the probability of
  the target element $\tau$ being at any specific index $N$ in the array is
  equal for all indices.
- Mathematically, this is expressed as $\mathbb{P}[X = n] = \frac{1}{N}$ for all
  $i$ in $\{0, 1, 2, \ldots, N-1\}$.
- This is based on the assumption that the target element $\tau$ is equally
  likely to be at any position in the array, reflecting a scenario where there
  is no prior knowledge about the position of $\tau$.

##### Calculating Expected Value $\mathbb{E}[\mathcal{T}(N)]$

1. **Calculating Expected Value of $X$**:

   - The expected value (average) of $X$, denoted as $\mathbb{E}[X]$, represents
     the average index position where $\tau$ is expected to be found.
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
   - When the linear search algorithm finds $\tau$ at index $n$, it performs
     $n + 1$ comparisons (since the first comparison is at index 0).
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

   - This result shows that on average, the linear search algorithm will perform
     $\frac{N+3}{2}$ comparisons to find $\tau$ in an array of size $N
    $
     under the assumption of uniform distribution.

##### Time Complexity Using Big O Notation $\mathcal{O}(g(N))$

1. **Definition of Big O Notation**:

   - Big O notation provides an upper bound on the growth rate of a function. In
     the context of time complexity, it is used to describe the worst-case
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

   - Let's choose $g(N) = N$. This choice reflects the linear nature of the time
     complexity function $\mathbb{E}[\mathcal{T}(N)]$.
   - Now, we need to find $C$ and $N_0$ such that
     $\mathbb{E}[\mathcal{T}(N)] \leq C \cdot g(N)$ for all $N \geq N_0$.

4. **Proof**:

   - We note that $\mathbb{E}[\mathcal{T}(N)] = \frac{N+3}{2}$.
   - Choose $C = 1$ and $N_0 = 1$. For all $N \geq N_0$:

     $$
     \begin{aligned}
     0 \leq \mathbb{E}[\mathcal{T}(N)] &= \frac{N+3}{2} \\
                                      &\leq N \quad (\text{since } N+3 \leq 2N \text{ for } N \geq 1) \\
                                      &= C \cdot g(N)
     \end{aligned}
     $$

   - This demonstrates that $\mathbb{E}[\mathcal{T}(N)] = \frac{N+3}{2}$ is
     bounded above by $C \cdot g(N)$, confirming that the average case time
     complexity of the linear search algorithm is $\mathcal{O}(N)$.
   - This shows that the average time complexity of linear search grows linearly
     with the size of the array $N$ and is bounded by a linear function of $N$.

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
  - $\mathcal{O}(\frac{N}{2})$
  - $\mathcal{O}(1)$
* - Element is not in the list
  - $\mathcal{O}(N)$
  - $\mathcal{O}(N)$
  - $\mathcal{O}(N)$
```

### Space Complexity

#### Input Space Complexity

- **Definition**: Input space complexity refers to the space used by the inputs
  to the algorithm.
- **Linear Search Context**: For linear search, the primary input is the list
  $\mathcal{A}$ of size $N$.
- **Analysis**: Since the list $\mathcal{A}$ is an essential input to the
  algorithm and occupies space proportional to its size, the input space
  complexity is directly related to the length of this list.
- **Space Complexity**: Thus, the input space complexity is $\mathcal{O}(N)$,
  where $N$ is the number of elements in $\mathcal{A}$.

#### Auxiliary Space Complexity

- **Definition**: Auxiliary space complexity accounts for the extra or temporary
  space used by an algorithm, excluding the space taken by the inputs.
- **Linear Search Context**: In linear search, the only additional space used is
  for a few variables, such as the index variable (and possibly a boolean flag).
- **Analysis**: The space required for these variables does not scale with the
  size of the input $N$; rather, it remains constant.
- **Space Complexity**: Therefore, the auxiliary space complexity for linear
  search is $\mathcal{O}(1)$.

#### Total Space Complexity

- **Definition**: Total space complexity combines both the input and auxiliary
  space complexities.
- **Linear Search Context**: The total space used by the algorithm includes the
  space for the list $\mathcal{A}$ and the constant extra space for the
  variables is just $\mathcal{O}(N)$.

### Implementation (Recursive)

```{code-cell} ipython3
def unordered_sequential_search_recursive(
    container: Iterable[T], target: T, index: int = 0
) -> int:
    """Recursive implementation of unordered Sequential Search."""
    if len(container) == 0:  # if not container is also fine
        return -1  # not found

    if container[0] == target:  # this is base case
        return index  # found

    # notice we increment index by 1 to mean index += 1 in the iterative case
    return unordered_sequential_search_recursive(
        container[1:], target, index + 1
    )  # recursive case

unordered_list = [1, 2, 32, 8, 17, 19, 42, 13, 0]

print(unordered_sequential_search_recursive(unordered_list, -1)) # smaller than smallest element
print(unordered_sequential_search_recursive(unordered_list, 45)) # larger than largest element
print(unordered_sequential_search_recursive(unordered_list, 13)) # in the middle
```

Let's see if our implementation obeys the 3 Laws of Recursion
({prf:ref}`axiom_three_laws_of_recursion`).

We need to shrink our `container` list from $n$ all the way down, and at the
same time, keep track of our `index` to point to the correct index of the
`container`.

1. We have two base cases: - in `lines 5-6`, we first check if the list is
   empty, if it is, means we reached till the end of the list and have not found
   the `target` element, and hence return `-1`. - in `lines 8-9`, if the list's
   first element is the `target`, then return the `index` since we found it.
2. Has our recursive algorithm change its state and move towards our base case?
   Yes, because after each function call at `lines 12-14`, we slice our list by
   `[1:]`, which means we drop the first element, and move on to check if the
   "next" element is our `target`. Here, we also need to increment `index` by 1
   since we need to recover the index if we found the `target`.
3. This is a recursive algorithm because the function calls itself at
   `lines 12-14`.

```{admonition} Tip
:class: tip

Time to revisit this recursion for revision, especially understand how
recursion is stacking function calls and popping it later.

I also think converting from an iterative solution to recursive is easier
than just thinking of recursion straight. You just need to observe
what variables are changing in **states** in iterative,
and try to do the same to its recursive counterpart.
```

Using Python Tutor to visualize recursive calls
[here](https://pythontutor.com/render.html#code=def%20f%28container,%20target,%20index%3D0%29%3A%0A%20%20%20%20if%20len%28container%29%20%3D%3D%200%3A%20%20%23%20if%20not%20container%20is%20also%20fine%0A%20%20%20%20%20%20%20%20return%20-1%20%20%23%20not%20found%0A%0A%20%20%20%20if%20container%5B0%5D%20%3D%3D%20target%3A%20%20%23%20this%20is%20base%20case%0A%20%20%20%20%20%20%20%20return%20index%20%20%23%20found%0A%0A%20%20%20%20%23%20notice%20we%20increment%20index%20by%201%20to%20mean%20index%20%2B%3D%201%20in%20the%20iterative%20case%0A%20%20%20%20return%20f%28container%5B1%3A%5D,%20target,%20index%20%2B%201%29%20%20%23%20recursive%20case%0A%20%20%20%20%0Aunordered_list%20%3D%20%5B1,%202,%2032,%208,%2017,%2019,%2042,%2013,%200%5D%0Aprint%28f%28unordered_list,%2013%29%29&cumulative=false&curInstr=0&heapPrimitives=nevernest&mode=display&origin=opt-frontend.js&py=3&rawInputLstJSON=%5B%5D&textReferences=false).

## Ordered Sequential Search

Previously, we showed how to perform sequential search on a list, which does not
assumes order.

We noticed that when the item is not in the list, the time complexity is
$\mathcal{O}(n)$, because we need to check every element in the list. This can
be alleviated if we assume that the list is ordered, and we can stop searching
when we reach an element that is greater than the element we are searching for.

For now, we will assume the list contains a list of integers, but this can be
generalized to other data types through mapping. For example, we can map the
alphabet to a list of integers, and then perform ordered sequential search on
the list of integers.

### Algorithm (Iterative)

```{prf:algorithm} Basic Ordered Linear Search Algorithm (Iterative)
:label: basic_ordered_linear_search_iterative

Given an ordered list $L$ of $n$ elements with values or records $L_0, L_1, ..., L_{n-1}$
such that $L_0 \leq L_1 \leq ... \leq L_{n-1}$, and target value $T$, the following subroutine uses ordered linear search to find the index of the target $T$ in $L$.

1. Set $i$ to 0.
2. If $L_i = T$, the search terminates successfully; return $i$. Else, go to step 3.
3. If $L_i > T$, the search terminates unsuccessfully; return $-1$.
```

### Implementation (Iterative)

```{code-cell} ipython3
def ordered_sequential_search(container: Iterable[T], target: T) -> Tuple[bool, int]:
    """Sequential search for ordered container."""
    is_found = False  # a flag to indicate so your return is more meaningful
    index = 0
    for item in container:
        if item == target:
            is_found = True
            return is_found, index
        index += 1
        if item > target:
            return is_found, -1
    # do not forget this if not if target > largest element in container, this case is not covered
    return is_found, -1
```

The reason for not using `enumerate` to get the index of a number in a list when
iterating is to minimize the usage of in-built functions.

```{code-cell} ipython3
ordered_list = [0, 1, 2, 8, 13, 17, 19, 32, 42]
print(ordered_sequential_search(ordered_list, -1)) # smaller than smallest element
print(ordered_sequential_search(ordered_list, 45)) # larger than largest element
print(ordered_sequential_search(ordered_list, 13)) # in the middle
```

#### Time Complexity

Note that for ordered sequential search, the time complexity does not change for
the case when the item is in the list.

However, for the case when the item is not in the list, we have our best case
scenario to be $\mathcal{O}(1)$, because upon checking our first element, and if
the first element is already greater than the element we are searching for, then
we can stop searching and return `False`.

For the worst case scenario, it is still $\mathcal{O}(n)$ since we have to check
every element in the list.

But, for the average case, it is now $\mathcal{O}(\frac{n}{2})$, because we can
stop searching when we reach an element that is greater than the element we are
searching for.

```{list-table} Time Complexity of Ordered Sequential Search
:header-rows: 1
:name: ordered_sequential_search_time_complexity

* - Case
  - Worst Case
  - Average Case
  - Best Case
* - Element is in the list
  - $\mathcal{O}(n)$
  - $\mathcal{O}(\frac{n}{2})$
  - $\mathcal{O}(1)$
* - Element is not in the list
  - $\mathcal{O}(n)$
  - $\mathcal{O}(\frac{n}{2})$
  - $\mathcal{O}(1)$
```

#### Space Complexity

Similarly, the space complexity is still $\mathcal{O}(1)$.

## Further Readings

- [GeeksforGeeks Linear Search](https://www.geeksforgeeks.org/linear-search/)
- [Runestone Academy Sequential Search](https://runestone.academy/ns/books/published/pythonds/SortSearch/TheSequentialSearch.html)
- [Wikipedia Linear Search](https://en.wikipedia.org/wiki/Linear_search)
- [Stack Overflow Recursive Linear Search](https://stackoverflow.com/questions/4295608/recursive-linear-search-returns-list-index)
- [Ozaner Sequential Search](https://ozaner.github.io/sequential-search/)
