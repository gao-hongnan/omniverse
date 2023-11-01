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

# Koko Eating Bananas

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![Question Number](https://img.shields.io/badge/Question-875-blue)](https://leetcode.com/problems/koko-eating-bananas/)
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)
![Tag](https://img.shields.io/badge/Tag-Array-orange)
![Tag](https://img.shields.io/badge/Tag-BinarySearch-orange)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

import rich
from IPython.display import HTML, display

from typing import List
import math

import sys
from pathlib import Path
parent_dir = str(Path().resolve().parents[2])
sys.path.append(parent_dir)

from utils.testing.test_decorator import TestFramework
```

## Learning Objectives

## Problem Statement

Koko loves to eat bananas. There are `N` piles of bananas, the `n-th` pile has
piles `piles[n]` bananas. The guards have gone and will come back in `h` hours.

Koko can decide her bananas-per-hour eating speed of `k`. Each hour, she chooses
some pile of bananas and eats `k` bananas from that pile. If the pile has less
than `k` bananas, she eats all of them instead and will not eat any more bananas
during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before
the guards return.

Return the _minimum_ integer `k` such that she can eat all the bananas within
`h` hours.

## Core Concepts, Analogies, and Problem Significance

The **Koko Eating Bananas** problem serves as a captivating example of
[optimization problems](https://en.wikipedia.org/wiki/Optimization_problem) in
computer science. At a high level, it tackles the challenge of resource
allocation under time constraints. The narrative may involve a monkey and her
bananas, but the core issue is universally relatable: _how to perform a set of
tasks in the most efficient manner possible given limited time?_

If we peel back the layers of this seemingly whimsical problem, we find a
scenario not unlike those encountered in areas like _network bandwidth
optimization_,
[job scheduling](https://en.wikipedia.org/wiki/Job_shop_scheduling) in
multi-core processors, or even time-management strategies in day-to-day
activities. The bananas can be thought of as data packets needing to be
processed, and Koko's eating speed is akin to a processor's clock speed.

The key attraction of this problem lies in its deceptive simplicity. It begins
as a straightforward question yet evolves into a puzzle requiring a nuanced
understanding of algorithms, particularly **binary search**. In solving this
problem, one isn't merely finding the minimum speed at which a fictional monkey
eats bananas but rather learning to make efficient decisions in constrained
environments.

For readers interested in broader contexts, this problem can be viewed as a
specific case within
[combinatorial optimization](https://en.wikipedia.org/wiki/Combinatorial_optimization)
and [decision theory](https://en.wikipedia.org/wiki/Decision_theory). Both are
fields that provide general frameworks for making optimal choices from a finite
set of possibilities.

Solving this problem doesn't just offer a method for optimizing a peculiar
banana-eating scenario. It provides a framework for tackling a wide range of
real-world optimization challenges. Thus, understanding its solution has both
_theoretical_ and _practical_ significance.

### Analogy: Job Scheduling in Data Centers

Imagine a **[data center](https://en.wikipedia.org/wiki/Data_center)** where
multiple tasks (akin to the piles of bananas) need to be processed by a cluster
of servers (akin to Koko). The task of the data center administrator (akin to
the algorithm you're trying to design) is to determine the **minimum processing
power** (_speed_) each server must have to complete all tasks within a given
time frame (_hours_).

#### Breaking it Down

1. **Tasks in Data Center**: These are equivalent to the piles of bananas. Each
   task could require different amounts of processing, just as piles can have
   different numbers of bananas.

2. **Processing Power of Servers**: This is akin to the _speed_ $k$ at which
   Koko eats bananas. The faster the processing power, the quicker the tasks get
   done.

3. **Time Frame**: Just as Koko has a deadline $h$, the data center has a
   _service level agreement_
   ([SLA](https://en.wikipedia.org/wiki/Service-level_agreement)) specifying the
   maximum time in which all tasks must be completed.

4. **Optimization**: The challenge lies in finding the _minimum server speed_
   that will allow all tasks to be completed within the time stipulated by the
   SLA, much like finding the _minimum_ $k$ in our problem.

This is a **real-world issue** with significant implications. Inefficient job
scheduling can lead to increased electricity costs, lower throughput, and
ultimately, less satisfied clients. Thus, solving this problem efficiently has
both **theoretical** and **practical significance**.

By understanding the Koko Eating Bananas problem, you gain insights into how you
might approach job scheduling optimization in a data center, a concept that has
far-reaching applications in the world of
**[cloud computing](https://en.wikipedia.org/wiki/Cloud_computing)** and
**[distributed systems](https://en.wikipedia.org/wiki/Distributed_computing)**.

## Problem Intuition

In any computational or mathematical endeavor, **intuition** serves as the
_compass_ that guides the journey from problem formulation to solution. While
**assumptions** lay the groundwork and **constraints** define the limits,
intuition enlightens us on the **_why_** behind a problem. It acts as the mental
model that helps us navigate the complexities, making the abstract concrete and
the intractable approachable.

Understanding the intuition behind a problem is akin to grasping the motivations
behind a story—it provides _context_, illuminates _significance_, and reveals
_underlying patterns_. It shapes our strategic approach, helping us decide which
algorithms to employ or what data structures would be most effective.

In essence, intuition corresponds to the **_why_** that breathes life into the
**_how_** and **_what_** of problem-solving. Just as a seasoned chess player
instinctively knows which moves lead to victory, a nuanced understanding of
problem intuition equips us with the foresight to make **informed decisions**
and **optimize solutions**.

### A Naive Approach

The **straightforward method** to solve this problem is to initialize $k = 1$
and incrementally test each value of $k$ to see if Koko can finish all the
bananas within $h$ hours.

**Why?**

This makes sense because we are interested in finding the **_minimum $k$_** that
allows Koko to meet the time constraint. The algorithm proceeds by _iteratively
increasing $k$_ and stops at the **first $k$** that allows Koko to consume all
the bananas in $h$ or **_fewer_** hours. This "_first_ $k$" will be the
**minimum $k$** satisfying the given condition.

This **naive approach** has a time complexity of
$\mathcal{O}(k_{\text{min}} \times N)$, where $k_{\text{min}}$ is the _smallest
speed_ at which Koko can consume all the bananas within $h$ hours, and $N$ is
the number of piles. Since $k_{\text{min}}$ is not known in advance and could be
large, this approach can be **computationally expensive** for large datasets or
**tight time constraints**. Therefore, the **crux of the problem** is to find
this*optimal $k$* without resorting to such a linear search through all
potential speeds.

### Reducing the Search Space

The observant reader should notice that actually the $k_{\text{min}}$ is
**_upper bounded_** by the **maximum number of bananas in a pile**:

$$
k_{\text{min}} \leq \max(\text{piles})
$$

**Why?**

Because we know from the _constraints_ that `piles.length <= h`. This is an
important observation because it allows us to **_reduce the search space_** from
$[1, \infty)$ to $[1, \max(\text{piles})]$. We will always yield a solution if
Koko is allowed to eat at a speed of $\max(\text{piles})$ bananas per hour. She
will always be able to finish all the bananas in $h$ hours or fewer. Therefore,
we can **_discard all speeds greater than $\max(\text{piles})$_**. So our time
complexity for the naive approach becomes
$\mathcal{O}(\max(\text{piles}) \times N)$ since we will at most iterate through
$\max(\text{piles})$ speeds.

## Example

In this section, we go through some small example(s) to illustrate the problem.

### Example 1

## Assumptions and Constraints

The relationship between assumptions and constraints in a research or
problem-solving context is akin to the interplay between axioms and theorems in
a mathematical framework. Assumptions serve as the foundational elements upon
which the rest of the work is constructed. They provide the initial conditions
or premises that guide the problem-solving approach. On the other hand,
constraints are often the derived limitations or boundary conditions that
naturally arise from these assumptions.

```{prf:remark} Externally Derived Constraints
:label: 875-koko-eating-bananas-externally-derived-constraints

It's important to understand that constraints can also be externally imposed or
arise from specific practical considerations, rather than being directly derived
from the problem's assumptions. In such cases, these constraints serve as
additional rules or limitations that must be adhered to when seeking a solution.
```

### Assumptions

Assumptions set the stage for problem-solving by defining the initial
conditions, parameters, or rules that are accepted without direct evidence. They
simplify complex situations, making them more manageable and tractable for
analysis or computational modeling.

In the **Koko Eating Bananas** problem, we make the following assumptions:

1. Koko can only eat a constant number of bananas per hour (`k`), from a single
   pile.
2. Koko will start a new pile only after finishing the current one.
3. If a pile has fewer bananas than `k`, Koko will take less than an hour to
   finish that pile but won't start another pile within the same hour.
4. Point 3 implicitly implies that Koko will not eat any more bananas from other
   piles in the same hour even if she finishes eating the current pile in less
   than an hour.
5. Koko has a fixed number of hours (`h`) to complete eating all the bananas.
6. We assume there exists a solution to the problem. In other words, we assume
   that there exists a speed `k` such that Koko can eat all the bananas in `h`
   hours or fewer.

### Constraints

These assumptions lead to certain constraints:

1. Minimum Speed: Koko can't have a speed of zero; she must eat at least one
   banana per hour.
2. Time Constraint: Koko has only `h` hours, a hard deadline to finish eating
   all bananas.
3. Integer Hours: Time is quantized in hours. Even if Koko takes less than an
   hour to finish a pile, she can't start another one within the same hour.
4. The number of piles is less than or equal to the number of hours. This is
   because if the number of piles is greater than the number of hours, then it
   is impossible for Koko to eat all the bananas in `h` hours or fewer.

In addition, leetcode provides the following constraints:

- $1 \leq \text{piles.length} \leq 10^4$
- $\text{piles.length} \leq h \leq 10^9$
- $1 \leq \text{piles[i]} \leq 10^9$

The additional constraints from LeetCode serve the following purposes:

1. **Computational Feasibility**: Limiting the array length and the value of `h`
   ensures that the problem can be solved within reasonable computational time,
   given the algorithmic techniques that candidates are expected to use.

2. **Problem Scope**: By setting minimum and maximum values, they define the
   problem's scope more precisely, thereby allowing for standardized assessment
   of solutions.

3. **Avoiding Edge Cases**: Constraints like $1 \leq \text{piles[i]} \leq 10^9$
   help in removing trivial or degenerate cases, focusing the problem on
   meaningful scenarios.

4. **Algorithmic Complexity**: The constraints provide bounds that can guide the
   choice of algorithm, helping one discern whether an $\mathcal{O}(N \log N)$
   or $\mathcal{O}(N)$ algorithm is appropriate, for example.

## Test Cases

### Standard Cases

1. **Multiple Piles, Limited Time**

   - `Input: piles = [30, 11, 23, 4, 20], h = 6`
   - `Output: 23`
   - **Explanation**: Koko needs a minimum speed of 23 bananas/hour to finish
     all bananas in 6 hours or fewer.

### Edge Cases

1. **Single Pile and Minimum Possible Input**

   - `Input: piles = [1], h = 1`
   - `Output: 1`
   - **Explanation**: This is the simplest case where the pile has the minimum
     number of bananas and Koko has the least amount of time. The speed is also
     1 banana/hour.

2. **Highly Skewed Piles**

   - `Input: piles = [100, 1, 1, 1], h = 4`
   - `Output: 100`
   - **Explanation**: The large pile dictates Koko's minimum eating speed. Koko
     needs to eat at 100 bananas/hour to finish all bananas in 4 hours.

3. **Large Data Set**

   - `Input: piles = [8]*10^6, h = 10^5`
   - `Output: 8`
   - **Explanation**: This tests the algorithm's ability to handle **large-scale
     scenarios** efficiently. Koko would need to eat at least 8 bananas/hour.

## Theoretical Best Time/Space Complexity and Space-Time Tradeoff

### Theoretical Best Time Complexity

In the "Koko Eating Bananas" problem, the goal is to minimize the eating speed
$k$ such that all bananas are eaten within $h$ hours. One common algorithmic
approach to solve this problem is using binary search on $k$.

The binary search would operate on the speed range $[1, \max(\text{piles})]$,
and for each candidate speed, we need to traverse all the piles to check if Koko
can finish eating in $h$ hours. This traversal takes $\mathcal{O}(N)$ time where
$N$ is the length of the `piles` array. Thus, the best theoretical time
complexity for solving this problem would be $\mathcal{O}(N \log M)$, where $M$
is the maximum number of bananas in a pile.

### Theoretical Best Space Complexity

For the binary search algorithm, we only need a constant amount of extra space
to store variables such as the low, high, and mid points of the search, as well
as a counter for the total hours Koko would take for a given $k$. Therefore, the
space complexity is $\mathcal{O}(1)$, which is the best you can achieve for this
problem assuming that the input size is not counted towards the space
complexity.

### Space-Time Tradeoff

In this specific problem, there's limited scope for a space-time tradeoff. The
time complexity is primarily determined by the need to iterate over all the
piles for each candidate $k$, and this is not something that can be pre-computed
or stored to save time later. Similarly, the space complexity is already at its
theoretical minimum $\mathcal{O}(1)$, so there isn't room for optimization by
using more space.

To sum up, this problem doesn't offer much room for a space-time tradeoff, given
its constraints and the nature of its optimal solution.

## Mathematical Formulation

First, we will introduce some mathematical notations and definitions to help us
formulate the problem and solution in a more concise manner.

---

Given a sequence of piles,

$$
\mathcal{P} = \left\{ p_1, p_2, \ldots, p_N \,|\, 1 \leq n \leq N \right\},
$$

each containing a non-negative integer number of bananas, and a positive integer
$h$ representing the total number of hours available, our goal is to find the
minimum constant integer eating speed $k$ such that Koko can finish all the
bananas in $\mathcal{P}$ within $h$ hours.

For reasons meantioned earlier and in the constraints, we can specify the search
space for $k$ as:

$$
\mathcal{K} = \left\{ k \in \mathbb{Z}^+ \,|\, 1 \leq k \leq \max_{n \in [1,N]} p_n \right\}.
$$

For convenience, we also define a shorthand notation for the maximum number of
bananas in a pile:

$$
M \stackrel{\text{def}}{=} \max_{n \in [1,N]} p_n
$$

In this setting, we define the time $\mathcal{T}(p_n, k)$ it takes Koko to eat a
pile $p_n$ at speed $k$ as:

$$
\mathcal{T}(p_n, k) = \left \lceil \frac{p_n}{k} \right \rceil
$$

where $\lceil \cdot \rceil$ is the ceiling function, which rounds a real number
$x$ **_up_** to the nearest integer. For example, $\lceil 2.3 \rceil = 3$.

```{prf:remark} Why ceiling?
:label: 875-koko-eating-bananas-why-ceiling

Because Koko can only eat a constant number of bananas per hour, so she will
always take at least $\lceil \frac{p_n}{k} \rceil$ hours to finish a pile $p_n$.
Consider $k=3$ and the pile $p_n=5$. Then Koko will take $2$ hours to finish
this pile. This is because $\frac{5}{3} = 1.6667$, which rounds up to $2$ (of
course you cannot round down).
```

Consequently, the total time $\mathcal{H}(k)$ required to eat all the bananas in
$\mathcal{P}$ at speed $k$ can be expressed as:

$$
\mathcal{H}(k) = \sum_{n=1}^{N} \mathcal{T}(p_n, k)
$$

The optimization problem can thus be formally stated as:

$$
\begin{aligned}
& \text{minimize}  && k \in \mathcal{K} \\
& \text{s.t.}      && \mathcal{H}(k) \leq h \\
& \text{where}     && k, h \in \mathbb{Z}^+, \; \mathcal{K} \subset \mathbb{Z}^+, \; 1 \leq k \leq M
\end{aligned}
$$

or equivalently:

$$
\begin{aligned}
& k^* = \arg\min_{k \in \mathcal{K}} k \\
& \text{s.t.}  \quad & \mathcal{H}(k) \leq h \\
& \text{where} \quad & k, h \in \mathbb{Z}^+, \; \mathcal{K} \subset \mathbb{Z}^+
\end{aligned}
$$

## Definitions

The definitions of this problem are fairly straightforward and self-contained.
The previous section already introduced most of the necessary definitions.

- $p_n$ is the number of bananas in the $n$th pile
- $k$ is the speed at which Koko eats bananas
- $h$ is the number of hours Koko has to eat all the bananas
- $n$ is the number of piles
- $\left\lceil \frac{p_n}{k} \right\rceil$ is the number of hours it takes Koko
  to eat the $n$th pile
- $\sum_{n=1}^{N} \left\lceil \frac{p_n}{k} \right\rceil$ is the total number of
  hours it takes Koko to eat all the bananas.
- $M$ is the maximum number of bananas in a pile $\max_{n \in [1,N]} p_n$. This
  is the upper bound of the search space for $k$.

We include two more definitions here for completeness since we will use it
later.

- **Feasibility Function $\mathcal{F}(\mathcal{P}, k, h)$**: A binary function
  indicating whether it is possible to eat all bananas within $h$ hours at speed
  $k$.

  $$
  \mathcal{F}(k, \mathcal{P}, h) =
  \begin{cases}
  1, & \text{if } \mathcal{H}(k) \leq h \\
  0, & \text{otherwise}
  \end{cases}
  $$

- **Optimal Eating Speed $k^*$**: The minimum constant integer eating speed that
  satisfies the problem's constraints.

  $$
  k^* = \min \{ k \in \mathcal{K} \,|\, \mathcal{H}(k) \leq h \}
  $$

## Solution: Binary Search

In solving this problem, the objective is to efficiently identify the minimum
eating speed $k$ that allows Koko to consume all the bananas in $h$ hours. A
straightforward method is to iterate over all potential speeds $k$ in the range
$1 \leq k \leq M$[^max-of-piles-is-M] and evaluate if $\mathcal{H}(k) \leq h$,
where $\mathcal{H}(k)$ represents the hours needed to eat all bananas at speed
$k$. This naive approach results in a time complexity of
$\mathcal{O}(N \times M)$, which could be computationally prohibitive when $N$,
the number of piles, **_and/or_** $M$, the maximum size of a pile, is large.
However, we can improve this to $\mathcal{O}(N \log M)$ by employing a binary
search technique.

### Solution Intuition

The first question that arises is, how can we be sure that binary search is an
appropriate technique for solving this problem? The answer often depends on
one's level of experience. However, beyond experience, there is an underlying
intuition that experienced programmers have developed. This intuition allows
them to instantly recognize problems that are well-suited for binary search. So,
what constitutes this intuition, and how can it be developed? In general, if a
specific type of algorithm adheres to a recognizable pattern or template, then
that pattern serves as the intuition behind choosing that algorithm.

Main intuition following the binary search framework.

> Minimize $k$ such that koko can _eat all the bananas within $h$ hours_.

#### The Precondition for Binary Search

The main precondition for applying binary search to this problem is the property
of **monotonicity** in the **feasibility**
function[^precondition-of-binary-search]. If a **feasibility** function
$\mathcal{F}(\cdot)$ is **monotonic**, then the problem lends itself well to a
binary search solution. Here, the feasible function $\mathcal{F}(\cdot)$ is a
boolean function that returns either `True` or `False`, or `1` or `0` since we
want to talk about monotonicity, which is better understood if the output is a
"number".

The second precondition is implicit. The search space $\mathcal{K}$ must be
inherently **ordered** in which you can sort it (sortable). This is because
binary search requires the search space to be sorted.

Consequently, the domain of the feasibility function $\mathcal{F}(\cdot)$ is the
search space $\mathcal{K}$, and the range is $\{0, 1\}$.

But why is having this monotonicity property important? Why is it that we need
this feasibility function to be monotonic? The reason is that we want to be able
to discard half of the search space at each step. This is the key to binary
search. The connection is that monotonicity implies there is a tipping point.
There exists a $k$ such that if $\mathcal{F}(k) = 1$, then $\mathcal{F}(k') = 1$
for all $k' > k$.

Understanding the problem through the lens of feasibility and monotonicity lends
itself to the concept of a _sorted boolean
array_[^first-true-in-a-sorted-boolean-array] where we can easily apply binary
search!

The only difficulty lies in defining the feasibility function
$\mathcal{F}(\cdot)$.

#### Framing the Problem as a Sorted Boolean Array

##### Feasibility Function

First, we define the feasibility function formally.

```{prf:definition} Feasibility Function
:label: 875-koko-eating-bananas-feasibility-function

The feasibility function $\mathcal{F}$ is a mapping from a decision variable
$k \in \mathcal{K}$ and a set of parameters $\Theta$ to a binary outcome,
subject to a set of constraints $C$.

$$
\mathcal{F}: \mathcal{K} \times \Theta \to \{0, 1\}
$$

Here, $\mathcal{K}$ is the domain of possible values for the decision variable,
$\Theta$ represents other relevant parameters, and $C$ represents the set of
constraints.

The function is defined as:

$$
\mathcal{F}(k, \Theta) = \begin{cases}
1, & \text{if } C(k, \Theta) \text{ is true} \\
0, & \text{otherwise} \end{cases}
$$

Here, $C(k, \Theta)$ is a function that returns `True` ($1$) or `False` ($0$)
based on whether all constraints are satisfied for the given $k$ and $\Theta$.
```

For our problem, the decision variable is $k$, the set of parameters is
$\Theta = \{ \mathcal{P}, h \}$, and the constraints is just whether Koko can
finish eating all the bananas in $h$ hours or fewer.

```{prf:definition} Feasibility Function for Koko Eating Bananas
:label: 875-koko-eating-bananas-feasibility-function-for-koko-eating-bananas

The feasibility function $\mathcal{F}$ for the Koko Eating Bananas problem is
defined as:

$$
\mathcal{F}(k, \mathcal{P}, h) = \begin{cases}
1, & \text{if } \mathcal{H}(k) \leq h \\
0, & \text{otherwise}
\end{cases}
$$

where $\mathcal{H}(k)$ is the total number of hours it takes Koko to eat all the
bananas at speed $k$.
```

This feasibility function looks reasonable, we remain to verify if this function
is monotonic on the search space $\mathcal{K}$.

##### Monotonicity

Here we first define the concept of monotonicity formally.

```{prf:definition} Monotonicity
:label: 875-koko-eating-bananas-monotonicity

In the context of a binary search problem, if the feasibility function
$\mathcal{F}: \mathcal{K} \to \{0, 1\}$ is monotonic over the search space
$\mathcal{K}$, it would satisfy one of the following conditions:

1. For all $k_1, k_2 \in \mathcal{K}$ such that $k_1 \leq k_2$, it holds that
   $\mathcal{F}(k_1) \leq \mathcal{F}(k_2)$ (Monotonically Increasing).
2. For all $k_1, k_2 \in \mathcal{K}$ such that $k_1 \leq k_2$, it holds that
   $\mathcal{F}(k_1) \geq \mathcal{F}(k_2)$ (Monotonically Decreasing).
```

We claim that the feasibility function $\mathcal{F}(k, \mathcal{P}, h)$ is
**_monotonically increasing_**. If Koko can eat all bananas in $h$ hours at a
speed $k_1$, then she can surely do so at any greater speed $k_2 > k_1$.
Note that the total number of hours to finish, $\mathcal{H}(k)$, however, is **_monotonically decreasing_** with
respect to $k$ because if Koko eats faster, she will take fewer hours to finish.

More concretely:

$$
\mathcal{F}(\mathcal{P}, k_1, h) = 1 \Rightarrow \mathcal{F}(\mathcal{P},
k_2, h) = 1 \quad \text{for all} \quad k_2 > k_1
$$

The logic

````{admonition} Proof
:class: dropdown

```{prf:proof}
We start off by a claim:

**Claim**:
$\mathcal{F}(k_1, \mathcal{P}, h) \leq \mathcal{F}(k_2, \mathcal{P}, h)$ for all
$k_1 < k_2$.

In other words, if $\mathcal{F}(k_1, \mathcal{P}, h) = 1$, then
$\mathcal{F}(k_2, \mathcal{P}, h) = 1$ for all $k_2 > k_1$.

**Proof**:

Given that $\mathcal{F}(k_1, \mathcal{P}, h) = 1$, we have:

$$
\sum_{n=1}^{N} \left \lceil \frac{p_n}{k_1} \right \rceil \leq h
$$

We aim to prove:

$$
\sum_{n=1}^{N} \left \lceil \frac{p_n}{k_2} \right \rceil \leq h
$$

Since $k_1 < k_2$, $\frac{p_n}{k_1} \geq \frac{p_n}{k_2}$ for all $n$.

This implies that
$\left \lceil \frac{p_n}{k_1} \right \rceil \geq \left \lceil \frac{p_n}{k_2} \right \rceil$
for all $n$.

Hence, if $\sum_{n=1}^{N} \left \lceil \frac{p_n}{k_1} \right \rceil \leq h$,
then $\sum_{n=1}^{N} \left \lceil \frac{p_n}{k_2} \right \rceil \leq h$.

This concludes the proof that $\mathcal{F}(k, \mathcal{P}, h)$ is monotonically
increasing with respect to $k$.
```
````

##### Translating to a First True in a Sorted Boolean Array Problem

Given the monotonic property of $\mathcal{F}$, we can effectively sort the
boolean outputs of $\mathcal{F}$ across the search space $\mathcal{K}$. That is,
as we traverse $\mathcal{K}$ from the slowest to the fastest eating speed, the
function $\mathcal{F}$ will transition from returning `False` to returning
`True`. Our objective is to find the smallest $k$ for which $\mathcal{F}$
returns `True`. Essentially, we have transformed the problem into a **_first
true in a sorted boolean array_** problem.

To rigorously connect these ideas, consider the feasibility function
$\mathcal{F}$ as a sequence
$\mathcal{F} = \{ \mathcal{F}(\mathcal{P}, k_1, h), \mathcal{F}(\mathcal{P}, k_2, h), \ldots \}$
where $k_1 < k_2 < \ldots$ are the sorted eating speeds in $\mathcal{K}$. The
sequence $\mathcal{F}$ is essentially a "Sorted Boolean Array". Given the
monotonic property of $\mathcal{F}$, $\mathcal{F}$ is a sorted boolean sequence
with a threshold point where it transitions from `False` to `True`. Finding this
point is equivalent to solving the original problem, and binary search is the
efficient way to do it.

Thus, we can see that the Koko Eating Bananas problem is fundamentally a "First
True in a Sorted Boolean Array" problem, revealed through the lens of
feasibility and its monotonic properties:

1. **Ordered Search Space**: The space of all possible eating speeds $k$ is
   well-defined and ordered (from 1 to $M$).

2. **Monotonicity of Feasibility**: The problem exhibits a monotonic property
   with respect to feasibility; if Koko can finish all the bananas in $h$ hours
   at speed $k$, she can also finish them at any speed greater than $
   k$.

3. **First True in Sorted Boolean Array**: With the above properties, one can
   treat the search space $\mathcal{K}$ as a sorted boolean array defined by the
   feasibility function $\mathcal{F}$. The problem reduces to finding the "first
   true" in this sorted boolean array.

To this end, we can reframe the earlier minimization statement:

$$
\begin{aligned}
& \text{minimize}  && k \in \mathcal{K} \\
& \text{s.t.}      && \mathcal{H}(k) \leq h \\
& \text{where}     && k, h \in \mathbb{Z}^+, \; \mathcal{K} \subset \mathbb{Z}^+, \; 1 \leq k \leq M
\end{aligned}
$$

to

$$
\begin{aligned}
& \text{minimize}  && k \in \mathcal{K} \\
& \text{s.t.}      && \mathcal{F}(k, \mathcal{P}, h) = 1 \\
& \text{where}     && k, h \in \mathbb{Z}^+, \; \mathcal{K} \subset \mathbb{Z}^+, \; 1 \leq k \leq M
\end{aligned}
$$

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

#### Whiteboarding

The domain of our search space is defined as $k \in [1, \max(\text{{piles}})]$:

- The lower bound $k = 1$ is set because Koko cannot eat 0 bananas per hour.
- The upper bound $k = \max(\text{{piles}})$ is chosen based on the constraint
  that eating at a faster rate than the largest pile is unproductive.

The problem lends itself to a binary search approach due to its inherent
monotonicity. To illustrate, consider an example with `piles = [3, 6, 7, 11]`
and $h = 8$. The initial search space spans from $k = 1$ to $k = 11$, with a
mid-point at $k = 6$.

We then evaluate whether Koko can consume all the bananas within $h = 8$ hours
at a speed of $k = 6$ bananas per hour. The answer is affirmative, requiring
`[1, 1, 2, 2]` hours for the respective piles. Consequently, all speeds
$k' > k = 6$ must also satisfy this condition. This enables us to discard the
right half of the search space, focusing our search on $k' \leq k = 6$.

Conversely, if we arrive at an insufficient speed, say $k = 3$, then all speeds
$k' < k = 3$ will also be insufficient. Therefore, we can discard the left half
of the search space, concentrating our search on $k' \geq k = 3$.

The monotonic property guarantees that if a speed $k$ enables Koko to consume
all bananas within $h$ hours, all speeds $k' > k$ will also satisfy this
condition. Hence, the binary search can be efficiently employed to halve the
search space iteratively until the minimum $k$ that meets the requirement is
found.

#### Pseudocode

````{prf:algorithm} Binary Search Pseudocode
:label: 875-koko-eating-bananas-binary-search-pseudocode

```
Algorithm Find_Optimal_Eating_Speed(P, h, M)
    Input: P = [p_1, p_2, ..., p_N] (list of piles),
           h (total hours available),
           M (maximum number of bananas in a pile)
    Output: k* (optimal eating speed)

    1: Initialize l = 1
    2: Initialize r = M

    3: while l < r do
        4:    m = floor((l + r) / 2)

        5:    if Feasibility_Function(P, m, h) == 1 then
        6:        r = m
        7:    else
        8:        l = m + 1
        9:    end if

       10: end while

    11: return l
```
````

#### Mathematical Representation

Given the search space $\mathcal{K}$ and the feasibility function
$\mathcal{F}(k, \mathcal{P}, h)$, we seek to find the optimal eating speed $k^*$
by exploring $\mathcal{K}$ via binary search.

We employ binary search in $\mathcal{K}$, the search space defined as:

$$
\mathcal{K} = \{ k \in \mathbb{Z}^+ \,|\, 1 \leq k \leq M \}
$$

The algorithm initiates with the lower bound as $l = 1$ and the upper bound as
$r = M$, and iteratively updates these bounds until $l \geq r$. Let $m_t$ be the
candidate speed at iteration $t$ calculated as
$m_t = \left\lfloor \frac{l_t + r_t}{2} \right\rfloor$.

The algorithm uses these bounds to iteratively tighten the search space.

1. **Initialization**:
    1. $l = 1$
    2. $r = M$
2. **Iterative Procedure**: For $k = m_t$:

    - Compute $\mathcal{F}(\mathcal{P}, k, h)$.
    - If $\mathcal{F}(\mathcal{P}, k, h) = 1$, then $r_{t+1} = m_t$.
    - Otherwise, $l_{t+1} = m_t + 1$.

3. **Termination**: The algorithm terminates when $l_{t+1} \geq r_{t+1}$. The
   optimal eating speed $k^* = l$.

#### Correctness

Prove the correctness of the algorithm

##### Claim

Statement claiming the correctness of the algorithm.

##### Proof

Proof showing the correctness of the algorithm.

### Implementation

```{code-cell} ipython3
class Solution:
    def feasible(self, piles: List[int], speed: int, h: int) -> bool:
        return self.total_hours_to_finish_eating(piles, speed) <= h

    def total_hours_to_finish_eating(self, piles: List[int], speed: int) -> int:
        total_hours = 0
        for pile in piles:
            hours = pile / speed      # num_bananas / speed -> hours needed to finish eating
            hours = math.ceil(hours)  # if hours = 2.5, it counts at 3
            total_hours += hours
        return total_hours

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        l, r = 1, max(piles)

        while l < r:
            m = l + (r - l) // 2
            if self.feasible(piles, speed=m, h=h):
                r = m
            else:
                l = m + 1
        return l
```

### Tests

Set of tests for validating the algorithm.

```{code-cell} ipython3
# Initialize the TestFramework class
tf = TestFramework()

minimum_speed = Solution().minEatingSpeed

@tf.describe("Testing minimum_speed function for Koko Eating Bananas")
def test_minimum_speed():

    @tf.individual_test("Multiple Piles, Limited Time")
    def _():
        tf.assert_equals(minimum_speed([30, 11, 23, 4, 20], 6), 23, "Should return 23")

    @tf.individual_test("Single Pile and Minimum Possible Input")
    def _():
        tf.assert_equals(minimum_speed([1], 1), 1, "Should return 1")

    @tf.individual_test("Highly Skewed Piles")
    def _():
        tf.assert_equals(minimum_speed([100, 1, 1, 1], 4), 100, "Should return 100")

    @tf.individual_test("Large Data Set")
    def _():
        tf.assert_equals(minimum_speed([8]*10**6, 10**5), 8, "Should return 8.")
```

### Time Complexity

Analysis of the time complexity of the solution.

### Space Complexity

#### Input Space Complexity

Analysis of the space complexity of the input.

#### Auxiliary Space Complexity

Analysis of the space complexity excluding the input and output space.

#### Total Space Complexity

Analysis of the total space complexity of the solution.

## References and Further Readings

Any useful references or resources for further reading.

[^max-of-piles-is-M]: Recall $\max(\mathcal{P}) = M$
[^precondition-of-binary-search]:
    See [Identifying Binary Search Problems](./concept.md) for a detailed
    discussion on the precondition for binary search.

[^first-true-in-a-sorted-boolean-array]:
    [**First True in a Sorted Boolean Array**](https://algo.monster/problems/binary_search_boundary)
