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

```{contents}
:local:
```

[![Question Number](https://img.shields.io/badge/Question-875-blue)](https://leetcode.com/problems/koko-eating-bananas/)
![Difficulty](https://img.shields.io/badge/Difficulty-Medium-yellow)
![Tag](https://img.shields.io/badge/Tag-Array-orange)
![Tag](https://img.shields.io/badge/Tag-BinarySearch-orange)

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

import rich
from IPython.display import HTML, display

from typing import List
import math
```

## Learning Objectives

## Problem Statement

Koko loves to eat bananas. There are `n` piles of bananas, the `i-th` pile has
piles `i` bananas. The guards have gone and will come back in `h` hours.

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
$\mathcal{O}(k_{\text{min}} \times n)$, where $k_{\text{min}}$ is the _smallest
speed_ at which Koko can consume all the bananas within $h$ hours, and $n$ is
the number of piles. Since $k_{\text{min}}$ is not known in advance and could be
large, this approach can be **computationally expensive** for large datasets or
**tight time constraints**. Therefore, the **crux of the problem** is to find
this_optimal $k$_ without resorting to such a linear search through all
potential speeds.

### Reducing the Search Space

The observant reader should notice that actually the $k_{\text{min}}$ is
**_upper bounded_** by the **maximum number of bananas in a pile**.

**Why?**

Because we know from the _constraints_ that `piles.length <= h`. This is an
important observation because it allows us to **_reduce the search space_** from
$[1, \infty)$ to $[1, \max(\text{piles})]$. We will always yield a solution if
Koko is allowed to eat at a speed of $\max(\text{piles})$ bananas per hour. She
will always be able to finish all the bananas in $h$ hours or fewer. Therefore,
we can **_discard all speeds greater than $\max(\text{piles})$_**. So our time
complexity for the naive approach becomes
$\mathcal{O}(\max(\text{piles}) \times n)$ since we will at most iterate through
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
   choice of algorithm, helping one discern whether an $\mathcal{O}(n \log n)$
   or $\mathcal{O}(n)$ algorithm is appropriate, for example.

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

    - `Input: piles = [1]*10^6, h = 10^5`
    - `Output: 10`
    - **Explanation**: This tests the algorithm's ability to handle
      **large-scale scenarios** efficiently. Koko would need to eat at least 10
      bananas/hour.

## Theoretical Best Time/Space Complexity and Space-Time Tradeoff

### Theoretical Best Time Complexity

In the "Koko Eating Bananas" problem, the goal is to minimize the eating speed
$k$ such that all bananas are eaten within $h$ hours. One common algorithmic
approach to solve this problem is using binary search on $k$.

The binary search would operate on the speed range $[1, \max(\text{piles})]$,
and for each candidate speed, we need to traverse all the piles to check if Koko
can finish eating in $h$ hours. This traversal takes $\mathcal{O}(n)$ time where
$n$ is the length of the `piles` array. Thus, the best theoretical time
complexity for solving this problem would be $\mathcal{O}(n \log m)$, where $m$
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

## Solution (Potentially Multiple)

### Intuition

Main intuition following the binary search framework.

> Minimize $k$ such that koko can _eat all the bananas within $h$ hours_.

The intuition is that the search space is from $k=1$ to $k=\max(\text{piles})$
because:

- Koko cannot eat $0$ bananas per hour, so the lower bound is $k=1$.
- Koko does not need to eat more than the maximum number of bananas in a pile
    per hour, so the upper bound is $k=\max(\text{piles})$. Because even if she
    does, there is no point as the question stated that once she finishes eating
    all the bananas in a pile, she will not eat any more bananas from other
    piles.

And why can we approach it from a binary search perspective? Consider first an
example where $k=1$ and say the `piles = [3, 6, 7, 11]` and `h=8`. Then we see
that the search space is from $k=1$ to $k=11$. We can see the mid number is
$k=6$. Then we can ask ourselves, is it possible for Koko to eat all the bananas
in $h=8$ hours if she eats at a speed of $k=6$ bananas per hour? The answer is
yes because she will take `[1, 1, 2, 2]` hours to eat all the bananas in the
piles. This is less than $h=8$ hours. So since it is a yes, we can say that all
$k' > k=6$ will also be a yes. So we can discard the right half of the search
space and only search the left half.

Conversely, if we somehow land ourselves in a situation where $k=3$, then we can
ask ourselves, is it possible for Koko to eat all the bananas in $h=8$ hours if
she eats at a speed of $k=3$ bananas per hour? The answer is no because she will
take `[1, 2, 3, 4]` hours to eat all the bananas in the piles. So in that case
we can say that all $k' < k=3$ will also be a no. So we can discard the left
half of the search space and only search the right half.

Because let's say there is a $k$ that satisfies the condition, then we can say
that all $k' > k$ will also satisfy the condition. This is because if $k$
satisfies the condition, then $k$ is the minimum speed at which Koko can eat all
the bananas in $h$.

"""Rephrase In this context, if we find a speed `k` such that Koko can eat all
the bananas within `h` hours, we know that for all speeds `k' > k`, Koko will
also be able to eat all the bananas within `h` hours because she is eating at a
faster rate. Therefore, we can discard the right side of the search space and
continue our search on the left side to find the minimum `k` that satisfies the
condition.

In the example you gave, we found that `k = 6` works, meaning Koko can eat all
the bananas in `h = 8` hours if she eats at a speed of 6 bananas per hour.
Therefore, all speeds `k' > 6` will also work. So we discard the right half of
the search space (speeds `k' > 6`) and continue our search on the left half
(speeds `k' ≤ 6`) to find the minimum speed that still allows Koko to eat all
the bananas within `h` hours.

In the binary search algorithm, we continue this process, halving the search
space at each step, until we find the minimum `k` that satisfies the condition.

The reason why we can approach this problem from a binary search perspective is
because the condition we are checking (`k` bananas per hour is enough for Koko
to eat all the bananas in `h` hours) has the property of monotonicity: if a
speed `k` is sufficient, then all speeds greater than `k` will also be
sufficient. This is exactly the kind of condition that binary search is designed
to handle efficiently. """

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

To formalize the earlier minimization problem, we can write it as:

$$
\min_{k} k \quad \text{s.t.} \quad \sum_{i=1}^{n} \left\lceil \frac{p_i}{k} \right\rceil \leq h
$$

where

- $p_i$ is the number of bananas in the $i$th pile
- $k$ is the speed at which Koko eats bananas
- $h$ is the number of hours Koko has to eat all the bananas
- $n$ is the number of piles
- $\left\lceil \frac{p_i}{k} \right\rceil$ is the number of hours it takes
    Koko to eat the $i$th pile
- $\sum_{i=1}^{n} \left\lceil \frac{p_i}{k} \right\rceil$ is the total number
    of hours it takes Koko to eat all the bananas
- $\min_{k} k$ is the minimum speed at which Koko can eat all the bananas in
    $h$ hours and is the quantity we are trying to minimize.

#### Pseudocode

Detailed description of the algorithm used to solve the problem.

#### Mathematical Representation

Math formulation

#### Correctness

Prove the correctness of the algorithm

### Claim

Statement claiming the correctness of the algorithm.

### Proof

Proof showing the correctness of the algorithm.

### Implementation

```{code-cell} ipython3
class Solution:
    def feasible(self, piles: List[int], speed: int, h: int) -> bool:
        return self.total_hours_to_finish_eating(piles, speed) <= h

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        l, r = 1, max(piles)

        while l < r:
            m = l + (r-l) // 2
            if self.feasible(piles, speed=m, h=h) is True:
                r = m
            else:
                l= m +1
        return l



    # speed = banana/hour

    def total_hours_to_finish_eating(self, piles: List[int], speed: int) -> int:
        total_hours = 0
        for pile in piles:
            # pile = num of bananas
            hours = pile / speed # recover time
            # if hours = 2.5, it counts at 3
            hours = math.ceil(hours)

            total_hours += hours
        return total_hours
```

### Tests

Set of tests for validating the algorithm.

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
