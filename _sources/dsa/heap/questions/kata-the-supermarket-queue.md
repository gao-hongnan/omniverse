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

# The Supermarket Queue

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from IPython.core.display import display, HTML

from typing import List
```

The **Supermarket Queue** problem serves as an illustrative example of a
[**multi-server queuing system**](https://en.wikipedia.org/wiki/M/M/c_queue) and
falls under the realm of
[**load balancing**](<https://en.wikipedia.org/wiki/Load_balancing_(computing)>).
It shares similarities with real-world challenges, such as optimizing the
allocation of computational resources or job scheduling in a multi-core
processor. In this scenario, the **customers** and the **checkout tills** (as we
shall see later) can be analogized to **jobs** and **processors**, respectively.

The essence of the problem lies in distributing a set of tasks (customers in
queue) across multiple service providers (checkout tills) in such a way that
_minimizes_ the total time required for _task completion_. Such problems are
ubiquitous in various domains, from telecommunications and cloud computing to
manufacturing and, of course, retail.

This is a real-world issue with substantial implications. Poor queue management
can lead to _customer dissatisfaction_, increased wait times, and ultimately, a
loss in revenue. Thus, solving this problem efficiently has both theoretical and
practical significance.

For those interested in digging deeper into the general concept, the problem is
a subset of [**queuing theory**](https://en.wikipedia.org/wiki/Queueing_theory),
a branch of mathematics that studies the dynamics of queues or waiting lines.

## Learning Objectives

...

## Problem Statement

At a supermarket, there is a queue for the self-checkout tills. Your task is to
write a function to calculate the total time required for all the customers to
check out.

The inputs are:

-   `customers`: An array of positive integers representing the queue. Each
    integer represents a customer, and its value is the amount of time they
    require to check out.
-   `N`: A positive integer representing the number of checkout tills.

The output is:

-   An integer representing the total time required.

## Problem Intuition

### Core Idea

The supermarket queue problem can be thought of as a specific instance of a
broader class of optimization problems dealing with resource allocation.

In real life, when multiple checkout tills are available, people don't line up
for each till one by one; rather, they tend to go to the till where they expect
to wait the least.

Translating this to our problem, each checkout till can be seen as a 'worker'
and each customer as a 'task' that takes a certain amount of time to complete.
The objective is to minimize the time it takes for the 'last' task to be
completed, which, in turn, will minimize the overall time for all tasks.

This problem exemplifies the underlying principle of
[**load balancing**](<https://en.wikipedia.org/wiki/Load_balancing_(computing)>),
where the tasks need to be efficiently distributed among the workers. The
challenge here also involves making an optimal or near-optimal "greedy choice"
at each step to achieve the global objective.

### Analogy

Beyond being a real-world scenario, this problem serves as an analogy to many
other resource allocation issues, such as distributing computational tasks among
servers in a data center. In other words, the problem is simple enough as it
relates to real life.

### Problem Categorization

This is fundamentally an optimization problem, falling under the category of
**combinatorial optimization**. Specifically, the objective function aims to
minimize the total time required for all tasks (customers) to be completed by
the available resources (checkout tills).

## Example

Let's look at a few examples.

### Example 1: One Till

First, let's consider the case where there is only one checkout till.

```python
customers = [5, 3, 4]
N         = 1
```

In this case, the total time required is simply the sum of the times required
for each customer to check out:

```python
5 + 3 + 4 = 12
```

Why? Since there is only one checkout till, the checkout process is strictly
sequential, each customer must wait for the customer in front to finish checking
out before they can begin.

### Example 2: Two Tills

Now, let's consider the case where there are two checkout tills.

```python
customers = [10, 2, 3, 3]
N         = 2
```

Consider that this is a real life scenario, and imagine that the checkout time
is displayed above each customer's head (in real life you would probably gauge
this by the number of items in their cart). As the first customer, with a
checkout time of 10, begins checking out at till 1, the second customer, with a
checkout time of 2, begins checking out at till 2. This is a natural consequence
of the fact that there are two tills, and the second customer does not have to
wait for the first customer to finish checking out before they can begin. What
is the deciding factor is that the third customer, with a checkout time of 3,
must now choose between the two tills. We further assume that the customer will
choose the till with minimum waiting time. Since the second customer has a
checkout time of 2, which is lesser than the checkout time of the first
customer, the third customer will choose to check out at till 2. Now, the fourth
and final customer, with a checkout time of 3, will have to choose between the
two tills. Again, the fourth customer realises that even though there are two
customers in till 2, the total checkout time at till 2 is lesser than the total
checkout time at till 1. Hence, the fourth customer will choose to check out at
till 2.

We see both tills, and take the maximum of the total checkout times at each
till:

```python
max(10, 2 + 3 + 3) = 10
```

In real life, you cannot see the number on top of the head, more realistic way of describing the above example
is that once the first two customers with 10 and 2 occupied the two tills. The third
customer will not know which till gets done first (ignoring hunch and eyeball), so he
will just wait, and it turns out that the second till gets done first, so he will go there.
Now the fourth customer will continue waiting, and it turns out till 2 again finishes first,
so he will go there.

## Intuition Revisited (TBD on what to name or put this section)

### Single Queue, Multiple Servers

Imagine the supermarket during a busy hour: the tills represent resources, and
the goal is to allocate these resources most efficiently. In our scenario, each
customer represents a task that requires a certain amount of time (resource) to
complete.

With multiple tills available, we're dealing with a classic "single queue,
multiple server" system. The intuition behind the solution stems from a
fundamental principle of queue theory: when multiple servers are available,
tasks (customers) should be allocated to servers (tills) in a way that minimizes
the overall completion time. This is akin to load balancing in computer systems,
where work is distributed among processors to optimize total processing time.

### Minimizing Wait Times

Customers naturally gravitate towards the path of least resistance: the till
with the shortest queue. However, it's not just the number of customers at each
till but the total projected service time, which influences their decision. A
till with two customers each with only two items might be preferable over a till
with a single customer unloading an entire cart.

Our approach simulates this decision-making process. As each new customer
arrives, they assess the tills and choose the one with the least cumulative
waiting time. This heuristic—while simple—mimics real-world behavior
effectively.

### Balancing the Load

The crux of our solution is balancing customer load across tills. When a new
customer arrives, rather than joining the end of a single line, they evaluate
which line is fastest, considering both the number of people waiting and the
number of items they have. By assigning each customer to the till that currently
has the least total checkout time, we ensure that work (checkout tasks) is
evenly distributed among the workers (tills), thereby optimizing throughput and
minimizing the total checkout time.

## Assumptions and Constraints

The relationship between assumptions and constraints in a research or
problem-solving context is akin to the interplay between axioms and theorems in
a mathematical framework. Assumptions serve as the foundational elements upon
which the rest of the work is constructed. They provide the initial conditions
or premises that guide the problem-solving approach. On the other hand,
constraints are often the derived limitations or boundary conditions that
naturally arise from these assumptions.

### Assumptions

In this problem, we make several assumptions to model the supermarket queue
scenario:

1. **Greedy Customer Behavior**: Each customer chooses the till with the least
   waiting time. This is a reasonable assumption as it mimics real-world
   behavior. As soon as a till becomes available, the next customer in the queue is automatically assigned to it, irrespective of the individual checkout times of the remaining customers. This approach does not optimize for minimizing the overall checkout time but rather assigns tills based on immediate availability. See the example `[1,1,1,1,30]`
   with `N=2` tills.

2. **Deterministic Checkout Times**: We assume that the time each customer takes
   to check out is known in advance and is deterministic. This eliminates any
   probabilistic elements that might otherwise be involved in estimating
   checkout times.

3. **Homogeneous Tills**: All the checkout tills are identical in terms of
   processing speed. One till does not have any advantage or disadvantage over
   the others in terms of how quickly it can process customers.

4. **Independent Customers**: Each customer in the queue acts independently of
   the others. Their checkout time does not depend on the presence or behavior
   of other customers in the line.

5. **FIFO Queue**: The queue follows a First-In-First-Out (FIFO) model. That is,
   customers are processed in the order in which they arrive, and there is no
   prioritization among them.

---

## TO REFINE

1. **Greedy Assignment**: The model assigns customers to the next available till without considering future customers in the queue. This immediate, local optimization may not lead to a global minimum for the total checkout time.

2. **FIFO Queue**: Customers are processed in the order in which they appear in the queue, aligning with a First-In-First-Out model. When a till becomes available, the decision about which customer to serve next is made based on their position in the queue, not on how this decision will impact the total checkout time.


3. **Greedy FIFO Assignment**: Customers are processed in a First-In-First-Out (FIFO) manner, and as soon as a till becomes free, the next customer in the queue is automatically assigned to it. This greedy assignment strategy focuses on immediate till availability and does not optimize for minimizing the overall checkout time.

This assumption combines the greedy nature of the assignment with the FIFO ordering, making it explicit why you can't get a total time of 30 minutes under these rules.

### Constraints

Based on these assumptions, we derive the following constraints that our
solution must adhere to:

1. **Fixed Customer Times**: Since checkout times are deterministic, the time
   taken for each customer is a fixed value and must be strictly adhered to in
   any simulation or calculation.

2. **Uniform Till Capacity**: Owing to the homogeneous nature of the tills, each
   till can handle customers at the same rate. This constrains us to distribute
   customers without considering the capabilities of individual tills.

3. **Non-Interactive Queues**: As customers are independent, there's no
   opportunity for interaction effects that could otherwise affect queue
   dynamics (e.g., a fast customer helping a slow one).

4. **Sequential Processing**: Due to the FIFO nature of the queue, a customer
   can only start being processed at a till once the customer before them in the
   same till has finished. This places a temporal constraint on the sequence of
   customer processing.

By clearly outlining these assumptions and constraints, we set the stage for a
rigorous approach to solving the supermarket queue problem. This formalization
aids in ensuring that the solution will be both valid and applicable within the
defined problem space.

## Test Cases

### Standard Cases

1. **Single Till, Multiple Customers**

    - `Input: customers = [5, 3, 4], N = 1`
    - `Output: 12`
    - **Explanation**: With only one till, the time is simply the sum of all
      customer times: $5 + 3 + 4 = 12$.

2. **Multiple Tills, Multiple Customers**

    - `Input: customers = [2, 2, 3, 3, 4, 4], N = 2`
    - `Output: 9`
    - **Explanation**: With two tills, the customers can be optimally divided
      into $[2, 3, 4]$ and $[2, 3, 4]$, resulting in a total time of $
      9$.

3. **Single Prolonged Service**
    - `Input: customers = [1, 1, 1, 1, 30], N = 2`
    - `Output: 30`
    - **Explanation**: One customer can dictate the total time regardless of
      other quick transactions, resulting in a total time of $30$.

      *Note here is a trap, as we shall see later!*

### Edge Cases

1. **Maximum Capacity**

    - `Input: customers = [1, 2, 3, 4], N = 4`
    - `Output: 4`
    - **Explanation**: With $N$ equal to the number of customers, each customer
      should proceed to a till immediately, making the total time the time of
      the slowest customer.

2. **One Till, Many Customers**

    - `Input: customers = [2, 3, 1, 3, 2, 3], N = 1`
    - `Output: 14`
    - **Explanation**: This scenario tests if the solution can handle a backlog
      of customers efficiently. The total time is
      $2 + 3 + 1 + 3 + 2 + 3 = 14
     $.

3. **Unexpected Zero Time**

    - `Input: customers = [0, 3, 4], N = 1`
    - `Output: 7`
    - **Explanation**: The solution should either handle or appropriately reject
      a zero-time customer. Here, the time is $0 + 3 + 4 = 7$.

4. **Excessively Large Input**
    - `Input: customers = [1] * 1000000, N = 1`
    - `Output: 1000000`
    - **Explanation**: This tests the algorithm's ability to handle large-scale
      scenarios without performance breakdown. The total time is $1000000$.

## Complexity Analysis and Tradeoffs

### Theoretical Best Time Complexity

In the realm of queue management and load-balancing problems, the optimal time
complexity often depends on how efficiently tasks (in this case, customers) can
be allocated among resources (tills). The naive approach would involve checking
every possible allocation, leading to exponential time complexity. However, more
efficient methods like greedy algorithms can achieve better performance. For
this specific problem, a greedy approach can achieve a time complexity of
$O(N \log N)$, given that $N$ tills are available and assuming that sorting the
customer times is a part of the process.

### Theoretical Best Space Complexity

For the space complexity, we need to consider the storage for the `customers`
array and the data structures to hold intermediate calculations, such as heap
data structures for greedy algorithms. In the best case, the extra space
required would be constant, $O(1)$, if we only use variables to keep track of
till times. However, if additional data structures are used to track the state
of each till, the space complexity could go up to $O(N)$, where $N$ is the
number of tills.

### Space-Time Tradeoff

In problems like this, there's often a tradeoff between time and space
complexity. For example, using a heap to manage the tills would increase space
requirements but could significantly speed up the task allocation process,
thereby improving time complexity. On the other hand, a simpler array-based
approach that uses constant extra space would likely have a worse time
complexity due to the need for repeated linear scans to find the next available
till. These considerations are crucial in scenarios where either time or space
is a limiting factor.

## Solution: Naive/Brute Force Approach

Before diving into sophisticated algorithms, it's crucial to consider the nature
of the problem and see how we can solve it in a brute force manner. The
supermarket queue problem essentially involves distributing customers to
available tills in such a way that the overall waiting time is minimized. A
naive approach could involve assigning each customer to the next available till
in a [**round-robin**](https://en.wikipedia.org/wiki/Round-robin_scheduling)
fashion. However, this approach doesn't account for the varying service times of
individual tills, leading to suboptimal load distribution.

## Solution: Priority Queue with List as Underlying Data Structure

In computer science terms, the naive approach would be similar to a round-robin
scheduling algorithm, but we're aiming for a more efficient solution. The idea
is to employ a priority queue to keep track of the next available time for each
till. Rather than using a heap-based priority queue, this particular solution
will implement the priority queue using a list. As we'll see, this
approach appears most natural for beginners (including me when I first
encountered this problem) and sets the stage for understanding more advanced
solutions.

### Solution Intuition

The essence of solving this problem lies in efficient resource allocation—in
this case, how to allocate customers to tills in a way that minimizes the total
time spent. We employ a simple but effective approach: we maintain a list
(`tills`) where the $in$-th element represents the total time that the $n$-th
till will take to process all its customers.

Our aim is to always allocate a new customer to the till that will finish
processing its current customers the earliest. In other words, we just check
which $n$-th till has the minimum time, and we add the new customer's time to
this till's time. This mimics the customer's behavior of choosing the till with
the least waiting time.

Although this approach requires scanning the entire list of tills for each
customer, each such scan takes $\mathcal{O}(N)$ time. Given that $N$, the number
of tills, is typically a small fixed number, and $M$, the number of customers,
is the variable that could be large, this results in a time complexity of
$\mathcal{O}(M \times N)$ and can be considered linear in $M$ if $N$ is much
smaller than $M$ (which is often the case in real-world scenarios).

### Visualization

...

### Definitions and Formulation

-   Let $M$ be the total number of customers.

-   Let $C = [c_1, c_2, \ldots, c_M]$ be the list of customer checkout times,
    where $c_m$ is the time required for the $m^{th}$ customer to check out,
    $1 \leq m \leq M$.

-   Let $N$ be the number of tills available for checkout.

-   Let $T = [t_1, t_2, \ldots, t_N]$ be the list representing the state of the
    tills, where $t_n$ is the total time the $n^{th}$ till will be busy,
    $1 \leq n \leq N$.
-   Let $T_{\min}$ and $T_{\max}$ be the minimum and maximum value in $T$.
-   Let $Q$ be the total queue time, i.e., the time it takes for the last
    customer to finish checking out.
-   Let $n^*$ be the index of the till with the minimum time. More formally:

    $$
    n^* = \underset{n}{\operatorname{argmin}} \; t_n, \quad 1 \leq n \leq N
    $$

-   Let $\mathcal{I}$ be the set of indices where the minimum occurs in $T$:

    $$
    \mathcal{I} = \{n \mid t_n = T_{\min}, \quad 1 \leq n \leq N\}
    $$

    The reason we need this set is that if there are multiple tills with the
    same minimum time, we need to choose the one with the smallest index. In other
    words, if there exists a tie, we choose the till that is first in line (first
    occurrence in $T$).

    This can have implications for load balancing among the tills, especially if ties occur frequently. It may lead to certain tills receiving more customers compared to others with the same minimum time. If a more balanced approach is desired, one could employ a different tie-breaking strategy, such as random selection among the tills with the minimum time, or round-robin allocation.

Then we can formulate the problem as follows:

Let $C = [c_1, c_2, \ldots, c_M]$ be the list of customer times, and let $T = [t_1, t_2, \ldots, t_N]$ be the list of tills. We seek to compute the maximum queue time $Q$ for all customers to check out.

### Algorithm

The algorithm employs a list-based priority queue to simulate the processing of
customers at multiple checkout tills. At a high level, the algorithm aims to
minimize the time each customer spends in the queue. It does this by always
assigning the next customer to the till that will be available the soonest. This
ensures a balanced load across all tills and optimizes for the quickest overall
checkout time for all customers.

#### Pseudocode

```{prf:algorithm}
:label: algorithm:naive

Algorithm: Queue Time Optimization

Input:

- List of customer times [c_1, c_2, ..., c_M] of size M,
- Number of tills N

Output: Maximum queue time Q

Base case:

1. If M = 0, return 0
2. If N = 1, return sum(C)
3. If N >= M, return max(C)

General case where N < M, M > 0 and N > 1:

1. Initialize an array `tills` of length N with all elements set to 0.
2. for each `customer` in `customers` do
    1. Find the index `n` of the minimum element in `tills`. If there are
       multiple tills with the same minimum time, choose the one with the
       smallest index or in other words, the first occurrence in `tills`.
    2. Update `tills[n] += customer`.
3. Return max(tills)
```

#### Mathematical Formulation

Let $C = [c_1, c_2, ..., c_M]$ be the list of customer times, and let $T = [t_1, t_2, ..., t_N]$ be the list of tills.

1. **Base Cases**

    1. **No Customers**: If $C$ is empty ($M = 0$), then the total queue time
       $Q$ is zero:

        $$
        Q = 0
        $$
    2. **One Till**: If $N = 1$, then $Q$ is the sum of all $t_n$:

        $$
        Q = \sum_{m=1}^{M} c_m
        $$
    3. **Tills >= Customers**: If $N \geq M$, then $Q$ is the maximum value in
       $C$:

        $$
        Q = \max(C)
        $$

    All three cases above warrants an immediate return of the respective $Q$.

2. **Initialization**

    - For $N > 1$, initialize $T$ with $N$ zeros to represent empty tills:

        $$
        T_i = 0 \quad \forall 1 \leq i \leq N
        $$

    After which, the algorithm iteratively updates $T$ as follows:

3. **Customer Distribution**

    - Iterate over each $c_m$ in $C$, where $1 \leq m \leq M$ and perform the following:

        1. Compute $n^*$ as the index of the minimum element in $T$. In case of ties, $n^*$ will be the smallest index in $\mathcal{I}$:

            $$
            n^* = \underset{n \in \mathcal{I}}{\operatorname{argmin}} \; t_n, \quad 1 \leq n \leq N
            $$

        2. Update the till time $t_{n^*}$ by adding $c_m$ to it:

            $$
            t_{n^*} = t_{n^*} + c_m
            $$

        This mimics the customer's behavior of choosing the till with the
        least waiting time.

4. **Total Time Calculation**

    - After iterating through all customers, the maximum queue time $Q$ can be computed as the maximum value in the updated $T$:

    $$
    Q = T_{\max}
    $$

    This $Q$ will be the output of the algorithm, representing the total queue time for the last customer to finish checking out. In order words, his represents the time when the last customer will finish checking
    out, which is the objective of the algorithm.

#### Claim Correctness and Proof TODO

...

#### Formal Explanation (Theory or Reasoning)

In a multi-server queuing system like this, customers distribute themselves
across available tills based on each till's workload, represented by $T_i$. The
algorithm aims to minimize the workload at each till by assigning the next
customer to the till with the minimum current workload.

The time at which the last customer finishes their checkout process is the time
it takes for the most heavily loaded till to clear all its customers.
Mathematically, this is represented as the maximum value in $T$, $\max(T)$.

Formally, if we let $T = [T_1, T_2, ..., T_n]$, then the total queue time $Q$
can be defined as:

$$
Q = \max(T)
$$

This is because $Q$ is essentially the "completion time" of the queuing system
for a given set of customers $C$. The system is considered "complete" when the
till with the maximum time finishes processing its last customer.

In other words, the system is as fast as its slowest component, and therefore,
it's all for one and one for all.

#### Intuitive Explanation (Analogy)

Imagine you're watching a series of races where each runner has a different
speed. You've got multiple tracks (representing tills), and each track has a
group of runners (representing customers). Even if one track finishes its race
early, you wouldn't say the entire set of races is complete until the track with
the slowest runners finishes too. The time taken for the last race to finish
dictates the total time for all races.

In the context of our supermarket queue, think of each till as a "mini-race."
Even if one till finishes early, it doesn't help the overall system because
we're interested in when the last customer, who is at the busiest till, finishes
checking out.

Hence, our total queue time $Q$ is determined by the slowest "race" or the
busiest till, which is $\max(T)$ in our list of tills $T$.

Both these explanations aim to clarify the same point: The system is as fast as
its slowest component, and therefore, $Q = \max(T)$.

### Implementation

#### Code

```{code-cell} ipython3
def queue_time(customers: List[int], N: int) -> int:
    # fmt: off
    if not customers:           # base case 1: M = 0
        return 0

    if N == 1:
        return sum(customers)   # base case 2: N = 1

    if N >= len(customers):     # base case 3: N >= M
        return max(customers)

    # N < M + N > 1 + M > 0
    tills = [0] * N             # Initialize N tills with zero time.

    for customer in customers:
        # Find the till with the minimum time and add the customer's time to it.
        shortest_till = tills.index(min(tills))
        tills[shortest_till] += customer

    return max(tills)
```

#### Test Cases

```{code-cell} ipython3
class TestFramework:

    def __init__(self):
        self.tests = []

    def describe(self, description):
        def wrapper(func):
            display(HTML(f"<span style='color: blue;'>Description: {description}</span>"))
            func()
        return wrapper

    def it(self, description):
        def wrapper(func):
            try:
                func()
                display(HTML(f"<span style='color: green;'>  [Pass]</span> {description}"))
            except AssertionError as e:
                display(HTML(f"<span style='color: red;'>  [Fail]</span> {description} - {e}"))
        return wrapper

    def assert_equals(self, actual, expected, message):
        assert actual == expected, message

test = TestFramework()

@test.describe("The Supermarket Queue")
def the_supermarket_queue():

    @test.it("Test single till with multiple customers")
    def test_single_till_multiple_customers():
        test.assert_equals(queue_time([5, 3, 4], 1), 12, "Failed: single till, multiple customers")

    @test.it("Test multiple tills with multiple customers")
    def test_multiple_tills_multiple_customers():
        test.assert_equals(queue_time([2, 2, 3, 3, 4, 4], 2), 9, "Failed: multiple tills, multiple customers")

    @test.it("Test single prolonged service")
    def test_single_prolonged_service():
        test.assert_equals(queue_time([1, 1, 1, 1, 30], 2), 30, "Failed: single prolonged service")

    @test.it("Test maximum capacity")
    def test_maximum_capacity():
        test.assert_equals(queue_time([1, 2, 3, 4], 4), 4, "Failed: maximum capacity")

    @test.it("Test one till with many customers")
    def test_one_till_many_customers():
        test.assert_equals(queue_time([2, 3, 1, 3, 2, 3], 1), 14, "Failed: one till, many customers")

    @test.it("Test unexpected zero time")
    def test_unexpected_zero_time():
        test.assert_equals(queue_time([0, 3, 4], 1), 7, "Failed: unexpected zero time")

    @test.it("Test excessively large input")
    def test_excessively_large_input():
        test.assert_equals(queue_time([1] * 1000000, 1), 1000000, "Failed: excessively large input")
```

#### Greedy Algorithm Does Not Imply Global Optimum

- What's local optimum?
- What's global optimum?
- What's greedy algorithm?

### Greedy Algorithms: A Rigorous Overview

A greedy algorithm is a heuristic-driven approach that makes a sequence of choices to produce a solution, where each choice is made based on a local optimization criterion. Mathematically, consider an optimization problem defined on a set $S$ with a cost function $f: S \rightarrow \mathbb{R}$. A greedy algorithm proceeds by iteratively making choices $s_i \in S$ such that each $s_i$ is a local minimizer (or maximizer) of $f$ over a restricted domain $D_i \subset S$, i.e.,

$$
s_i = \arg\min_{s \in D_i} f(s)
$$

### Greediness in the Supermarket Queue Problem

Our problem is to minimize the total queue time for $N$ tills and a given list $C = [c_1, c_2, \ldots, c_n]$ of customer times. Our solution involved selecting the till with the shortest current queue time and appending the next customer to it. Mathematically, let $Q = [q_1, q_2, \ldots, q_N]$ represent the total queue time for each of the $N$ tills. For each $c_i$, our algorithm selects $q_j$ such that

$$
q_j = \min_{1 \leq k \leq N} q_k
$$

and updates $q_j := q_j + c_i$. This choice is a local minimum since it makes the smallest increase in the total time across all queues at that instant.

### Local Optimum vs Global Optimum

In the context of optimization, a local optimum is a solution that is better than all neighboring solutions in the search space, while a global optimum is a solution that is better or equal to all feasible solutions. For greedy algorithms, each $s_i$ is a local optimum in $D_i$, but there is no guarantee that the sequence $(s_1, s_2, \ldots)$ leads to a global optimum.

### The Case of 32 vs 30

Now, let's examine why the test case `test.assert_equals(queue_time([1, 1, 1, 1, 30], 2), 30)` failed and instead produced 32.

Consider the two tills $T_1$ and $T_2$. Here's how our greedy algorithm works:

- $T_1: [1], T_2: []$ (Time = 1)
- $T_1: [1], T_2: [1]$ (Time = 1)
- $T_1: [1, 1], T_2: [1]$ (Time = 2)
- $T_1: [1, 1], T_2: [1, 1]$ (Time = 2)
- $T_1: [1, 1, 30], T_2: [1, 1]$ (Time = 32)

As you can see, our greedy algorithm fails to see that it would be more efficient to distribute the '30' to $T_2$ for a total time of 30 instead of 32. This discrepancy arises because the greedy algorithm does not consider future possibilities and global optimality; it focuses on the local optimum at each decision step.

Therefore, the greedy nature of the algorithm causes it to make locally optimal choices that do not necessarily lead to a globally optimal solution, explaining the incorrect output of 32 instead of 30.

---

### The Nature of Greedy Algorithms

A greedy algorithm constructs a solution to a problem by making a series of choices that are locally optimal—each choice looks like the best one at that moment without regard for the global solution. The algorithm makes decisions based on the current input and state, without considering how that decision may affect future choices or the quality of the overall solution.

Mathematically, let $f(x)$ be an objective function that the algorithm aims to optimize (maximize or minimize). At each step $i$, a greedy algorithm chooses the option $x_i$ that maximizes (or minimizes) $f(x_i | x_1, x_2, \ldots, x_{i-1})$ under the constraints of the problem and the choices made so far $x_1, x_2, \ldots, x_{i-1}$.

### How It Applies to Our Problem

In our supermarket queue problem, the greedy choice is to always assign the next customer to the till that currently has the smallest waiting time. The algorithm does not consider how this choice might affect the distribution of customers to tills in the future. It simply takes the action that appears to be the best at that specific point in time.

### Local vs Global Optimum

A "local optimum" is a solution that cannot be improved by considering a neighboring set of candidate solutions, whereas a "global optimum" is the best possible solution among all possible solutions. Greedy algorithms are generally good at finding local optima but are not guaranteed to find a global optimum. In many cases, the local optimum is sufficient, but sometimes the locally optimal choices can lead to a poor global solution.

### Why The Answer is 32, Not 30

The greedy algorithm assigns customers to tills based on the current state of the system, without considering the total time taken for each till to be free. In the test case `queue_time([1, 1, 1, 1, 30], 2)`, here's how the algorithm's assignments might look at each step:

1. First customer with 1 unit of time goes to till 1. State: $[1, 0]$
2. Second customer with 1 unit of time goes to till 2. State: $[1, 1]$
3. Third customer with 1 unit of time goes to till 1. State: $[2, 1]$
4. Fourth customer with 1 unit of time goes to till 2. State: $[2, 2]$
5. Fifth customer with 30 units of time goes to till 1. State: $[32, 2]$

At each step, the algorithm makes a locally optimal decision to minimize the waiting time for the next customer. However, this local decision-making leads to a global solution where the total time is 32, not 30. The global optimum would have been achieved by assigning the customer who takes 30 units of time earlier in the process, specifically to the first till right after the first customer. The greedy algorithm, focused on immediate benefits, fails to account for this longer-term optimization.

This example reveals the primary weakness of the greedy approach: while it can provide solutions quickly and often approximates the correct answer well, it can fail to find the globally optimal solution in certain instances.

### Time Complexity

1. **Base Cases:**
    - If there are no customers ($M = 0$), the algorithm has a constant time complexity $\mathcal{O}(1)$.
    - If there's only one till ($N = 1$), the time complexity for summing all the customer times is $\mathcal{O}(M)$.
        - This is because `sum` is a built-in function that runs in linear time.
    - If there are more tills than customers ($N \geq M$), the time complexity for finding the maximum customer time is $\mathcal{O}(M)$.
        - This is because `max` is a built-in function that runs in linear time as it involves a linear scan/search of the list.

2. **Till Initialization:**
    - We initialize the `tills` array with $N$ zeros, which takes $\mathcal{O}(N)$ time.
3. **Loop Over Customers:** For each customer ($M$ in total), we go through the $N$ tills to find the minimum.
    1. **Finding the Shortest Till**
        - we use the `min` function on the `tills` array, which has a time complexity of $\mathcal{O}(N)$.
        - Then we use the `index` function on the `tills` array, which has a time complexity of $\mathcal{O}(N)$.
        - So it takes total of $\mathcal{O}(N + N) = \mathcal{O}(2N) = \mathcal{O}(N)$ time to find the shortest till.
        - Note it is not a nested operation, so it is not $\mathcal{O}(N^2)$.
    2. **Updating the Shortest Till**
        - We update the shortest till by adding the customer time to it, which takes $\mathcal{O}(1)$ time.
        - This is because accessing an element in an array is a constant-time operation.
        - Then adding the customer time to it is also a constant-time operation.

    So in total we have $\mathcal{O}(2N) + \mathcal{O}(1) = \mathcal{O}(2N + 1) = \mathcal{O}(2N) = \mathcal{O}(N)$
    for each loop, and for $M$ such loops, we have $\mathcal{O}(MN)$.

4. **Finding the Maximum Element:**
    - At the end of the algorithm, we find the maximum element in `tills`, which takes $\mathcal{O}(N)$.

So in total, we have:

$$
\mathcal{T}(M, N) = \mathcal{O}(1) + \mathcal{O}(N) + \mathcal{O}(MN) = \mathcal{O}(MN + N) = \mathcal{O}(MN)
$$

#### Summary

- **Worst-Case Time Complexity:** $\mathcal{O}(N \times M)$
- **Average-Case Time Complexity:** $\mathcal{O}(N \times M)$
- **Best-Case Time Complexity:** $\mathcal{O}(1)$ (when $M = 0$) or $\mathcal{O}(M)$ (when $N = 1$)

### Space Complexity

- input
- auxiliary
- total


The space complexity is mostly influenced by the storage required for the `tills` array, which is $O(N)$. The rest of the variables take constant space $O(1)$. Hence, the total space complexity is:


1. **Customer and Till Arrays:**
    - We have two arrays: `customers` of size $M$ and `tills` of size $N$.

2. **Variables:**
    - There are a few integer variables ($N$, $M$, `shortest_till`, etc.), which take constant space.

$$
\mathcal{S}(N) = O(N) + O(1) = O(N)
$$

#### Summary:

- **Worst-Case Space Complexity:** $\mathcal{O}(N + M)$
- **Average-Case Space Complexity:** $\mathcal{O}(N + M)$
- **Best-Case Space Complexity:** $\mathcal{O}(1)$ (when $M = 0$)

This analysis shows that while the algorithm is not highly efficient in terms of time complexity, it is reasonably efficient in terms of space complexity. Improvements can be made to the time complexity by using more efficient data structures like min-heaps.

### Dump

To solve this problem, you can use a priority queue (or a min heap) data
structure. The priority queue will help you manage the checkout tills
efficiently, ensuring that the customer who can finish earliest gets the
checkout opportunity first.

Here's how you can approach it:

1. **Priority Queue/Min Heap**: You will store the 'busy' time for each till in
   a priority queue. A priority queue will always give you the till that gets
   free the earliest.

2. **Customers Queue**: This is the input list of customers, where each customer
   has an associated checkout time.

Here's a step-by-step breakdown of the algorithm:

1. Initialize the priority queue with 'N' zeros, representing the initial 'busy'
   time for each of the 'N' tills (since they are free at the beginning).

2. For each customer in the queue (iterating through the customer list): a.
   Extract the minimum element from the priority queue (this represents the till
   that will get free the earliest). b. Add the current customer's checkout time
   to this extracted time (this is simulating assigning the customer to this
   till). c. Add the new 'busy' time back to the priority queue (as this till is
   now occupied until the new 'busy' time).

3. After processing all customers, the priority queue contains 'N' 'busy' times.
   The maximum of these is the total time required for all customers to check
   out because it represents the checkout completion time for the last customer.

Here is a pseudo-code for the function:

```
function calculateTotalTime(customers, N):
    // Initialize a min heap with N zeros
    minHeap = new MinHeap()
    for i from 1 to N:
        minHeap.add(0)

    // Process each customer
    for each time in customers:
        // Get the next available till (the one that finishes earliest)
        nextAvailable = minHeap.pop()

        // The till is now occupied with the new customer
        newOccupiedTime = nextAvailable + time

        // Push the new occupied time into the min heap
        minHeap.push(newOccupiedTime)

    // The last customer finishes at the maximum time in the min heap
    totalTime = max element in minHeap

    return totalTime
```

You'll need to handle the priority queue operations (like extracting the
minimum, adding elements, etc.) based on the specific programming language's
available data structures or libraries. The above approach ensures that at every
step, we are assigning a customer to the till that gets available the earliest,
ensuring minimum waiting time and hence, the total time calculation.

```python
def queue_time(customers, N):
    # base cases
    if not customers:
        return 0
    if N == 1:
        return sum(customers)

    # Initialize N tills with zero time.
    tills = [0] * N

    for customer in customers:
        # Find the till with the minimum time and add the customer's time to it.
        shortest_till = tills.index(min(tills))
        tills[shortest_till] += customer

    return max(tills)


def queue_time(C, N):
	if not C:
		# no customers
		return 0

	if N == 1:
		return sum(C)

	tills = [0] * N

	for index, c in enumerate(C):
		till_with_min_time = tills.index(min(tills))
		tills[till_with_min_time] += c

	return max(tills)
```

1. Base cases
    1. If there are no customers, then the total queue time is zero.
    2. If there is only 1 till, then the total queue time is the sum of all
       customer times since it is a strictly sequential process.
2. For all `N` that are greater than 1, we initialize `N` tills with zero time.
   Why? Because we want to simulate the scenario where all tills are free at the
   beginning since there are no customers.
    1. In code that can be a list/array of `N` zeros.
3. Now assume we have `m` customers, where `m` is greater than `N`.
    1. Then we iterate through this list of `m` customers, and recall that each
       customer has a time `t_i` where `0 < i <=m` associated with them.
    2. We want to assign each customer to a till such that the total time is
       minimized.
    3. We can do this by assigning each customer to the till with the minimum
       time.
    4. We then add the customer's time to this till.
    5. We repeat this process until all customers have been assigned to a till.
4. Finally, we return the maximum time from the list of tills. Why? Because the
   last customer to finish will be the one that takes the longest time.

## Solution (Priority Queue/Heap)
