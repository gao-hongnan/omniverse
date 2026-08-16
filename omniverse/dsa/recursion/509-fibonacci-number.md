# Question Name

## Problem

A clear and concise statement of the problem.

## Example

Small example(s) to illustrate the problem.

## Intuition

Brief explanation of the initial thought process for solving the problem.

## Assumptions

List of any assumptions made in the problem solving process.

## Constraints

### What are the Constraints for?

Explanation of the constraints and their impact on the problem and solution.

## Test Cases

Set of test cases for validating the solution.

## Edge Cases

Discussion of any potential edge cases in the problem.

## Walkthrough / Whiteboarding

Detailed walkthrough of the problem-solving process.

## Theoretical Best Time Complexity

Discussion of the theoretical best time complexity for this problem.

## Theoretical Best Space Complexity

Discussion of the theoretical best space complexity for this problem.

## Space-Time Tradeoff

Analysis of the tradeoff between space and time complexity for the problem.

## Solution (Potentially Multiple)

### Intuition

Brief explanation of the core ideas or insights that form the basis for the
solution.

### Visualization

Visual representation of the problem and solution (if applicable).

### Algorithm

```
fib(n):
    if n = 0 then     // base case
        return 0
    elseif n = 1 then // base case
        return 1
    else
        return fib(n - 1) + fib(n - 2)
    endif
```

### Claim

**Claim:** The algorithm, `fib(n)` is correct and returns the $n$th Fibonacci
number.

(Proof by Strong Induction)

### Proof

<https://cs.stackexchange.com/questions/14025/prove-correctness-of-recursive-fibonacci-algorithm-using-proof-by-induction>

The intuition is that we assume that the algorithm is correct for all values of
$k \leq n$ and then prove that it is correct for $n+1$. The distinction between
strong and weak induction is that in strong induction, we assume that the
algorithm is correct for all values of $k \leq n$ whereas in weak induction, we
assume that the algorithm is correct for $k=n$.

Why do we need to use strong induction here? Because we need to look back at
**two** previous values of $k$ to compute the $k+1$th value. So, weak induction
would not work here.

In certain problems like this one, the property at a certain step $n$ depends
not just on the property at the step $n-1$, but potentially on multiple or even
all previous steps. This is where strong induction becomes useful.

In the case of the Fibonacci sequence, $F_{k+1} = F_k + F_{k-1}$, to prove the
property for $k+1$, we need the property to hold for both $k$ and $k-1$ steps,
not just $k$.

So in such situations where a step depends on multiple prior steps, we use
strong induction because it allows us to assume that the statement holds for all
previous steps, which provides a stronger base for the inductive step. This
'looking backwards more', as you phrased it, is indeed why we often use strong
induction.

**Base Case:** for inputs $0$ and $1$, the algorithm returns $0$ and $1$
respectively. So this is correct.

**Induction Hypothesis:** Here, we assume that the algorithm defined as `fib(k)`
is correct and returns the **_correct Fibonacci number_** **for all values of
$k \leq n$**, where $n,k\in \mathbb{N}$.

We must now show that the algorithm is correct for $n = k+1$. That means show
that

$$
F_{k+1} = F_k + F_{k-1}
$$

where $F_k$ is the $k$th Fibonacci number.

**Inductive Step:**

1.  let `fib(k)` be true **for all values until $n$**, this means that `fib(k)`
    correctly computes $F_k$ for $k = 0,1,2,\ldots,n$.
2.  From the induction hypothesis, we know `fib(k)` correctly computes $F_k$ and
    `fib(k-1)`correctly computes $F_{k-1}$
3.  So, we can then say the below:

    $$
    \begin{align*}
     \text{fib(k+1)} &= \text{fib(k)  + fib(k-1)} \\ &\text{(by definition of the Fibonacci function)} \\ \\
      &= F_k + F_{k-1} & \\ \\
      &= F_{k+1}    \\ &\text{(By definition of Fibonacci numbers)}\\
    \end{align*}
    $$

4.  Thus by rules of mathematical Inducion, `fib(n)` always returns the correct
    result for all values of $n$.

### Implementation

Code implementation of the algorithm.

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
