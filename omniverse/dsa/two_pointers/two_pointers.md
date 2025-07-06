---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Two Pointers

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Two_Pointers-orange)

```{contents}
```

The **Two Pointers Technique** is a method where two pointers are used to
traverse an array from two different points. Typically, this method is employed
in a sorted array or list where the goal is to solve problems that demand the
use of less time complexity, which is a primary factor in optimizing algorithms.

The two pointers technique is very useful in solving problems with linear data
structures like arrays and linked lists. It's most often used when we need to
find a pair of elements in the array that meet certain conditions.

Here is a breakdown of the two pointers technique:

1. **Initialization**: You initialize two pointers, a left pointer (usually
   denoted as `left` or `start`) that starts from the beginning of the array,
   and a right pointer (usually denoted as `right` or `end`) that starts from
   the end of the array.

2. **Traversal**: Depending upon the condition, both pointers start moving
   towards each other. This movement can either be step-by-step or in a
   skip-over fashion (like in binary search).

3. **Meeting Point**: The process continues until the two pointers meet, which
   will be the case if the array is traversed completely.

The two pointers technique can be classified into two types based on the
direction of the two pointers:

-   **Same Direction Two Pointers**: In this, both pointers traverse in the same
    direction. An example is the Sliding Window algorithm.

-   **Opposite Direction Two Pointers**: Here, pointers traverse in the opposite
    direction. A typical example is to find the target sum in a sorted array or
    to sort a binary array.

Applications of the two pointers technique in computer science are numerous:

-   **Array and List Problems**: The technique is used to solve various array
    and list problems efficiently, such as finding a pair with a given sum,
    removing duplicates from a sorted list, and others.

-   **String Manipulation**: The two pointers technique is frequently used in
    string manipulation problems, like checking if a string is a palindrome or
    not.

-   **Linked List Problems**: The technique can be used to solve problems like
    finding the middle element of a linked list, detecting a cycle in a linked
    list, finding the kth element from the end in a single pass, and others.

Despite its apparent simplicity, the two pointers technique underpins many
algorithmic problems and solutions. Mastering how it works is crucial to
understanding, designing, and implementing effective algorithms and data
structures.

For more information about the two pointers technique, LeetCode has a
[Two Pointers category](https://leetcode.com/tag/two-pointers/) dedicated to
problems that can be solved using this technique. The problems range in
difficulty levels, allowing learners to gradually master this technique.

## The Two Pointers Technique

The two pointers technique is a common algorithmic strategy that utilizes two
pointers that traverse through an array in a synchronized manner, often with the
objective of finding a pair of elements satisfying a particular condition or
reducing the time complexity of an otherwise expensive operation.

### Two Pointers Same Direction

Suppose we have a sequence $S$ defined as:

$$
S = [s_1, s_2, ..., s_n]
$$

of $n$ elements where $s_i \in \mathbb{R}$ for all $1 \leq i \leq n$. We have
two pointers $i$ and $j$, such that $1 \leq i, j \leq n$ and $i \leq j$.

The task often involves finding a pair of elements $s_i$ and $s_j$ that satisfy
a particular condition $C(s_i, s_j)$. The pointers are moved according to the
evaluation of condition $C$ and the specific problem requirements. The general
algorithm can be described as follows:

```{prf:algorithm} Two Pointers Technique
:label: two-pointers-technique-algorithm

Input: Sequence $S$ of $n$ elements, Condition $C$

1.  Initialize $i, j = 1$
2.  While $i \leq n$ and $j \leq n$ do:
    - If $C(s_i, s_j)$ is True:
        1. Process valid pair $(s_i, s_j)$
        2. Increment $i$ by $1$
    - Else, increment $j$ by $1$
3.  End while
```

In this algorithm, we start with both pointers at the beginning of the sequence.
We evaluate condition $C(s_i, s_j)$. If it's satisfied, we process the pair and
move the left pointer forward. If it's not, we move the right pointer forward.
This continues until we've traversed the whole sequence.

This technique is particularly useful when the sequence is sorted, and condition
$C$ represents some form of order relationship between $s_i$ and $s_j$, but it
can be used in many different contexts.

### Removing Duplicates from a Sorted Array

In the context of the "Remove Duplicates from Sorted Array" problem, the
sequence $S$ can be defined as $S = [s_1, s_2, ..., s_n]$, where $s_i$ denotes
the $i$-th element in the array.

For this problem, we start with two pointers, $i$ and $j$, at the beginning of
the array. The role of these two pointers is to identify duplicates and help
maintain the "uniqueness" of elements in the processed part of the array:

-   The "slow" pointer $i$ marks the position where the next unique element
    should be placed.
-   The "fast" pointer $j$ scans through the array to find the next unique
    element.

The condition $C(s_i, s_j)$ that guides the movement of the pointers in this
case is $s_i \neq s_j$, which checks if the current element and the next element
in the array are different.

-   If $C(s_i, s_j)$ is True (i.e., $s_i \neq s_j$), we have found a
    non-duplicate element. In this case, we increment $i$ by 1 and update the
    element at the $i$-th position with the value of $s_j$, i.e., $s_i = s_j$.
-   If $C(s_i, s_j)$ is False (i.e., $s_i = s_j$), we have found a duplicate. In
    this case, we increment $j$ by 1 to continue scanning for the next
    non-duplicate.

This approach works because the array is sorted, so any duplicate values must be
adjacent in the array. The two pointers, $i$ and $j$, help us keep track of the
"processed" part of the array (up to index $i$) and the "unprocessed" part (from
index $j$ onwards).

By moving the pointers according to condition $C(s_i, s_j)$, we effectively
"remove" duplicates by overwriting them with unique elements, and the first
$i+1$ elements in the array will be the unique elements.

### Two Pointers Opposite Direction

The two-pointer technique can certainly be used with pointers moving in
different directions. This is often referred to as the "two-pointer technique
with meet in the middle". It's typically used in sorted arrays where the goal is
to find two elements that meet a certain condition.

Here's a mathematical formulation for the case where the pointers move towards
each other:

Suppose we have a sequence $S$ defined as:

$$
S = [s_1, s_2, ..., s_n]
$$

of $n$ elements where $s_i \in \mathbb{R}$ for all $1 \leq i \leq n$. We have
two pointers, $i$ and $j$, initially positioned at the start and end of the
sequence, respectively.

The task often involves finding a pair of elements $s_i$ and $s_j$ that satisfy
a particular condition $C(s_i, s_j)$. The pointers are moved according to the
evaluation of condition $C$ and the specific problem requirements.

The general algorithm can be described as follows:

```{prf:algorithm} Two Pointers Technique with Meet in the Middle
:label: two-pointers-technique-meet-in-middle-algorithm

Input: Sequence $S$ of $n$ elements, Condition $C$

1.  Initialize $i = 1, j = n$
2.  While $i < j$ do:
    - If $C(s_i, s_j)$ is True:
        1. Process valid pair $(s_i, s_j)$
        2. Increment $i$ by 1
    - Else, decrement $j$ by 1
3.  End while
```

In this algorithm, we start with one pointer at the beginning of the sequence
and the other at the end. We evaluate condition $C(s_i, s_j)$. If it's
satisfied, we process the pair and move the left pointer forward. If it's not,
we move the right pointer backward. This continues until the two pointers meet.

This variation of the two-pointer technique is particularly useful when the
sequence is sorted, and condition $C$ represents some form of order relationship
between $s_i$ and $s_j$. It's often used in problems related to pair sum, where
you need to find two numbers in the array that add up to a given target.

### Two Sum II - Input Array Is Sorted

We see a classic problem
["Find Two Numbers That Add Up to a Given Target"](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/):

```{admonition} Two Sum II - Input Array Is Sorted
Given a **1-indexed** array of integers `numbers` that is already sorted in
**non-decreasing order**, find two numbers such that they add up to a specific
`target` number. Let these two numbers be `numbers[index1]` and
`numbers[index2]` where
`1 \leq \text{index1} < \text{index2} \leq \text{numbers.length}`.

Return the **indices of the two numbers**, `index1` and `index2`, **added by
one** as an integer array `[index1, index2]` of length 2.

The tests are generated such that there is **exactly one solution**. You may
**not use the same element twice**.

Your solution must use only constant extra space.
```

We can define the sequence $S$ can be defined as $S = [s_1, s_2, ..., s_n]$,
where $s_i$ denotes the $i$-th element in the array.

For this problem, we start with two pointers, $i$ and $j$, at the beginning and
end of the array, respectively. The role of these two pointers is to identify
pairs of elements that add up to the given target:

-   The "left" pointer $i$ starts from the beginning of the array (smallest
    element) and moves towards larger values.
-   The "right" pointer $j$ starts from the end of the array (largest element)
    and moves towards smaller values.

The condition $C(s_i, s_j)$ that guides the movement of the pointers in this
case checks whether the sum of the current pair of elements equals, is less
than, or is greater than the target:

-   If $C(s_i, s_j)$ is $s_i + s_j = \text{target}$, we have found a pair of
    elements that add up to the target. In this case, we return the pair
    $(i, j)$.
-   If $C(s_i, s_j)$ is $s_i + s_j < \text{target}$, the current sum is less
    than the target. In this case, we increment $i$ by 1 to increase the sum, as
    we can get a larger sum by considering a larger $s_i$ due to the sorted
    nature of the array.
-   If $C(s_i, s_j)$ is $s_i + s_j > \text{target}$, the current sum is more
    than the target. In this case, we decrement $j$ by 1 to decrease the sum, as
    we can get a smaller sum by considering a smaller $s_j$ due to the sorted
    nature of the array.

This approach works because the array is sorted, so by moving the pointers
inward based on the comparison of the current sum with the target, we can find a
pair that adds up to the target, if it exists. The two pointers, $i$ and $j$,
help us control the sum and traverse the array in $\mathcal{O}(n)$ time.
