# Template

-   Terminating condition: the search space is empty.

    -   We don't use `len(nums) == 0` because this question usually want us to
        have auxiliary space complexity of $\mathcal{O}(1)$. And thus `nums` may
        not be mutated directly here (retrospectively).
    -   We generally use left, right pointers and if `left > right` then the
        search space is empty. Why?

        -   The `left` pointer moves right (`left = mid + 1`), and the `right`
            pointer moves left (`right = mid - 1`), so if they cross, it means
            we've checked all possible elements.
        -   The `left <= right` condition ensures that when `left` and `right`
            are pointing to the same element (i.e., the search space has only
            one item left), we still check this last element.
        -   When `left > right`, there are no elements left to check in the
            search space, and the algorithm can terminate.

        This condition works for the binary search paradigm where we exclude the
        middle element at each step after checking it. There are other paradigms
        where the `left` and `right` pointers do not exclude the middle element
        after checking it, and the terminating condition for those may be
        `left < right`. However, for the classic binary search, the
        `left > right` condition is used to indicate an empty search space.

-   Distinguishing Syntax:
    -   Initial Condition: `left = 0`, `right = length-1`
    -   Termination: `left > right`
    -   Searching Left: `right = mid-1`
    -   Searching Right: `left = mid+1

## When to use Binary Search?

If we can discover some kind of **monotonicity**, for example, if `condition(k)`
is `True` then `condition(k + 1)` is `True`, then we can consider binary search.

More formally, we have:

The essential precondition to apply binary search is the presence of a
**monotonic property**. This is a property that allows us to decide which half
of the search space should be eliminated based on the comparison between the
target value and the value at the current index.

```{prf:definition} Monotonicity
:label: monotonicity

In more formal terms, a function or sequence is said to have the property of
monotonicity if it is either entirely non-increasing or non-decreasing. A
function that increases monotonically does not necessarily increase constantly,
but it does not decrease at any point. Similarly, a function that decreases
monotonically does not necessarily decrease constantly, but it does not increase
at any point.

1. A sequence or function $f$ is said to be **monotone increasing** (or
   non-decreasing) on an interval $I$ if for all $x, y \in I$, if $x \leq y$,
   then $f(x) \leq f(y)$. In simple terms, as we move along the interval, the
   function value does not decrease; it either increases or stays the same.

2. Similarly, a sequence or function $f$ is said to be **monotone decreasing**
   (or non-increasing) on an interval $I$ if for all $x, y \in I$, if
   $x \leq y$, then $f(x) \geq f(y)$. That is, as we move along the interval,
   the function value does not increase; it either decreases or stays the same.
```

In the context of binary search, when the `condition` function has a monotonic
property (either always `True` to `False`, or always `False` to `True`), it
means that there is a clear threshold or tipping point in the sorted array that
divides the array into two halves - the first half where the `condition`
function is `True` and the second half where the `condition` function is `False`
(or vice versa).

That's where binary search comes into play: it allows us to effectively locate
that threshold by iteratively narrowing down the search space. If we find that
the `condition` is `True` for a given middle element (let's call it `mid`), we
know that all elements on the right of `mid` will also satisfy `condition`
(because of the monotonic property), so we can safely ignore the right half.
Conversely, if `condition(mid)` is `False`, we can ignore the left half.

If we can't establish such a monotonic property, it's difficult (or even
impossible) to decide which half of the array to eliminate, rendering binary
search ineffective or incorrect. Therefore, confirming the existence of this
monotonicity is crucial before deciding to use binary search.

## Solution (Minimize $k$, $s.t.$ condition($k$) is True)

Before going into details, we see the below:

```{prf:remark} Finding Target and Finding First True
:label: finding-target-and-finding-first-true

Finding a target in a sorted array and finding the "first True" in a sorted
Boolean array are conceptually similar because both rely on a monotonic
condition. In the first case, the condition is "Is the element at the current
index greater or equal to the target?" In the second case, it's "Is the element
at the current index True?"

To bridge the gap:

1. Consider the feasible function $f(x)$ that maps each element in the sorted
   array to either True or False based on whether the element is greater or
   equal to the target. This makes the problem equivalent to finding the "first
   True" in a sorted Boolean array derived from $f(x)$.
2. In both problems, once you identify an element that satisfies the condition
   (either being the target or being True), you can be sure that no elements
   satisfying the condition exist in the half of the array that is 'less' than
   the current element.

In the context of finding a specific target element $x$ in a sorted array, the
feasible function $f(i)$ would map to True for all elements greater than or
equal to $x$ and False for all elements less than $x$. So, if the array is
$[1, 3, 5, 7]$ and the target is $5$, the mapped Boolean array based on $f(i)$
would be $\text{FFFFTTT}$, making it a sorted Boolean array. In this setup,
"finding the first True" indeed corresponds to "finding the target element."
```

The problem of finding a target number in a sorted array is a "minimize k s.t.
condition(k) is True" problem because you're essentially looking for the
smallest (left-most) index `k` where the condition "array value at `k` is
greater than or equal to the target" is True.

In other words, you're trying to find the minimum `k` such that
`nums[k] >= target`. This can either be the first occurrence of the target in
the array (if the target exists in the array) or the position where the target
could be inserted to maintain the sorted order of the array (if the target does
not exist in the array).

This fits the structure of "minimize k s.t. condition(k) is True" because you
are minimizing the index `k` subject to a condition (i.e., `nums[k] >= target`).

In the binary search template, this is implemented as the `condition(mid)`
function. The binary search algorithm keeps adjusting the search boundaries
(i.e., `left` and `right`) based on whether the condition is met at the
mid-point, and keeps narrowing down to the smallest `k` (left-most position)
where the condition is True. This is why this problem fits into the "minimize k
s.t. condition(k) is True" structure.

```{code-cell} ipython3
def binary_search(nums: List[int], target: int) -> int:
    def condition(k: int, nums: List[int]) -> bool:
        return nums[k] >= target

    left, right = 0, len(array)
    while left < right:
        mid = left + (right - left) // 2
        if condition(k=mid, nums=nums):
            right = mid
        else:
            left = mid + 1
    return left

# Example 1
array = [1, 2, 3, 4, 5]
target = 3
result = binary_search(nums=array, target=target)
assert result == 2
```

So you essentially combined the two steps of finding the target and finding the
left-most occurrence of the target into one step.
