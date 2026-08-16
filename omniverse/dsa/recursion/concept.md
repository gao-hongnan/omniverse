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

## The Three Laws/Axioms of Recursion

```{prf:axiom} The Three Laws of Recursion
:label: axiom_three_laws_of_recursion

1. A recursive algorithm must have a base case.
2. A recursive algorithm must change its state and move toward the base case.
3. A recursive algorithm must call itself, recursively.
```

Useful latex alignment.

$$
\begin{align*}
F_0+F_1+F_2+\cdots+F_n+F_{n+1} &=& \left(F_0+F_1+F_2+\cdots+F_n\right)+F_{n+1} \\
&=& \left(F_{n+2}-1\right)+F_{n+1} \\
&=& \left(F_{n+2}+F_{n+1}\right)-1 \\
&=& F_{n+3}-1 \tag*{(by assumption)}
\end{align*}
$$


We start with the well-ordering principle: every nonempty set of positive
integers contains a least element.

From this, we can prove the principle of mathematical induction. Suppose that
$P(n)$ is a property defined for every positive integer $n$, and we want to
prove that $P(n)$ holds for all positive integers.

\textbf{Step 1:} We prove that $P(1)$ is true.

\textbf{Step 2:} We assume that $P(k)$ is true for some fixed positive integer
$k$, and use this assumption to prove that $P(k+1)$ is also true.

Suppose, to the contrary, that there is at least one positive integer for which
$P(n)$ is false. Let $S$ be the set of all positive integers for which $P(n)$ is
false, and by the well-ordering principle, $S$ must have a smallest element, say
$m$.

By our induction hypothesis, $P(1)$ is true, so $m$ is not 1. Moreover, because
$m$ is the smallest counterexample, $P(k)$ is true for all positive integers $k$
less than $m$. In particular, $P(m-1)$ is true.

But, by the induction step, if $P(m-1)$ is true, then $P((m-1)+1)$ or $P(m)$ is
also true, a contradiction. Therefore, $S$ must be empty, which implies that
$P(n)$ is true for all positive integers $n$.

---

Time Complexity:

- Amortized $\mathcal{O}(1)$: This term is used to describe an algorithm's
    average time taken per operation, over a worst-case sequence of operations.
    In the context of this problem, while the 'pop' operation may take
    $\mathcal{O}(n)$ time in some cases (when stack `s2` is empty and elements
    need to be moved from `s1` to `s2`), most other operations take
    $\mathcal{O}(1)$ time (when 'pop' is called while `s2` has elements, or
    'push'/'peek'/'empty' operations). So, when averaged over a large number of
    operations, the time complexity can be said to be $\mathcal{O}(1)$.

- Worst-case $\mathcal{O}(n)$: This is the scenario where the operation takes
    the maximum time. In this problem, it happens when the 'pop' operation is
    called while stack `s2` is empty and we need to move all the elements from
    `s1` to `s2`. The time complexity for this single operation is
    $\mathcal{O}(n)$, where n is the number of elements in the queue.

---

```{code-cell} ipython3
from typing import List
from rich.jupyter import print

def list_sum(nums: List[int]) -> int:
    stack = []
    stack.append(nums)

    result = 0

    while stack:
        current = stack.pop()
        # print(current)

        # Base case: list is empty
        if not current:
            continue

        # Recursive case: Split list into head and tail, add head to result, and push tail back onto the stack
        head, *tail = current
        result += head
        print(tail)
        stack.append(tail)

    return result




list_sum([1,3,5,7,9])
```

## References and Further Readings

- [Real Python](https://realpython.com/fibonacci-sequence-python/)
    - The post has wonderful stack call visualizations.
- https://brilliant.org/wiki/strong-induction/
- https://brilliant.org/wiki/induction/#induction-introduction
- https://math.libretexts.org/Bookshelves/Combinatorics_and_Discrete_Mathematics/A_Spiral_Workbook_for_Discrete_Mathematics_(Kwong)/03%3A_Proof_Techniques/3.06%3A_Mathematical_Induction_-_The_Strong_Form
- http://mathfoundations.lti.cs.cmu.edu/class2/induction.html#:~:text=The%20principle%20of%20induction%20is,the%20set%20of%20natural%20numbers.
- https://towardsdatascience.com/train-your-mind-to-think-recursively-in-5-steps-8f85c0e0eb81
- https://stackoverflow.com/questions/23301682/how-can-i-visualize-this-recursive-implementation-of-finding-the-maximum-number

- https://stackoverflow.com/questions/10959874/what-is-the-relationship-between-recursion-and-proof-by-induction
- http://vaidehijoshi.github.io/blog/2014/12/14/to-understand-recursion-you-must-first-understand-recursion/
- https://www.reddit.com/r/learnprogramming/comments/ajwi3j/why_is_recursion_so_hard/
- <https://www.freecodecamp.org/news/how-recursion-works-explained-with-flowcharts-and-a-video-de61f40cb7f9/#:~:text=Recursive%20functions%20use%20something%20called,things%20one%20at%20a%20time>.
- <https://www.youtube.com/watch?v=rf6uf3jNjbo>
- <https://www.youtube.com/watch?v=boS4N1_TLBk>
- <https://codecrucks.com/tower-of-hanoi/>
- <https://www.mathsisfun.com/games/towerofhanoi.html>
- <https://math.libretexts.org/Bookshelves/Mathematical_Logic_and_Proof/Book%3A_Mathematical_Reasoning__Writing_and_Proof_(Sundstrom)/04%3A_Mathematical_Induction/4.03%3A_Induction_and_Recursion>
- <https://yourbasic.org/algorithms/induction-recursive-functions/>
- <https://proofwiki.org/wiki/Tower_of_Hanoi>