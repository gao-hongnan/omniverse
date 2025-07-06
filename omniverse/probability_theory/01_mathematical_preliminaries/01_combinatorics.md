# Permutations and Combinations

```{contents}
```

## The Counting Principles

To fill in by referring to Choo Yan Min's book on Counting Principles.

## The Multiplication Principle (MP)

We start off with a simple example. Let us assume we have to choose lunch and
dinner on an airplane. We were offered 2 choices in lunch menu and 3 choices in
dinner menu which we denote $l_1, l_2$ and $d_1, d_2, d_3$ respectively.

We claim that there are a total of $2 \times 3 = 6$ choices. We first convince
ourselves this is true by enumerating manually:

$$
\left|\left\{(l_1, d_1), (l_1, d_2), (l_1, d_3), (l_2, d_1), (l_2, d_2), (l_2, d_3)\right\}\right| = 6
$$

Recall **Multiplication** is nothing but **Addition**, and $2 \times 3$ can be
understood as $3 + 3$, where we can understood it as if we choose $l_1$ and fix
it as it is, how many choices can we have for dinner, the answer is 3 choices
and therefore we have a total of $3 + 3$ since we have 2 lunch choices, so we
add twice. The same can be said if we see $2 \times 3$ as $2 + 2 + 2$, as we can
fix dinner as $d_1$ and ask, how many choices do we have now? The answer is 2,
and we see that there are 3 dinner, so add them thrice.

## Why do you multiply probabilities?

Also appear in Chapter 2 summary.

Consider rolling a dice twice, what is the probability that you roll a 5 and 6
respectively.

We all know the answer is $\dfrac{1}{6} \times \dfrac{1}{6} = \dfrac{1}{36}$.
But why?

This can be first understood that our denominator is the total outcomes in our
sample space $\S$. This is $36$, why? By our counting principle on
multiplication, we know that if we have $6$ choices in roll $1$ and $6$ choices
in roll 2, then the cross-product is $6 \times 6 = 36$ total choices. One can
enumerate $\{(1,1), (1,2), \ldots, (6,6)\}$ to see why.

Now the numerator is also related to the counting principle of multiplication as
well! In roll 1, rolling a 5 is 1 choice, rolling a 6 next is 1 choice, so total
there is a only one combination choice $1 \times 1$!

Now if we reframe the problem to what is the probability that you roll a 1, 2 or
3 in the first roll and 2 or 3 in the second roll. Then of course our
denominator don't change as $36$, but our numerator changes, since in roll 1 we
have 3 choices, and roll 2 have 2 choices, by the multiplicative principle we
have a total of $3 \times 2 = 6$ choices, and so our probability is
$\dfrac{6}{36}$ now. You can verify that there are indeed $6$ choices manually.

> **Now the most important part is we can use this if both events are
> independent! If not we need to be careful!**
