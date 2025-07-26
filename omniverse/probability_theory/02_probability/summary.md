# Summary

```{contents}
```

## Conditional

Therefore, after the intuition, one should not be constantly checking what the
formula represents, if we have $\P(A ~|~ B)$, then it just means given $B$ has
happened, what is the probability of $A$ happening? The logic becomes apparent
when we reduce the whole sample space $\S$ to be only $B$ now, and that whatever
$A$ overlaps with $B$ will be the probability of this conditional.

## Independence

So the intuition can be understood by using conditional, say $\P(A ~|~ B)$, if
$A$ and $B$ are truly independent, then even if $B$ happened, the probability of
$A$ should remain unchanged, which means that $\P(A ~|~ B) = \P(A)$, but also
recall the definition of conditionals, $\P(A ~|~ B) = \dfrac{\P(A \cap B)}{B}$,
so equating them we have the nice equation of $\P(A)\P(B) = \P(A \cap B)$.

## Bayes\' Theorem

The observant reader should find it easy that this formula is deduced by abusing
the fact that $\P(A \cap B) = \P(B \cap A)$, and therefore

$$
\P(A|B) = \dfrac{\P(A\cap B)}{\P(B)} = \dfrac{\P(B \cap A)}{\P(B)} = \dfrac{\P(B ~|~ A)\P(A)}{\P(B)}
$$

since $\P(B ~|~ A) = \dfrac{\P(B \cap A)}{\P(A)}$.

## The Law of Total Probability

The intuition of this is also apparent after understanding that if a sample
space can be broken down into disjoint events $A_1, A_2, \ldots, A_k$, then any
event $B$ that is in the sample space must touch some of these disjoint events,
whenever they overlap, we can represent them as $\P(A_i \cap B)$, and if we add
them up it is just the event $B$ itself. If they don\'t overlap, we can still
add them anyways since $P(A_k \cap B) = 0$. Note the loose usage of the index
here, $i$ means overlap, $k$ means non-overlap.

## Why do you multiply probabilities?

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
denominator don\'t change as $36$, but our numerator changes, since in roll 1 we
have 3 choices, and roll 2 have 2 choices, by the multiplicative principle we
have a total of $3 \times 2 = 6$ choices, and so our probability is
$\dfrac{6}{36}$ now. You can verify that there are indeed $6$ choices manually.

> **Now the most important part is we can use this if both events are
> independent! If not we need to be careful!**.

## Conditional, Priori, Posterior

Common terms in ML world.

Suppose there are three types of players in a tennis tournament: A, B, and C.
Fifty percent of the contestants in the tournament are A players, 25% are B
players, and 25% are C players. Your chance of beating the contestants depends
on the class of the player, as follows:

-   0.3 against an A player
-   0.4 against a B player
-   0.5 against a C player

If you play a match in this tournament, what is the probability of your winning
the match? Supposing that you have won a match, what is the probability that you
played against an A player?

Let $W$ be the event that you win and $A$ be the event that you played vs player
$A$, then

-   Conditional: $\P(W~|~A)$ = given you played player $A$, what is your
    probability of winning?
-   Priori: $\P(A)$ = **without entering the game**, what is your probability of
    facing player $A$?
-   Posterior: $\P(A~|~W)$ = **after entering the game and winning the match**,
    what is your probability that you have actually played with $A$?
-   Machine Learning: In many practical engineering problems, the question of
    interest is often the last one. That is, supposing that you have observed
    something, what is the most likely cause of that event? For example,
    supposing we have observed this particular dataset, what is the best
    Gaussian model that would fit the dataset? Questions like these require some
    analysis of conditional probability, prior probability, and posterior
    probability.
