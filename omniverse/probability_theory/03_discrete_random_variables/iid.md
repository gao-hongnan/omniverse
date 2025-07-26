# Independent and Identically Distributed (IID)

```{contents}
```

## Definition

```{prf:definition} Independent and Identically Distributed (IID)
:label: def_iid

Let $X_1, X_2, \ldots, X_n$ be a sequence of random variables.

We say that the random variables are ***independent and identically distributed (i.i.d.)*** if the following two conditions hold:

1. The random variables are **independent** of each other. That is, $\P(X_i = x_i | X_j = x_j, j \neq i) = \P(X_i = x_i)$ for all $i, j$.
2. The random variables have the **same distribution**. That is, $\P \lsq X_1 = x \rsq = \P \lsq X_2 = x \rsq = \ldots = \P \lsq X_n = x \rsq$ for all $x$.
```

## Examples

### Example 1

```{prf:example} Example 1
:label: ex_iid_1

The PMF of the height distribution is given by $\P \lsq X = x \rsq = \frac{1}{10}$ for all $x \in \{1, \ldots, 10\}$.
In other words, $X \sim \uniform(1, 10)$.

Then if you draw 100 people from this population/distribution, with replacement, then this is an example of IID random variables.
This is because with replacement, with each draw, the probability will be the same
(all draws will follow $\uniform(1, 10)$), and the draws are independent of each other.
```

Confusion in notation is when you index random variables with $i$. See
[here](https://stats.stackexchange.com/questions/200741/let-random-variables-x-1-dots-x-n-be-a-iid-random-sample-from-fx-wha)
sometimes $X_1, X_2, \ldots, X_{10}$ can mean 10 different people drawn and
[here](https://stats.stackexchange.com/questions/141416/example-of-sample-x-1-x-2-ldots-x-n?rq=1).

## Further Readings

-   Chan, Stanley H. "Chapter 5.1.6 Independent random variables." In
    Introduction to Probability for Data Science, 253-254. Ann Arbor, Michigan:
    Michigan Publishing Services, 2021.
-   [Wikipedia](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables)
-   [On the importance of the i.i.d. assumption in statistical learning](https://stats.stackexchange.com/questions/213464/on-the-importance-of-the-i-i-d-assumption-in-statistical-learning)
