# Exercises

## Example 5.7 ([link](https://www.probabilitycourse.com/chapter5/5_1_3_conditioning_independence.php))

```{admonition} Problem
:class: note

Suppose that the number of customers visiting a fast food restaurant in a given day
follows a Poisson distribution with parameter $\lambda$. We further note that each customer
purchases a drink with probability $p$, independently from other customers and independently
from Poisson distribution. Then, let $X$ be the number of customers who purchase drinks, find $\mathbb{E}[X]$.
```

Let $Y$ be the number of customers visiting the restaurant in a given day. Then, we have

$$
\begin{aligned}
Y &\sim \text{Poisson}(\lambda) \\
\end{aligned}
$$

The problem mentioned that each customer purchases a drink with probability $p$, independently from other customers $(\iid)$.
This allows us to model each customer as a Bernoulli random variable with parameter $p$. More concretely,
let $W$ be the indicator random variable of whether a customer purchases a drink or not, then we have
$W \sim \text{Bernoulli}(p)$.

We can then model $X$ as the number of customers who purchase drinks **in a day**, which is a binomial random variable with
parameters $n$ and $p$. More concretely, we have $X$ as a sum of $n$ independent Bernoulli random variables $W$
with parameter $p$, then we have $X \sim \text{Binomial}(n, p)$.

The question asks us to find $\mathbb{E}[X]$. We do know that $\mathbb{E}[X] = np$,
but we do not know what $n$ is in a single day. Our intuition says that since the number of customers visiting the restaurant
in a day follows a Poisson distribution, then the average number of customers visiting the restaurant in a day is $\lambda$,
by definition {prf:ref}`prop:poi_exp`. Then we can instead say $\mathbb{E}[X] = \lambda p$. It turns out
our intuition is correct, and we can prove this result using the Law of Total Expectation.

The Law of Total Expectation states that

$$
\mathbb{E}[X] = \sum_{y} \mathbb{E}[X \mid Y=y] p_Y(y)
$$

The tricky part is we need to sum all possible values of $Y$, which is an infinite set $\mathbb{N}$,
let's write it as follows:

$$
\mathbb{E}[X] = \sum_{n=0}^{\infty} \mathbb{E}[X \mid Y=n] p_Y(n)
$$

For $p_Y(n)$, we know that $Y$ follows a Poisson distribution with parameter $\lambda$, then we have

$$
\begin{aligned}
p_Y(n) &= \frac{\lambda^n e^{-\lambda}}{n!} \\
\end{aligned}
$$

For $\mathbb{E}[X \mid Y=n]$, we need to find a general expression for this, since this should be
a expression in terms of $X$, it means that $X \mid Y=n$ follows a Binomial distribution with parameters $n$ and $p$.

$$
\begin{aligned}
X \mid Y=n &\sim \text{Binomial}(n, p) \\
\end{aligned}
$$

where $n$ is the number of customers visiting the restaurant in a day, and $p$ is the probability of a customer
purchasing a drink. Then

$$
\mathbb{E}[X \mid Y=n] = np
$$

Putting everything together, we have

$$
\begin{aligned}
\mathbb{E}[X] &= \sum_{n=0}^{\infty} \mathbb{E}[X \mid Y=n] p_Y(n) \\
&= \sum_{n=0}^{\infty} np \frac{\lambda^n e^{-\lambda}}{n!} \\
&= p \sum_{n=0}^{\infty} n \frac{\lambda^n e^{-\lambda}}{n!} \\
&= p \sum_{n=0}^{\infty} n p_Y(n) \\
&= p \mathbb{E}[Y] \\
&= p \lambda \\
\end{aligned}
$$

where we used the fact that $\sum_{n=0}^{\infty} n p_Y(n) = \mathbb{E}[Y]$.