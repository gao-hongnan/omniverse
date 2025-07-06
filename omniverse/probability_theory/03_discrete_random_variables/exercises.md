# Exercises

```{contents}
```

## Problem 1

[Question 11 of Chapter 3 in Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/chapter3/3_3_0_chapter3_problems.php)

The number of emails that I get in a weekday (Monday through Friday) can be
modeled by a Poisson distribution with an average of $\frac{1}{6}$ emails per
minute. The number of emails that I receive on weekends (Saturday and Sunday)
can be modeled by a Poisson distribution with an average of $\frac{1}{30}$
emails per minute.

a) What is the probability that I get no emails in an interval of length 4 hours
on a Sunday?

b) A random day is chosen (all days of the week are equally likely to be
selected), and a random interval of length one hour is selected on the chosen
day. It is observed that I did not receive any emails in that interval. What is
the probability that the chosen day is a weekday?

a) Let $X$ be the number of emails received on a weekend, in a time interval of
length $1$ minute. Then $X \sim \text{Poisson}(\frac{1}{30})$ has a Poisson
distribution with parameter $\lambda = \frac{1}{30}$.

In Poisson's {ref}`poisson_assumptions`, the **linearity assumption** states
that the probability of an event occurring is proportional to the length of the
time period. As a consequence, the value of $\lambda$ is proportional to the
length of the time period.

And since the problem asked for a time period of length $4$ hours, we have that
the $\lambda$ is now,

$$
\lambda = 60 \times 4 \times \frac{1}{30} = 8
$$

This should be intuitive because $\lambda$ is the average number of occurences
of an event in a time period $T$. Thus, if in $1$ minute, there is
$\frac{1}{30}$ email, then in $240$ minutes (4 hours), there should be $8$
emails.

We can rephrase our initial statement as follows:

Let $X$ be the number of emails received on a weekend, in a time interval of
length $4$ hours. Then $X \sim \text{Poisson}(8)$ has a Poisson distribution
with parameter $\lambda = 8$.

Subsequently, the probability of getting no emails in a time interval of length
$4$ hours is given by

$$
\P \lsq X = 0 \rsq = \frac{e^{-8} 8^0}{0!} = e^{-8} \approx 3.4 \times 10^{-4}
$$

b) Let $X$ be the number of emails received on a weekday, in a time interval of
length $1$ hour, and
$X \sim \poisson \lpar 60 \times \frac{1}{6} \rpar = \poisson(10)$.

Let $Y$ be the number of emails received on a weekend, in a time interval of
length $1$ hour, and
$Y \sim \poisson \lpar 60 \times \frac{1}{30} \rpar = \poisson(2)$.

Let $Z$ be the disjoint union of $X$ and $Y$, and $Z = X \sqcup Y$. $Z$ is also
a random variable with a Poisson distribution $Z \sim \poisson(12)$. This is a
consequence of the the **additivity property** in Poisson's
{prf:ref}`prop_sum_poi`.

Let $W$ be a random chosen day in a week, and $W$ is a random variable with a
Uniform distribution $W \sim \uniform \lpar 1, 7 \rpar$.

Now we can further decompose $W$ to $W = W_1 \sqcup W_2$, where $W_1$ is the
random variable that indicates whether the chosen day is a weekday, and $W_2$ is
the random variable that indicates whether the chosen day is a weekend. Note in
particular that, $W$ is a random variable, a function $W: \S \to \R$, where $\S$
is the sample space, and $\R$ is the range of $W$, as indicated in
{prf:ref}`random_variables`.

The sample space $\S$ is just the set of days in a week, where we denote as
$\S = \lset 1, 2, 3, 4, 5, 6, 7 \rset$.

Similarly, $Z$ has a sample space that can be understood as the disjoint union
of the sample spaces of $X$ and $Y$. This is important as we want to invoke the
Law of Total Probability.

Now we can formulate the problem as follows:

$$
\P \lsq W_1 \lvert Z = 0 \rsq
$$

which means what is the probability of a chosen day is a weekday given that I
did not receive any emails in the time interval of $1$ hour of a random chosen
week.

By Bayes' Theorem ({prf:ref}`def:bayes-theorem`), we have that

$$
\begin{aligned}
\P \lsq W_1 \lvert Z = 0 \rsq &\defa \dfrac{\P \lsq Z = 0 \lvert W_1 \rsq \P \lsq W_1 \rsq}{\P \lsq Z = 0 \rsq} \\
       &\defb \dfrac{e^{-10} \cdot \frac{5}{7}}{\P \lsq Z = 0 \rsq} \\
\end{aligned}
$$ (eq:prob_w1_given_z0)

where $\defb$'s $e^{-10}$ is derived from the Poisson formula when $\lambda = 10$, because
$\P \lsq Z = 0 \lvert W_1 \rsq = \P \lsq X = 0 \rsq = e^{-10}$ since conditional probability shrinked
the sample space of $Z$ to $X$; the $\frac{5}{7}$ is just the probability of a chosen day is a weekday.

The denominator $\P \lsq Z = 0 \rsq$ is the probability of not receiving any emails in a time interval of length $1$ hour
of a random chosen week. This is not straightforward and we have to use the Law of Total Probability ({prf:ref}`thm:law-total-probability`).
That is,


$$

\begin{aligned} \P \lsq Z = 0 \rsq &= \P \lsq Z = 0 \lvert W_1 \rsq \P \lsq W_1
\rsq + \P \lsq Z = 0 \lvert W_2 \rsq \P \lsq W_2 \rsq \\ &= \P \lsq X = 0 \rsq
\P \lsq W_1 \rsq + \P \lsq Y = 0 \rsq \P \lsq W_2 \rsq \\ &= e^{-10} \cdot
\frac{5}{7} + e^{-2} \cdot \frac{2}{7} \\ \end{aligned}

$$

With this, we have solved the problem,


$$

\P \lsq W_1 \lvert Z = 0 \rsq = \dfrac{e^{-10} \cdot \frac{5}{7}}{e^{-10} \cdot
\frac{5}{7} + e^{-2} \cdot \frac{2}{7}} $$
