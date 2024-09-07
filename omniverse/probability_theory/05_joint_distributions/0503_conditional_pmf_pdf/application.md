# Application

## Medical Diagnosis

This example is taken from [Dive into Deep Learning, Section 2.6.5](https://d2l.ai/chapter_preliminaries/probability.html#an-example).

Assume that a doctor administers an HIV test to a patient.
This test is fairly **accurate** and it fails only with $1\%$ probability
if the patient is healthy but reporting him as diseased.
Moreover, it never fails to detect HIV if the patient actually has it.

In machine learning lingo, we can treat the diagnosis as a **classifier**. More concretely,
let $Y$ be the ground truth label of the patient, in this case, $Y \in \{0, 1\}$,
where $0$ (negative) indicates that the patient is healthy and $1$ (positive)
indicates that the patient has HIV.

Let $\hat{Y}$ be the hard label predicted by the test, i.e., $\hat{Y} \in \{0, 1\}$.

To keep the notation similar to the original example, denote the first test as $\hat{Y}_1$ and the second test as $\hat{Y}_2$.

Then we can define the following to describe the relationship between $Y$ and $\hat{Y}$:

| Conditional probability          | $Y=1$ | $Y=0$ |
| :------------------------------- | ----: | ----: |
| $\mathbb{P}(\hat{Y} = 1 \mid Y)$ |     1 |  0.01 |
| $\mathbb{P}(\hat{Y} = 0 \mid Y)$ |     0 |  0.99 |

In machine learning, a similar way is to use a **confusion matrix**.

Indeed, the true positive in the above table is $100\%$, which means that if a patient
has HIV, the predictions will always predict correctly. Consequently, the false negative
is $0\%$. On the other hand, the false positive is $1\%$, which means that if a patient
is healthy but the predictions predict him as diseased, the probability is $1\%$. Consequently,
the true negative is $99\%$, which means that if a patient is healthy, the predictions
indeed predict correctly with $99\%$ probability.

Note that the column sums are all 1 (but the row sums don't),
since they are conditional probabilities.

For our first task, let's compute the probability of the patient having HIV
if the (first) test (classifier) comes back (predicts) positive,

$$
P(Y = 1 \mid \hat{Y}_1 = 1) = \frac{P(\hat{Y}_1 = 1 \mid Y = 1) P(Y = 1)}{P(\hat{Y}_1 = 1)}.
$$

Intuitively this is going to depend on how common the disease is,
since it affects the number of false alarms.

We further assume the prior probability of the patient having HIV is $0.0015$.

$$
\mathbb{P}(Y = 1) = 0.0015.
$$

Then we can invoke Bayes' theorem,

$$
\begin{aligned}
P(Y = 1 \mid \hat{Y}_1 = 1)
=& \frac{P(\hat{Y}_1 = 1 \mid Y = 1) P(Y = 1)}{P(\hat{Y}_1 = 1)} \\
\end{aligned}
$$

We already know $\mathbb{P}(\hat{Y}_1 = 1 \mid Y = 1) = 1$ and $\mathbb{P}(Y = 1) = 0.0015$.
We just need to compute $\mathbb{P}(\hat{Y}_1 = 1)$, where we need to apply marginalization
{prf:ref}`def-marginal-pmf-pdf` to determine.

$$
\begin{aligned}
\mathbb{P}(\hat{Y}_1 = 1) =& \mathbb{P}(\hat{Y}_1 = 1, Y = 1) + \mathbb{P}(\hat{Y}_1 = 1, Y = 0) \\
=& \mathbb{P}(\hat{Y}_1 = 1 \mid Y = 1) \mathbb{P}(Y = 1) + \mathbb{P}(\hat{Y}_1 = 1 \mid Y = 0) \mathbb{P}(Y = 0) \\
=& 1 \times 0.0015 + 0.01 \times 0.9985 \\
=& 0.011485.
\end{aligned}
$$

A point to note, $\mathbb{P}(\hat{Y}_1 = 1, Y = 1) + \mathbb{P}(\hat{Y}_1 = 1, Y = 0)$ makes us the sum of the probabilities of the patient having HIV and not having HIV, respectively, while fixing $\hat{Y}_1 = 1$. This is the same as the marginal probability of $\hat{Y}_1 = 1$. And in turn we can use the law of total probability to compute the marginal probability of $\hat{Y}_1 = 1$.

This leads us to

$$
\begin{aligned}
\mathbb{P}(Y = 1 \mid \hat{Y}_1 = 1) =& \frac{1 \times 0.0015}{0.011485} \\
=& 0.1306.
\end{aligned}
$$

In other words, there is only a $13.06\%$ chance
that the patient actually has HIV,
despite using a very accurate test.
As we can see, probability can be counterintuitive.

What should a patient do upon receiving such terrifying news?
Likely, the patient would ask the physician
to administer another test to get clarity.
The second test has different characteristics
and it is not as good as the first one. You can think of it as using
a second type of classifier (i.e. a different model) to predict the label of the patient.

| Conditional probability            | $Y=1$ | $Y=0$ |
| :--------------------------------- | ----: | ----: |
| $\mathbb{P}(\hat{Y}_2 = 1 \mid Y)$ |  0.98 |  0.03 |
| $\mathbb{P}(\hat{Y}_2 = 0 \mid Y)$ |  0.02 |  0.97 |

Unfortunately, the second test comes back positive, too.
Now our question becomes, what is the probability that the patient has HIV
given that both tests come back positive? Formally stated as below:

$$
P(Y = 1 \mid \hat{Y}_1 = 1, \hat{Y}_2 = 1).
$$ (eq:prob-hiv-both-tests)

As usual, we represent {eq}`eq:prob-hiv-both-tests` using Bayes' theorem:

$$
\begin{aligned}
P(Y = 1 \mid \hat{Y}_1 = 1, \hat{Y}_2 = 1)
=& \frac{P(\hat{Y}_1 = 1, \hat{Y}_2 = 1 \mid Y = 1) P(Y = 1)}{P(\hat{Y}_1 = 1, \hat{Y}_2 = 1)} \\
\end{aligned}
$$

Now, $\mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1 \mid Y = 1)$ is the probability that both tests come back positive given that the patient has HIV. This involves the joint probability of the two tests, and is not easily computed. Instead, we can use the conditional independence assumption to simplify the computation. More concretely,
we assume that $\hat{Y}_1$ and $\hat{Y}_2$ are independent given $Y$. This means,
given $Y=1$, the probability that both tests come back positive is the product of the probabilities that each test comes back positive.

$$
\begin{aligned}
\mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1 \mid Y = 1) =& \mathbb{P}(\hat{Y}_1 = 1 \mid Y = 1) \mathbb{P}(\hat{Y}_2 = 1 \mid Y = 1) \\
=& 1 \times 0.98 \\
=& 0.98
\end{aligned}
$$

We now need to compute $P(\hat{Y}_1 = 1, \hat{Y}_2 = 1)$, which is the marginal probability of both tests coming back positive. We can use the law of total probability to compute this.

$$
\begin{aligned}
\mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1) =& \mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1, Y = 1) + \mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1, Y = 0) \\
=& \mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1 \mid Y = 1) \mathbb{P}(Y = 1) + \mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1 \mid Y = 0) \mathbb{P}(Y = 0) \\
=& 0.98 \times 0.0015 + 0.0003 \times 0.9985 \\
=& 0.00176955
\end{aligned}
$$

Note that $\mathbb{P}(\hat{Y}_1 = 1, \hat{Y}_2 = 1 \mid Y = 0)$ is calculated similarly using
the conditional independence assumption.

Finally, we can compute the probability that the patient has HIV given that both tests come back positive.

$$
\begin{aligned}
\mathbb{P}(Y = 1 \mid \hat{Y}_1 = 1, \hat{Y}_2 = 1) =& \frac{0.98 \times 0.0015}{0.00176955} \\
=& 0.8307
\end{aligned}
$$

That is, the second test allowed us to gain much higher confidence that not all is well.
Despite the second test being considerably less accurate than the first one,
it still significantly improved our estimate.
The assumption of both tests being conditional independent of each other
was crucial for our ability to generate a more accurate estimate.
Take the extreme case where we run the same test twice.
In this situation we would expect the same outcome in both times,
hence no additional insight is gained from running the same test again.
The astute reader might have noticed that the diagnosis behaved
like a classifier hiding in plain sight
where our ability to decide whether a patient is healthy
increases as we obtain more features (test outcomes).

(simplified-probit-model)=
## Simplified Probit Model

This example is taken from {cite}`chan_2021`, section 5.3.2, example 5.20.

Let $X$ be a random bit such that

$$
X= \begin{cases}+1, & \text { with prob } 1 / 2 \\ -1, & \text { with prob } 1 / 2\end{cases}
$$

Suppose that $X$ is transmitted over a noisy channel so that the observed signal is

$$
Y=X+N,
$$

where $N \sim \operatorname{Gaussian}(0,1)$ is the noise, which is independent of the signal $X$. Find the probabilities $\mathbb{P}[X=+1 \mid Y>0]$ and $\mathbb{P}[X=-1 \mid Y>0]$.

1. $Y=X+N$ such that $N$ is a normal random variable with mean 0 and variance 1.
   1. Note that $X$ and $N$ are independent by definition. This means that $N$ happening does
        not change the probability of $X$ happening. Ask why $N = Y - X$ does not imply that $X$
        is dependent on $N$.
   2. Note that $X$ and $Y$ are **not** independent. Can we justify or our intuition is wrong.

2. To find $P[X=1 | Y>0]$, we need the following:
   1. We first recall that to find the conditional probability, we need to find the conditional PDF $f_{X|Y}(x|y)$ first, or more concretely, $f_{X|Y}(x=1|y>0)$.
   2. We first note $f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$.
   3. This is by definition of conditional probability. We can see that in  section 5.3.1 and also in the chapter on conditional probability.
   4. In particular, note that $f_{X, Y}(x, y) = f_X(x) \cap f_Y(y)$, so it is indeed the numerator of the conditional probability.
   5. Overall, it is not clear how to find $f_Y(y)$, though we can find $f_X(x)$.
   6. We will leave finding the denominator for later.
   7. Note that $P[X=1|Y>0]$ is equivalent to integrating $f_{X|Y}(x=+1|y>0)$ for all $y>0$. But we will soon hit a wall when we try to find an expression form for this PDF, furthermore, we could make use of the fact that the marginal PDF of $X$ is given to solve this problem.
   8. Now we instead use Bayes to say that

    $$
    P[X=+1|Y>0] = \dfrac{P[Y>0|X=+1]P[X=+1]}{P[Y>0]}
    $$

    which translates to finding the RHS of the equation. Note the numerator is a consequence of $P(X = +1, Y > 0) = P(Y > 0 | X = +1)P(X = +1)$, which is the definition of conditional probability. The denominator is the marginal probability of $Y>0$, which we will find later.

   9. Note $P[X=+1]$ is trivially equals to $\frac{1}{2}$, since $X$ is a Bernoulli random variable with $p=0.5$. Even though it is not mentioned explicitly, we can assume that $X$ is a Bernoulli random variable with $p=0.5$ since it does seem to fulfil the definition of a Bernoulli random variable provided it is independent trials.

   10. Now to find $P[Y>0|X=+1]$, we need to find $f_{Y|X}(y>0|x=+1)$.

3. To find the conditional distribution $f_{Y|X}(y>0|x=+1)$, we first must be clear that this is a conditional PDF and not a probability yet, i.e. $P[Y>0|X=1]$ is found by integrating this PDF! We must also be clear that this probability is all about $y$ and therefore we will integrate over $dy$ only instead of the usual double integral. Why? Because we are given $X=+1$, this means $X$ is fixed and there is nothing ***random*** about it, you can imagine in the 2D (3D) space PDF where the axis $X$ is fixed at 1, and we are integrating over the curve under $Y>0$ with $X=1$, i.e. $(x=1, y=0.1), (x=1, y=0.2), \ldots$
   1. Now the difficult question is what is $f_{Y|X}(y>0|x=1)$? We can find clues by looking at the equation $Y=X+N$. In laymen terms, $Y=X+N$ means what is $Y$ given $X=1$? So we can simplify $Y=X+N$ to $Y=1+N$. We emphasise that this PDF is a function of $y$ only, and not $x$. But this does not mean $f_{Y|X} = f_Y$, which we will soon see.
   2. Next, by the definition of shifting (linear transformation), if $N$ is a normal random variable of mean $\mu$ and $\sigma$, then shifting it by $1$ merely shifts the mean by $1$ and the variance remains the same [^1]. This shows that $Y$ is actually still a gaussian family, same as $N$, but with a different mean and same variance.
   3. Therefore, $Y=1+N$ is a normal random variable with mean $1+\mu$ and variance $\sigma^2$, $Y \sim \mathcal{N}(1+\mu, \sigma^2)$. With $\mu=0$ and $\sigma=1$, we have $Y \sim \mathcal{N}(1, 1)$.
   4. Now we can find $f_{Y|X}(y>0|x=1)$ by plugging in $y>0$ into the PDF of $\mathcal{N}(1, 1)$, which is $f_{Y|X}(y>0|x=1) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y-1)^2}$. Note that this is a PDF, not a probability yet.
   5. To recover the probability, we must integrate over $dy$.

    $$
    P[Y>0|X=1] = \int_{y>0} \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y-1)^2} dy = \int_{0}^{\infty} \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y-1)^2} dy
    $$

    this is because $y>0$ is equivalent to $0<y<\infty$.

   6. We can now use the standard normal table to find the probability, which is $0.8413$. See chan's solution which is $1 - \Phi(-1) = 0.8413$.
   7. Similarly, we can find $P[Y>0|X=-1]$ by plugging in $y>0$ into the PDF of $\mathcal{N}(-1, 1)$, which is $f_{Y|X}(y>0|x=-1) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}(y+1)^2}$. We can then integrate over $y>0$ to find the probability, $1-\Phi(1)$.

4. As of now, we have recovered $P[X=+1]$ and $P[Y>0|X=+1]$, what is left is the denominator $P[Y>0]$. By the law of total probability, we have

    $$
    \begin{aligned}
        P[Y>0] &= P[Y>0|X=+1]P[X=+1] \\
        &+ P[Y>0|X=-1]P[X=-1]
    \end{aligned}
    $$

    which is $0.8413 \times 0.5 + 0.1587 \times 0.5 = 0.5$.

5. Finally, we can now recover $P[X=+1|Y>0]$ by plugging in the values we have found.

    $$
    P[X=+1|Y>0] = \dfrac{P[Y>0|X=+1]P[X=+1]}{P[Y>0]} = \dfrac{0.8413 \times 0.5}{0.5} = 0.8413
    $$

    which is the same as the answer given in the question.

6. Last but not least, to find $P[X=-1|Y>0]$, it is simply the complement of $P[X=+1|Y>0]$, which is $1 - 0.8413 = 0.1587$.

    $$
    P[X=-1|Y>0] = 1 - P[X=+1|Y>0] = 1 - 0.8413 = 0.1587
    $$

    which is the same as the answer given in the question.

[^1]: You can easily plot it out to see that the bell curve shifting 1 on the x axis merely shifts the curve right by 1, and since mean is the center of the bell curve, the mean is shifted by 1. The variance remains the same because the bell curve is symmetric about the mean, and the variance is the width of the bell curve, which remains unchanged.
