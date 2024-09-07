# Concept

## Conditional PMF

```{prf:definition} Conditional PMF
:label: def:conditional-pmf

Let $X$ and $Y$ be two discrete random variables with joint PMF $p_{X, Y}(x, y)$.
The **conditional PMF** of $X$ given $Y$ is

$$
p_{X \mid Y}(x \mid y)=\frac{p_{X, Y}(x, y)}{p_{Y}(y)} .
$$ (eq:conditional-pmf-2d)
```

```{prf:remark} Some Intuition
:label: remark:conditional-pmf

It is relatively easy to associate the definition of conditional PMF with the definition of conditional probability
in the chapter on [Conditional Probability](../../02_probability/0204_conditional_probability.md).

Indeed, $p_{X \mid Y}(x \mid y)$ can be viewed as $\mathbb{P}[X=x \mid Y=y]$.

To see why this perspective makes sense, let us recall the definition of a conditional probability:

$$
\begin{aligned}
p_{X \mid Y}(x \mid y) &\overset{(a)}{=} \frac{p_{X, Y}(x, y)}{p_{Y}(y)} \\
&\overset{(b)}{=} \frac{\mathbb{P}[X=x \cap Y=y]}{\mathbb{P}[Y=y]}\\
&\overset{(c)}{=} \mathbb{P}[X=x \mid Y=y] .
\end{aligned}
$$

In $(a)$, we unpack the definition of conditional PMF. In $(b)$, we convert the numerator
to our familiar notation $\mathbb{P}[X=x \cap Y=y]$, and the denominator to $\mathbb{P}[Y=y]$.
In $(c)$, we use the definition of conditional probability. We see that the definition of conditional PMF
is equivalent to the definition of conditional probability.
```

```{prf:remark} Conditional Distribution is a Distribution for a Sub-Population
:label: remark:conditional-distribution-is-a-distribution-for-a-sub-population

Here's a very important concept mentioned by Professor Chan. in his book. To interpret
the notation $p_{X \mid Y}(x \mid y)$, we need to be clear that there is nothing random about
the random variable $Y$, since it is already fixed. What remains is a sub-population distribution
in terms of $X$ (i.e. the reduced sample space of $\Omega_X$ where $Y$ happened). Consequently,
the conditional PMF $p_{X \mid Y}(x \mid y)$ is a distribution in terms of $X$, and is in fact
a legitimate probability mass function.
```

```{prf:example} Two Coins
:label: ex:two-coins

This example is taken from Professor Chan's book, {cite}`chan_2021`.

Consider a joint PMF given in the following table. Find the conditional
$\operatorname{PMF~}_{X \mid Y}(x \mid 1)$ and the marginal $\operatorname{PMF} p_{X}(x)$.

$$
\begin{array}{r|cccc}
& {Y=1} & {Y=2} & {Y=3} & {Y=4} \\
\hline \mathrm{X}=1 & \frac{1}{20} & \frac{1}{20} & \frac{1}{20} & \frac{0}{20} \\
\mathrm{X}=2 & \frac{1}{20} & \frac{2}{20} & \frac{3}{20} & \frac{1}{20} \\
\mathrm{X}=3 & \frac{1}{20} & \frac{2}{20} & \frac{3}{20} & \frac{1}{20} \\
\mathrm{X}=4 & \frac{0}{20} & \frac{1}{20} & \frac{1}{20} & \frac{1}{20}
\end{array}
$$

Before we even look at the solution in the book, we can test our conceptual understanding using
intuition. Since we want to find the distribution of $X$ given $Y=1$ has happened, this means our
sub-population is restricted to $X$ where $Y=1$ has happened. This means we are only interested in
the column $Y=1$ in the table.

In that sub-population, the total is $\frac{3}{20}$, and for each state $x=1, 2, 3, 4$, the probability
is merely the fraction of the total:

$$
\begin{array}{ll}
x=1: & \frac{\frac{1}{20}}{\frac{3}{20}}=\frac{1}{3}, \\
x=2: & \frac{\frac{1}{20}}{\frac{3}{20}}=\frac{1}{3}, \\
x=3: & \frac{\frac{1}{20}}{\frac{3}{20}}=\frac{1}{3}, \\
x=4: & \frac{\frac{0}{20}}{\frac{3}{20}}=0, \\
\end{array}
$$

It turns out our intuition is correct.
```

```{prf:theorem} Conditional PMF of an Event $A$ Given $Y=y$
:label: thm:conditional-pmf-of-an-event-a-given-y

Let $X$ and $Y$ be two discrete random variables, and let $A$ be an event.

Then the probability $\mathbb{P}[X \in A \mid Y=y]$ is given by

$$
\mathbb{P}[X \in A \mid Y=y]=\sum_{x \in A} p_{X \mid Y}(x \mid y)
$$

and the probability $\mathbb{P}[X \in A]$ is given by

$$
\mathbb{P}[X \in A]=\sum_{x \in A} \sum_{y \in \Omega_{Y}} p_{X \mid Y}(x \mid y) p_{Y}(y)=\sum_{y \in \Omega_{Y}} \mathbb{P}[X \in A \mid Y=y] p_{Y}(y) .
$$
```

```{prf:definition} Conditional CDF
:label: def:conditional-cdf-of-a-discrete-random-variable

Let $X$ and $Y$ be discrete random variables. Then the conditional CDF of $X$ given $Y=y$ is

$$
F_{X \mid Y}(x \mid y)=\mathbb{P}[X \leq x \mid Y=y]=\sum_{x^{\prime} \leq x} p_{X \mid Y}\left(x^{\prime} \mid y\right) .
$$
```

## Conditional Independence

```{prf:definition} Conditional Independence
:label: def:conditional-independence

Two discrete random variables $X$ and $Y$ are conditionally independent given a third discrete random variable $Z$ if and only if they are independent in their conditional probability distribution given $Z$. That is, $X$ and $Y$ are conditionally independent given $Z$ if and only if, given any value of $Z$, the probability distribution of $X$ is the same for all values of $Y$ and the probability distribution of $Y$ is the same for all values of $X$. Formally:

$$
(X \perp Y) \mid Z \quad \Longleftrightarrow \quad F_{X, Y \mid Z=z}(x, y)=F_{X \mid Z=z}(x) \cdot F_{Y \mid Z=z}(y) \quad \text { for all } x, y, z
$$

where $F_{X, Y \mid Z}=z(x, y)=\operatorname{Pr}(X \leq x, Y \leq y \mid Z=z)$ is the conditional cumulative distribution function of $X$ and $Y$ given $Z$.

In terms of events $A$, $B$ and $C$, we can write

$$
\mathbb{P}(A, B \mid C)=\mathbb{P}(A \mid C) \cdot \mathbb{P}(B \mid C) \quad \text { for all } A, B, C
$$
```

```{prf:example} Conditional Independence
:label: example:conditional-independence

The example is from [Dive into Deep Learning, Section 2.6.4](https://d2l.ai/chapter_preliminaries/probability.html#multiple-random-variables).

Interestingly, two variables can be independent in general but become dependent when conditioning on a third. This often occurs when the two random variables  and  correspond to causes of some third variable . For example, broken bones and lung cancer might be independent in the general population but if we condition on being in the hospital then we might find that broken bones are negatively correlated with lung cancer. That’s because the broken bone explains away why some person is in the hospital and thus lowers the probability that they have lung cancer.

And conversely, two dependent random variables can become independent upon conditioning on a third. This often happens when two otherwise unrelated events have a common cause. Shoe size and reading level are highly correlated among elementary school students, but this correlation disappears if we condition on age.
```


## Conditional PDF

We now discuss the conditioning of a continuous random variable.

```{prf:definition} Conditional PDF
:label: def:conditional-pdf-of-a-continuous-random-variable

Let $X$ and $Y$ be two continuous random variables. The conditional PDF of $X$ given $Y$ is

$$
f_{X \mid Y}(x \mid y)=\frac{f_{X, Y}(x, y)}{f_{Y}(y)} .
$$ (eq:conditional-pdf-2d)
```

Even though {eq}`eq:conditional-pdf-2d` is in the [form of Bayes' rule](https://en.wikipedia.org/wiki/Bayes%27_theorem#Simple_form), this definition becomes hazy since we cannot treat the conditional PDF of a continuous random variable
the same way we treat the discrete counterpart earlier. That is to say, we cannot just say that the conditional
PDF stems from the Bayes' rule. This is because the
denominator $p_{Y}(y)$ is already $0$ by definition (i.e. $\mathbb{P}[Y=y]=0$) when
$Y$ is continuous.

To answer this question, we first define the conditional $\mathrm{CDF}$ for continuous random variables.

```{prf:definition} Conditional CDF
:label: def:conditional-cdf-of-a-continuous-random-variable

Let $X$ and $Y$ be continuous random variables. Then the conditional CDF of $X$ given $Y=y$ is

$$
F_{X \mid Y}(x \mid y)=\frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y\right) d x^{\prime}}{f_{Y}(y)} .
$$
```

Professor Chan probed further by asking why should the conditional CDF of continuous random variable be defined in this way? One way to interpret $F_{X \mid Y}(x \mid y)$ is as the limiting perspective.

```{prf:definition} Limiting Perspective
:label: def:limiting-perspective

We can define the conditional CDF as

$$
\begin{aligned}
F_{X \mid Y}(x \mid y) & =\lim _{h \rightarrow 0} \mathbb{P}(X \leq x \mid y \leq Y \leq y+h) \\
& =\lim _{h \rightarrow 0} \frac{\mathbb{P}(X \leq x \cap y \leq Y \leq y+h)}{\mathbb{P}[y \leq Y \leq y+h]}
\end{aligned}
$$ (eq:limiting-perspective)

This should not come as a surprise as we are merely taking the limit for the conditioned variable
$Y$ here, since $Y$ being continuous is the one giving us troubles.
```

With some calculations, we can express {eq}`eq:limiting-perspective` in terms of the conditional PDF of $X$ given $Y$.

$$
\begin{aligned}
\lim _{h \rightarrow 0} \frac{\mathbb{P}(X \leq x \cap y \leq Y \leq y+h)}{\mathbb{P}[y \leq Y \leq y+h]} & =\lim _{h \rightarrow 0} \frac{\int_{-\infty}^{x} \int_{y}^{y+h} f_{X, Y}\left(x^{\prime}, y^{\prime}\right) d y^{\prime} d x^{\prime}}{\int_{y}^{y+h} f_{Y}\left(y^{\prime}\right) d y^{\prime}} \\
& =\lim _{h \rightarrow 0} \frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y^{\prime}\right) d x^{\prime} \cdot h}{f_{Y}(y) \cdot h} \\
& =\frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y^{\prime}\right) d x^{\prime}}{f_{Y}(y)} .
\end{aligned}
$$

The key here is that the small step size $h$ in the numerator and the denominator will cancel each other out. Now, given the conditional CDF, we can verify the definition of the conditional PDF. It holds that

$$
\begin{aligned}
f_{X \mid Y}(x \mid y) & =\frac{d}{d x} F_{X \mid Y}(x \mid y) \\
& =\frac{d}{d x}\left\{\frac{\int_{-\infty}^{x} f_{X, Y}\left(x^{\prime}, y\right) d x^{\prime}}{f_{Y}(y)}\right\} \stackrel{(a)}{=} \frac{f_{X, Y}(x, y)}{f_{Y}(y)},
\end{aligned}
$$

where (a) follows from the fundamental theorem of calculus ({prf:ref}`fundamental_theorem_of_calculus`).

Just like the conditional PMF, we can calculate the probabilities using the conditional PDFs. In particular, if we evaluate the probability where $X \in A$ given that $Y$ takes a particular value $Y=y$, then we can integrate the conditional PDF $f_{X \mid Y}(x \mid y)$, with respect to $x$.

```{prf:theorem} Conditional PDF of an Event $A$ Given $Y=y$
:label: thm:conditional-pdf-of-an-event-given-y

Let $X$ and $Y$ be continuous random variables, and let $A$ be an event.

Then the probability $\mathbb{P}\left[X \in A \mid Y=y\right]$ is given by

$$
\mathbb{P}[X \in A \mid Y=y]=\int_{A} f_{X \mid Y}(x \mid y) d x
$$

and the probability $\mathbb{P}\left[X \in A\right]$ is given by

$$
\mathbb{P}[X \in A]=\int_{\Omega_{Y}} \mathbb{P}[X \in A \mid Y=y] f_{Y}(y) d y
$$
```

