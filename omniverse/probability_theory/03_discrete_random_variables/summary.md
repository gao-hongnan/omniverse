---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Important

```{contents}
```

This section is a placeholder as need time to digest. Later on we can put the
content below back to their respective sections.

-   Revisit the idea of the histogram, even a section during PMF is good as a
    refresher!
-   Try to think about how to write PMF as an ideal histogram into my notes,
    especially from the machine learning perspective.
-   Early mention of empirical CDF vs CDF, empirical histogram vs PMF

## Important Brain Dump

Some important stuff here which I need to remember and yet to be able to
formalize.

**Always refer to my example setup.**

### Important Topics

See `0307_discrete_uniform_distribution` for my whole setup.

### The Empirical vs Theoretical Distribution Setup

-   https://inferentialthinking.com/chapters/intro.html
-   See `0307_discrete_uniform_distribution` for my whole setup.

### The Problem

#### The ticket model and what is a r.v

-   https://stats.stackexchange.com/questions/50/what-is-meant-by-a-random-variable/54894#54894
-   https://stats.stackexchange.com/questions/68599/distribution-of-correlation-coefficient-between-two-discrete-random-variables-an/68782#68782

-   A true population, say the height of all people in the world. To simplify
    the problem imagine living in a world of people whose height is a whole
    number ranging from 1-100 cm, ok I know it is absurd but it is just for the
    sake of the example, since discrete numbers are easier to visualize. Also
    secretly imagine the population is 1000 people with 100 people of each
    height (important here as we will see later).
-   Imagine a ticket box called the "population box" which has a ticket for each
    person in the world.
    -   every person in the world write their height on a ticket and put it in
        the box.
-   So note that we can think the population box as our sample space, each
    ticket is an outcome.
-   Note that it can be the case that more than 1 ticket has the same height, so
    to be very concise, our population box not a "set", so not a sample space.
-   Now recall that a random variable is a function that maps outcomes to real
    numbers.
-   In our case if we treat $X: \S \to \R$ as a r.v, then it is obvious that the
    sample space $\S$ is all from 0 to 100 and for this case our mapping to $\R$
    is just the identity function, since the height of a person is a real number
    ( a random variable is a way to assign a numerical code to each possible
    outcome). This aside, what is more important is that we say $X$ is a r.v.
    that represents the height of a person (when we pick 1 ticket). And the
    randomness comes from we don't know which ticket we pick, as it could be any
    person representing any height. But once **it is picked**, the
    **realization** of the r.v. is the height of the person, say hongnan with
    height 175 cm.

#### Why data points are considered random variables

-   See https://mathstat.slu.edu/~speegle/_book/SimulationRV.html
-   In ml context, Random variables $X_1, X_2, \ldots X_{???}$ are the data
    points of the height of people, and the true population space is the set of
    all possible data points (in our case it is actually 1 million people).
-   I was confused because $X: \S \to \R$ is a r.v. that represents the height
    of a person, say if $100 cm \in \S$, then $X(100) = 100 cm$ is the realized
    outcome. Then why do we need to index the data points? Because the mapping
    of $X$ is already well defined for any outcome in $\S$, so for each person
    in the world we already can represent the single random variable $X$ that
    represents the height of that person. So why do we need to index the data
    points?
-   For example, most cited definition is the iid assumption: random variables
    $X_1, X_2, \ldots X_{n}$ are called independent and identically distributed
    or iid if the variables are mutually independent, and each $X_i$ has the
    same probability distribution. Say $n=10$ people. It turns out we should
    think of it this way, in the true population box, all the tickets (height)
    of the people are **numbered**, and each $X_i$ is actually remember is a
    deterministic answer after realization, and therefore the numbering makes
    sense. We treat each draw of the ticket as a random variable, and the
    numbering is just a way to index the random variables.
-   Furthermore, we usually take a random sample of size $n$ from the population
    box, and treat each draw as a rv.

#### Super important

Above has a hazy concept, one one hand sample space is a set and therefore
should be unique, but on the other hand, if there's 1000 people in the true
population, and we treat the population box as the sample space, then the sample
space is no longer "unique" since there are only 100 distinct heights? However,
if we number the tickets, then the sample space is unique, in a way we are
playing abuse of notation that two different people with the same height 175 cm
are two distinct outcomes in the sample space. This is very important to
realize!!!

#### Empirical distribution/histogram

To put more concrete example, consider the same experiment above, define the rv
$X$ to be the height of a person, then find probability of getting a person with
height 175 cm, say $P(X=175) = ?$.

To find this answer, we need to find PMF. Note PMF is ideal means it is
deterministic and hinged upon the true population.

So we secretly know the true PMF of the above distribution is actually simply

$$
\begin{align}
\P(X=x) = \frac{10}{1000} = \frac{1}{10} \quad \forall x \in \{1, \ldots, 100\}
\end{align}
$$

$$
\begin{align}
\P(X=x) = \begin{cases}
\frac{1}{10} \text{ if } x=1 \\
\frac{1}{10} \text{ if } x=2 \\
\frac{1}{10} \text{ if } x=3 \\
\vdots \\
\frac{1}{10} \text{ if } x=100 \\
\end{cases}
\end{align}
$$

since it is equally likely to get any person with height 1-100 cm over 1000
people. Note that for each height, we have exactly 10 people.

Recall the example

$$
\P(A) = \dfrac{n(A)}{n(\S)}
$$

and since our $\S$ is 1000 people.

Now we randomly pick 10 people from the true population, then we can plot a
histogram of the heights of the 10 people, and the histogram says 3 people have
height 175 cm, 2 people have height 180 cm, and so on. This is the empirical
distribution, and it is not the true PMF, and in our case we have
$\dfrac{3}{10}$ probability that a person has 175cm $\P(X=175) = \dfrac{3}{10}$.
But note carefully here is **empirical** distribution and is non-deterministic.

## Questions Asked

-   https://stats.stackexchange.com/questions/590792/empirical-histogram-and-pmf
-   https://stats.stackexchange.com/questions/593079/interpretation-of-bernoulli-and-binomial-random-variables-in-a-call-simulation-c/593084?noredirect=1#comment1097323_593084

### Ideal histogram

-   Experiment: roll a dice once to see what number comes up.
-   Prof drew a very important diagram towards the end of lecture 10, on how
    histograms tend to their PMF as a number of experiments tend to infinity,
    and how each histogram mean (sample mean) tends to its PMF's expectation
    (population mean). I believe this part is very very important and needs to
    be bookmarked in my notes and treated as a header. This is because he
    mentioned how sample mean is random but population mean (expectation) is
    deterministic, which implies histograms are random (i.e randomly drawn
    samples from the population while PMF is deterministic (?)).

### Expectation is deterministic

### Variance

-   In lecture video around 28 minute onwards prof gave a good intuition of
    visualizing variance.
-   Intuition why why DC shift dont shift var.
-   Min 45 onwards PMF vs Histogram same holds for sample variance vs population
    variance etc.
