# Probability Space

```{contents}
:local:
```

Most definitions and theorems below are cited from {cite}`chan_2021`.

## Experiment

Real-world scenarios often involve chance. We can model such scenarios
mathematically. For this purpose, we\'ll use a mathematical object named the
**experiment** (probability space), typically denoted as an ordered triplet:

$$\text{Experiment} = (\S, \E, \P)$$

```{prf:definition} Experiment/Probability Space
:label: experiment

An **experiment** is an ordered triplet defined as

$$\text{Experiment} = (\S, \E, \P)$$

where:

- $\S$ is a set of possible outcomes of the experiment;
- $\E$ is a set of events;
- $\P$ is a probability measure on $\E$.
```

We will go through the notations in the following sections.

## Sample Space $\mathbf{\Omega}$

```{prf:definition} Sample Space $\mathbf{\Omega}$
:label: sample_space

A **sample space** $\Omega$ is a set of all possible outcomes of an experiment.
We denote $\xi \in \Omega$ as an *outcome*.

Note that a **sample space** can be **uncountable**.
```

## Event $E$ and Event space $\mathcal{F}$

```{prf:definition} Event $E$
:label: event

An **event** $E$ is a subset of the sample space $\Omega$.
```

```{prf:definition} Event Space $\mathcal{F}$
:label: event_space

An **event space** $\mathcal{F}$ is the set of all possible events $E$.
```

## Probability Law and Function $\mathbb{P}$

```{prf:definition} Probability Law
:label: probability_law

A **probability law** is a probability measure on the event space $\mathcal{F}$.

More concretely, a probability law is a mapping defined as

$$
\begin{align}
    \P: \mathcal{F} &\to [0, 1] \\
    E &\mapsto \P(E)
\end{align}
$$
```

## Probability Law is a Measure

The notion of measure is important for us to understand.

See chapter 2, section 2.2.3 of {cite}`chan_2021` for more details.

## Measure Zero Sets

This definition here will be more obvious when we visit **continuous
distributions** where it does not make sense to ask the question of the
probability of a single point as it will always be zero.

```{prf:definition} Measure Zero Sets
:label: measure_zero_sets

Let $\S$ be the sampel space. A set $A \in \S$ is said to have **measure zero**
if for any $\epsilon > 0$,

- There exists a countable number of subsets $A_i \in \S$ such that
  $A = \bigcup_{i=1}^{\infty} A_i$ and
- $\sum_{i=1}^{\infty} \P\lsq A_i \rsq < \epsilon$.
```

## Example of Probability Space

Real-world scenarios often involve chance. We can model such scenarios
mathematically. For this purpose, we\'ll use a mathematical object named the
**experiment** (probability space), typically denoted as an ordered triplet:

$$\text{Experiment} = (\S, \E, \P)$$

defined as such:

-   **Sample Space $\S$:** This is simply the set of all possible outcomes. And
    an outcome is defined as a result of an experiment.

-   **Event Space $\E$:** This is the collection of all possible events. An
    event $E$ is simply any set of possible outcomes, which means that an event
    is a subset of the **sample space**. In turn, the event space $\E$ is simply
    the set of all events.

-   **Probability Function $\P$:** A mapping function from an event $E$ to a
    number $\P(E)$ which ideally measures the size of the event; usually, $\P$
    simply assigns to each event some probability between $0$ and $1$. This
    probability is interpreted as the likelihood of that particular event
    occurring.

```{list-table}
:header-rows: 1
:name: table:probability_space

* - Definition
  - Example
* - An Experiment is a scenario involving chance or probability
  - Throwing a coin
* - **Outcome**: Result of an experiment/Probability Space
  - When you toss a coin, only outcomes are **Heads (H) or Tails (T)**
* - **Sample Space**: Set of all possible outcomes
  - $\S = \{\text{Heads, Tails}\}$
* - **Event Space**: Set of all possible events
  - $\E = \{\emptyset, \{\text{Heads}\}, \{\text{Tails}\}, \{\text{Heads, Tails}\}\}$
* - **Events**: Subset of the Sample Space
  - An event A can be getting a heads = $\{\text{Heads}\}$
* - **Probability Function**: $\P: \E \to \R$ assigns to each event a $\textbf{probability}$ which is a number between 0 and 1
  - $\P(\{\text{Heads}\}) = \frac{1}{2}$
```

```{admonition} Note
:class: note

If the sample space $\S$ consists of a
finite number of equally likely outcomes, the probability of an event
$A$, denoted by $\P(A)$, is given by:

$$
\P(A) = \dfrac{n(A)}{n(\S)}
$$

where $n(A)$ denotes the number of outcomes in the event $A$. So if
event $A$ is the event where you roll a 6, we consider $n(A)$ to
be 1 since $A = \lset 6 \rset$ implies that there is only one
outcome. Therefore, $\P(A) = \frac{1}{6}$ as $n(\S) = 6$ (6 outcomes
for a dice roll).
This will be more formally stated later, but for now, understand
the probability function as a frequency counter.
```

## The Notion of Experiments and Defining the Probability Space

### How to define a Sample Space?

Consider the following example:

> We have 10 balls, 6 red and 4 black. We also know that 3 of the red balls is
> round, and only 1 of the black balls is round. The rest of the balls which are
> not round are assumed to be square (may not be important here). So we were
> asked, what is the probability of picking one of the ball that is round, given
> that the ball is red? What is our sample space $\S$, event space $\E$?

The beginner in probability (me) might not immediately see what the experiment
is, and therefore cannot construct the sample space $\S$ immediately.

In this context, we can define our **experiment** to be the action of picking a
ball from the $10$ balls. Notice that we did not mention any adjectives on the
ball - we did not specify if the ball is red, blue or round. This can be
specified here, but can also be mentioned in the sample space later.

Now to define the sample space $\S$, it is necessarily for us to know what
**outcomes** are there from the **experiment**? Well, from the problem, we know
that the balls we pick can be red, black, round or square (not round). So our
naive construction can be:

$$
\S = \{\text{red}, \text{black}, \text{round}, \text{not round}\}
$$

but this soon does not make sense because this **sample space** does not obey
the **Probability Axioms (which we will go in depth in the next section)**, a
simple inspection tells us that if we add the probability of drawing a red and a
black ball, it sums up to $1$, but when we also sum up the probability of
drawing a round ball and non-round ball, the probability gives us $1$ again,
when you sum these outcomes it becomes $2$, violating the fact that the
probability of observing all possible outcomes is $1$. Hence, we cannot have
such a $\S$.

With some thought, we can define our sample space $\S$ to be a set of tuples,
where each tuple is **parametrized** by the balls\' **color and shape**. We now
denote red and black to be r and b respectively, and round and not round to be c
and d respectively. We thus have:

$$
\S = \{(r, c), (r, d), (b, c), (b, d)\}
$$

as our sample space. Then the rest should follow, just like the experiment of
rolling 2 die.

---

However, what\'s more provoking is that this is the only way to define our
sample space $\S$, which may be confusing. A well known user on
[mathstackexchange](https://math.stackexchange.com/questions/3382903/misconception-in-sample-space-event-space-of-a-picking-a-ball-question)
has the following reply.

Sometimes probability is so much more confusing than the rest of the fields in
math - intuition is totally not there for me, notwithstanding the fact that most
textbooks go straight into the axioms and formulas, which make me only want to
rote learn the subject. One major hurdle that I find myself stuck in at first
was how to define a sample space, a event and subsequently an event space. The
textbooks examples seem too simple and when further questions present
themselves, one tends to be overwhelmed.

One should know that given an experiment, the sample space is entirely up to you
to define it, as long as it satisfies the below:

-   In the sample space, the outcomes $s_1,s_2,...,s_n$ must be mutually
    exclusive.
-   The outcomes must be collectively exhaustive;
-   The sample space must have the right granularity depending on what we are
    interested in. We must remove irrelevant information from the sample space.
    In other words, we must choose the right abstraction (forget some irrelevant
    information).

It is entirely up to us so long as the events that we are interested in
discussing the probabilities of are subsets of the sample space that we chose
and each result of an experiment is described uniquely by exactly one of the
outcomes in the sample space. But looking carefully at our experiment, it says
we want to pick a ball that is in a sense, both red and round. So it is
important that in our outcome, there should and must be one unique outcome that
describes this. In my previous attempt of coming up with a sample space
pertaining to this experiment, I claimed that

$$
\S = \{\text{red}, \text{black}, \text{round}, \text{not round}\}
$$

The problem is there is no unique outcome in the sample space that **desribes my
red and round ball**. We can say that 2 of the outcomes (red) and (round)
describes my red and round ball, but we need one unique outcome to describe this
red and round ball. Hence we have to tweak a little.

There are some **suggestions, not compulsory**, however when choosing a sample
space. First is simplicity. For instance, in the trial of tossing a coin, we
could have as a sample space

$$
\S_1 = \{H,T\}
$$

Another possible sample space that we can come up with could be

$$
\S_2= \{H\&R,H\&NR,T\&R,T\&NR\}
$$

Here, $R$ stands for rains and $NR$ not rains. Obviously, $\S_1$ is a better
choice than $\S_2$ as we do not care about how the weather affects the tossing
of a coin.

Secondly, we also tend to prefer sample spaces in which the outcomes are
equiprobable, that is, each outcome in the sample space being equally likely to
occur. We like these choices of sample spaces since in such a scenario we can
use counting techniques to calculate probabilities by taking the ratio of the
size of the event compared to the size of the sample space, something which
cannot be done in general.

Finally, back to our specific problem, we may _temporarily assume the balls are
uniquely numbered_! Yes, the balls might not have been numbered in reality, but
by temporarily assuming they were numbered you should be able to convince
yourself that the probabilities of selecting a ball of a certain type would stay
the same as if they weren\'t and this now allows us to describe the problem with
an equiprobable sample space.

Here, we can let the balls be numbered $1,2,3,\dots,10$. The three red round
balls being labeled $1,2,3$, the not round red balls being $4,5,6$, the black
round ball being $7$, and the black not round balls being $8,9,10$.

Our sample space then is

$$
\S = \{1,2,3,4,5,6,7,8,9,10\}
$$

And our event space is the set of all subsets of our sample space.

$$
\Sigma = \{\emptyset, \{1\},\{2\},..,\{10\},\{1,2\},\{1,3\},..,\{9,10\},...,\{1,2,3,...,10\}\}
$$

and of course any element of the event space is considered an event. For
example, let $A$ be the event such that the ball is round, and correspondingly,
$A = \{1,2,3,7\}$. Let $B$ be the event such that the ball is red, and the
correspondingly, $B = \{1,2,3,4,5,6\}$.
