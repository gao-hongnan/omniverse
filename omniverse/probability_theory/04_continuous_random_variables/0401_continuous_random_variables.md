# Continuous Random Variables

## Definition

````{prf:definition} Uncoutable Set
:label: def_uncountable_set

A set $S$ is uncountable if there is no bijection between $S$ and the set of natural numbers $\mathbb{N}$.
````


````{prf:definition} Continuous Random Variables
:label: def_continuous_random_variable

A **continuous random variable** $X$ is a random variable[^random_variable] whose 
[cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function)
(CDF) is continuous.

We can also define it via its [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) and say that $X$ is continuous if there is a function $f(x)$
such that for any $a \leq b$, we have

$$
P(a \leq X \leq b) = \int_a^b f(x) dx
$$

in which case $f(x)$ is called the **probability density function** (PDF) of $X$.
````

We have not yet defined what the PDF and CDF are but we will do so in the next section. To be less
pedantic, we can follow the definition in {prf:ref}`def_discrete_random_variables` and say that
$X$ is continuous if its range is uncountably infinite. This however is not formal as stated [here](https://stats.stackexchange.com/questions/455668/defining-continuous-random-variables-via-uncountable-sets).


[^random_variable]: See {prf:ref}`random_variables` for the definition of a random variable.
