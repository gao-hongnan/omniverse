---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Convolution and Sum of Random Variables

```{contents}
```

```{code-cell} ipython3
:tags: [remove-input]
import sys
from pathlib import Path
parent_dir = str(Path().resolve().parents[2])
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
```

In {cite}`chan_2021`, the author gave us an intuitive treatment of the origin of
Gaussian random variables before the formal mathematical treatment of the
Central Limit Theorem in a later chapter.

Let's say we are rolling a fair dice and we can model the problem as a discrete
random variable $X$ with discrete uniform distribution, $X \sim \uniform(1, 6)$.
The empirical histograms after repeating, say, $10000$ times can be seen below.
The histogram is almost equal in height across the 6 states, as expected.

Now, let's say we are rolling two fair dice and we are interested in the sum of
the two dice. We can denote roll 1 as $X_1$ and roll 2 as $X_2$, both of which
follow the same discrete uniform distribution. The sum of the two dice is then
$Z = X_1 + X_2$. To find the empirical distribution of $Z$, we list out the
sample space of $Z$ to be $\lset (1, 1), (1, 2), \ldots, (6, 6) \rset$ and this
is mapped to the set of integers $\lset 2, 3, \ldots, 12 \rset$. The empirical
histogram of $Z$ is shown below. It should not be a surprise that the histogram
has a triangular shape, since we do expect the sum of two dice to be more likely
to be closer to 7 than 2 or 12, simply because there are more way to get a sum
of 7 (e.g. $(1, 6)$, $(2, 5)$, $(3, 4)$ ...) than to get a sum of 2 or 12 (e.g.
$(1, 1)$, $(6, 6)$).

The logic follows when we throw $6$ dice and consider
$Z = X_1 + X_2 + \cdots + X_6$, and if we throw $100$ dice, we will get
$Z = X_1 + X_2 + \cdots + X_{100}$. The triangle shape will "evolve" and become
more and more like a bell curve, as shown below.

```{code-cell} ipython3
:tags: [hide-input]
from omnivault.utils.probability_theory.plot import plot_sum_of_uniform_distribution

plot_sum_of_uniform_distribution()
```
