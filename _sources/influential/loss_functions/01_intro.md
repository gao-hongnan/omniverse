# Loss Functions

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

Learning is impossible without a way to measure how wrong a hypothesis is. A
loss function assigns to every prediction a non-negative penalty, and the
machinery of
[empirical risk minimization](../empirical_risk_minimization/01_intro.md) —
and therefore of training itself — consists of minimizing its average over
data.

This chapter first defines loss functions formally, including the view of a
loss as a random variable, which is what licenses talking about the _expected_
risk. It then studies two workhorses rooted in information theory: entropy
together with information gain, and the cross-entropy loss.

## Table of Contents

```{tableofcontents}

```

## References and Further Readings

-   Jung, Alexander. "Chapter 2.3. The Loss." In Machine Learning: The Basics.
    Springer Nature Singapore, 2023.
-   [Wikipedia: Loss Function](https://en.wikipedia.org/wiki/Loss_function)
-   [Wikipedia: Empirical Risk Minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization)
