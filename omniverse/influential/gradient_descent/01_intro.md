# Gradient Descent

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

Gradient descent is a first-order iterative optimization algorithm for finding a
local minimum of a differentiable function. It is the workhorse behind the
training of most machine learning models: linear regression, logistic regression
and deep neural networks are all, in practice, fitted by some variant of it.

This chapter first builds the calculus vocabulary needed to understand the
algorithm — gradient vectors and directional derivatives — and then proves the
statement that gives gradient descent its name: the gradient points in the
direction of steepest ascent, so stepping against it decreases the function
fastest. A from-scratch implementation closes the chapter.

## Table of Contents

```{tableofcontents}

```

## References and Further Readings

-   Zhang, Aston, Zachary C. Lipton, Mu Li, and Alexander J. Smola. "Chapter
    12.3. Gradient Descent." In Dive into Deep Learning, 2021.
-   [Khan Academy: The Gradient](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivative-and-gradient-articles/a/the-gradient)
