---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Linear Regression

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
```

In this chapter, we discuss linear regression, which is a very widely used
method for predicting a real-valued output (also called the dependent variable
or target) $y \in \mathbb{R}$, given a vector of real-valued inputs (also called
independent variables, explanatory variables, or covariates)
$\boldsymbol{x} \in \mathbb{R}^D$. The key property of the model is that the
expected value of the output is assumed to be a linear function of the input,
$\mathbb{E}[y \mid \boldsymbol{x}]=\boldsymbol{w}^{\top} \boldsymbol{x}$, which
makes the model easy to interpret, and easy to fit to data {cite}`pml1Book`.

There are two views to solving Linear Regression, we will focus on the
probabilistic aspect of it and leave the geometry/linear algebra interpretation
for further readings.

Regression in itself is a very broad topic, the analysis in itself can be made
as a course/book. Therefore, I will just touch and go on the important parts of
it.

## Table of Contents

```{tableofcontents}

```

## References and Further Readings

-   https://d2l.ai/
-   Murphy, Kevin P. "Chapter ." In Probabilistic Machine Learning: An
    Introduction. MIT Press, 2022.
-   James, Gareth, Daniela Witten, Trevor Hastie, and Robert Tibshirani.
    "Chapter ." In An Introduction to Statistical Learning: With Applications in
    R. Boston: Springer, 2022.
-   Jung, Alexander. "Chapter " In Machine Learning: The Basics. Singapore:
    Springer Nature Singapore, 2023.
-   Bishop, Christopher M. "Chapter ." In Pattern Recognition and Machine
    Learning. New York: Springer-Verlag, 2016.
-   Hal Daumé III. "Chapter ." In A Course in Machine Learning, January 2017.
-   [Machine Learning from Scratch](https://dafriedman97.github.io/mlbook/content/introduction.html)
-   **GOOD**: https://github.com/NathanielDake/intuitiveml
-   https://github.com/goodboychan/goodboychan.github.io/tree/main/_notebooks
