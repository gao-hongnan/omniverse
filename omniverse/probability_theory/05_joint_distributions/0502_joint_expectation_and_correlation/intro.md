# Joint Expectation and Correlation

## Introduction

This section quantifies the degree in which two random variables are related to each other.
For a [simple example](https://online.stat.psu.edu/stat414/lesson/18), consider you are a boss of a cafe,
and you want to know if coffee and cheese cake sales are related to each other. Then we can model
the number of coffee and cheese cake sold as random variables $X$ and $Y$ respectively, if the
**correlation** between $X$ and $Y$ is high, then we should include both coffee and cheese cake
on the same day's menu.

Another example is in [ensemble learning in machine learning](https://stats.stackexchange.com/questions/281856/why-do-ensemble-models-work-better-when-we-ensemble-models-of-low-correlation)
, it is typical to ensemble different
models' predictions to get a better prediction. In this case, we can use the correlation between
each model's prediction to decide how much weight to give to each model's prediction via a weighted
average. We also want to ensure that the models are not too correlated with each other, otherwise
the ensemble will not be better than the best model.

But how do we measure the **correlation** between two random variables? This is what we will
discuss in this section.

## Further Readings

- Chan, Stanley H. "Chapter 5.2. Joint Expectation." In Introduction to Probability for Data Science, 257-266. Ann Arbor, Michigan: Michigan Publishing Services, 2021.
- [PSU Stat 414. Lesson 18: The Correlation Coefficient](https://online.stat.psu.edu/stat414/lesson/18)