---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Stage 5.4. Model Testing

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

Testing in machine learning is a bit different from traditional software
testing. Here we just give an intuition on _how_ different it is.
[Eugene Yan](https://eugeneyan.com/) has written a series of articles on testing
in machine learning.

Eugene highlighted that software involves having some _input data_ and some
_handcrafted logic_ that processes the data to produce some _output data_, which
is then compared against the expected output - a deterministic process. In
contrast, machine learning involves having some _input data_ and _output data_
and with a suitable learning algorithm $\mathcal{A}$, we can learn a model
$\mathcal{G}$ that can predict the output data from the input data. The process
involves a _learned logic_ and when we want to test the model with the learned
logic, we would then actually need to load the model and run it on some input
data to get the output data to compare against the expected output. And it is
also common to compare loss for each epoch against a threshold to see if the
model is learning.

In my own experience, I always prepare a debug dataset, that is usually sampled
(stratified, grouped if needed) from the training dataset. This dataset can be
used as your fixture in testing. But more importantly, one should also use this
debug dataset to test sanity of your training pipeline. For example, like what
Eugene and Karpathy suggested, run your model for a certain number of steps and
check if the loss is decreasing, and you can even craft it to be overfit on the
debug dataset to see if your model $\mathcal{G}$ has capacity to learn the data.
Furthermore, during hyperparameter tuning, it is very expensive to run the model
on the full dataset, so you can use the debug dataset to test if your model is
reacting well to the hyperparameters (i.e. learning rate finder).

## References and Further Readings

-   [How to Test Machine Learning Code and Systems](https://eugeneyan.com/writing/testing-ml/)
-   [Writing Robust Tests for Data & Machine Learning Pipelines](https://eugeneyan.com/writing/testing-pipelines/)
-   [Don't Mock Machine Learning Models In Unit Tests](https://eugeneyan.com/writing/unit-testing-ml/)
-   [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
