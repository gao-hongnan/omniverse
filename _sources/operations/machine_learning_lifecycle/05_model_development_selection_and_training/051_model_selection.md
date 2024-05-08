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

# Stage 5.1. Model Selection

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

"Just use gradient boosting", or "Just use the biggest Transformer model" are
pretty much irresponsible advice for beginners. As a data scientist or machine
learning engineer, you are not a gambler and hope to hit the jackpot with the
biggest or SOTA model. Instead, you should understand the trade-offs between
different models and choose the one that best fits your problem. On top of that,
having a solid understanding of the assumptions and limitations of the model is
crucial for making an informed decision.

## Understanding Model And Data Assumptions

Before selecting a model, it's important to understand the assumptions and
limitations of the model. Different models make different assumptions about the
data, and these assumptions can impact the model's performance. We would take
examples from Chip Huyen's book _Designing Machine Learning Systems_ as she
provided us with some useful points to consider when selecting a model for your
problem.

[^chip-chapter6]

```{list-table} Factors to Consider in Model Selection
:header-rows: 1
:name: ml-lifecycle-model-selection-factors

-   -   Consideration
    -   Description
-   -   Dataset size
    -   Some algorithms, like deep learning models, perform better with a large
        amount of data. Conversely, simpler models like linear regression or
        decision trees may be more appropriate for smaller datasets.
-   -   Feature characteristics
    -   The nature of your features also impacts model choice. For instance,
        decision trees and random forests are less affected by feature scaling
        and can handle mixtures of features (binary, categorical, numerical),
        whereas logistic regression or support vector machines usually require
        feature scaling for high performance.
-   -   Non-linearity
    -   If your data isn't linearly separable or the relationship between
        features is non-linear, linear models like Linear Regression or Logistic
        Regression may not be the best choice. You may need non-linear models
        like Neural Networks, Support Vector Machines with non-linear kernels,
        or Tree-Based models.
-   -   Dimensionality
    -   If you have a high-dimensional dataset, some models may suffer from the
        curse of dimensionality, such as k-nearest neighbors (k-NN). In such
        scenarios, dimensionality reduction techniques or models less prone to
        this issue like Random Forests or Gradient Boosting Machines could be
        beneficial.
-   -   Interpretability vs. Accuracy
    -   Depending on the use case, you may prioritize interpretability over
        prediction accuracy or vice versa. Models like Linear Regression,
        Logistic Regression, and Decision Trees are highly interpretable, while
        Neural Networks, SVMs, and Ensemble methods trade-off interpretability
        for higher accuracy.

        However, there are many cases where a more complex model is necessary to
        achieve the desired performance. For example, in image classification, a
        classical model might failed spectacularly, but a deep learning model
        like a Convolutional Neural Network (CNN) might be able to achieve much
        better performance. There are indeed tools such as Grad-CAM to help with
        the interpretability of deep learning models.

-   -   Real-time prediction
    -   If you need to make real-time predictions, consider models that are not
        only fast at prediction time but also have a smaller memory footprint.
        Simpler models like Logistic Regression, Decision Trees, or k-NNs
        (provided that the dataset is not too large) could be good choices here.
-   -   Computation resources
    -   Training models like deep learning or large ensembles can be
        resource-intensive. If you have computational resource constraints, you
        might prefer simpler or more efficient models.
```

Remember, there's rarely a one-size-fits-all model for any given problem.
Typically, you'll experiment with multiple models and choose the one that
performs the best on your validation data and aligns with your project
requirements. The
[no free lunch theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem) in
machine learning states that no single algorithm works best for every problem.
Therefore, after knowing what limitations that the models may impose for your
specific problem, you should then experiment with a few models to see which one
is doing well.

## Baseline

Before proceeding with the selection of more complex models, it is beneficial to
establish a baseline model. The baseline model is typically simple, or even
naive, serving as the minimum benchmark for model performance.

A common practice is to use a simple algorithm or heuristic that can quickly be
implemented and run on your data. This baseline approach depends on the nature
of the problem at hand. For instance:

-   In regression tasks, you might predict the average of the target variable
    for all instances.
-   In classification tasks, a simple classifier such as predicting the most
    common class can serve as a baseline.
-   In time series forecasting, a persistence model that predicts the next step
    to be the same as the last observed step can be used.

> It is worth noting that scikit-learn's `Dummy` module provides a convenient
> way to create baseline models for regression and classification tasks.

The baseline model provides a point of comparison for future, more complex
models. Any sophisticated model we train should perform significantly better
than this baseline.

## Model Selection Revisited

Following the baseline model, the next step is model selection. Here, you
identify potential algorithms that could be used for solving the given problem.

The difference here is you are actually running a few model classes from
different hypothesis spaces $\mathcal{H}$ based on the model selection process
earlier.

The choice of models usually depends on the problem type (regression,
classification, clustering, etc.), the nature of your data, and practical
considerations such as computational resources and time constraints.

For example, if you have labeled data and a binary output, you may consider
supervised learning algorithms such as logistic regression, decision trees, or
support vector machines. If your dataset is large and complex, you might
consider more powerful models like random forests or neural networks.

In this step, it's also important to consider the interpretability of the model.
In some cases, a simpler model might be preferred if it offers similar
performance to a more complex model but is easier to interpret and explain.

Remember, no one model fits all scenarios. A good practice is to try multiple
models and see which one performs best on your specific dataset.

---

Sometimes using the biggest model do yield the best results but if your latency
is too high for inference and usable by the end-user, then it's not a good
model - and that's why many current research directions emphasize on how to
quantize large models, prune models, or even distill models to serve powerful
models in a reasonable latency.

## References and Further Readings

-   [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://sebastianraschka.com/pdf/manuscripts/model-eval.pdf)
-   Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.

[^chip-chapter6]:
    Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
