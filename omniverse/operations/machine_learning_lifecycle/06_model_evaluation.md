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

# Stage 6. Model Evaluation (MLOps)

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

Once the model has been trained, it's crucial to evaluate its performance. This
evaluation involves various techniques:

-   **Out-of-Fold Predictions**: In k-fold cross-validation, the model makes
    predictions on the validation set in each fold of the process. These
    predictions are "out of fold" because they are made on data that the model
    hasn't seen during training. Collecting the out-of-fold predictions from
    each fold and comparing them to the actual targets can give a more robust
    estimate of the model's performance.

-   **Holdout Evaluation**: This is a simpler form of model validation, where
    the dataset is split into two sets: a training set and a validation (or
    holdout) set. The model is trained on the training set and evaluated on the
    holdout set. The key is to ensure that the model never sees the validation
    data during training, which gives us an unbiased estimate of the model's
    performance on unseen data.

-   **Bias-Variance Tradeoff**: This is a fundamental concept in machine
    learning that refers to the tradeoff between a model's ability to fit
    training data (bias) and its ability to generalize to unseen data
    (variance). An optimal model should strike a balance between the two, which
    is typically achieved through techniques like regularization and
    hyperparameter tuning. This is usually not a separate step in the model
    evaluation, but a principle that guides the entire model building process.

-   **Model Metrics Evaluation**: Use the appropriate evaluation metrics (as
    decided in the Model Training stage) to measure the model's performance.
    This could include metrics like accuracy, precision, recall, F1 score,
    AUC-ROC for classification problems; and Mean Absolute Error (MAE), Mean
    Squared Error (MSE), Root Mean Squared Error (RMSE) for regression problems,
    among others.

-   **Model Interpretability Evaluation**: If applicable, evaluate how
    interpretable the model is. Some models, like decision trees or linear
    regression, are quite interpretable (i.e., we can understand how they make
    predictions), but others, like deep neural networks, are more like "black
    boxes". Depending on the application, a model's interpretability may be very
    important.

These evaluation techniques allow us to get a more complete picture of how well
the model is likely to perform on unseen data.

## Bias and Variance

See
[mlxtend's bias-variance decomp](https://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/).

1. Draw `num_rounds` bootstrap samples from the training set. Each bootstrap
   sample `X_boot` and `y_boot` is the same size as the original training set
   and is drawn with replacement.

2. Fit the model to each bootstrap sample and make predictions on the test set.
   This gives you a matrix of predictions, `all_pred`, with `num_rounds` rows
   (one for each round of bootstrapping) and `n_test` columns (one for each
   example in the test set).

3. Calculate the main predictions. If the loss is '0-1_loss', this is the mode
   of the predictions in `all_pred` for each test example (i.e., the most common
   prediction across bootstrap rounds). If the loss is 'mse', this is the mean
   of the predictions in `all_pred` for each test example.

4. Now calculate the three quantities of interest:

    - The average expected loss is the average loss across all bootstrap rounds
      and all test examples. For '0-1_loss', this is the mean number of times
      the prediction in each round is not equal to the true test label. For
      'mse', this is the mean of the squared differences between the prediction
      in each round and the true test label.

    - The average bias is the loss of the main predictions. For '0-1_loss', this
      is the number of times the main prediction is not equal to the true test
      label, averaged over all test examples. For 'mse', this is the mean of the
      squared differences between the main prediction and the true test label.

    - The average variance is the average variance of the predictions across all
      bootstrap rounds and all test examples. For '0-1_loss', this is the mean
      number of times the prediction in each round is not equal to the main
      prediction. For 'mse', this is the mean of the squared differences between
      the prediction in each round and the main prediction.

The average expected loss, average bias, and average variance are then returned.

This method provides a way to quantify the bias and variance of a model, giving
insight into the model's tendency to overfit (high variance) or underfit (high
bias) the data. By using bootstrapping, it simulates the scenario of having
multiple different samples from the population and provides an estimate of how
the model's predictions would vary across these different samples.

## References and Further Readings

-   [Fast.ai](https://www.fast.ai/)
-   [Kaggle](https://www.kaggle.com/)
-   [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://sebastianraschka.com/pdf/manuscripts/model-eval.pdf)
-   Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
-   [Madewithml](https://madewithml.com/)
-   Training pipeline by me
