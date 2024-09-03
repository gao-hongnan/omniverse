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

# Stage 5. Model Development and Training (MLOps)

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

In this stage, the prepared data is used to train machine learning models, some
key stages are listed below.

```{list-table} Some Key Stages In Model Development and Training
:header-rows: 1
:name: ml-lifecycle-model-development-training-some-key-stages

-   -   Step
    -   Description
-   -   Model Selection
    -   Choose suitable models based on the problem type (regression,
        classification, clustering, etc.), data characteristics, and size.
-   -   Evaluation Metric Selection
    -   Decide on appropriate metrics to evaluate model performance. The choice
        of metrics depends on the problem type and the business objective. For
        example, accuracy, precision, recall, F1-score may be suitable for
        classification problems, whereas Mean Absolute Error (MAE), Root Mean
        Squared Error (RMSE), or R-squared might be chosen for regression
        problems.
-   -   Baseline Model
    -   Train a simple or naive model using a standard technique. This model
        serves as a reference point or benchmark to compare with more
        sophisticated models. You can refer to scikit-learn's
        [`Dummy`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
        module for this purpose. For instance, if you are training a binary
        classification model, predicting the majority class could serve as the
        baseline model.
-   -   Training
    -   Use the training dataset to train the selected models. This process
        involves adjusting the model parameters to minimize a certain loss
        function, relevant to the task at hand.
-   -   Cross-Validation
    -   This method involves splitting the training dataset into $K$ subsets,
        then training the model $K$ times, each time using a different subset as
        the validation set and the remaining data as the training set. This
        provides a robust estimate of model performance, as the model's ability
        to generalize is tested on different subsets of data. Common methods
        include $K$-fold cross-validation, stratified $K$-fold (especially for
        imbalanced datasets), and time series cross-validation (for time series
        data).
-   -   Hyperparameter Tuning
    -   Enhance model performance by optimizing the model's hyperparameters.
        This often involves methods like grid search, random search, or Bayesian
        optimization to find the optimal set of hyperparameters.

        In the Model Training stage, cross-validation is typically used in
        conjunction with hyperparameter tuning. For example, you might use
        cross-validation to estimate the performance of different sets of
        hyperparameters and choose the set that yields the best cross-validated
        performance. In this context, cross-validation is a tool to prevent
        overfitting during the model training process. You're not so much
        interested in the exact cross-validated performance estimate, but in
        which set of hyperparameters performs the best on average.

-   -   Final Model Training
    -   Once the best hyperparameters are identified, train the final model
        using these optimized configurations. Ensure to evaluate the model's
        performance on a validation dataset to verify its ability to generalize
        well to unseen data.
```

In the training stage, cross-validation is used primarily for model selection
and hyperparameter tuning. Here, cross-validation helps estimate how well
different models or hyperparameters will perform on unseen data, based on
different splits of the training data. The goal is to tune and select a model
that is expected to perform well on new data.

Remember that the model training process can involve iterative loops of steps
like feature selection, model selection, and hyperparameter tuning until the
satisfactory performance is achieved. This process must be automated and
reproducible via pipelines - and this is where MLOps practices come into play.

## The Model, Loss and Data Paradigm

Basically to construct a model, you need to define the model architecture
$\mathcal{G}$, the loss function $\mathcal{L}$, and the data
$\mathcal{S} \overset{\mathrm{iid}}{\sim} \mathcal{D}$.

The model $\mathcal{G}$ is one such choice in the hypothesis space
$\mathcal{H}$, the loss function $\mathcal{L}$ is the objective function where
we typically employ a learning algorithm $\mathcal{A}$ to minimize the loss
function over the data $\mathcal{S}$.

This series is more on MLOps, and we won't go too deep into the theory like the
empirical risk minimization principle, or learning theory like VC dimension,
bias-variance tradeoff, etc - which is really important to have a basic
understanding of when doing model development. For example, knowing the VC
dimension allows you to understand the model's capacity and generalization
ability.

### Learning Curves

Although one might need to fully grasp things like VC dimension in rigour. But
one should definitely look at things like learning curves. Note that learning
curves is not just you plotting the training and validation loss over the number
of epochs. You would have the number of samples on the x-axis and the error on
the y-axis - so you can have a good gauge on the scalability of the model.

```{admonition} See Also
:class: seealso

-   [Plotting Learning Curves and Checking Models’ Scalability](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
```

## Ablation Studies

We can run a few chosen models with default hyperparameters and see which one is
doing good. But no one's stopping you from trying multiple models. But be sure
to understand why things work or don't work. One can do ablation studies to
understand the importance of each component in the model - like for example when
you did not expect the model to perform well, but it did, you can do ablation
studies to understand why it worked (i.e. say remove or replace a component and
see if the performance drops).

## Hyperparameter Tuning

Once you've selected a model, the next step is to tune its hyperparameters.
Hyperparameters are the configuration settings of the model that are set before
the learning process begins. They control the behavior of the model and can
significantly impact its performance.

For instance, in a decision tree, the maximum depth of the tree is a
hyperparameter. In a neural network, the learning rate, number of layers, and
number of neurons per layer are all hyperparameters.

There are various strategies for hyperparameter tuning, including:

-   **Grid Search**: This involves exhaustively testing a predefined set of
    hyperparameters to see which combination yields the best performance.
-   **Random Search**: Rather than testing all combinations, random search
    selects random combinations of the hyperparameters to test. This can be more
    efficient than grid search, especially when dealing with a large number of
    hyperparameters.
-   **Bayesian Optimization**: This is a more advanced method that uses the
    concept of probability to find the minimum of a function, which in this case
    is the model's error function.

Hyperparameter tuning can be computationally expensive, but it can also
significantly improve model performance.

## Cross-Validation

Cross-validation is a technique used to assess the predictive performance of a
model and ensure that it's not overfitting to the training data.

It involves splitting the dataset into $K$ groups or 'folds'. Then, we train the
model on $K$-1 folds and test it on the remaining fold. We repeat this process
$K$ times, each time testing on a different fold. The average performance across
all $K$ trials is then used as the overall performance measure of the model.

Cross-validation provides a more robust estimate of the model's performance on
unseen data compared to using a single train-test split. The most common form of
cross-validation is $K$-fold cross-validation, where $K$ is often set to 5
or 10.

### A Caution on Data Leakage

We have seen in earlier chapters that data leakage can occur when information
from the validation/test set is used to train the model. This can lead to overly
optimistic results. The most common pitfalls is for instance, perform
preprocessing on the whole dataset before splitting.

Another common mistake is about groups, consider the following example:

Consider a dataset of patients, where multiple rows belong to the same patient,
recording different visits to a medical facility. Each row represents a
different visit and includes features like the symptoms reported, tests
conducted, and the final diagnosis.

Let's suppose we're trying to build a model to predict a specific disease based
on the symptoms and tests. We could use cross-validation to estimate our model's
performance, where we randomly split our data into training and validation sets.
However, this method could lead to data leakage.

Why? Because information about the same patient could end up in both the
training and validation set. Our model might seem to perform well because it's
not so much learning the relationship between symptoms and disease but instead
memorizing information about specific patients.

In such cases, it is more appropriate to use a technique like GroupKFold from
scikit-learn. This method ensures that the same group (in this case, the same
patient) does not appear in both the training and validation set. It essentially
treats each patient as a separate group and ensures that all entries from a
particular patient are either in the training set or the validation set, but not
in both.

By doing this, we ensure our model generalizes better to new patients since the
validation set only contains patients that the model hasn't seen during
training. This would give us a more realistic estimate of how well our model
would perform in a real-world setting, where it needs to make predictions for
new patients it hasn't seen before.

Furthermore, in this specific example, it is common that the label/target is
whether the patient has a certain disease, for instance, cancer. And the
positive label of cancer is rare, so you need to use `StratifiedGroupKFold` to
ensure that the positive label is distributed evenly across the folds, alongside
the grouping.

## Final Model Training

After a comprehensive process of baseline modeling, iterative experimentation,
and hyperparameter tuning, we will have identified the optimal configurations
for our chosen model. The final stage in model training involves using these
configurations to train our final model.

Here's what that might look like:

-   **Training with Optimized Configurations**: Utilize the optimal
    hyperparameters discovered during the tuning process to train the model.

-   **Full Dataset Utilization**: Often, the final model is trained on the full
    training dataset. We've already determined that our model and its
    configurations are robust and reliable, so we can now use as much data as
    available to optimize the model's learning.

-   **Validation Performance**: Despite training on the full dataset, it remains
    crucial to evaluate the final model's performance on a held-out validation
    dataset. This will provide the last check on the model's ability to
    generalize to unseen data, ensuring its robustness and reliability.

Remember, this entire process, from initial baseline model to final model
training, might be iterated several times as new insights, data, or resources
become available. Machine learning model development is a highly iterative and
evolving process, constantly moving towards better and more reliable
predictions.

This iterative nature of model development highlights the importance of
meticulous experiment tracking, allowing for comparison, reproducibility, and
efficient backtracking when needed.

## Training Chronicles

One can write a book if they want to document all the tricks, tips and insights
they have learned during the model development and training process. For
example, a beginner might take a while to realise that the learning rate is
perhaps one of the most important hyperparameters to tune when training a large
deep neural network. One can always use a learning rate finder where the idea is
to fit a few batches to see the initial learning rate and loss and treat it as
hyperparameter tuning on a very small subset. Having a small debug dataset is
also useful for quick tuning.

And perhaps a bit more advanced users would know that tracking gradient norms,
activations and their distributions are important to diagnose issues such as
gradient explosion or vanishing problem. After all, deep learning models are
_chaotic_ systems, deterministic yet sensitive to initial conditions.

Fast.ai and Kaggle offer a wealth of model development tips and tricks,
including SOTA techniques and best practices. You can couple the tricks with
papers and you are good to go!

## A Note On Cross-validation

The idea is usually as such, in training, say we split the dataset into 5 folds,
then we have five models, each trained on 4 folds and evaluated on the remaining
fold.

In $K$-fold cross-validation (along with its variants), hyperparameters are
typically chosen based on the average performance across all folds. The idea is
to identify the hyperparameters that, on average, lead to the best model
performance. Here's how this process usually works:

1. **Set up a grid of hyperparameters**: You specify a range of possible values
   for each hyperparameter that you want to optimize.

2. **Train and evaluate a model for each combination of hyperparameters**: For
   each combination, you perform a $K$-fold cross-validation. This means
   training and evaluating a model on each fold and calculating the average
   performance across all folds.

    Here you must be careful! The hyperparameters chosen are fixed for all $K$
    folds!

3. **Select the best hyperparameters**: The best hyperparameters are the ones
   that led to the best average performance across all folds.

In Kaggle, we always average $K$ folds, this is more geared towards
stacking/ensembling, and is not part of our scope here.

## Continuous Training (Dependent on Monitoring of Drifts)

Continuous Training refers to the ongoing process of re-training machine
learning models on fresh data. This process is necessary because the performance
of a model might degrade over time as the underlying data distribution changes -
a phenomenon known as concept drift. So one can say that it is an extension or
part of CI/CD.

Machine Learning models might not always maintain their predictive power due to
changes in data over time (concept drift). Continuous Training addresses this by
regularly retraining models on new data, or whenever the model performance
degrades beyond an acceptable level. This step involves monitoring model
performance over time, collecting new training data, and re-running the training
and evaluation steps. Automated retraining pipelines can be set up for this
purpose to ensure the models stay up-to-date.

Note that the success of Continuous Training relies on robust monitoring, as it
is the feedback from the model monitoring that typically triggers the retraining
process. If there's a significant drop in performance or a detected change in
the input data distribution, the model can be flagged for retraining. Therefore,
it's an iterative process that spans across multiple stages in the MLOps
lifecycle.

## Where Is The MLOps?

The model and metric selection process is more of a design process. The real
operations is to package the whole training, tuning, and evaluation process into
a pipeline that can be run automatically. At the same time, ensuring the trained
model has artifacts tracked, logged and versioned into a central store (can be
feature store) or model registry. This is especially in the context of deploying
your model to production.

```{figure} ../assets/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd.svg
---
name: mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-4-ml-automation-ci-cd
---

CI/CD and automated ML pipeline.

Image Credits: [Google - MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
```

## References and Further Readings

-   [Fast.ai](https://www.fast.ai/)
-   [Kaggle](https://www.kaggle.com/)
-   [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://sebastianraschka.com/pdf/manuscripts/model-eval.pdf)
-   Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
-   [Madewithml](https://madewithml.com/)
-   [Scikit-Learn: Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
-   [Google: Why model calibration matters and how?](https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html)
-   [A Recipe for Training Neural Networks - Karpathy](https://karpathy.github.io/2019/04/25/recipe/)

[^chip-chapter6]:
    Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
