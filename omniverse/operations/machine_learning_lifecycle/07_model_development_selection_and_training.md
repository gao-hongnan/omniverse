# Model Development and Training (MLOps)

In this stage, the prepared data is used to train machine learning models:

-   **Model Selection**: Choose suitable models based on the problem type
    (regression, classification, clustering, etc.), data characteristics, and
    size.

-   **Evaluation Metric Selection**: Decide on appropriate metrics to evaluate
    model performance. The choice of metrics depends on the problem type and the
    business objective. For example, accuracy, precision, recall, F1-score may
    be suitable for classification problems, whereas Mean Absolute Error (MAE),
    Root Mean Squared Error (RMSE), or R-squared might be chosen for regression
    problems.

-   **Baseline Model**: Train a simple or naive model using a standard
    technique. This model serves as a reference point or benchmark to compare
    with more sophisticated models. You can refer to scikit-learn's `Dummy`
    module for this purpose. For instance, if you are training a binary
    classification model, predicting the majority class could serve as the
    baseline model.

-   **Training**: Use the training dataset to train the selected models. This
    process involves adjusting the model parameters to minimize a certain loss
    function, relevant to the task at hand.

-   **Cross-Validation**: This method involves splitting the training dataset
    into $$K$$ subsets, then training the model $$K$$ times, each time using a
    different subset as the validation set and the remaining data as the
    training set. This provides a robust estimate of model performance, as the
    model's ability to generalize is tested on different subsets of data. Common
    methods include $K$-fold cross-validation, stratified $K$-fold (especially
    for imbalanced datasets), and time series cross-validation (for time series
    data).

-   **Hyperparameter Tuning**: Enhance model performance by optimizing the
    model's hyperparameters. This often involves methods like grid search,
    random search, or Bayesian optimization to find the optimal set of
    hyperparameters.

    In the Model Training stage, cross-validation is typically used in
    conjunction with hyperparameter tuning. For example, you might use
    cross-validation to estimate the performance of different sets of
    hyperparameters and choose the set that yields the best cross-validated
    performance. In this context, cross-validation is a tool to prevent
    overfitting during the model training process. You're not so much interested
    in the exact cross-validated performance estimate, but in which set of
    hyperparameters performs the best on average.

-   **Final Model Training**: Once the best hyperparameters are identified,
    train the final model using these optimized configurations. Ensure to
    evaluate the model's performance on a validation dataset to verify its
    ability to generalize well to unseen data.

> In the training stage, cross-validation is used primarily for model selection
> and hyperparameter tuning. Here, cross-validation helps estimate how well
> different models or hyperparameters will perform on unseen data, based on
> different splits of the training data. The goal is to tune and select a model
> that is expected to perform well on new data.

Remember that the model training process can involve iterative loops of steps
like feature selection, model selection, and hyperparameter tuning until the
satisfactory performance is achieved. This process should ideally be automated
and reproducible.

## Theoretical Underpinnings

### The Model, Loss and Data Paradigm

See Alexander Jung's book. Basically to construct a model, you need to define
the model, the loss function and the data. The model is the architecture, the
loss function is the objective function, and the data is the data.

### The Empirical Risk Minimization Principle

...

### Learning Theory (VC Dimension, Bias-Variance Tradeoff, etc.)

...

## Model Selection

Model Selection is one of the most critical steps in machine learning workflows,
and it's often a mixture of art, science, and domain expertise. Following are
the critical aspects to consider while choosing the right model for your
problem.

-   **Data size**: Some algorithms, like deep learning models, perform better
    with a large amount of data. Conversely, simpler models like linear
    regression or decision trees may be more appropriate for smaller datasets.

-   **Feature characteristics**: The nature of your features also impacts model
    choice. For instance, decision trees and random forests are less affected by
    feature scaling and can handle mixtures of features (binary, categorical,
    numerical), whereas logistic regression or support vector machines usually
    require feature scaling for high performance.

-   **Non-linearity**: If your data isn't linearly separable or the relationship
    between features is non-linear, linear models like Linear Regression or
    Logistic Regression may not be the best choice. You may need non-linear
    models like Neural Networks, Support Vector Machines with non-linear
    kernels, or Tree-Based models.

-   **Dimensionality**: If you have a high-dimensional dataset, some models may
    suffer from the curse of dimensionality, such as k-nearest neighbors (k-NN).
    In such scenarios, dimensionality reduction techniques or models less prone
    to this issue like Random Forests or Gradient Boosting Machines could be
    beneficial.

-   **Interpretability vs. Accuracy**: Depending on the use case, you may
    prioritize interpretability over prediction accuracy or vice versa. Models
    like Linear Regression, Logistic Regression, and Decision Trees are highly
    interpretable, while Neural Networks, SVMs, and Ensemble methods trade-off
    interpretability for higher accuracy.

    However, there are many cases where a more complex model is necessary to
    achieve the desired performance. For example, in image classification, a
    classical model might failed spectacularly, but a deep learning model like a
    Convolutional Neural Network (CNN) might be able to achieve much better
    performance. There are indeed tools such as Grad-CAM to help with the
    interpretability of deep learning models.

-   **Real-time prediction**: If you need to make real-time predictions,
    consider models that are not only fast at prediction time but also have a
    smaller memory footprint. Simpler models like Logistic Regression, Decision
    Trees, or k-NNs (provided that the dataset is not too large) could be good
    choices here.

-   **Tolerance to errors**: Some models are more sensitive to errors in the
    data. For instance, outliers can significantly affect the performance of
    linear regression models. In such cases, robust models like Decision Trees,
    Random Forests, or SVMs might be preferred.

-   **Computation resources**: Training models like deep learning or large
    ensembles can be resource-intensive. If you have computational resource
    constraints, you might prefer simpler or more efficient models.

Remember, there's rarely a one-size-fits-all model for any given problem.
Typically, you'll experiment with multiple models and choose the one that
performs the best on your validation data and aligns with your project
requirements. The no free lunch theorem in machine learning states that no
single algorithm works best for every problem. Therefore, it is always a good
practice to try a handful of different algorithms when starting with model
selection.

## Metric Selection

Before beginning model training, it's crucial to define the metrics by which the
model's performance will be evaluated. These metrics will guide the optimization
process and provide a quantitative measure of the model's quality.

The choice of metric depends on the problem type (classification, regression,
ranking, etc.) and the business requirements:

The appropriate metric depends on the type of machine learning task, the data,
and the specific requirements of the project:

-   **Classification**: Here, we're predicting categorical outcomes. Common
    metrics include Accuracy (percentage of correct predictions), Precision
    (proportion of true positive predictions out of all positive predictions),
    Recall (proportion of true positive predictions out of all actual
    positives), F1-Score (harmonic mean of precision and recall), and AUC-ROC
    (area under the Receiver Operating Characteristic curve, representing the
    model's ability to distinguish between classes).

    However, these metrics might not be sufficient when dealing with imbalanced
    datasets. In such cases, other metrics like the Matthews correlation
    coefficient, Cohen's kappa, or area under the Precision-Recall curve could
    be more insightful.

-   **Regression**: In regression problems, we're predicting continuous values.
    Metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean
    Squared Error (RMSE), and R-squared are typically used.

    MAE measures the average magnitude of errors in a set of predictions,
    without considering their direction. MSE and RMSE are similar to MAE but
    amplify the impact of large errors. R-squared (coefficient of determination)
    explains how much of the dependent variable's variation is explained by the
    independent variable(s).

-   **Ranking**: In ranking problems, we're interested in the order of
    predictions. The Normalized Discounted Cumulative Gain (NDCG) measures the
    quality of a ranking, taking into account the positions of the relevant
    items. Precision at K is another commonly used metric, providing the
    proportion of relevant items among the top K items.

Remember that the choice of a metric should align with the business objective.
For instance, if false positives and false negatives have a different cost, you
might want to optimize for Precision or Recall rather than overall Accuracy.

Another thing to consider is whether you are more interested in the ranking of
predictions (which prediction is ranked highest) or in the absolute values of
predictions (predicting the exact value). Depending on this, you might choose to
optimize for AUC-ROC or Log Loss for a classification problem, or for RMSE or
Mean Absolute Percentage Error (MAPE) for a regression problem.

Finally, when comparing different models, it's important to use the same metric
for a fair comparison. If one model is optimized for Accuracy and another for
F1-Score, they might perform differently when evaluated using a single, common
metric.

It's not uncommon to use multiple metrics to evaluate a model's performance, as
they can each highlight different aspects of the model's behavior. For instance,
in a binary classification problem, accuracy might not be a good metric if the
classes are imbalanced; precision and recall or the F1-score might be more
informative.

### Benefit Structure

The "benefit structure" or "cost-benefit analysis" is an approach to model
evaluation that goes beyond standard metrics like accuracy, precision, and
recall. It involves assigning specific costs and benefits to different types of
errors and correct decisions the model makes, based on their real-world
implications. This analysis can be especially crucial in fields like healthcare,
where different types of errors can have drastically different consequences.

Let's illustrate this with an example of cancer detection.

Suppose we are building a machine learning model to detect whether a patient has
cancer based on various diagnostic tests. The prediction can either be positive
(cancer detected) or negative (no cancer detected). However, these predictions
can either be correct or incorrect, leading to four possible outcomes:

-   **True Positive (TP)**: The model correctly identifies a cancer patient.
-   **True Negative (TN)**: The model correctly identifies a healthy patient.
-   **False Positive (FP)**: The model incorrectly identifies a healthy patient
    as having cancer.
-   **False Negative (FN)**: The model incorrectly identifies a cancer patient
    as healthy.

We can construct a table like this:

| -                         | Cancer (actual)                                     | No Cancer (actual)                            |
| ------------------------- | --------------------------------------------------- | --------------------------------------------- |
| **Cancer (predicted)**    | TP (Benefit: early treatment, higher survival rate) | FP (Cost: unnecessary worry, further tests)   |
| **No Cancer (predicted)** | FN (Cost: delayed treatment, lower survival rate)   | TN (Benefit: peace of mind, no further tests) |

The benefits and costs associated with each outcome can be estimated:

-   **True Positive (TP)**: Early detection of cancer can lead to early
    treatment, which significantly increases the survival rate. We might assign
    a high benefit value here, say +100.

-   **True Negative (TN)**: Correctly identifying a patient without cancer can
    provide peace of mind and avoid unnecessary further tests. We might assign a
    moderate benefit value here, say +10.

-   **False Positive (FP)**: A false positive result can lead to unnecessary
    worry and further tests for the patient. We could assign a moderate cost
    value here, say -20.

-   **False Negative (FN)**: A false negative is very harmful because it might
    lead to delayed treatment, significantly decreasing the survival rate.
    Therefore, we could assign a high cost value here, say -200.

Using this benefit structure, we can evaluate the model not just on how often
it's right or wrong, but also on the real-world impact of its predictions.

A model with a higher overall benefit score (i.e., a weighted sum of TP, TN, FP,
FN using the assigned benefit/cost values) would be preferred over one with
lower benefit score, even if the latter has higher accuracy.

In conclusion, benefit structure allows for a more nuanced evaluation of model
performance by considering the specific implications of each type of prediction
error. This type of analysis is particularly relevant in critical
decision-making contexts, such as medical diagnosis, credit scoring, or fraud
detection.

### Calibration

A good understanding of calibration can be found in the
[unofficial google blog](https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html).

#### Why do we need to calibrate models?

The output of classification models can often be interpreted as the probability
that a given instance belongs to the positive class. For example, in binary
classification, a model might predict a value of 0.7 for a certain instance,
which can be interpreted as a 70% probability that this instance belongs to the
positive class.

However, these predicted probabilities are not always accurate. For example, a
model might predict a probability of 70% for a set of instances, but only 60% of
them actually belong to the positive class. This discrepancy between the
predicted probabilities and the true probabilities is a sign that the model is
not well-calibrated.

A well-calibrated model is one where, for all instances where it predicts a
probability of P, the proportion of those instances that are positive is
actually P. For example, if we gather all instances where the model predicts a
70% probability, 70% of those instances should indeed be positive. Calibration
is particularly important in cases where the predicted probabilities are used to
make decisions that depend not just on the class labels, but also on the
certainty of the prediction.

#### Example

Let's consider a model that predicts whether a patient has a certain disease
(the positive class) or not (the negative class) based on some diagnostic tests.
Let's assume we have a set of 1000 patients and the model makes predictions on
this set.

We could compile the model's predictions and the actual outcomes into a table
like this:

| Predicted Probability | Number of Predictions | Actual Positives |
| --------------------- | --------------------- | ---------------- |
| 0.1-0.2               | 200                   | 30               |
| 0.2-0.3               | 180                   | 50               |
| 0.3-0.4               | 150                   | 60               |
| 0.4-0.5               | 120                   | 65               |
| 0.5-0.6               | 100                   | 70               |
| 0.6-0.7               | 90                    | 65               |
| 0.7-0.8               | 80                    | 65               |
| 0.8-0.9               | 50                    | 45               |
| 0.9-1.0               | 30                    | 30               |

In this table, we've binned the predictions into ranges like 0.1-0.2, 0.2-0.3,
etc. Then we count the number of predictions in each bin and the number of
actual positives in each bin.

Ideally, the proportion of actual positives in each bin should match the
predicted probability. For example, in the 0.3-0.4 bin, the model predicts a
probability of 30% to 40%, and the actual proportion of positives is 60/150 =
40%, which is well within the range.

But if we look at the 0.7-0.8 bin, the model predicts a 70% to 80% probability,
but the actual proportion of positives is 65/80 = 81.25%. This is a sign that
the model is not well-calibrated for this range of probabilities, as the actual
positive rate is higher than the predicted probability.

In practice, we often visualize this data with a calibration plot, where the
x-axis represents the predicted probabilities and the y-axis represents the
actual positive rates. A well-calibrated model would lie along the line y = x,
while deviations from this line indicate miscalibration.

#### Calibrating Models

There are several methods to calibrate a model after it has been trained. One of
the most popular methods is Platt Scaling, which fits a logistic regression
model to the model's scores. This logistic regression model can adjust the
predicted probabilities to make them more accurate. Another popular method is
Isotonic Regression, which fits a piecewise-constant non-decreasing function to
the model's scores.

#### Calibration and Evaluation (Brier + AUROC combo)

Once a model is calibrated, we need a way to evaluate how well-calibrated it is.
This can be done using calibration plots, also known as reliability diagrams. In
a calibration plot, the x-axis represents the predicted probabilities and the
y-axis represents the proportion of instances that are positive. A
well-calibrated model will have its calibration plot close to the diagonal.

Another common way to evaluate calibration is using the Brier score. The Brier
score is a metric that combines calibration and refinement. The Brier score is
given by:

$$
BS = \frac{1}{N}\sum_{t=1}^{N}(f_t-o_t)^2
$$

where:

-   $N$ is the total number of instances,
-   $f_t$ is the predicted probability for instance $t$,
-   $o_t$ is the true outcome for instance $t$ (1 for positive and 0 for
    negative).

The Brier score ranges from 0 for a perfect model to 1 for a constantly wrong
model. However, the Brier score alone is not sufficient to evaluate a model, as
it does not account for the model's ability to discriminate between classes. For
this reason, it is often used in combination with the AUROC (Area Under the
Receiver Operating Characteristic curve) metric.

While the AUROC measures how well the model can distinguish between classes, it
does not take into account the accuracy of the predicted probabilities. Thus,
using the Brier score and the AUROC together can provide a more complete
evaluation of a model's performance. By optimizing these two metrics
simultaneously, we can obtain a model that not only discriminates well between
classes, but also provides accurate predicted probabilities.

## Experiment Tracking

During the iterative process of model development and training, keeping track of
the numerous experiments run, their parameters, and outcomes is a crucial step.
As models are trained, tested, tweaked, and retrained, it becomes increasingly
complex to manage and compare these various experiments.

This is where experiment tracking comes into play. By meticulously tracking each
experiment, data scientists can easily compare the results of different models,
configurations, hyperparameters, and even completely different approaches.

Here are some aspects you may want to track:

-   **Model Parameters**: The settings and hyperparameters used for each model.
-   **Model Performance**: How each model performed according to the selected
    evaluation metrics.
-   **Feature Importance**: Which features were most influential in the model's
    predictions.
-   **Artifacts**: Any output files, such as trained models or plots.

It's also worth noting that several tools can facilitate experiment tracking,
such as MLflow, TensorBoard, and Weights & Biases. By adopting such tools, teams
can create a central repository of experiments that foster collaboration and
reproducibility.

With proper experiment tracking in place, data science teams can ensure that
their model development process is transparent, reproducible, and effective. It
becomes easier to revisit old experiments, share findings with team members, and
ultimately make more informed decisions about which models and configurations to
move forward with.

Proceeding with this strategy, the next steps in the model development process
might include tuning model hyperparameters and performing model validation.

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

> We can run a few chosen models with default hyperparameters and see which one
> is doing good. But no one's stopping you from trying multiple models.

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

## Training Strategies

### Regularization

...

### Early Stopping

### Debug Mode Hyperparameter Tuning

Examples:

-   Learning Rate fit over the first 10 batches to see the initial LR and loss.
-   Overfitting on a few batches to see if the model is learning.
-   Basically idea is to train on a few batches to see if your hyperparameters
    are ok.

### Other Deep Learning Tricks (See fastai)

...

## Cross-validation Confusion

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

    1. Here you must be careful! The hyperparameters chosen are fixed for all
       $K$ folds!

3. **Select the best hyperparameters**: The best hyperparameters are the ones
   that led to the best average performance across all folds.

In Kaggle, we always average $K$ folds, this is more geared towards
stacking/ensembling, and is not part of our scope here.

## The Training Pipeline

Mention my config based training pipeline.

## Continuous Training (Dependent on Monitoring of Drifts)

Continuous Training, often related to the concept of Continuous Learning or
Lifelong Learning in machine learning, fits within the model training stage but
is more of a cyclic process. It can be framed as part of the overall model
lifecycle management in MLOps.

We will see this step in details in
[stage 12](12_continuous_integration_deployment_learning_and_training.md).

Continuous Training refers to the ongoing process of re-training machine
learning models on fresh data. This process is necessary because the performance
of a model might degrade over time as the underlying data distribution changes -
a phenomenon known as concept drift.

-   **Continuous Training**: Machine Learning models might not always maintain
    their predictive power due to changes in data over time (concept drift).
    Continuous Training addresses this by regularly retraining models on new
    data, or whenever the model performance degrades beyond an acceptable level.
    This step involves monitoring model performance over time, collecting new
    training data, and re-running the training and evaluation steps. Automated
    retraining pipelines can be set up for this purpose to ensure the models
    stay up-to-date.

Note that the success of Continuous Training relies on robust monitoring, as it
is the feedback from the model monitoring that typically triggers the retraining
process. If there's a significant drop in performance or a detected change in
the input data distribution, the model can be flagged for retraining. Therefore,
it's an iterative process that spans across multiple stages in the MLOps
lifecycle.

## References and Further Readings

Fast.ai and Kaggle offer a wealth of model development tips and tricks,
including SOTA techniques and best practices.

-   [Fast.ai](https://www.fast.ai/)
-   [Kaggle](https://www.kaggle.com/)
-   [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://sebastianraschka.com/pdf/manuscripts/model-eval.pdf)
-   Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
-   [Madewithml](https://madewithml.com/)
-   [Scikit-Learn: Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
-   [Google: Why model calibration matters and how?](https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html)
-   Training pipeline by me
