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

# Stage 5.2. Metric Selection

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

## Metric Selection

Before beginning model training, it's crucial to define the metrics by which the
model's performance will be evaluated. These metrics will guide the optimization
process and provide a quantitative measure of the model's quality.

The appropriate metric depends on the type of machine learning task, the data,
and the specific requirements of the project:

### Classification

Here, we're predicting categorical outcomes. Common metrics include Accuracy
(percentage of correct predictions), Precision (proportion of true positive
predictions out of all positive predictions), Recall (proportion of true
positive predictions out of all actual positives), F1-Score (harmonic mean of
precision and recall), and AUC-ROC (area under the Receiver Operating
Characteristic curve, representing the model's ability to distinguish between
classes).

### Regression

In regression problems, we're predicting continuous values. Metrics like Mean
Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
and R-squared are typically used.

MAE measures the average magnitude of errors in a set of predictions, without
considering their direction. MSE and RMSE are similar to MAE but amplify the
impact of large errors. R-squared (coefficient of determination) explains how
much of the dependent variable's variation is explained by the independent
variable(s).

### Clustering

In clustering problems we typically see Adjusted Mutual Information Score (AMI),
Adjusted Rand Index (ARI), Completeness Score, Homogeneity Score, and Silhouette
Score. These metrics evaluate the quality of the clustering results by comparing
the predicted clusters to the true clusters.

### Ranking, Detection, Pairwise, Retrieval And Other Metrics

There are many many more different types of metrics, for a comprehensive list,
you can see
[TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html)
or
[Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html).

### Choosing The Right Metric

Remember that the choice of a metric should align with the business objective.
For instance, if false positives and false negatives have a different cost, you
might want to optimize for Precision or Recall rather than overall Accuracy.

When comparing different models, it's important to use the same metric for a
fair comparison. If one model is optimized for Accuracy and another for
F1-Score, they might perform differently when evaluated using a single, common
metric.

It's not uncommon to use multiple metrics to evaluate a model's performance, as
they can each highlight different aspects of the model's behavior. For instance,
in a binary classification problem, accuracy might not be a good metric if the
classes are imbalanced; precision and recall or the F1-score might be more
informative.

## Choose Your Metrics Wisely

If you choose a misleading metric, you might end up with a model that performs
well on paper but fails in practice simply because it reports a misleading and
optimistic performance whereas it is not doing well.

Consider a training dataset consisting of 1000 patients where we want to train a
classifier to "accurately" classify whether a patient has cancer (positive
class 1) or no cancer (negative class 0). The dataset is dichotimized by 950
benign patients and the remaining 50, cancerous.

The ZeroR classifier (baseline classification model) predicts only the majority
class. And in our case, the in-sample training set accuracy will be $0.95\%$
since it predicts all sample to be benign; for completeness, we also assume a
validation set with 1000 patients (990 benign and 10 cancerous), it follows that
our validation set's accuracy will be $0.99\%$.

Euphoria is at all time high over this result, surprised that a baseline model
can perform so well, you then happily reported this result to your boss, and
gets fired immediately. You drown in tears and googled "Is accuracy a bad
metric?", only to learn a valuable lesson: since the model you trained will
predict benign no matter the input, this means it completely missed out every
single cancerous patient even though you get a 99 percent accuracy.

## Benefit Structure

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

## Calibration

Calibration in the context of machine learning models refers to how well the
model's predicted probabilities of outcomes match the actual outcomes'
frequencies in the real world. A well-calibrated model means that if the model
predicts an event with a probability of $p$, that event should indeed occur
approximately $p$ percent of the time. For example, if a model predicts that
there's a $70 \%$ chance of rain tomorrow, then, ideally, it should rain 70 out
of 100 similar days with that prediction.

```{admonition} See Also
:class: seealso

- [Why model calibration matters and how to achieve it](https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html).
```

### Why Does Calibration Matter?

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

### Example

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

### Calibrating Models

There are several methods to calibrate a model after it has been trained. One of
the most popular methods is Platt Scaling, which fits a logistic regression
model to the model's scores. This logistic regression model can adjust the
predicted probabilities to make them more accurate. Another popular method is
Isotonic Regression, which fits a piecewise-constant non-decreasing function to
the model's scores.

```{admonition} See Also
:class: seealso

-   [1.16. Probability calibration (Scikit-Learn)](https://scikit-learn.org/stable/modules/calibration.html)
```

### Calibration and Evaluation (Brier + AUROC combo)

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

-   [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html)
-   [Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html).
-   [Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning](https://sebastianraschka.com/pdf/manuscripts/model-eval.pdf)
-   Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
-   [Scikit-Learn: Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
-   [Google: Why model calibration matters and how?](https://www.unofficialgoogledatascience.com/2021/04/why-model-calibration-matters-and-how.html)

[^chip-chapter6]:
    Huyen, Chip. "Chapter 6. Model Development and Offline Evaluation." In
    Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
