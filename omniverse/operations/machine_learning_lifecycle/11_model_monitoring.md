# Stage 11. Model Monitoring

Model monitoring is about continuously tracking the performance of models in
production to ensure that they continue to provide accurate and reliable
predictions.

- **Performance Monitoring**: Regularly evaluate the model's performance
  metrics in production. This includes tracking metrics like accuracy,
  precision, recall, F1 score for classification problems, or Mean Absolute
  Error (MAE), Root Mean Squared Error (RMSE) for regression problems, etc.

- **Data Drift Monitoring**: Over time, the data that the model receives can
  change. These changes can lead to a decrease in the model's performance.
  Therefore, it's crucial to monitor the data the model is scoring on to
  detect any drift from the data the model was trained on.

- **Model Retraining**: If the performance of the model drops or significant
  data drift is detected, it might be necessary to retrain the model with new
  data. The model monitoring should provide alerts or triggers for such
  situations.

- **A/B Testing**: In case multiple models are in production, monitor their
  performances comparatively through techniques like A/B testing to determine
  which model performs better.

In each of these stages, it's essential to keep in mind principles like
reproducibility, automation, collaboration, and validation to ensure the
developed models are reliable, efficient, and providing value to the
organization.

## Intuition

Even though we've trained and thoroughly evaluated our model, the real work
begins once we deploy to production. This is one of the fundamental differences
between traditional software engineering and ML development. Traditionally, with
rule-based, deterministic software, the majority of the work occurs at the
initial stage and once deployed, our system works as we've defined it. But with
machine learning, we haven't explicitly defined how something works but used
data to architect a probabilistic solution. This approach is subject to natural
performance degradation over time, as well as unintended behavior, since the
data exposed to the model will be different from what it has been trained on.
This isn't something we should be trying to avoid but rather understand and
mitigate as much as possible. In this lesson, we'll understand the shortcomings
from attempting to capture performance degradation in order to motivate the need
for **drift detection**.

## System Health

The first step to ensure that our model is performing well is to ensure that the
actual system is up and running as it should. This can include metrics specific
to service requests such as latency, throughput, error rates, etc. as well as
infrastructure utilization such as CPU/GPU utilization, memory, etc.

## System Health Dashboard

Fortunately, most cloud providers and even orchestration layers will provide
this insight into our system's health for free through a dashboard. In the event
we don't, we can easily use [Grafana](https://grafana.com/),
[Datadog](https://www.datadoghq.com/), etc. to ingest system performance metrics
from logs to create a customized dashboard and set alerts.

## Examples

The plot shows both the cumulative mean squared error (MSE) and the sliding MSE
over time.

The x-axis represents the hour since the model has been deployed, and the y-axis
represents the MSE. The "Threshold" line is a hypothetical threshold that we set
for the MSE; we could say that any MSE above this threshold would indicate that
the model's performance is not satisfactory.

We can see that both the cumulative and sliding MSE start reasonably low, but
after some time, they start to increase, indicating that the model's performance
is degrading. The sliding MSE is more responsive to recent changes in the
model's performance and hence it rises above the threshold before the cumulative
MSE. This is why it's beneficial to monitor both cumulative and sliding
metrics - they give us different insights into the model's performance over
time.

## MY OLD EXAMPLE TO REFINE

Model monitoring and data drift monitoring are key components in maintaining the
performance of any predictive model over time. Here are the key aspects to
consider for your Bitcoin price prediction model:

1. **Model Performance Monitoring**: Regularly evaluate your model's performance
   metrics such as accuracy, precision, recall, F1-score, AUC-ROC, etc.,
   depending on the nature of your problem (classification, regression, etc.).
   Track these metrics over time. If there's a substantial decrease, your model
   might need retraining or updating.

2. **Data Drift Monitoring**: This involves checking if the distribution of the
   model's input data is changing over time. You want to make sure that the data
   your model was trained on is representative of the data it is making
   predictions on. If the data drifts too much, the model’s performance might
   decrease.

To monitor data drift, you can use a two-sample t-test, which compares the means
of two groups to determine if they're significantly different. Here's how to do
it:

- Consider one feature at a time. For instance, start with 'Volume'.
- From your current data, take a sample. Compute its mean (let's call it
  mean1) and standard deviation (std1).
- Take a sample of the same size from the data your model was trained on.
  Compute its mean (mean2) and standard deviation (std2).
- Use the t-test formula to calculate the t-score. The formula is
  `t = (mean1 - mean2) / sqrt((std1^2/n1) + (std2^2/n2))`, where n1 and n2 are
  the sizes of your samples.
- If the absolute t-score is large (greater than the critical t-value for your
  desired confidence level), then the means are significantly different,
  indicating data drift.

The resulting `t` value (t-score) is a measure of the size of the difference
relative to the variation in your data. A large absolute t-score means that the
difference in means is large relative to the variability of the data, which
suggests that the means are significantly different. This would be an indication
of data drift.

Remember to conduct this test for all relevant features ('Open', 'Close',
'Volume', etc.) and over regular intervals (daily, weekly, etc.) to ensure
continuous monitoring.

3. **Concept Drift Monitoring**: Sometimes, even if the data distribution stays
   the same, the underlying relationship between the input features and the
   target variable might change. This is called concept drift. To monitor this,
   you can look for a decrease in your model's performance over time, even when
   there's no significant data drift.

Lastly, while t-tests can help in identifying drifts, they are just one part of
the puzzle. Monitoring residuals (the differences between your model’s
predictions and the actual values) can also provide insights into whether the
model is continuing to perform well.

## Monitor 2

When monitoring your model and data for drifts in the context of predicting
Bitcoin price increase or decrease, there are a few techniques you can consider,
including statistical tests such as t-tests. Here's a general overview:

1. Model Monitoring: It involves monitoring the performance of your predictive
   model over time to ensure its accuracy and reliability. Some techniques for
   model monitoring include:

   - Tracking key performance metrics like accuracy, precision, recall, or mean
     absolute error.
   - Monitoring model output distributions to detect significant changes or
     shifts.
   - Comparing model predictions with actual outcomes to identify
     discrepancies.

2. Data Monitoring: It involves monitoring the input data used by your model to
   detect any changes or drifts that may impact the model's performance. Here
   are a few methods for data monitoring:
   - Statistical tests: T-tests can be used to compare statistical properties
     (e.g., means) of different data subsets or time periods. For example, you
     can compare Bitcoin price increase predictions for different time
     intervals to identify significant differences.
   - Control charts: These graphical tools help detect shifts or anomalies in
     data distribution, allowing you to identify potential drifts.
   - Concept drift detection: Techniques like change point detection algorithms
     or sliding window approaches can be employed to detect significant changes
     in the underlying data distribution.

Remember, model and data monitoring should be an ongoing process to ensure the
reliability of your predictions. Regularly evaluating and updating your model
can help account for evolving market dynamics and improve its performance.

## Drift _madewithml_

$$
\begin{array}{|lll|}
\hline \text { Entity } & \text { Description } & \text { Drift } \\
\hline X & \text { inputs (features) } & \text { data drift } \rightarrow P(X) \neq P_{\text {ref }}(X) \\
\hline y & \text { outputs (ground-truth) } & \text { target drift } \rightarrow P(y) \neq P_{\text {ref }}(y) \\
\hline P(y \mid X) & \text { actual relationship between } X \text { and } y & \text { concept drift } \rightarrow P(y \mid X) \neq P_{r e f}(y \mid X) \\
\hline
\end{array}
$$

## Concept Drift

Sometimes, even if the data distribution stays the same, the underlying
relationship between the input features and the target variable might change.
This is called concept drift. To monitor this, you can look for a decrease in
your model's performance over time, even when there's no significant data drif

Let's consider a concrete example of concept drift in the context of a movie
recommendation system.

Suppose you have a movie recommendation algorithm that takes into account
factors like user's age, genre preferences, and ratings of previously watched
movies to recommend new movies. The model is trained on a dataset and works well
initially, giving good recommendations and having a high click-through rate.

Over time, however, you notice that even though the distribution of the user's
age, genre preferences, and ratings of previously watched movies (your input
features) stays the same, the click-through rate of the recommended movies (your
target variable) starts to decrease. This indicates that the relationship
between the input features and the target variable has changed.

Why might this happen? One possible reason is a change in movie trends. Perhaps
when the model was initially trained, action movies were very popular. But over
time, the popularity of action movies has decreased and documentaries have
become more popular. The model, however, is still biased towards recommending
action movies because that's what worked when it was initially trained. This is
an example of concept drift: the underlying concept – what kind of movies are
likely to be clicked on – has changed, even though the distribution of the input
features has not.

To monitor this, you would track the performance of your model over time. If you
see a decrease in performance – in this case, a decrease in the click-through
rate – that could be an indication of concept drift. You might then decide to
retrain your model on more recent data, or to revise it to take into account
more recent trends in movie popularity.
