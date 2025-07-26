---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Stage 1. Problem Formulation

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
```

## Introduction

In the development of an AIOps system, the first step is to clearly define the
problem you intend to solve and appropriately scope the project. This stage
involves **identifying the business problem**, **clarifying requirements**,
**determining constraints**, and **establishing metrics for success**.

Understanding business requirements and stakeholder motivations is crucial to
ensure the AIOps system meets the needs of the users and organization. Discuss
these requirements in-depth, along with any technical constraints or
assumptions, to guide the project effectively and to meet stakeholder needs.

## Business Problem and Objectives

The initial task is to
**[identify the business problem and its objectives](https://medium.com/codex/how-to-scope-a-machine-learning-project-d74d4025e04c)**.

### Identify the Business Problem

First, we _identify the business problem_ that the AIOps system aims to solve.
This part of the task involves clearly understanding and defining the specific
issue or challenge that the business is facing. It could relate to an
inefficiency, a lost opportunity, a market need, or any other problem that the
organization wants to address.

In addition to identifying the problem, this task requires setting out the
_goals_ or _objectives_ that need to be achieved in order to address or solve
the problem. These objectives guide the solution and provide criteria for
measuring success.

### An Example

-   **Business Problem**: A streaming platform has a vast catalog of movies, but
    users often find it **challenging** to discover movies that align with their
    interests. This leads to **decreased engagement and satisfaction** with the
    platform.

-   **Objective**: **Enhance user engagement and satisfaction** by developing a
    movie recommendation system that provides **personalized recommendations**
    to users based on their viewing history, preferences, and behavior.

## Measuring Success

Having identified the specific business problem and outlined the objectives, it
is essential to determine how success will be measured. This requires the
establishment of clear and relevant **business metrics**, which align with the
overall goals and allow for the continuous assessment of the AIOps system's
impact. The selection of these metrics will be driven by the unique aspects of
the problem being solved and may include aspects related to digital marketing
and recommendation systems.

### Business metrics

Business metrics are quantifiable measures that track and assess the performance
of a business process. They are used to gauge the success of a business strategy
or initiative, and are often tied to specific business objectives.

In the context of machine learning, beyond our familiar machine learning
metrics, we need to ensure that these metrics can translate into business value.

### An Example On Movie Recommendation System

**Movie recommendation systems** utilize algorithms to analyze user behavior and
preferences to suggest movies that a user is likely to enjoy. They play a
critical role in driving user engagement and revenue for streaming platforms.

```{list-table}
:name: business-metrics-movie-recommendation
:header-rows: 1

-   -   Metric
    -   Description
-   -   User Engagement Rate
    -   Measures the percentage of users interacting with recommended movies.
        High engagement rates may indicate that the recommendations are aligned
        with user preferences.
-   -   Click-Through Rate (CTR) on Recommended Movies
    -   The percentage of users who click on a recommended movie to view more
        details or watch the trailer. A high CTR may reflect the relevancy of
        the recommendations.
-   -   Conversion Rate
    -   In this context, conversion rate refers to the percentage of users who
        proceed to watch a recommended movie. This is a key metric in assessing
        the effectiveness of the recommendation system in driving viewership.
-   -   Average Viewing Duration of Recommended Movies
    -   This metric assesses how long users are watching the recommended movies.
        If viewers are watching these movies for a significant duration, it may
        suggest that the recommendations are resonating with their preferences.
-   -   Churn Rate
    -   Measures the percentage of subscribers who cancel their subscription
        within a given period. If the recommendation system is ineffective,
        leading to dissatisfaction, the churn rate may increase.
```

By applying these business metrics to a movie recommendation system, a streaming
platform can gauge the success of its recommendation algorithms, understand how
users are interacting with recommendations, and make data-driven decisions to
enhance the user experience and meet business objectives.

### From Business Metrics to Machine Learning Metrics

While it's essential to define the business metrics that align with the overall
goals of the project, when framing the problem within the context of machine
learning, a transformation of these metrics is often necessary.

Take, for example, the customer churn rate, defined as the proportion of
customers who discontinue a company's product or service within a specified
period. In a business context, minimizing the churn rate is a prevalent goal, as
it often proves more economically efficient to retain existing customers than to
procure new ones.

This goal can be translated into the machine learning domain by formulating it
as a binary classification problem. Here, the task becomes predicting whether a
given customer will churn or not. By structuring the problem in this manner, it
becomes feasible to apply machine learning techniques to achieve the business
objective of reducing churn and evaluate using typical classification metrics
such as accuracy, precision, recall, F1 score, AUROC, etc.

## Clarifying Requirements

When given a problem, many often rush to build a model without fully
understanding the requirements. This can lead to suboptimal solutions that do
not meet the needs of the stakeholders. It is always good to take a step back
and clarify the requirements before committing to a solution.

Clarifying requirements involves posing key questions such as the in
{ref}`table below <clarifying-requirements-aiops-system>`.

```{list-table} Clarifying Requirements for an AIOps System
:name: clarifying-requirements-aiops-system
:header-rows: 1

-   -   Requirement Category
    -   Questions and Details
-   -   Baseline
    -   Is there a pre-existing solution or system for this problem? If so, how
        well is it performing?
-   -   Benefit Structure
    -   What are the anticipated benefits or improvements with the
        implementation of the AIOps system? Do you favor precision or recall
        (assuming a classification problem)?
-   -   Features
    -   Are there specific features that stakeholders expect in the solution?
        For instance, in a TikTok video recommendation system, stakeholders may
        expect a feedback mechanism, such as like/dislike options, to gather
        data for continuously improving the system.
-   -   Data
    -   What kind and amount of data is available for developing and training
        the system?
-   -   Constraints
    -   Are there any technical, financial, or operational constraints? These
        might include system latency requirements, project budget, or data
        privacy issues.
-   -   Scale
    -   How large is the problem? For example, how many users will the system
        serve? How much data will the system handle?
-   -   Performance
    -   What are the performance requirements? Does the solution need to
        function in real-time, or can offline processing suffice?
```

This list is by no means exhaustive, and one must be prepared that requirements
may evolve or change as the project progresses. However, having a clear
understanding of the requirements at the outset can help guide the project in
the right direction.

### Baseline

What is a baseline? Baseline is the assessment of any existing **solutions** or
**systems** that address the **problem**, including an evaluation of their
**performance**. It serves as a starting point for measuring **improvements**.

Establishing a baseline is crucial in the early stages of project development. A
baseline could be an existing model or system that your AIOps solution will be
compared against. For instance, suppose your task is to improve the vacation
rental recommendation system. If there's already a model in place, its
performance (accuracy, precision, recall, F1 score, etc.) can serve as a
benchmark. If no such model exists, a simple statistical approach or a
rule-based system might serve as your baseline. Understanding the baseline helps
to set realistic performance goals and provides context on how much improvement
is needed.

### Benefit Structure

The **analysis** of anticipated **advantages** or **enhancements** that come
with the **implementation** of the **system**. In the context of
**classification problems**, it can involve deciding the importance between
**precision** and **recall**.

Benefit structure is the expected outcome or value from implementing the AIOps
system. Often it can manifest as a cost-benefit analysis, where the costs of
different outcomes are weighed against the benefits. Let's see an example of a
benefit structure in a binary classification problem.

#### Classification

In the context of a spam detection system, the business metrics would likely
revolve around enhancing user experience, trust, and operational efficiency. We
can naively define two business metrics:

1. **User Experience and Trust**:

    - **Metric**: User Satisfaction Rate
    - **Description**: The percentage of users who are satisfied with the spam
      filtering system, reflecting the balance between filtering out spam and
      not misclassifying legitimate emails.

2. **Operational Efficiency**:
    - **Metric**: Spam Filtering Efficiency
    - **Description**: The number of spam emails successfully filtered out as a
      proportion of total spam emails received. This measures how effectively
      the system is working in real operational terms.

And we can define the corresponding machine learning metrics:

1. **False Positive Impact**:

    - **Metric**: False Positive Rate (FPR)
    - **Description**: The number of legitimate emails marked as spam as a
      proportion of total legitimate emails received. This could have business
      implications such as lost opportunities, missed communications, or
      decreased user trust.

2. **False Negative Impact**:
    - **Metric**: False Negative Rate (FNR)
    - **Description**: The number of spam emails not marked as spam as a
      proportion of total spam emails received. This could have business
      implications such as decreased user trust or increased operational costs.

As we know, there is a trade-off between precision (favoring FPR) and recall
(favoring FNR). In the context of spam detection, the business may prioritize
minimizing false positives over false negatives, as the cost of misclassifying a
legitimate email as spam could lead to missed opportunities or dissatisfaction,
and thus, it's associated with a higher cost. Therefore, the model might be
tuned to favor precision over recall. However, this trade-off should be
carefully evaluated based on the specific business needs and the relative
importance of user trust, operational efficiency, and other relevant factors.

We can then formulate a benefit structure using mathematical terms. Below, I'm
assuming that the cost and benefit are expressed in monetary units (e.g.,
dollars), and these values are hypothetical and adjustable based on the specific
business context.

| Outcome             | Description                                 | Benefit/Cost  |
| ------------------- | ------------------------------------------- | ------------- |
| True Positive (TP)  | Spam correctly identified as spam           | Benefit: \$10 |
| True Negative (TN)  | Legitimate email correctly identified       | Benefit: \$5  |
| False Positive (FP) | Legitimate email incorrectly marked as spam | Cost: -\$200  |
| False Negative (FN) | Spam incorrectly marked as legitimate       | Cost: -\$100  |

-   **True Positive (TP)**: Correctly identifying spam might save resources and
    enhance user experience, so it's associated with a benefit.
-   **True Negative (TN)**: Correctly identifying a legitimate email might be
    less valuable but still provides a benefit in terms of user trust and
    engagement.
-   **False Positive (FP)**: Misclassifying a legitimate email as spam could
    lead to missed opportunities or dissatisfaction, so it's associated with a
    cost. This is usually worse than a false negative since it could lead to
    lost business.
-   **False Negative (FN)**: Failing to identify spam could lead to more
    significant issues like security risks, thus incurring a higher cost.

This table helps in quantifying the trade-offs in designing the spam detection
system. By assigning specific monetary values to each outcome, it provides a
concrete way to evaluate different models and to tune the system according to
business priorities. It can guide decision-making around the trade-off between
precision and recall and the overall cost-effectiveness of the system.

```{admonition} Medical Diagnosis Favours Recall
:class: note

In medical diagnosis (positive = cancer), the cost of a false negative is
usually higher than the cost of a false positive. For instance, a false
negative in cancer detection could lead to a delayed diagnosis and treatment,
which could be life threatening. In this case, the cost of a false negative is
higher than the cost of a false positive. However, in the case of a spam
detection system, a false positive is usually worse than a false negative, as
it could lead to missed opportunities or dissatisfaction. In this case, the
cost of a false positive is higher than the cost of a false negative.
```

Similarly, for other problems like recommendation systems, the benefit structure
could be different. For instance, in a recommendation system, the benefit
structure could be based on user engagement and satisfaction. The corresponding
metrics could be things like Precision@K, Recall@K, or NDCG@K, where K is the
number of recommendations shown to the user.

### Features

The specific **functionalities**, **attributes**, or **characteristics** that
**stakeholders** expect in the **solution**. These elements directly influence
**clinician and patient interactions** and **diagnostic outcomes**. Note that at
this stage, features often refer to the **end product features** or
**functionalities** that the **solution** will offer, rather than the **model
features** used in the **machine learning** algorithm.

For this illustration, let’s consider a cancer diagnosis system. The term
"features" in this context refers to various characteristics or attributes used
both to describe the patient data and the diagnostic criteria employed by the
system. Features serve as inputs to the machine learning model to facilitate
accurate predictions or diagnoses based on patterns recognized in the clinical
data.

```{list-table} Features of a Cancer Diagnosis System
:name: features-cancer-diagnosis
:header-rows: 1

-   -   Feature Category
    -   Description
-   -   End Product Features
    -   These might include the functionalities or characteristics of the cancer
        diagnosis system itself, such as its user interface for clinicians, the
        integration with existing hospital information systems, visualization
        tools for medical imaging, and decision-support systems that present
        diagnostic options.
-   -   Model Features
    -   In a cancer diagnosis system, model features would include medical data
        such as patient medical history, genetic information, biomarkers, lab
        test results, and imaging data. These attributes help the algorithm
        analyze and predict the presence or type of cancer in patients.
-   -   Feature Engineering
    -   This involves the process of selecting, transforming, and constructing
        relevant model features from complex medical data. Effective feature
        engineering is vital in medical applications as the precision of the
        features significantly affects diagnostic accuracy. This might include
        deriving new biomarkers from genetic sequences or enhancing image
        features from scans for better machine learning model performance.
```

Stakeholders might want this end product to not only provide accurate diagnoses
but also be able to integrate their legacy systems (yes healthcare has many of
those) and provide a user-friendly interface for clinicians.

### Data

The examination of the **types**, **quantities**, and **qualities** of **data**
available for **training** and **developing** the **system**. This includes
understanding the **sources**, **formats**, and how the **data aligns** with the
**problem** being solved.

The type and amount of data available are crucial for an AIOps project. For a
cancer diagnosis system, the data might include patient medical records, genetic
information, lab test results, and medical imaging data. Without data, what can
you do? Pray to the AI gods for a miracle? No, you can't. Generate synthetic
data using few-shot since generative AI is "mature"? Maybe, but it's not always
feasible.

### Constraints

The identification of any **technical**, **financial**, or **operational
limitations** that may impact the **development** or **deployment** of the
**system**. Constraints define the **boundaries** within which the **solution**
must operate.

In machine learning projects, it is essential to consider various constraints
that might impact the choice of the model and its deployment. Some key factors
include [**data availability**](https://en.wikipedia.org/wiki/Data_availability)
and the inherent
[**rarity of minority classes**](https://en.wikipedia.org/wiki/Class_imbalance)
in classification problems. Additionally,
[**hardware constraints**](https://en.wikipedia.org/wiki/Computer_hardware) like
the availability of sufficient GPUs and the deployment environment are crucial
for complex tasks, especially
[**deep learning**](https://en.wikipedia.org/wiki/Deep_learning) tasks. Finally,
consider the
[**latency requirements**](<https://en.wikipedia.org/wiki/Latency_(engineering)>)
of the application, as some use cases demand real-time inference, such as
[**Google's autocomplete**](https://en.wikipedia.org/wiki/Autocomplete), while
others can tolerate slower processing times, like
[**Customer Segmentation**](https://en.wikipedia.org/wiki/Customer_segmentation).

By addressing these constraints, you can optimize your model's performance and
ensure successful deployment in various industrial use cases.

<table>
  <tr>
    <th>Constraint Category</th>
    <th>Subcategory</th>
    <th>Question/Consideration</th>
    <th>Use Cases</th>
  </tr>
  <tr>
    <td rowspan="3"><strong>Data</strong></td>
    <td>Data Availability</td>
    <td>How much <strong><a href="https://en.wikipedia.org/wiki/Data/">data</a></strong> is accessible? Smaller datasets may require less complex models, while larger datasets may need more complex models like deep neural networks.</td>
    <td>Spam detection, sentiment analysis</td>
  </tr>
  <tr>
    <td>Data Collection</td>
    <td>Can more data be collected? Consider the cost and time required for data collection.</td>
    <td>Image recognition, speech recognition</td>
  </tr>
  <tr>
    <td>Minority Class</td>
    <td>Is the minority class inherently rare (e.g., cancer cases)? If so, it may be reasonable to keep the data as-is.</td>
    <td>Fraud detection, disease diagnosis</td>
  </tr>
  <tr>
    <td rowspan="2"><strong>Hardware Constraints</strong></td>
    <td>Hardware Resources</td>
    <td>Are sufficient <strong><a href="https://www.nvidia.com/en-us/data-center/gpu-accelerated-applications/">GPUs</a></strong> available for complex tasks, especially deep learning tasks?</td>
    <td>Autonomous vehicles, facial recognition</td>
  </tr>
  <tr>
    <td>Deployment Environment</td>
    <td>Does the model need to be deployed on a specific device, such as a smartphone or web application?
    For example, YOLOv5 has an iOS app that detects objects in real-time,
    indicating the model is performing real-time inference within the app.</td>
    <td>Mobile apps, IoT devices</td>
  </tr>
  <tr>
    <td rowspan="2"><strong>Latency</strong></td>
    <td>Online (Real Time)</td>
    <td>Does the application require real-time inference, such as <strong><a href="https://en.wikipedia.org/wiki/
    Autocomplete">Google's autocomplete</a></strong> feature? It will be useless if it takes more than a few seconds to autocomplete.</td>
    <td>Chatbots, real-time translations</td>
  </tr>
  <tr>
    <td>Offline (Non-Real Time)</td>
    <td>Can the application afford slower processing times, like <strong><a href="https://en.wikipedia.org/wiki/Market_segmentation">Customer Segmentation</a></strong>?</td>
    <td>Recommendation systems, credit scoring</td>
  </tr>
</table>

### Scale

The evaluation of the **magnitude** of the **problem**, encompassing aspects
such as the **number of users**, the **amount of data handled**, and the
**geographical spread** of the **system**. It helps in understanding the
**capacity needs** of the **solution**.

The scale of the problem refers to the size of the task that your AIOps system
needs to address. For a recommendation system, you might need to consider the
number of users, the number of potential recommendations, and the volume of
interaction data. Large-scale problems may require distributed computing
resources and algorithms designed to handle big data.

### Performance

The determination of how the **system** must **perform**, considering factors
like **speed**, **accuracy**, **reliability**, and **efficiency**. This includes
understanding whether **real-time processing** is required or if **offline
processing** can suffice.

The performance requirements will significantly impact your AIOps system design.
Real-time systems, such as a recommendation system that updates as users
interact with the site, need fast, lightweight models and efficient data
handling. In contrast, offline systems, such as a model that updates
recommendations overnight, can afford to use more complex models and
computationally intensive processes.

## References and Further Readings

-   Huyen, Chip. "Chapter 2. Introduction to Machine Learning Systems Design."
    In Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
-   Kleppmann, Martin. "Chapter 1. Reliable, Scalable, and Maintainable
    Applications." In Designing Data-Intensive Applications. Beijing:
    O'Reilly, 2017.
-   [Designing Twitter's Trending Hashtags Solution](https://mlops-discord.github.io/blog/designing-twitters-trending-hashtags-solution/)

[^benefit-structure]:
    An example is in a binary classification problem, where there are $4$
    possible outcomes: true positive, false positive, true negative, and false
    negative. The benefit structure is the cost of each outcome. For example, in
    spam detection, a false positive is more costly than a false negative, and
    therefore a possible benefit structure is $[1, 10, 1, 0]$.
