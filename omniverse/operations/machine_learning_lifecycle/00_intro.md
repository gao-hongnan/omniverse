---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# The Lifecycle of an AIOps System

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
```

**Artificial Intelligence for ML Operations (AIOps)** has emerged as a powerful
approach to managing complex Machine Learning (ML) environments. By leveraging
[AI](https://www.ibm.com/cloud/learn/what-is-artificial-intelligence),
[machine learning](https://www.expert.ai/blog/machine-learning-definition/), and
[data analytics](https://www.oracle.com/data-analytics/what-is-data-analytics/),
AIOps enables ML teams to automate and optimize various aspects of their
operations, improving efficiency and reducing downtime. In this documentation,
we'll explore the lifecycle of an AIOps system, providing insights into how
these systems are developed, implemented, and maintained.

In Google's article
[MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning),
the authors defined 9 key steps in the lifecycle of an AIOps system. We will
refine it below.

1. [Problem Formulation](01_problem_formulation.md)
2. [Project Scoping](02_project_scoping.md)
3. [Data Pipeline, Data Engineering and DataOps](./03_dataops_pipeline/03_dataops_pipeline.md)
4. [Data Extraction, Analysis and Preparation](04_mlops_data_pipeline.md)
5. [Model Development, Selection and Training](./05_model_development_selection_and_training/05_ml_training_pipeline.md):
   The data scientist implements different algorithms with the prepared data to
   train various ML models. In addition, you subject the implemented algorithms
   to **hyperparameter tuning** to get the best performing ML model. The output
   of this step is a **trained model**.
6. [Model Evaluation](06_model_evaluation.md): The model is evaluated on a
   [**holdout test set**](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets#Holdout_dataset)
   to evaluate the model quality. The output of this step is a set of
   **metrics** to assess the quality of the model.
7. [Model Validation, Registry and Pushing Model to Production](07_model_validation_registry_and_pushing_model_to_production.md):
   The model is confirmed to be adequate for deployment—that its **predictive
   performance** is better than a certain baseline.
8. [Model Deployment and Serving](08_model_deployment_and_serving.md): The
   validated model is deployed to a target environment to serve predictions.
   This deployment can be one of the following:

    - **Microservices** with a REST API to serve online predictions.
    - An **embedded model** to an edge or mobile device.
    - Part of a **batch prediction system**.

9. [Model Monitoring](09_model_monitoring.md): The model is monitored to ensure
   that it continues to perform as expected. This monitoring can be one of the
   following:

    - **Anomaly detection** to detect unexpected behavior in the model.
    - **Drift detection** to detect changes in the data distribution.
    - **Performance monitoring** to detect changes in the model performance.

10. [Continuous Integration, Deployment, Learning and Training](010_continuous_integration_deployment_learning_and_training.md)

The lifecycle gif below by
[Deepak](https://www.linkedin.com/in/mr-deepak-bhardwaj/) captures a
_simplified_ lifecycle of an AIOps system.

```{figure} ./assets/ml-lifecycle.gif
---
name: ml-lifecycle
height: 400px
---

MLOps Lifecycle.

**Image Credit:**
[Deepak](https://www.linkedin.com/in/mr-deepak-bhardwaj)
```

He also has a DataOps lifecycle gif below.

```{figure} ./assets/dataops-lifecycle.gif
---
name: dataops-lifecycle
height: 400px
---

DataOps Lifecycle.

**Image Credit:**
[Deepak](https://www.linkedin.com/in/mr-deepak-bhardwaj)
```

## Table of Contents

```{tableofcontents}

```

## References and Further Readings

-   [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
-   [MLOps Maturity Model (Azure)](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)
-   [Made With ML: Designing Machine Learning Products](https://madewithml.com/courses/mlops/design/)
-   Huyen, Chip. "Chapter 2. Introduction to Machine Learning Systems Design."
    In Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
-   Kleppmann, Martin. "Chapter 1. Reliable, Scalable, and Maintainable
    Applications." In Designing Data-Intensive Applications. Beijing:
    O'Reilly, 2017.
-   [Designing Twitter's Trending Hashtags Solution](https://mlops-discord.github.io/blog/designing-twitters-trending-hashtags-solution/)
-   [Machine Learning System Design Interview](https://bytebytego.com/intro/machine-learning-system-design-interview)
