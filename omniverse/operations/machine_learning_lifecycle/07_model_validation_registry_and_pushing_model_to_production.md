---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Stage 7. Model Validation, Registry and Pushing Model to Production (MLOps)

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
```

After a model has been trained and evaluated, it's crucial to thoroughly
validate it before promoting it to production. This validation process involves
multiple stages, each designed to ensure the model's reliability, performance,
and suitability for real-world deployment. We first take a look at offline
validation.

## Offline Validation

Offline validation is a critical step that goes beyond simple metric evaluation.
This includes producing evaluation metric values using the trained model on a
test dataset to assess the model's predictive quality. Compare the evaluation
metric values produced by your newly trained model to the current model,
baseline model, or other business-requirement models. Ensure that the new model
produces better performance than the current model before promoting it to
production. Check that the performance of the model is consistent on various
segments of the data.

```{list-table} Offline Validation Steps
:header-rows: 1
:name: offline-validation-steps

* - Validation Step
  - Details
* - Rigorous Metric Evaluation
  - - Produce evaluation metric values using the trained model on a held-out test dataset.
    - Employ a diverse set of metrics relevant to the problem domain (e.g., accuracy, F1-score, AUC-ROC, RMSE for regression tasks).
    - Utilize techniques like k-fold cross-validation to ensure robustness of results.
* - Comparative Analysis
  - - Benchmark against current production model, baseline models, and business-defined thresholds.
    - Conduct statistical significance tests (e.g., t-tests, McNemar's test) to ensure improvements are not due to chance.
* - Data Segment Performance
  - - Evaluate model performance across various data segments (e.g., demographic groups, time periods, geographical regions).
    - Identify and address any performance disparities that could indicate bias or overfitting.
* - Error Analysis
  - - Conduct in-depth analysis of misclassified instances or high-error predictions.
    - Use techniques like SHAP (SHapley Additive exPlanations) values to understand feature importance and model decisions.
```

## Deployment Testing

Deployment testing ensures the model's compatibility with the production
environment:

-   **Infrastructure Compatibility**:

    -   Verify model compatibility with target hardware (CPU, GPU, TPU) and
        software stack.
    -   Test model loading, initialization, and inference times under various
        load conditions.

-   **API Consistency**:

    -   Ensure the model's input/output format aligns with the prediction
        service API specifications.
    -   Implement comprehensive unit tests and integration tests for the model
        serving pipeline.

-   **Scalability Testing**:

    -   Conduct load testing to verify the model's performance under expected
        and peak traffic conditions.
    -   Measure and optimize latency and throughput to meet SLAs (Service Level
        Agreements).

## Online Validation

Online validation assesses the model's performance in a real-world environment:

```{list-table} Online Validation Techniques
:header-rows: 1
:name: online-validation-techniques

* - Technique
  - Details
* - Canary Deployment
  - - Gradually roll out the new model to a small percentage of live traffic (e.g., 5-10%).
    - Monitor key performance indicators (KPIs) and business metrics in real-time.
    - Implement automated rollback mechanisms if predefined thresholds are breached.
* - A/B Testing
  - - Design statistically rigorous A/B tests to compare the new model against the current production model.
    - Define clear success criteria and run-time for the experiment.
    - Analyze both model-specific metrics and downstream business impacts.
* - Shadow Mode Deployment
  - - Deploy the new model in parallel with the existing production model.
    - Log predictions from both models without affecting user experience.
    - Analyze discrepancies and potential improvements offline.
```

These stages ensure that the model is not only good theoretically but also
performs well in practical, real-world scenarios.

## Model Registry and Promotion (MLOps)

A model registry is a centralized place where developers, data scientists, and
MLOps engineers can share and collaborate on different versions of machine
learning models. It serves as a single source of truth for all models developed
and deployed within an organization.

-   **Versioning**: Every time a model is trained, updated or tuned, a new
    version of the model is created and registered. This helps in tracking the
    evolution of models and enables easy rollback to any previous version if
    required.

-   **Metadata Management**: Along with the model binaries, the model registry
    also stores metadata about each model such as the date of creation, the
    person who created it, its version, its performance metrics, associated
    datasets, etc.

-   **Model Lineage**: The registry keeps track of the model's lineage, which
    includes the detailed process of how a model was built, including data
    sources, feature transformations, algorithms used, model parameters, etc.
    This is crucial for debugging, audit, compliance, and collaboration.

## Model Promotion

After successful validation, models that meet the desired performance criteria
are promoted to production. The promotion process involves setting the model
status to "production" in the registry, and potentially deploying it to a
production environment. This could involve replacing an older version of the
model that's currently in production, or it might involve deploying the model to
a new environment or application.

The model promotion process should be systematic and auditable:

1. **Staging Environment**:

    - Deploy candidate models to a staging environment that closely mimics
      production.
    - Conduct final integration tests and performance benchmarks.

2. **Approval Workflow**:

    - Implement a formal approval process involving data scientists, ML
      engineers, and business stakeholders.
    - Use a checklist-based approach to ensure all validation steps are
      completed.

3. **Automated Promotion**:

    - Develop CI/CD pipelines for automated model deployment upon approval.
    - Implement blue-green deployment strategies for zero-downtime updates.

4. **Monitoring and Alerting**:

    - Set up real-time monitoring of model performance post-deployment.
    - Implement automated alerts for performance degradation or data drift.

5. **Rollback Strategy**:
    - Maintain the ability to quickly revert to previous model versions.
    - Conduct regular drills to ensure rollback procedures are effective.

MLFlow or similar experiment tracking tools can be used to implement a model
registry. These tools provide a centralized place to track and manage models,
including model versioning, lineage, and metadata management. It allows you to
tag models with different stages, such as "staging", "production", "archived",
etc. This helps in keeping track of the model's lifecycle and enables easy
rollback to previous versions if required.
