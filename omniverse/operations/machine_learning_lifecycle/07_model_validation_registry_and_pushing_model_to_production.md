# Stage 9. Model Validation, Registry and Pushing Model to Production (MLOps)

After a model has been trained and evaluated, it's essential to validate it
before promoting it to production:

-   **Offline Validation**: This includes producing evaluation metric values
    using the trained model on a test dataset to assess the model's predictive
    quality. Compare the evaluation metric values produced by your newly trained
    model to the current model, baseline model, or other business-requirement
    models. Ensure that the new model produces better performance than the
    current model before promoting it to production. Check that the performance
    of the model is consistent on various segments of the data.

-   **Deployment Testing**: Test your model for deployment, including
    infrastructure compatibility and consistency with the prediction service
    API.

-   **Online Validation**: After offline validation, a newly deployed model
    undergoes online validation — in a canary deployment or an A/B testing setup
    — before it serves predictions for the online traffic. This helps to assess
    the model's performance in a real-world environment with live data.

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

### Model Promotion

After successful validation, models that meet the desired performance criteria
are promoted to production. The promotion process involves setting the model
status to "production" in the registry, and potentially deploying it to a
production environment. This could involve replacing an older version of the
model that's currently in production, or it might involve deploying the model to
a new environment or application.

MLFlow or similar experiment tracking tools can be used to implement a model
registry. These tools provide a centralized place to track and manage models,
including model versioning, lineage, and metadata management. It allows you to
tag models with different stages, such as "staging", "production", "archived",
etc. This helps in keeping track of the model's lifecycle and enables easy
rollback to previous versions if required.