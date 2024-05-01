# Stage 6. Data Preparation (MLOps)

Data Preparation is the stage where insights and decisions from the Data
Analysis phase are operationalized. We are essentially **executing the data
analysis decisions**.

At this juncture, we prime the data for machine learning models through the
following activities:

-   **Data Cleaning**: Execute decided strategies for handling missing values,
    outliers, or incorrect data.

-   **Data Preprocessing**: Conduct necessary preprocessing tasks such as
    encoding categorical variables, normalizing or standardizing numerical
    variables.

-   **Feature Engineering**: Develop new features from the initial data to
    enhance model performance.

-   **Resampling (Class Imbalance)**: For imbalanced datasets, consider using
    resampling techniques to balance the classes.

-   **Feature Selection**: If the data has high dimensionality, apply feature
    selection techniques to shrink the feature space and prevent overfitting.

-   **Dimensionality Reduction**: Especially when dealing with high-dimensional
    data, techniques like Principal Component Analysis (PCA) or t-Distributed
    Stochastic Neighbor Embedding (t-SNE) might be used to reduce the number of
    features and simplify the data structure. This can help mitigate the curse
    of dimensionality and improve computational efficiency.

-   **Data Encoding**: Ensure that the data is in a format that machine learning
    algorithms can process. This might entail one-hot encoding for categorical
    data or other forms of encoding.

-   **Handling Temporal Data**: In the case of time series data, additional
    steps may be required, such as generating time-based features, managing
    missing time steps, or resampling the data to a different frequency.

-   **Data Augmentation**: For specific use cases like image and sound
    classification tasks, data augmentation techniques can enhance the training
    data volume and improve model performance.

Data Preparation is about converting raw data into a format suitable for model
training. The steps executed at this stage can significantly influence the
effectiveness of subsequent machine learning operations and the quality of the
final outputs.

Remember that this stage is iterative. Based on the results of model training
and evaluation, you might need to revisit data analysis or data preparation
steps. Therefore, an adaptable and iterative approach should be upheld
throughout this stage.

## Resampling Methods - Cross Validation

Lastly, we have the splitting:

-   **Resampling (Cross Validation)**: It's standard practice at this stage to
    partition the data into training, validation, and test sets. This is
    fundamental for evaluating model performance and avoiding overfitting. The
    method and ratios for splitting should be carefully chosen. For certain
    types of data, such as time series, random splits might not be appropriate.

## Summary

Data preparation: The data is prepared for the ML task. This preparation
involves data cleaning, where you split the data into training, validation, and
test sets. You also apply data transformations and feature engineering to the
model that solves the target task. The output of this step are the data splits
in the prepared format.

## References and Further Readings

-   Huyen, Chip. "Chapter 4. Training Data." In Designing Machine Learning
    Systems: An Iterative Process for Production-Ready Applications, O'Reilly
    Media, Inc., 2022.
-   Huyen, Chip. "Chapter 5. Feature Engineering." In Designing Machine Learning
    Systems: An Iterative Process for Production-Ready Applications, O'Reilly
    Media, Inc., 2022.
