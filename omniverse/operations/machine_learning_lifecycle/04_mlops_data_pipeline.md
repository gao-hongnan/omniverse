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

# Stage 4. Data Extraction (MLOps), Data Analysis (Data Science), Data Preparation (Data Science)

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

## Stage 4.1. Data Extraction (MLOps)

Data extraction in MLOps usually involves pulling data from a Data Warehouse or
Data Lake that has already been transformed. The transformed data is generally
agnostic to downstream tasks, such as the specific machine learning models being
used.

Remember, the data at this stage has been cleaned and transformed — but this
doesn't imply it's been preprocessed or engineered for machine learning tasks.
This cleaned data serves as a versatile resource, used not just by machine
learning teams, but also by business intelligence, data analytics, and other
teams that require transformed data.

### Batch Extraction/Ingestion

This type of data extraction is used when there are large volumes of data that
do not require immediate analysis or use. The data is collected over a period of
time and then ingested into the data lake all at once. This can be scheduled to
occur at regular intervals, like daily, weekly, or monthly, depending on the
requirements. This method is particularly beneficial when dealing with large
amounts of structured and unstructured data and where latency is not a critical
factor.

#### An Example of Batch Extraction

Let's consider an example involving a traffic surveillance system for a city.

Imagine a city has installed surveillance cameras at all major intersections to
monitor and analyze traffic patterns. These cameras generate an enormous amount
of visual data every day, and analyzing this data in real-time isn't always
necessary. The primary goal is to understand traffic patterns over time,
identify peak hours, detect unusual congestion, and recognize patterns that
could suggest infrastructure changes, like adding a new traffic light or
creating a roundabout.

In this case, the video feeds from each camera could be processed in batches. At
the end of each day, a batch ingestion process could collect the video data,
preprocess it (e.g., converting video into frames), and store it in a suitable
format (e.g., image files, video files, or even encoded vector representations)
in a data lake or similar storage system.

A computer vision algorithm can then process these batches of data to extract
relevant features and analyze them. For instance, it could estimate the number
of vehicles, identify the type of vehicles (bikes, cars, trucks), recognize
traffic jams, or detect accidents. These analyses can then provide actionable
insights to traffic management authorities.

Consequently, batch ingestion in this scenario is suitable because the data
doesn't need to be processed in real-time.

### Real-time Ingestion (Stream Ingestion)

Real-time data ingestion involves the continuous input of data into the data
lake as it is generated. It's used when the data needs to be analyzed and acted
upon immediately or near real-time, such as in event-driven applications or live
dashboards. This method is commonly used in scenarios like tracking user
activity on a website, real-time analytics, monitoring, fraud detection, etc.
Real-time ingestion can be more complex and resource-intensive than batch
processing due to the need for continuous processing and potential volume and
velocity of data.

#### An Example of Real-time Ingestion

TikTok, like other social media platforms, operates in a real-time environment
where quick decision making is key to user engagement and satisfaction.

For example, when users interact with TikTok, they generate vast amounts of data
through their actions, such as likes, shares, comments, and even the duration
for which they watch a video. These user interactions are captured and processed
in real-time.

As users scroll through the "For You" page, TikTok's algorithm needs to decide
which video to show next to keep the user engaged. This decision is informed by
the user's past behavior and the behavior of similar users. The real-time data
ingestion pipeline continuously feeds the machine learning model with fresh
data, helping it make the most relevant recommendations.

Moreover, TikTok also needs to monitor user-generated content in real-time for
any inappropriate or harmful content. With millions of videos uploaded every
day, this task needs to be automated. Machine learning models, continuously
updated with real-time ingested data, can help identify and flag such content.

In both these examples - content recommendation and content moderation -
real-time data ingestion is needed to ensure timely and accurate
decision-making.

## Stage 4.2. Data Analysis (Data Science)

Data analysis forms the basis of any successful machine learning project. At
this stage, we dive deeper into the dataset's characteristics by doing
**Exploratory Data Analysis (EDA)**. EDA involves getting a better understanding
of the data, including variables' relationships, detecting outliers, and
understanding patterns.

Note that this stage is usually not part of the MLOps pipeline. Instead, it's
what we need to do before model building.

In the {ref}`table below <ml-lifecycle-04-data-analysis>`, we list some common
types of data analysis that are performed during this stage.

```{list-table} Types Of Data Analysis
:header-rows: 1
:name: ml-lifecycle-04-data-analysis

-   -   Stage
    -   Description
-   -   Distribution Analysis
    -   Evaluate the distribution of the data to identify issues like class
        imbalance, skewed data, and outliers.
-   -   Feature Engineering Planning
    -   Determine the need for creating new features based on the data analysis,
        such as interaction variables, polynomial features.
-   -   Data Cleaning and Preprocessing Decision
    -   Decide on the appropriate steps for data cleaning and preprocessing,
        including handling missing values, encoding, and scaling.
-   -   Correlation Analysis
    -   Examine relationships between variables to identify influential features
        and detect multicollinearity.
-   -   Outlier Detection and Treatment
    -   Identify and decide how to treat outliers, which may involve
        transformations or removal.
-   -   Anomaly Detection
    -   Focus on finding patterns that do not conform to expected behavior,
        which could be indicative of data issues or unique insights.
-   -   Statistical Analysis
    -   Perform hypothesis testing and other statistical methods to understand
        the significance and interactions of variables.
-   -   Feature Selection and Importance Evaluation
    -   If you have a preliminary model or an easy-to-use model (like a Decision
        Tree), you might run a quick test to see which features seem to be most
        predictive. This can help guide feature engineering and selection. This
        is a common technique for “feature selection”.
-   -   Data Visualization
    -   Visualize the data to uncover insights not apparent from raw data
        analysis.
```

EDA is not just about visualization and show summary statistics for the sake of
it. Why does people say that it is an important step before model building is
because one can draw insights from the data to help make _informed_ decisions
about the model building process. For example, if you find that the data is
highly imbalanced, you might need to use techniques like weighted loss functions
and use apply appropriate evaluation metrics as well as any variation of
stratified resampling methods.

### Sampling

Sampling is a strategy to select a subset of data from your dataset, and it is
crucial when the dataset is too large to handle effectively. There are different
types of sampling methods, such as random sampling, stratified sampling,
reservoir sampling, cluster sampling and many more.

Note at this stage this isn't really about the usual resampling or
cross-validation techniques. Here we won't talk about every reason why we need
to sample data. However, let's consider a simple scenario where you have
terabytes of text data, ready to be analyzed for downstream pre-training tasks.
However, it may well be inefficient to load all the data and perform EDA, with
high performing system, you might be able to, but a better and more efficient
way is to sample a representative subset of the data for EDA.

### Labeling

Labeling is the process of assigning definitive labels to each instance in your
dataset. This is necessary for supervised learning tasks. Automated labeling
techniques like weak/semi-supervised learning, pseudo-labeling, can be used, but
in some cases, manual labeling by subject matter experts is required. Though I
would say with the advent of generative AI models, we can make this task less
labor-intensive.

### Class Imbalance

Class imbalance occurs when the classes in your target variable are not
represented equally. This can lead to biased models and inaccurate predictions.
Various techniques such as over-sampling, under-sampling, or SMOTE can be used
to deal with class imbalance.

One should always note that class imbalance is difficult to deal with. It can be
worsened if you use a misleading metric like accuracy - consider the case where
99.9% of the data is in class A, and 0.1% is in class B. If your model naively
predict everything as class A, you will get an accuracy of 99.9%, but you are
not actually predicting anything in class B.

There are many literature out there on how to handle class imbalance, some argue
that over/under sampling is not a good idea and instead should use weighted loss
functions. However, in practice, it is often a combination of these methods that
work best. Just remember, for most classification problem (discriminative
models), you really just want to force the model to learn the decision boundary
for the minority class.

### Data Augmentation

Data augmentation is a strategy to increase the diversity of your data by
applying random but realistic transformations to the data. In the context of
image data, this could mean rotations, flips, or color changes.

### Feature Engineering

Feature engineering is the process of creating new features from the existing
ones to improve model performance. The running joke is that deep learning is
everywhere now and it is sold as a way to avoid manually handcrafting features.

Like the simplest example is the engineering of word embeddings, classical ways
include Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF).
However, in deep learning models like GPT, we would just have these embeddings
learning by themselves.

However, we are still far from using deep learning everywhere, and sometimes
even in deep learning models, feature engineering may still be necessary. We
leave the readers to have a read on "Chapter 5. Feature Engineering." In
Designing Machine Learning Systems for more information.

### Data Leakage

Data leakage is a problem where information from outside the training dataset is
used to create the model. This can lead to overly optimistic performance
estimates. During EDA, we don't need to worry about data leakage as we're not
using the data to train or test the model. It's crucial to guard against data
leakage when we move into model building. Like it is meaningless to have
validation metrics that are super good only to realise the answers are already
leaked from the training set to the validation set.

## Stage 4.3. Data Preparation (MLOps)

Data Preparation is the stage where insights and decisions from the Data
Analysis phase are operationalized. We are essentially **executing the data
analysis decisions** and package it into a pipeline to adhere to the MLOps,
DataOps and DevOps principles.

To reiterate, we have some common steps in the data preparation stage:

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

-   **Resampling (Cross Validation)**: It's standard practice at this stage to
    partition the data into training, validation, and test sets. This is
    fundamental for evaluating model performance and avoiding overfitting. The
    method and ratios for splitting should be carefully chosen. For certain
    types of data, such as time series, random splits might not be appropriate.

Remember that this stage is iterative. Based on the results of model training
and evaluation, you might need to revisit data analysis or data preparation
steps. Therefore, an adaptable and iterative approach should be upheld
throughout this stage.

## References and Further Readings

-   Huyen, Chip. "Chapter 4. Training Data." In Designing Machine Learning
    Systems: An Iterative Process for Production-Ready Applications, O'Reilly
    Media, Inc., 2022.
-   Huyen, Chip. "Chapter 5. Feature Engineering." In Designing Machine Learning
    Systems: An Iterative Process for Production-Ready Applications, O'Reilly
    Media, Inc., 2022.
-   [Madewithml: Data Engineering](https://madewithml.com/courses/mlops/data-stack/)
