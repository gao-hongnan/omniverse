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

# Stage 2. Project Scoping And Framing The Problem

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

## Introduction

After defining the problem statement from a business point of view, the next
step is to scope the project and frame the problem in the context of machine
learning (that is if the problem warrants a machine learning solution in the
first place). This involves identifying the inputs, outputs, objective function,
and the type of machine learning task that best suits the problem.

The following sections we will go deeper into the process of framing real-world
problems as machine learning tasks, knowing how to define inputs and outputs,
identifying objective functions, and recognizing the specific machine learning
tasks that best suit different types of problems.

## Frame a Business Problem as an Machine Learning Task

In the first section, we discuss a simplified example taken from Chip Huyen's
book _Design Machine Learning Systems_.

### An Example On E-commerce Recommendation

Let's take the example of a popular online e-commerce platform. They are looking
to increase the user engagement on their website, which in turn will lead to
increased purchases, thereby raising the company's overall profits.

At first glance, increasing user engagement might not seem like a machine
learning problem, but with a little bit of investigation and framing, it can be
turned into one.

After doing a thorough analysis, the ML engineering team finds out that users
often abandon the platform because they struggle to find the items they are
interested in. The vast array of products on the website becomes overwhelming,
resulting in a frustrating experience for the users.

With these insights, the team decides to frame the problem as a machine learning
task - **recommendation**.

1. **Input**: User's past browsing and purchase history, demographics, and other
   user metadata.
2. **Output**: A personalized list of recommended products.
3. **Objective Function**:
   **[Maximize click-through rate (CTR)](https://en.wikipedia.org/wiki/Click-through_rate)**
   of the recommended items.
4. **Machine Learning Task**:
   **[Recommendation](https://en.wikipedia.org/wiki/Recommender_system)**.

By formulating this as a recommendation problem, the team can now use machine
learning to learn the user's preferences and suggest items they would be
interested in, making the browsing process more efficient and enjoyable for the
user. This would increase user engagement and eventually lead to an increase in
purchases.

### Key Components

The process of framing the problem can be systematically broken down into four
key components, each playing a critical role in defining the
**[machine learning task](https://en.wikipedia.org/wiki/Machine_learning)**.

```{list-table} Key Components of Framing a Machine Learning Problem
:header-rows: 1
:name: ml-lifecycle-02-project-scoping-key-components

-   -   Component
    -   Description
-   -   Inputs
    -   These are the raw materials that the model will use to make predictions
        or decisions. Inputs can range from numerical data, text, images,
        videos, to audio, or any combination thereof. Identifying the right
        inputs is essential, and note that sometimes you have a combination of
        inputs (i.e. text + image - multi-modal inputs).
-   -   Outputs
    -   These represent the predictions or decisions that the model will make
        based on the given inputs. Outputs might be continuous values, discrete
        categories, sequences, or more complex structures.
-   -   Objective Function
    -   This defines the specific goal that the machine learning algorithm aims
        to optimize. It might be the minimization of error in a regression task,
        maximization of accuracy in a classification task, or any other
        quantifiable metric that guides the learning process. The choice of the
        objective function must align with the overall goals of the task and
        drive the model towards meaningful results (i.e. fullfil the business
        objectives).
-   -   Type of Machine Learning Task
    -   Identifying the type of machine learning task involves categorizing the
        problem into a specific paradigm, such as classification, regression,
        clustering, recommendation, nlp or computer vision.
        This categorization informs the
        selection of appropriate algorithms, techniques, and evaluation metrics.
```

## Features, Labels and Outputs (Inputs/Outputs)

Understanding the nature of the inputs and outputs for your machine learning or
deep learning task is crucial as it directly influences your data collection
strategy, the selection and design of the model architecture, and even the
deployment of the model. For instance, if you are working on a text
classification task, the input data would be the text documents, and the output
would be the predicted class labels along with the confidence scores (or
logits).

### Labels

The terms "features" and "labels" are frequently used in the context of
supervised learning. The features represent the input data, which can take a
multitude of forms including numerical data, categorical data, text, images,
video, audio, and more. The labels, on the other hand, represent the outputs or
the target variables we want the model to predict.

The nature of the task dictates the format of these features and labels. For
example, in an image classification task, the features would be the raw image
pixels and the labels would be the class or category of each image.

On the other hand, if labels are not readily available or are difficult and
costly to obtain, you might consider alternative strategies such as
semi-supervised or self-supervised learning. In these approaches, the model
leverages both labeled and unlabeled data to learn. For example, Generative
Pre-trained Transformer (GPT) models are trained on vast amounts of text data
without explicit labels, learning to predict the next word in a sentence where
we use the model's input as the label.

### Features (Inputs)

**Inputs** (features) represent the raw data or features that the algorithm will
process to generate predictions or decisions.

#### Types of Inputs

In {ref}`the table below <ml-lifecycle-02-types-of-data-inputs>`, we outline the
different types of data inputs that machine learning models can process. These
include numerical data, categorical data, text data, and multimedia data.

```{list-table} Types of Data Inputs for Machine Learning
:header-rows: 1
:name: ml-lifecycle-02-types-of-data-inputs

-   -   Data Type
    -   Description
-   -   Numerical Data
    -   Values that can be measured and quantified, such as
        [temperature](https://en.wikipedia.org/wiki/Temperature),
        [age](https://en.wikipedia.org/wiki/Ageing), or
        [income](https://en.wikipedia.org/wiki/Income).
-   -   Categorical Data
    -   Discrete categories or labels, such as
        [gender](https://en.wikipedia.org/wiki/Gender),
        [product type](<https://en.wikipedia.org/wiki/Product_(business)>), or
        [country](https://en.wikipedia.org/wiki/Country).
-   -   Text Data
    -   Unstructured text that can include
        [reviews](https://en.wikipedia.org/wiki/Review),
        [comments](<https://en.wikipedia.org/wiki/Comment_(computer_programming)>),
        or [social media posts](https://en.wikipedia.org/wiki/Social_media).
-   -   Multimedia Data
    -   Complex data types like
        [images](https://en.wikipedia.org/wiki/Digital_image),
        [videos](https://en.wikipedia.org/wiki/Video), or
        [audio](https://en.wikipedia.org/wiki/Audio_signal).
```

Again, the above are not mutually exclusive and you might have a combination of
these data types as inputs.

#### An Example On Medical Diagnosis

In the context of a medical diagnosis on cancer, inputs might include patient
**[medical history](https://en.wikipedia.org/wiki/Medical_history)**,
**[symptoms](https://en.wikipedia.org/wiki/Symptom)**,
**[lab results](https://en.wikipedia.org/wiki/Medical_test)** and
**[medical images](https://en.wikipedia.org/wiki/Medical_imaging)**.

This could jolly well be a multi-modal input, where the model takes in both
images and text as inputs. For example, if you were to predict the presence of
cancer in a patient, you might use the patient's medical history and symptoms as
text inputs and medical images (like X-rays or MRIs) as image inputs.

### Outputs (Predictions)

**Outputs** refer to the predictions, decisions, or insights that a machine
learning model generates based on the defined inputs.

#### Relationship to Business Objectives

Outputs must be carefully crafted to reflect the ultimate goals and success
metrics of the organization. Whether it's predicting
**[customer churn](https://en.wikipedia.org/wiki/Customer_attrition)**,
optimizing
**[supply chain operations](https://en.wikipedia.org/wiki/Supply_chain)**, or
identifying **[fraudulent transactions](https://en.wikipedia.org/wiki/Fraud)**,
the outputs must provide actionable insights that drive decision-making and
align with overarching business strategies.

This may mean that interpretable models are preferred over black-box models when
the business requires explainable results. What this means is that some model
like neural networks are not well-calibrated. What this means that the
probability predictions (logits) of the model do not reflect the _actual
likelihood_ of the event happening. In simpler terms, if such a model predicts a
90% probability of an event happening, it might not actually happen 90% of the
time.

### Example: Personal Protective Equipment (PPE) Detection

#### Inputs and Labels

For instance, suppose we want to develop a model to detect Personal Protective
Equipment (PPE) in construction sites and although we have abundance of input
images (inputs) but we do not have enough bounding box labels.

Here's two possible approaches to tackle this problem:

1. **Self-Supervised Learning**: Since we do not have enough resources for
   manual labelling, we can use a pre-trained model like YOLO to infer human
   bounding boxes.

2. **Pseudo-Labeling**: We manually label a small set of about 200 images with
   PPE details. Then, we train a model to overfit these 200 images, ensuring it
   becomes adept at identifying PPE in these specific images. After this, we can
   use this model to generate pseudo-labels for the rest of our data, thus
   significantly reducing the amount of manual labelling required.

To proceed with this, we could prepare an input CSV file with the following
columns:

-   Human ID (for grouping CV strat)
-   Label (for stratification)
-   Image ID
-   Image Path (for loading image)

This CSV based approach is useful for small datasets, but for larger datasets,
you might consider using a database like MongoDB or PostgreSQL or even those
optimized data lake/data warehouse solutions like AWS S3 or Google BigQuery.

#### Outputs

The outputs of the model will be the bounding boxes and class labels of detected
PPE gear in each image or video frame. Each detected object (helmet, mask, vest,
etc.) would have a bounding box (indicating its location and size in the image)
and a class label (identifying the type of PPE). The model might also provide a
confidence score showing the model's certainty about the detected object's
class.

#### Further Processing

These raw outputs can be processed further to generate more useful information,
such as:

-   Binary indication of PPE compliance for each worker.
-   An overall PPE compliance score for the site at a given time.
-   Alerts or notifications triggered by any detected non-compliance.
-   Aggregated statistics over time, such as compliance rates by worker, time of
    day, area of the site, etc.

We can also save the embeddings for further analysis such as model
explainability and interpretability. Embeddings can also detect data drift and
concept drift.

## Machine Learning Objective (Establishing Metrics)

One of the key steps in formulating a machine learning problem is defining the
business objectives and then translating them into quantifiable and measureable
metrics. Business objectives could range from increasing revenue and viewership
to improving click-through rates. But these high-level objectives are often not
directly measurable and can't be used to evaluate the performance of the machine
learning system directly.

So, how can we connect these business objectives to machine learning objectives?
The first step is to translate the business problem into a machine learning
problem. Once this is achieved, we can then define a well-specified objective
function for the machine learning problem.

This is not a formal machine learning theory review, so we will be less pedantic
and mix the idea of loss, cost, objective and performance metrics in one
section.

While **establishing metrics**, we need to consider both
**[offline metrics](https://en.wikipedia.org/wiki/Offline_learning)** (used
during the model development phase to evaluate the model's performance on a
validation set) and
**[online metrics](https://en.wikipedia.org/wiki/Online_machine_learning)**
(used to measure the model's performance after deployment in a production
environment).

Here's a brief overview of various types of metrics (non-exhaustive):

<table>
  <tr>
    <th>Metric Type</th>
    <th>Problem Type</th>
    <th>Metric Examples</th>
    <th>Description</th>
  </tr>
  <tr>
    <td rowspan="2"><strong>Offline Metrics</strong></td>
    <td><strong>Classification</strong></td>
    <td>AUROC, F1</td>
    <td>Select appropriate <strong><a href="https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234">offline metrics</a></strong> based on the type of machine learning problem. Note that accuracy may not be the best metric for classification problems, as it can be misleading.</td>
  </tr>
  <tr>
    <td><strong>Regression</strong></td>
    <td>R-squared, MSE, RMSE, MAPE</td>
    <td>Choose suitable metrics for regression problems to evaluate the model's performance.</td>
  </tr>
  <tr>
    <td rowspan="2"><strong>Online Metrics</strong></td>
    <td><strong>Web Applications</strong></td>
    <td>Click-through Rate (CTR)</td>
    <td>Design metrics to evaluate the AIOps system's performance in a production environment, such as <strong><a href="https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093">click-through rates</a></strong> for web applications.</td>
  </tr>
  <tr>
    <td><strong>Credit Card Scanning</strong></td>
    <td>Failures and Manual Keying</td>
    <td>Measure the number of times users fail and return to manual keying for credit card scanning applications.</td>
  </tr>
  <tr>
    <td><strong>Non-Functional Metrics</strong></td>
    <td><strong>Efficiency</strong></td>
    <td>Training Speed, Resource Utilization</td>
    <td>Consider metrics like <strong><a href="https://www.softwaretestinghelp.com/non-functional-testing/">training speed or resource utilization</a></strong> to assess the system's efficiency.</td>
  </tr>
</table>

### Offline Metrics

Offline metrics are used during the model development and validation phases. For
instance, in a classification problem where we are trying to detect if an email
is spam or not, we might use metrics such as precision, recall, F1 score, and
AUROC to evaluate the model's performance on a held-out validation set.

For classification problems, consider using offline metrics like
**[AUROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)**
and **[F1](https://en.wikipedia.org/wiki/F1_score)**, while for regression
problems, metrics like
**[R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination)**,
**[MSE](https://en.wikipedia.org/wiki/Mean_squared_error)**,
**[RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)**, and
**[MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)** can be
used.

### Online Metrics

When the model is in a production environment, online metrics become important.
For web applications, metrics such as
**[Click-through Rate (CTR)](https://en.wikipedia.org/wiki/Click-through_rate)**
might be appropriate, while for applications such as credit card scanning,
metrics such as Failures and Manual Keying might be more suitable.

For instance, after deploying a model in a production environment, such as a web
application that predicts the animal in user-uploaded images, it's crucial to
develop online metrics to assess its performance. One such metric is
Click-Through Rate (CTR), which indicates user engagement and helps attract more
visitors to the website.

Consider another scenario where a smartphone application that scans credit card
details to autofill user information. This process can be seen as an object
detection task, where the app first localizes and classifies the credit card
amongst other objects in the camera's view. Subsequently, Optical Character
Recognition (OCR) is performed to extract the necessary details.

In this context, an effective online metric would be the number of times users
fail and revert to manual data entry. This metric helps gauge the app's
performance and user experience.

### Non-Functional Metrics

Lastly, don't forget to consider **non-functional metrics** like training speed
and resource utilization to evaluate the system's efficiency. For example, in
the pretraining of Large Scale Deep Learning models, we are often interested in
TeraFLOPS and MFU (Model FLOPS Utilization) to understand how efficiently the
model is being trained. I mean you don't want to spend a fortune on training a
model that is severely underutilized. Often there are already benchmarks for
these, so you may be able to compare the model's training speed against these
benchmarks.

### Understanding the Connection: Business Goals and Corresponding Machine Learning Tasks

| Case Study                                      | Business Objective                                             | Machine Learning Objective                                                              |
| ----------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Spam Email Filter                               | Reduce the number of spam emails users receive                 | Classify emails as spam or not spam                                                     |
| Recommendation System for an E-commerce Site    | Increase user engagement and sales                             | Recommend items that users are likely to purchase                                       |
| Music Streaming App                             | Improve user experience by providing tailored content          | Predict the next song a user might want to listen to based on their listening history   |
| Credit Card Fraud Detection                     | Minimize the amount of fraudulent transactions                 | Predict fraudulent transactions based on transaction history                            |
| Autonomous Vehicle                              | Ensure the safety and efficiency of the autonomous system      | Predict and react appropriately to the surrounding environment                          |
| Customer Churn Prediction for a Telecom company | Retain more customers and minimize customer churn              | Predict which customers are likely to churn based on their usage patterns and behaviors |
| Sentiment Analysis for Social Media Platform    | Improve understanding of user sentiment towards certain topics | Classify user comments/posts as positive, negative, or neutral                          |
| Image Recognition for Medical Diagnosis         | Improve the accuracy and speed of disease diagnosis            | Identify and classify diseases from medical images                                      |

The table is intentionally vague and I omit the specific metrics for each
machine learning objective. The idea is to show the connection between the
business objectives and the corresponding machine learning tasks - what metrics
to use will depend on the specific context and the nature of the problem.

## Identifying the Type of Machine Learning Task

We need to identify the type of machine learning task that best suits the
problem at hand. This involves categorizing the problem into a specific
paradigm, such as classification, regression, clustering, recommendation, NLP or
computer vision etc.

```{admonition} Multi-Stage and Multi-Modal Paradigms
:class: warning

In any complex enough problem, often you cannot just categorize the problem into
one type of machine learning task. For instance, in image captioning, you might
have to use a combination of computer vision and natural language processing
techniques.
```

### Machine Learning Tasks

```{figure} ./assets/mermaid-diagram-2024-05-01-173743.svg
---
name: ml-life-cycle-02-ml-tasks
---

Mermaid Diagram Of Machine Learning Tasks.
```

````{admonition} Mermaid Diagram Of Machine Learning Tasks
:class: dropdown

```text
graph TD
  ML(Machine Learning)
  ML --> Supervised
  ML --> Unsupervised
  ML --> Reinforcement
  ML --> Semi-supervised
  ML --> Self-supervised
  Supervised --> Classification
  Supervised --> Regression
  Unsupervised --> Clustering
  Unsupervised --> Dimensionality_Reduction[Dimensionality Reduction]
  Reinforcement --> Q_Learning[Q-Learning]
  Reinforcement --> Policy_Gradients[Policy Gradients]
  Semi-supervised --> Label_Propagation[Label Propagation]
  Semi-supervised --> Multi_Instance_Learning[Multi-Instance Learning]
  Self-supervised --> Autoencoders
  Self-supervised --> GANs[Generative Adversarial Networks]
```
````

In this tree diagram, we start with the main categories of machine learning
tasks, including Supervised Learning, Unsupervised Learning, Reinforcement
Learning, Semi-Supervised Learning, and Self-Supervised Learning. We then
further break these categories down into specific methods and techniques.

However, the scope of machine learning is even wider. For instance, the
classification task under supervised learning can be further differentiated into
Binary Classification, Multi-Class Classification, and Multi-Label
Classification.

#### Examples of Machine Learning Tasks

| Machine Learning Task                                                                  | Sub-Task                                                                                    | Example                                                                                                                     |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **[Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)**           | **[Classification](https://en.wikipedia.org/wiki/Statistical_classification)** (Binary)     | [Spam Email Detection](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)                                            |
| **[Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)**           | **[Classification](https://en.wikipedia.org/wiki/Statistical_classification)** (Multiclass) | [Handwritten Digit Recognition](https://en.wikipedia.org/wiki/MNIST_database) (MNIST Dataset)                               |
| **[Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)**           | **[Classification](https://en.wikipedia.org/wiki/Statistical_classification)** (Multilabel) | [Tagging Customer Queries](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff) |
| **[Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)**           | **[Regression](https://en.wikipedia.org/wiki/Regression_analysis)**                         | [Housing Price Prediction](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)                            |
| **[Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)**           | **[Recommendation Systems](https://en.wikipedia.org/wiki/Recommender_system)**              | [Movie Recommendations](https://en.wikipedia.org/wiki/Netflix_Prize) (Netflix)                                              |
| **[Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning)**       | **[Clustering](https://en.wikipedia.org/wiki/Cluster_analysis)**                            | [Market Segmentation](https://en.wikipedia.org/wiki/Market_segmentation)                                                    |
| **[Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning)**       | **[Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)**      | [Visualizing High-Dimensional Data](https://lvdmaaten.github.io/tsne/) (t-SNE)                                              |
| **[Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)**     | N/A                                                                                         | [Game Playing](https://deepmind.com/research/case-studies/alphago-the-story-so-far) (AlphaGo)                               |
| **[Self-Supervised Learning](https://en.wikipedia.org/wiki/Self-supervised_learning)** | N/A                                                                                         | [Pretraining Transformers](https://arxiv.org/abs/1810.04805) (BERT)                                                         |

### Deep Learning Tasks

```{figure} ./assets/mermaid-diagram-2024-05-01-174201.svg
---
name: ml-life-cycle-02-deep-learning-tasks
---

Mermaid Diagram Of Deep Learning Tasks.
```

There are many many more that I missed, things like Graph Neural Networks,
Speech and Audio Processing, and many more. Take a look at
[HuggingFace's Tasks page](https://huggingface.co/tasks) for a comprehensive
list of tasks.

````{admonition} Mermaid Diagram Of Deep Learning Tasks
:class: dropdown

```text
graph TD
  DL(Deep Learning)
  DL --> CV[Computer Vision]
  DL --> NLP[Natural Language Processing]
  DL --> ST[Speech and Audio Processing]
  DL --> GNN[Graph Neural Networks]
  CV --> Image_Classification[Image Classification]
  CV --> Object_Detection[Object Detection]
  CV --> Segmentation[Image Segmentation]
  CV --> SR[Super Resolution]
  NLP --> Sentiment_Analysis[Sentiment Analysis]
  NLP --> Machine_Translation[Machine Translation]
  NLP --> Text_Summarization[Text Summarization]
  NLP --> Question_Answering[Question Answering]
```
````

#### Examples of Deep Learning Tasks

| Deep Learning Task          | Sub-Task                     | Example                                            |
| --------------------------- | ---------------------------- | -------------------------------------------------- |
| Computer Vision             | Image Classification         | Categorizing images of cats and dogs               |
| Computer Vision             | Object Detection             | Detecting pedestrians in autonomous driving        |
| Computer Vision             | Image Segmentation           | Medical imaging - tumor detection                  |
| Natural Language Processing | Sentiment Analysis           | Analyzing social media sentiment towards a product |
| Natural Language Processing | Machine Translation          | Translating text from English to French            |
| Speech and Audio Processing | Automatic Speech Recognition | Converting spoken language into written form       |

### Combination of Machine Learning and Deep Learning Tasks

The integration of traditional machine learning methods with deep learning
techniques are very common in practice. We list very simple and naive examples.

1. **[Feature Engineering for Deep Models](https://en.wikipedia.org/wiki/Feature_engineering)**:
   Traditional machine learning techniques can be utilized to create significant
   features for deep learning models. For example,
   **[TF-IDF](https://en.wikipedia.org/wiki/Tf–idf)** vectors from text data can
   be integrated as input for a neural network.

2. **[Deep Learning for Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)**:
   Deep learning approaches like
   **[autoencoders](https://en.wikipedia.org/wiki/Autoencoder)** can
   significantly reduce dimensions, and these reduced features can enhance
   traditional machine learning models.

3. **Postprocessing Deep Learning Outputs**: The outputs from deep learning,
   such as **[embeddings](https://en.wikipedia.org/wiki/Embedding)** from a
   neural network, can be refined through traditional machine learning models
   like **[SVM](https://en.wikipedia.org/wiki/Support_vector_machine)** or
   **[decision trees](https://en.wikipedia.org/wiki/Decision_tree)**.

### Multimodal Deep Learning (Yes ChatGPT Is Multimodal)

**[Multimodal deep learning](https://en.wikipedia.org/wiki/Multimodal_learning)**
fuses data from various formats like text, images, and audio, leading to richer
data representations. By integrating information from different modalities,
multimodal models can learn representations that capture a richer understanding
of the data - and I believe a step closer to AGI (?)

Just have a look at ChatGPT, it uses multimodal learning. Why? Users can not
only ask questions in text and receive text responses but also upload images and
receive responses that incorporate information from both the text and the
image - a classic example of multimodal learning.

The main challenge in multimodal learning is the fusion of the different
modalities. Techniques such as early fusion (concatenating all modalities at the
input level), late fusion (combining the outputs of separate models for each
modality), and hybrid fusion (a mix of early and late fusion) are commonly used.

#### Early Fusion

Early fusion is the simplest form of fusion, where all the different types of
data are concatenated at the input level before going through the model. The
model is then trained on this combined data.

A classic example of early fusion is multimodal sentiment analysis where text
and audio data are combined to better understand user sentiment. Textual data
might be user reviews or comments, while audio data might include user
recordings. The text and audio data are processed separately initially (i.e.,
transformed into numerical form), and then concatenated to form a single input
vector to the model.

Why is it naive? For one, the homogenization of features from vastly different
modalities might lead to loss of information. Text is text data and tokenized,
for audio data, it might be in the form of signals with temporal information.
However, if you are able to extract features that are compatible across the
different modalities, early fusion might be good - for example, converting audio
data into spectrograms and then concatenating them with text data, or better,
using speech-to-text to convert audio data into text data.

#### Late Fusion

Late fusion trains individual models on each data modality, later combining
their outputs for the final decision. For example medical diagnosis of cancer
might have image data passing through a typical Convolutional Neural Network
(CNN) or Vision Transformer (ViT) and text data passing through a Encoder based
model, numerical data like lab results etc could pass through a traditional
machine learning model like Random Forest or Gradient Boosting.

Then the concatenation happens maybe at the last layer, where the previous
results across the three modalities are pooled (averaged, concatenated, etc) to
make the final decision. This is better than concatenating at the input level as
the models can learn the best representations (i.e. embeddings for each
modality) for each modality and then combine them at the end.

#### Hybrid Fusion

Hybrid fusion combines the strengths of both early and late fusion. It involves
fusing data both at the input level and at the model output level.

A healthcare monitoring system could use hybrid fusion for patient monitoring.
Here, data like heart rate, blood pressure, and temperature (numerical data)
could be combined with data from a wearable device (like step count and sleep
hours) and textual data from patient self-reports. Early fusion could be used to
combine the numerical data and processed wearable device data, and separate
models could be trained on this combined data and the textual data. Then, late
fusion would combine the outputs of these models for final analysis and
predictions. The rationale here is that certain features might be more
interrelated (justifying early fusion), while others might have unique patterns
that are best captured separately (justifying late fusion).

#### Choosing the Right Fusion Strategy

How does one choose the right fusion strategy? Honestly, this is not a trivial
matter and often requires experimentation. However, there are some factors to
consider from the book
["Machine Learning System Design Interview"](https://bytebytego.com/intro/machine-learning-system-design-interview)
by Ali Aminian.

Firstly, the **_nature of the data sources_** is important. If the data sources
are homogeneous (e.g. of the same type), simpler fusion techniques like early
fusion might suffice and simple average pooling might be enough. However, if the
sources are heterogeneous (e.g. images and text), more complex fusion methods
might be needed.

Furthermore, one might build a super good fusion model but if it is not able to
serve due to it being computationally intensive, then it is not a good model to
say, serve in real time. Of course, modern engineering methods like
quantization, pruning, and distillation might help but it is something to
consider.

In some cases, the fusion method must be easily interpretable (e.g. in medical
applications). This might favor simpler, more transparent methods.

## Minimum Viable Product (MVP)

Lastly, at this stage, you likely have an idea of the business problem you want
to solve. Consequently, it is not uncommon to start thinking about a Minimum
Viable Product (MVP) that can be used to solve the problem. I won't go into
details on the difference between a demo, a prototype, and an MVP, but the idea
is to have a basic version of the product that can be shown as a proof of
concept.

## References and Further Readings

-   Huyen, Chip. "Chapter 2. Introduction to Machine Learning Systems Design."
    In Designing Machine Learning Systems: An Iterative Process for
    Production-Ready Applications, O'Reilly Media, Inc., 2022.
-   Huyen, Chip. "Chapter 4. Training Data." In Designing Machine Learning
    Systems: An Iterative Process for Production-Ready Applications, O'Reilly
    Media, Inc., 2022.
-   [Machine Learning System Design Interview - Ali Aminian](https://bytebytego.com/intro/machine-learning-system-design-interview)
-   [HuggingFace Tasks](https://huggingface.co/tasks)
