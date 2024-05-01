# Stage 4. Data Extraction (MLOps)

Data extraction in MLOps usually involves pulling data from a Data Warehouse or
Data Lake that has already been transformed. The transformed data is generally
agnostic to downstream tasks, such as the specific machine learning models being
used.

Remember, the data at this stage has been cleaned and transformed — but this
doesn't imply it's been preprocessed or engineered for machine learning tasks.
This cleaned data serves as a versatile resource, used not just by machine
learning teams, but also by business intelligence, data analytics, and other
teams that require transformed data.

## Batch Ingestion

This type of data ingestion is used when there are large volumes of data that do
not require immediate analysis or use. The data is collected over a period of
time and then ingested into the data lake all at once. This can be scheduled to
occur at regular intervals, like daily, weekly, or monthly, depending on the
requirements. This method is particularly beneficial when dealing with large
amounts of structured and unstructured data and where latency is not a critical
factor.

### Batch Ingestion Example

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

Batch ingestion in this scenario allows efficient and cost-effective processing
of a large amount of data without demanding real-time analysis.

## Real-time Ingestion (Stream Ingestion)

Real-time data ingestion involves the continuous input of data into the data
lake as it is generated. It's used when the data needs to be analyzed and acted
upon immediately or near real-time, such as in event-driven applications or live
dashboards. This method is commonly used in scenarios like tracking user
activity on a website, real-time analytics, monitoring, fraud detection, etc.
Real-time ingestion can be more complex and resource-intensive than batch
processing due to the need for continuous processing and potential volume and
velocity of data.

### Real-time Ingestion Example

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
real-time data ingestion plays a crucial role in enabling TikTok's fast-paced,
dynamic environment.

## Key Considerations for Data Ingestion Strategies

For both types, you would need to consider various factors like:

-   **Data quality**: Ensuring the data ingested is of high quality is vital.
    You might need to set up processes to clean and standardize data, handle
    missing or inconsistent data, etc.

-   **Data transformation**: Depending on your specific use case, you might need
    to transform the data into a suitable format for your data lake. This could
    involve processes like data normalization, conversion, aggregation, etc.

-   **Metadata**: Metadata provides information about the data you're storing in
    your data lake. This includes information about the data's source, when it
    was ingested, what kind of transformations have been applied to it, etc.

-   **Security and compliance**: Ensuring the data is securely ingested and
    stored is vital, especially for sensitive information. You would need to
    consider things like data encryption, access controls, audit logs, etc.
    Also, you would need to ensure you're complying with any relevant
    regulations, like GDPR or HIPAA.

Remember that the specific methods and tools you use for data ingestion would
depend on your specific use case, the type of data you're working with, the
specific requirements of your data lake, and the infrastructure you're working
with.

## Data Ingestion Tools

Here look at my modern tech stack documentation, things like Airbyte, dbt, and
airflow etc.


## References and Further Readings

-   [Data Engineering Fundamentals](../../designing_machine_learning_systems/03_data_engineering_fundamentals.md)
-   [Madewithml: Data Engineering](https://madewithml.com/courses/mlops/data-stack/)
