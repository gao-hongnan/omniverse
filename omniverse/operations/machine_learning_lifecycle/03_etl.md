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

# Stage 3. Extract, Transform, Load (ETL)

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

## The Evolution of Data Engineering (Don't Quote Me On This!)

With the basic understanding of **Data Engineering** and its essential role in
**Machine Learning**, it's important to recognize the evolution of data handling
practices. Traditional **ETL (Extract, Transform, Load)** methodologies have
long been the backbone of data pipeline design. They set the stage for
collecting, processing, and storing data in a structured manner.

However, the modern era of data-driven applications demands a more agile and
responsive approach. This is where **DataOps**, encompassing principles of
**Continuous Integration/Continuous Deployment (CI/CD)**, comes into play. The
process builds on the ETL framework but now with automation, collaboration,
monitoring, and quality assurance.

In traditional ETL or ELT processes, the main focus is on extracting data from
various sources, transforming it into the required format, and then loading it
into a target system. These processes are typically batch-oriented and can be
run on schedules or triggered manually.

In a CI/CD DataOps pipeline, the focus expands to the entire data lifecycle and
emphasizes automation, continuous integration, and continuous deployment. This
means that the process not only includes the basic ETL or ELT steps but also
involves:

-   **Continuous Integration**: Automating the process of integrating code
    changes from multiple contributors into a shared repository, often followed
    by automated building and testing.
-   **Continuous Deployment**: Automating the process of deploying the
    integrated and tested code to production environments, ensuring that the
    data pipeline remains stable and updated.
-   **Monitoring and Alerting**: Keeping track of the performance and health of
    the data pipeline, triggering alerts if anomalies or issues are detected.
-   **Testing and Quality Assurance**: Embedding rigorous testing within the
    pipeline to ensure data quality, integrity, and compliance with business
    rules.

## The ETL/ELT Framework

### ETL (Extract, Transform, Load)

**ETL** is a process in data handling that involves three main stages:

1. **Extract**: Gathering data from various sources.
2. **Transform**: Processing this data to fit the desired format, usually
   outside the target system. This might include cleaning, aggregating,
   filtering, etc.
3. **Load**: Finally, loading the transformed data into the destination data
   warehouse or database.

### ELT (Extract, Load, Transform)

**ELT** is a variant of ETL, but with a different order of operations:

1. **Extract**: Gathering data from various sources.
2. **Load**: Loading the raw data into the destination system.
3. **Transform**: Performing transformations within the target system itself,
   utilizing the processing capabilities of modern data warehouses.

### ELTL (Extract, Load, Transform, Load)

This combination could represent a two-step process:

1. **Extract**: Gathering data from various sources.
2. **Load**: Loading the raw data into a staging area or temporary storage.
3. **Transform**: Performing transformations within this temporary storage.
4. **Load**: Loading the transformed data into the final destination, such as a
   data warehouse or database.

This approach might be beneficial when working with massive datasets, allowing
for an initial raw data consolidation, followed by transformation and final
loading into the target system.

### Intuition on When to Use ETL vs ELT

In certain scenarios, companies opt for the ELT (Extract, Load, Transform)
process, particularly when dealing with complex and unstructured data. During
the **extraction** phase, data is collected from various sources and then
immediately **loaded** or dumped into a **data lake**, which is a storage
repository that holds a vast amount of raw data in its native format.

This approach has the advantage of quickly making the data available, preserving
its raw state for future use. However, this raw, unstructured data can become
unwieldy, particularly when dealing with large volumes.

When it's time to analyze or utilize the data, it must be **extracted** again
from the data lake. This is followed by the **transformation** phase, where the
data is processed and converted into a structured format suitable for analysis.

While the ELT paradigm allows for greater flexibility and the ability to
accommodate diverse data types, it can lead to inefficiencies when searching
through large and unstructured data sets within the data lake. The process of
extracting and transforming data from the data lake can be time-consuming and
resource-intensive, particularly if the data needs to be combed through
extensively.

In essence, the ELT approach with a data lake can be both a boon and a
challenge. It enables faster data ingestion and provides a flexible repository
for raw data, but the subsequent handling and processing of that data might
require significant effort, especially when dealing with large quantities of
unstructured information.

### ETL versus ELT

Here's a table that breaks down the comparison, advantages, and disadvantages of
both ETL and ELT.

| Criteria                | ETL                                                                                      | ELT                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Basic Process**       | Extract, Transform, Load                                                                 | Extract, Load, Transform                                                                         |
| **Data Latency**        | May introduce delays, affecting real-time analysis; near-real-time possible with tooling | Often reduces delays between data collection and availability                                    |
| **Scalability**         | Can be scalable with proper architecture and parallel processing                         | Typically leverages modern data warehouses for scalability                                       |
| **Flexibility**         | Less adaptable to changing requirements; can be mitigated with design                    | More adaptable to changes in data structure or requirements                                      |
| **Pipeline Complexity** | May involve complex transformations, increasing development and maintenance efforts      | Might simplify some aspects of the pipeline, depending on tools and requirements                 |
| **Accessibility**       | May require specialized skills, limiting accessibility                                   | Might allow more team participation, especially with common languages like SQL                   |
| **Advantages**          | Suitable for complex transformations with structured data; control over transformation   | Flexibility, scalability, and potentially reduced latency; useful for unstructured data handling |
| **Disadvantages**       | Potential bottlenecks; complexity; potential rigidity                                    | Might lead to inefficiencies in processing unstructured data; simplicity is context-dependent    |
| **Use Case**            | When precise control over transformations and structured data processing is needed       | When handling diverse or unstructured data, or when flexibility and scalability are priorities   |

### Sample ELTL Pipeline

#### Extract

In the extraction phase, data is pulled from various sources which could be
structured, semi-structured or unstructured, and could be located in databases,
data lakes, data warehouses, or external APIs. The key is to capture the
necessary data without losing or modifying any of the original data during the
process.

#### Data Analysis

Post extraction, data analysis provides insights into the nature and quality of
the data. This stage involves examining the distribution of the data,
identifying potential anomalies or outliers, and assessing the overall data
quality. Techniques such as descriptive statistics and data visualization are
commonly used.

It is important to note that this process is generally applicable and not
specific to machine learning. The outcome of this stage guides the
decision-making process for subsequent data cleaning and transformation tasks,
thereby setting up a solid foundation for downstream tasks such as reporting,
analytics, or model training.

#### Validate Raw

Validation of raw data ensures that the extracted data meets the requirements
and constraints for the subsequent stages. It involves checking for data
completeness, consistency, and accuracy. This stage might include checking if
all expected data has been extracted, if there are any unexpected null or
missing values, and if the data aligns with known constraints (like a field that
should always be positive).

#### Load

The load stage involves transferring the extracted data into a target system for
storage. The target system can be a data warehouse, a data lake, or a specific
database depending on the use case. The focus during this phase is on efficiency
and reliability, ensuring that all data is accurately loaded without disrupting
existing data or processes.

#### Transform

The transformation phase involves changing the raw data into a format that is
suitable for downstream tasks. This may include cleaning operations (like
handling missing values or outliers), integrating data from different sources,
aggregating or summarizing data, and converting data types. Additionally,
feature engineering for machine learning tasks often takes place in this stage.

#### Validate Transformed

After transformation, the data needs to be validated again to ensure it meets
the specific requirements for downstream tasks. This might involve checking the
data against predefined rules or statistical properties (like a specific
distribution), checking for unexpected null or missing values after
transformation, or comparing a sample of the transformed data against the
expected output.

#### Load Transformed

After the transformed data has been validated, it can be loaded into the target
system for storage. This might be a data warehouse, a data lake, or a specific
database depending on the use case.

### Summary

The choice between ETL and ELT depends on various factors like data volume,
real-time requirements, team skills, technology stack, and the nature of the
transformations. ETL might be more suitable for complex transformations with
structured data, while ELT might be preferred for more flexible, scalable
handling of diverse or unstructured data.

It's crucial to acknowledge that both ETL and ELT can be implemented effectively
or poorly, depending on the specific context, tools, and design principles
applied. Neither approach is universally superior, and the decision should be
based on a comprehensive understanding of the project's unique requirements and
constraints.

## Batch Processing vs. Stream Processing (TODO as not familiar with stream processing)

For real-time or near-real-time ML applications, traditional batch processing of
ETL might not be suitable. Instead, stream processing frameworks like Apache
Kafka or Apache Flink allow for continuous data processing and may be used as
alternatives or complements to ETL.
