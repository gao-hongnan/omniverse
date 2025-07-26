---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Stage 3.1. Data Source and Formats

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
```

## Identify and Scope the Data Source

In this stage, we identify and scope the data source. This involves determining
the type of data, locating the data, assessing accessibility and compliance,
gauging the data volume, and understanding data characteristics.

### Intuition (What comes before Data Extraction?)

As we have seen in the pipeline and in a later section, the ELT/ETL framework,
the first step is data extraction. However, before we can extract data, we need
to first identify the data source and scope it.

In what follows, we will discuss the steps involved in identifying and scoping
the data source, as well as the tools and methods for extracting data from the
source.

### Steps to Identify and Scope the Data Source

```{list-table} Table of Steps to Identify and Scope the Data Source
:header-rows: 1
:name: ml-lifecycle-03-steps-identify-scope-data-source

-   -   Step
    -   Action
    -   Rationale
-   -   Define the Type of Data
    -   Determine whether the data is numerical, categorical, time-series,
        text-based, images, or audio.
    -   This can affect the model design and the choice of data sources.
-   -   Locate the Data
    -   Identify the location, such as databases (SQL or NoSQL), APIs, log
        files, Excel or CSV files, etc.
    -   Enables the selection of the suitable tools and methods for extraction.
-   -   Assess Accessibility and Compliance
    -   Understand permissions, authentication, privacy concerns, and
        restrictions on data extraction.
    -   Ensures adherence to legal and organizational policies.
-   -   Gauge the Data Volume
    -   Determine the size of the dataset.
    -   Influences the choice of tools for extraction and storage. This is
        important because large dataset need to be stored in a way that is
        efficient and scalable.
-   -   Understand Data Characteristics
    -   Recognize and address special characteristics, for example, if you are
        collecting images of apples for classification, you need to be sure
        that the images have say, in RGB format, and not in grayscale.
    -   Facilitates proper processing, validation, and utilization of the data.
```

### Data Types in Machine Learning Systems

Before we scope the data source, a logical question to first ask is, what
_types_ of data are we dealing with? Knowing the data types will help us
**understand the nature and structure of information that we need to obtain.**
This understanding, in turn, informs our choice of **data sources** that are
best suited to provide this specific type of data.

For example, if we are working with time-series data, our data sources might be
sensors, logs, or financial market feeds. If we are dealing with textual data,
the sources might be documents, websites, or social media platforms.

Here's a brief overview of the different types of data in the form of a table.

| Main Type                | Subtype          | Specific Types       | Description                                                                    |
| ------------------------ | ---------------- | -------------------- | ------------------------------------------------------------------------------ |
| **Structured Data**      | Numerical        | Continuous, Discrete | Continuous data can take any value, while Discrete data takes specific values. |
|                          | Categorical      | Nominal, Ordinal     | Nominal data has no inherent order; Ordinal data has a meaningful order.       |
|                          | Time-Series Data |                      | Data collected at specific time intervals.                                     |
|                          | Geospatial Data  |                      | Information that includes geographical attributes.                             |
|                          | Boolean Data     |                      | True/false or yes/no values.                                                   |
| **Semi-Structured Data** | Multimodal       |                      | Combines data from multiple sources or types.                                  |
|                          | Graph Data       |                      | Represents relationships using nodes and edges.                                |
|                          | Mixed Data Types |                      | A combination of various data types.                                           |
| **Unstructured Data**    | Text-Based Data  |                      | Unstructured textual information.                                              |
|                          | Image Data       |                      | Visual information in a grid of pixels.                                        |
|                          | Audio Data       |                      | Sound or speech data.                                                          |
|                          | Binary Data      |                      | Data represented in a binary format.                                           |
|                          | Embeddings       |                      | Representations of categorical, text, or complex data as continuous vectors    |

### Data Sources in Machine Learning Systems

Having identified the _types_ of data that our machine learning system will
handle, we now turn our attention to the various **sources** from which this
data can be obtained. Different data types require specific sources, both in
terms of format compatibility and functional alignment. Here's an overview of
various data sources, categorized by their characteristics and aligned with the
types of data they typically provide:

| **Category**                     | **Type**                    | **Examples/Details**                                                                                                                     |
| -------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Databases**                    | Relational Databases (SQL)  | [MySQL](https://www.mysql.com/), [PostgreSQL](https://www.postgresql.org/), [MS SQL Server](https://www.microsoft.com/en-us/sql-server/) |
|                                  | NoSQL Databases             | [MongoDB](https://www.mongodb.com/), [Cassandra](https://cassandra.apache.org/), [Redis](https://redis.io/)                              |
| **File-Based Sources**           | Flat Files                  | CSV, Excel, TSV                                                                                                                          |
|                                  | Binary Files                | Parquet, Avro                                                                                                                            |
|                                  | Image and Video Files       | JPEG, PNG, MP4                                                                                                                           |
|                                  | Text Files                  | TXT, PDF, DOC                                                                                                                            |
| **Web Sources**                  | Web APIs                    | RESTful APIs, SOAP, [GraphQL](https://graphql.org/)                                                                                      |
|                                  | Web Scraping                | HTML, XML                                                                                                                                |
|                                  | Social Media                | Twitter, Facebook, Reddit                                                                                                                |
| **Streaming Data Sources**       | Message Brokers             | [Kafka](https://kafka.apache.org/), [RabbitMQ](https://www.rabbitmq.com/)                                                                |
|                                  | Real-Time Feeds             | Stock prices, sensor data                                                                                                                |
| **Sensor Data**                  | IoT Devices                 | Smart devices, wearable tech                                                                                                             |
|                                  | Industrial Sensors          | Temperature, pressure, humidity sensors                                                                                                  |
| **Scientific Sources**           | Genomic Data                | DNA sequences, proteomics                                                                                                                |
|                                  | Meteorological Data         | Weather stations, satellites                                                                                                             |
| **Financial Data Sources**       | Stock Market Data           | Exchanges, trading platforms                                                                                                             |
|                                  | Banking Transactions        | Credit card swipes, ATM transactions                                                                                                     |
| **Healthcare Data Sources**      | Electronic Health Records   | Patient medical records                                                                                                                  |
|                                  | Medical Imaging             | MRI, CT scans, X-rays                                                                                                                    |
| **Government and Public Data**   | Census Data                 | Demographics, economics                                                                                                                  |
|                                  | Legislation and Regulations | Law documents, policy papers                                                                                                             |
| **Educational Data Sources**     | Academic Databases          | Research papers, thesis documents                                                                                                        |
|                                  | Learning Management Systems | Student grades, course content                                                                                                           |
| **Human-Generated Data Sources** | Surveys and Questionnaires  | Market research, feedback forms                                                                                                          |
|                                  | Crowdsourcing Platforms     | [Amazon Mechanical Turk](https://www.mturk.com/)                                                                                         |
| **Third-Party Data Providers**   | Commercial Data Providers   | Market trends, consumer habits                                                                                                           |
|                                  | Open Data Repositories      | [Kaggle](https://www.kaggle.com/), [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)                           |

## Data Formats in Machine Learning Systems

In other words, once you scope the data source and data types, and manage to
extract them, you need to store it in a format that is easy to work with. By
easy I mean that the data should be easily accessible, scalable, and efficient
to work with. As such, storing data isn't straightforward because data can be of
different types and one must be experienced or knowledgeable enough to know what
storage to use when storing data of a particular type.

Some questions to ask when choosing a data format:

-   Where do you store the data? In a database? In a file system? In a key-value
    store? We want it to be ideally cheap and fast to retrieve the data.
-   How to store complex models so they can be loaded and run on different
    devices (e.g. mobile phones, web browsers, etc.). In ML, it can be GPU, CPU,
    etc.

### Distributed Data Parallelism And Data Sharding

Sometimes data is too large to fit into a single machine's memory. In such
cases, we can use distributed data parallelism (DDP) and data sharding to
distribute the data across multiple machines. In DDP the model and trainer are
replicated across multiple instances/nodes/ranks and each instance processes a
different subset of the data. In this case, data sharding is often necessary in
DDP to ensure that each process (or GPU) gets a unique subset of the data during
each training iteration. Each process needs to handle a different portion of the
data to ensure diversity in learning across replicas and prevent data
redundancy, which can skew the learning process.

In practice, data loaders that are DDP-aware (like those in PyTorch)
automatically shard the dataset across the available GPUs. For instance, if you
have a dataset of 1000 samples and 10 GPUs, each GPU might be assigned 100
unique samples per iteration. This distribution ensures that all the data gets
utilized without overlap between GPUs (their sampler ensures that each GPU gets
a unique subset of the data).

### An Example on Multimodal Data Storage For E-Commerce

In e-commerce platforms, product pages often contain rich multimedia
information, including images and corresponding textual descriptions. Storing
and retrieving this information efficiently can involve the following.

1. **Storing Images in a Binary Format**: Rather than embedding the raw image
   tensor within a data structure, it's often more efficient to store the image
   in a binary format (e.g., JPEG, PNG) and keep a reference to its location
   (e.g., file path or URL).

2. **Utilizing a Database for Textual Information**: The textual information,
   including descriptions and metadata, can be stored in a relational database.
   This approach provides scalable storage and efficient query capabilities.

3. **Creating a Unified Schema**: A unified schema or data model could
   encapsulate both the image references and the corresponding textual data.
   This schema acts as a bridge between the two data types, allowing them to be
   treated as a cohesive unit.

Consider the below code snippet:

```python title="Sample Data Schema Encoding Image and Text"
sample_data_schema = {
    "product_id": 123,
    "image_url": "https://path/to/image.jpg",
    "description": "This is a picture of a cat.",
    "additional_metadata": { ... }  # Additional textual or numerical information.
}
```

and in tabular form:

| Field Name            | Data Type       | Description                                                                               |
| --------------------- | --------------- | ----------------------------------------------------------------------------------------- |
| `product_id`          | Integer         | A unique identifier for the product.                                                      |
| `image_url`           | String (URL)    | The URL or file path to the product's image.                                              |
| `description`         | String (Text)   | The textual description of the product.                                                   |
| `additional_metadata` | Dictionary/JSON | Additional textual or numerical information, such as categories, tags, or specifications. |

In this approach, the `"image_url"` field stores a reference to the location of
the image, and the `"description"` field contains the textual description. The
additional metadata can encapsulate other relevant information, such as
categories, tags, or product specifications.

This design offers several advantages:

-   **Scalability**: By storing images in a binary format and using database
    storage for text, this approach can scale to handle large product catalogs.
-   **Efficiency**: Leveraging specialized storage mechanisms for different data
    types ensures that retrieval and updates are efficient.

### Data Formats

We will describe a few choices of data formats below.

#### Data Serialization vs Data Deserialization

The process of transforming data structures or object states into a format that
can be saved (e.g., in a file like JSON) and later rebuilt in the same or a
different computing environment is known as serialization. The opposite process,
called deserialization, involves retrieving data from the stored formats. In
simpler terms, serialization refers to storing data, while deserialization
refers to accessing data from the saved formats.

In other words, **storing data** is called **serialization**, and **retrieving
data from the stored formats** is called **deserialization**.

#### JSON

[**JSON**](https://www.json.org/json-en.html), which stands for JavaScript
Object Notation, is a lightweight data-interchange format that uses a key-value
pair paradigm. It is human-readable, easy to parse, and simple to generate,
making it an ideal choice for data exchange between a server and a client in
machine learning applications. JSON's structure allows for easy storage in
databases and can represent a wide variety of data types, including strings,
numbers, booleans, objects, and arrays.

```json title="example.json"
{
    "name": "John",
    "age": 30,
    "cars": [
        { "name": "Ford", "models": ["Fiesta", "Focus", "Mustang"] },
        { "name": "BMW", "models": ["320", "X3", "X5"] },
        { "name": "Fiat", "models": ["500", "Panda"] }
    ]
}
```

While JSON has many advantages, it does have some drawbacks, such as increased
storage requirements due to its text-based nature. However, its simplicity and
ease of use have made it one of the most popular data formats in machine
learning and other applications.

In addition to the key-value pair structure, JSON also supports nesting of
objects and arrays, which allows for more complex data representation. This
makes JSON a versatile choice for a variety of use cases, from simple
configuration files to complex machine learning model inputs and outputs.

Furthermore, JSON has extensive support in many programming languages, with
built-in libraries or third-party packages available for parsing and generating
JSON data.

In summary, JSON's human-readable format, easy parsing, support for complex data
structures, and widespread language support make it an excellent choice for data
exchange and storage in machine learning applications, despite its increased
storage requirements compared to binary formats.

#### Row and Columnar Formats

##### Concept of Row-major vs Column-major order

Row-major and column-major order describe two ways to store multi-dimensional
arrays in linear memory. In row-major order, the elements of a multi-dimensional
array are stored row by row, whereas in column-major order, the elements are
stored column by column.

##### Examples of Row-major vs Column-major order

In row-major order, the elements of each row of a matrix are stored together in
contiguous memory locations, with the elements of successive rows appearing
consecutively in memory. For example, consider a 3x2 matrix:

$$
\boldsymbol{A} = \begin{bmatrix}
    1 & 2 \\
    3 & 4 \\
    5 & 6
\end{bmatrix}
$$

In row-major order, the elements are stored in memory as:

```python
[1, 2, 3, 4, 5, 6]
```

In contrast, in column-major order, the elements of each column are stored
together in contiguous memory locations, with the elements of successive columns
appearing consecutively in memory. For the same matrix, the column-major order
would be:

```python
[1, 3, 5, 2, 4, 6]
```

Row-major and column-major order can make a difference in performance when
accessing multi-dimensional arrays, especially for large arrays. For example,
when accessing elements of a row in row-major order, consecutive elements of the
row are likely to be cached together, which can improve access time. Similarly,
when accessing elements of a column in column-major order, consecutive elements
in the column are likely to be cached together, which can improve performance.

#### Pros and cons of Row-major vs Column-major order

##### Row-major order

**Pros:**

-   It is the default order used in many programming languages, including C and
    C++.
-   It can be more intuitive for humans to understand, as rows are typically
    used to represent entities (e.g., students, observations) and columns are
    used to represent attributes (e.g., grades, measurements).
-   When iterating over the elements of a matrix row-by-row, row-major order
    ensures that the elements accessed are contiguous in memory, which can
    improve cache locality and reduce the number of cache misses.
-   Many linear algebra libraries, such as BLAS and LAPACK, use row-major order
    by default.

**Cons:**

-   When iterating over the elements of a matrix column-by-column, row-major
    order can lead to poor cache locality and a higher number of cache misses.
    This is because consecutive elements in the same column are not necessarily
    contiguous in memory.
-   When transposing a matrix, row-major order requires copying the entire
    matrix into a new block of memory in column-major order, which can be costly
    for large matrices.
-   Some hardware architectures may be optimized for column-major order, leading
    to lower performance for row-major order.

##### Column-major order

**Pros:**

-   Column-major order is used by default in some programming languages, such as
    Fortran.
-   When iterating over the elements of a matrix column-by-column, column-major
    order ensures that the elements accessed are contiguous in memory, which can
    improve cache locality and reduce the number of cache misses.
-   Some hardware architectures, such as GPUs, are optimized for column-major
    order, leading to potentially better performance.

**Cons:**

-   Column-major order can be less intuitive for humans to understand, as it is
    not the standard representation used in many fields.
-   When iterating over the elements of a matrix row-by-row, column-major order
    can lead to poor cache locality and a higher number of cache misses. This is
    because consecutive elements in the same row are not necessarily contiguous
    in memory.
-   Many linear algebra libraries, such as BLAS and LAPACK, use row-major order
    by default, so using column-major order may require additional memory copies
    or transpositions.

Overall, the choice between row-major and column-major order depends on the
specific use case and hardware architecture.

#### Modern Row and Columnar Formats

| Library | Order for Multidimensional Arrays              |
| ------- | ---------------------------------------------- |
| NumPy   | Row-Major Order                                |
| MATLAB  | Column-Major Order                             |
| OpenGL  | Column-Major Order                             |
| CUDA    | Column-Major Order                             |
| OpenCV  | Row-Major Order                                |
| Eigen   | Supports both Row-Major and Column-Major Order |
| CSV     | Row-Major Order                                |
| Parquet | Column-Major Order                             |

Column-major formats are better for accessing specific columns of large datasets
with many features, while row-major formats are better for faster data writes
when adding new individual examples to data. Row-major formats are better for a
lot of writes, while column-major formats are better for a lot of column-based
reads.

When you have a dataset with many features, storing the data in a column-major
format is more efficient because it allows for direct access to individual
columns without having to scan through all the other data in the rows. This
means that when you need to extract a specific subset of columns from the
dataset, you can do so more efficiently because the system doesn't need to read
through all the other data in the rows to access the desired columns.

In contrast, with a row-major format, the data for each row is stored together
in memory, meaning that to access a specific column, you have to read through
all the other columns in the row before you get to the desired column. This can
be especially inefficient when dealing with large datasets with many features,
as the system has to read through a lot of data to extract the desired subset of
columns.

For example, consider a dataset of ride-sharing transactions with 1,000
features, but you only need to extract four specific columns: time, location,
distance, and price. With a column-major format, you can directly access these
columns, whereas with a row-major format, you have to read through all the other
996 columns in each row before getting to the desired four columns. This can be
slow and inefficient, especially if you need to access the subset of columns
frequently or if the dataset is very large.

In summary, storing data in a column-major format is more efficient for datasets
with many features because it allows for direct access to individual columns,
which can significantly speed up data retrieval and processing.

#### Examples in code (Python) of Row-major vs Column-major order and its effect on performance

```python
import functools
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Timer decorator."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper


@timer
def traverse_dataframe_by_row(df: pd.DataFrame) -> None:
    for col in df.columns:
        for _ in df[col]:
            pass


@timer
def traverse_dataframe_by_column(df: pd.DataFrame) -> None:
    num_rows = df.shape[0]
    for row_idx in range(num_rows):
        for _ in df.iloc[row_idx]:
            pass


df = pd.DataFrame(np.random.rand(5000, 5000))
print(df.shape)

traverse_dataframe_by_row(df)
traverse_dataframe_by_column(df)


# Row-major traversal (C-like order)
df_np = df.to_numpy()
df_np = np.array(df_np, order="C")  # Row-major traversal (C-like order)
n_rows, n_cols = df_np.shape


@timer
def traverse_numpy_by_row(array: npt.NDArray[np.floating[Any]]) -> None:
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            _ = array[row_idx, col_idx]


@timer
def traverse_numpy_by_column(array: npt.NDArray[np.floating[Any]]) -> None:
    for col_idx in range(n_cols):
        for row_idx in range(n_rows):
            _ = array[row_idx, col_idx]


traverse_numpy_by_row(df_np)
traverse_numpy_by_column(df_np)

df_np_col = np.array(df_np, order="F")  # Column-major traversal (Fortran-like order)

traverse_numpy_by_row(df_np_col)
traverse_numpy_by_column(df_np_col)
```

### Text vs Binary Formats

CSV and JSON are text files, while Parquet files are binary files. Text files
are human-readable, while binary files are only readable by programs that can
interpret the raw bytes. Binary files contain only 0s and 1s and are more
compact than text files. Binary files can save space compared to text files; for
example, storing the number 1000000 requires 7 bytes in a text file and only 4
bytes in a binary file as int32. Parquet files are more efficient than text
files in terms of storage and processing speed. For example, AWS recommends
using the Parquet format because it consumes up to 6x less storage and is up to
2x faster to unload in Amazon S3 compared to text formats.

For example, if you want to store the number $1000000$, and if you store it in
text file it takes 7 characters (1, 0, 0, 0, 0, 0, 0), taking up 7 bytes of
storage if 1 character is 1 byte. But if you store it in binary format as
`int32`, then it takes 32 bits, which is 4 bytes.

## Workflow

Once the data source is scoped and well-defined, before we even start extracting
the data, we need to know what kind of data we are dealing with and **how** and
**where** we are going to store the extracted data.

Determining the **storage format** is critical. Will the data be stored in its
raw form, or does it need to be processed and converted into a different format
like CSV, JSON, or Parquet? The chosen data format can have significant
implications on storage costs, access speed, and compatibility with your data
processing tools.

The **storage location** is equally important. Depending on the volume of the
data, your budget, and security requirements, you might opt for on-premises
servers, cloud storage, or even a hybrid solution. Cloud storage, like Google
Cloud Storage, Amazon S3, or Azure Blob Storage, offer scalable and secure
solutions. However, you need to consider data privacy regulations and compliance
requirements when deciding where to store the data.

You should also consider how the data will be organized. Will it be stored in a
structured database like MySQL, a NoSQL database like MongoDB, or a distributed
file system like Hadoop HDFS? The data's nature, the need for scalability, and
the types of queries you'll be running, all factor into this decision.

Finally, the choice of **storage technology** also depends on the **data
operations** you anticipate. For instance, if your data needs frequent updates,
a database might be more suitable. If your data is largely static but needs to
be read frequently, a file system might be a better choice.

## References and Further Readings

-   Huyen, Chip. "Chapter 3. Data Engineering Fundamentals." In Designing
    Machine Learning Systems: An Iterative Process for Production-Ready
    Applications, O'Reilly Media, Inc., 2022.
-   Kleppmann, Martin. "Chapter 2. Data Models and Query Languages." In
    Designing Data-Intensive Applications. Beijing: O'Reilly, 2017.
