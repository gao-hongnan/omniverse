### Data Storage Options

... list table of different data storage options for different use cases.

As we see shortly in the next few sections, once the data source is defined,
before we even start extracting the data, we need to know what kind of data we
are dealing with and **how** and **where** we are going to store it.

### Data Lake

### Data Warehouse

### Data Lakehouse

### Delta Lake

### SQL vs NoSQL

### Vector Database (A High-dimensional Playground for Large Language Models)

[see here](https://learn.microsoft.com/en-us/semantic-kernel/memories/vector-db)

A **vector database** is an ingenious data storage system that capitalizes on
the properties of vectors — mathematical objects that possess magnitude and
direction. The high-dimensional vectors stored in these databases embody the
features or attributes of data, which could range from text, images, audio, and
video to even more complex structures.

#### Transformation and Embeddings

The crucial task of converting raw data to their vector representations
(embeddings) is typically achieved by utilizing machine learning models, word
embedding algorithms, or feature extraction techniques.

For instance, a movie review text can be represented as a high-dimensional
vector via word embedding techniques like Word2Vec or GloVe. Similarly, an image
can be transformed into a vector representation using deep learning models like
convolutional neural networks (CNNs).

#### The Power of Similarity Search

Vector databases deviate from the conventional way databases work. Rather than
retrieving data based on exact matches or predefined criteria, vector databases
empower users to conduct searches based on vector similarity. This facilitates
the retrieval of data that bears semantic or contextual similarity to the query
data, even if they don't share exact keyword matches.

Consider this example: Given an image of a cat, a vector database can find
images that are visually similar (e.g., other images of cats, or perhaps images
of small, furry animals), even if "cat" isn't explicitly tagged or described in
the metadata of those images.

#### The Working Mechanism

Here's how the magic happens: A query vector, which symbolizes your search
criterion, is used to scour the database for the most similar vectors. This
query vector can be either generated from the same data type as the stored
vectors (image for image, text for text, etc.) or from different types.

A similarity measure, such as cosine similarity or Euclidean distance, is then
employed to calculate the proximity between the query vector and stored vectors.
The result is a ranked list of vectors — and their corresponding raw data — that
have the highest similarity to the query.

#### Use Cases: From NLP to Recommendation Systems

The potential applications for vector databases are wide-ranging. They can be
utilized in natural language processing, computer vision, recommendation
systems, and any domain requiring a deep understanding and matching of data
semantics.

For example, a large language model (LLM) like GPT-3 can be complemented with a
vector database to generate more relevant and coherent text. Let's say you want
the LLM to write a blog post about the latest trends in artificial intelligence.
While the model can generate text based on the prompt, it may lack the most
recent information or context about the subject matter.

This is where a vector database comes into play. You could maintain a vector
database with the latest information, articles, and papers about AI trends. When
you prompt the LLM to write the blog post, you could use a query to pull the
most relevant and recent vectors from the database, and feed this information
into the model along with your prompt. This would guide the model to generate
text that is not only contextually accurate but also up-to-date with current
information.

Keep in mind, though, that building and maintaining such a vector database
requires careful consideration of your data update strategy, storage
requirements, and search efficiency, among other things.

#### The New Kid in Town

As the world of data continues to expand in volume and complexity, the need for
intelligent and efficient databases becomes more apparent. Vector databases,
with their high-dimensional storage and similarity-based search capabilities,
provide a promising solution to manage and make sense of the deluge of data in
various application areas.
