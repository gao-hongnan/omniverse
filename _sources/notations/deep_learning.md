# Deep Learning Notations

We largely follow the
[Machine Learning: The Basics](https://link.springer.com/book/10.1007/978-981-16-8193-6)
book in terms of notations.

## NLP Terminologies

Consider a toy corpus of five text units:

```text
1. This is Foo, how can I help you
2. It might rain today
3. I love football
4. Crazy, Stupid & Love
5. I shot the sheriff
```

The terminology below follows common usage (see
[this discussion](https://stackoverflow.com/questions/72550835/corpus-vs-vocabulary-vs-document-in-nlp)):

-   A **document** is the basic unit of text. What counts as a document is
    task-dependent — it can be an entire article, a passage, or a single
    sentence. In the toy corpus above, each line is one document, so "This is
    Foo, how can I help you" is a document and "It might rain today" is another.
-   A **corpus** (or collection) is a set of documents. The toy corpus above is
    composed of five documents.
-   The **vocabulary** is the set of all distinct words contained in the corpus.
    Here it is [&, can, crazy, foo, football, help, how, i, is, it, love, might,
    rain, sheriff, shot, stupid, the, this, today, you].

```{list-table} Basic NLP Terminologies
:header-rows: 1
:name: basic-nlp-terminologies

-   -   Notation
    -   Description
-   -   $V$
    -   The number of distinct vocabulary words in the corpus.
-   -   Corpus/Collection
    -   A collection of documents.
```

These terms are used throughout the
[vector semantics and embeddings chapter](../influential/vector_semantics_and_embeddings/01_intro.md).
