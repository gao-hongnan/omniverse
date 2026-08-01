# Vector Semantics and Embeddings

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

How can a computer represent the _meaning_ of a word? Vector semantics answers
this by representing a word as a point in a high-dimensional space, derived from
the contexts in which it appears. The idea is load-bearing for modern natural
language processing: virtually every language model begins by mapping tokens to
vectors — an embedding.

This chapter develops the classical pipeline of vector semantics:

1. **Words and vectors** — representing words and documents through
   co-occurrence matrices (term-document and term-term matrices).
2. **TF-IDF** — re-weighting raw counts so that words which are frequent in a
   document but rare across documents dominate, with an application that builds
   a small movie recommender.
3. **Cosine similarity** — measuring the closeness of two vectors by the angle
   between them, with an implementation and an application to word similarity.

The treatment follows closely Jurafsky and Martin {cite}`jurafsky_martin_2022`.

## Table of Contents

```{tableofcontents}

```

## References and Further Readings

-   Jurafsky, Dan, and James H. Martin. "Chapter 6. Vector Semantics and
    Embeddings." In Speech and Language Processing: An Introduction to Natural
    Language Processing, Computational Linguistics, and Speech Recognition.
    Pearson, 2022.
