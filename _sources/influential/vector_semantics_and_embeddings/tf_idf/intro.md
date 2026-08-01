# Term Frequency-Inverse Document Frequency (TF-IDF)

```{contents}
:local:
```

TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a popular
technique used in information retrieval and natural language processing to
quantify the importance of each word in a document or corpus of documents.

## Motivation

Consider the example in
[the section on representing words as vectors using document dimension](words-as-vectors-document-dimensions).
Let's add one more vocab to the matrix, the most frequent word **the**, which
appears many times across all documents.

```{list-table} Number of occurrences of selected words (modified) in four Shakespeare plays.
:header-rows: 1
:name: term-document-table-modified

* -
  - As You Like It
  - Twelfth Night
  - Julius Caesar
  - Henry V
* - battle
  - 1
  - 0
  - 7
  - 13
* - good
  - 114
  - 80
  - 62
  - 89
* - fool
  - 36
  - 58
  - 1
  - 4
* - wit
  - 20
  - 15
  - 2
  - 3
* - the
  - 1000
  - 1200
  - 900
  - 1100
```

Then by the cosine similarity metric, we will see that many words will be
similar to the word `the`. This is skewed information, if you consider the table
above and see the word `the` and `good`, then if you want to do document
retrieval based on these two words alone (hypothetically), will yield you the
document that contains the word `the` the most, which is not what we want. This
kind of violate the idea "similar documents tend to have similar words"
mentioned in [words and vectors](../words_and_vectors/concept.ipynb), because
the word `the` is so common that it appears in almost all documents, so it is
not a good indicator of similarity.

In what follows, the basic idea behind TF-IDF is to give a high weight to words
that appear frequently in a document but rarely in other documents, while giving
a low weight to words that are common across all documents. The weight assigned
to each word is calculated based on two factors: term frequency (TF) and inverse
document frequency (IDF). This is a solution to the aforementioned problem.

## Defining TF-IDF

Formally, the TF-IDF weight of a term $t$ in a document $d$ is the product of
two factors {cite}`jurafsky_martin_2022`:

$$
\operatorname{tf-idf}(t, d) = \operatorname{tf}(t, d) \times \operatorname{idf}(t)
$$

where

-   the **term frequency** $\operatorname{tf}(t, d)$ measures how often $t$
    occurs in $d$. Because a word that appears $100$ times in a document is not
    $100$ times more indicative of its topic, the raw count is usually squashed
    logarithmically,
    $\operatorname{tf}(t, d) = \log_{10}\left(1 + \operatorname{count}(t, d)\right)$;
-   the **inverse document frequency**
    $\operatorname{idf}(t) = \log_{10}\left(\frac{N}{\operatorname{df}(t)}\right)$
    rewards rarity: $N$ is the number of documents in the corpus and
    $\operatorname{df}(t)$ is the number of documents containing $t$. A word
    such as **the** that occurs in every document has
    $\operatorname{idf} = \log_{10}(1) = 0$, and its TF-IDF weight vanishes —
    exactly the fix the motivation above asked for.

Libraries differ in the exact variant: scikit-learn, used in the application
that follows, computes
$\operatorname{idf}(t) = \ln\left(\frac{1 + N}{1 + \operatorname{df}(t)}\right) + 1$
by default (smoothed, natural logarithm) and then length-normalizes each
document vector.

## Table of Contents

```{tableofcontents}

```
