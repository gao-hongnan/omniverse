## Motivation: Understanding Similarity in High-Dimensional Spaces

In machine learning, particularly in fields like **natural language processing
(NLP)** and **computer vision**, a fundamental challenge is to understand and
quantify the notion of similarity. Consider the task of image recognition or
word meaning interpretation. How do we determine that two images are similar, or
that two words have similar meanings?

### Representing Images and Words as Vectors

In high-dimensional spaces like $\mathbb{R}^D$, complex entities like images and
words can be represented as vectors. For instance:

-   **Images**: Each image can be represented as a vector, where each dimension
    corresponds to a pixel or a feature extracted from the image.
-   **Words**: In NLP, words are represented as vectors in a space where
    distances between vectors are related to semantic similarities between
    words. This is achieved through techniques like **word embeddings**.

### The Need for Similarity Measures

Once we have these vector representations, we need a way to quantify how 'close'
or 'similar' they are. This is where the concept of **similarity** comes in.
Similarity measures in vector spaces enable us to:

-   **Compare Images**: Determine how similar two images are based on their
    vector representations. This has applications in image search, face
    recognition, and more.
-   **Understand Word Semantics**: In NLP, measure the closeness of words in the
    embedding space to capture semantic relationships (like synonyms,
    analogies).
-   **Cluster and Categorize**: Group similar items together, whether they're
    images in a photo library or words in a document.

### Role of Norms and Distance Metrics

To quantify similarity, we often use norms and distance metrics like the
Euclidean norm ($L_2$ norm) or the Manhattan norm ($L_1$ norm). These
mathematical tools give us a way to compute distances in high-dimensional
spaces, translating into measures of similarity or dissimilarity:

-   **Closer Vectors**: Indicate more similarity (e.g., images with similar
    features, words with related meanings).
-   **Further Apart Vectors**: Suggest less similarity or greater dissimilarity.

### Limitations of $L_1$ and $L_2$ Norms in Measuring Similarity

The $L_1$ and $L_2$ norms focus on the magnitude of vectors, which can be a
limitation in certain scenarios:

-   **Dominance of Magnitude**: In high-dimensional spaces, especially with
    sparse vectors (common in NLP), the magnitude of vectors can dominate the
    similarity measure. Two vectors might be pointing in the same direction
    (hence, similar in orientation) but can be deemed dissimilar due to
    differences in magnitude.
-   **Insensitive to Distribution of Components**: These norms don't
    differentiate how vector components contribute to the overall direction. Two
    vectors with similar orientations but different distributions of values
    across components can have the same $L_1$ or $L_2$ norm.

Consequently,
**[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)** emerges
as a critical concept, especially in NLP and document classification. Unlike
Euclidean or Manhattan norms that focus on magnitude, cosine similarity
concentrates on the angle between vectors, making it exceptionally suited for
comparing the orientation (and thus the semantic direction) of word embeddings
in high-dimensional space. We will explore cosine similarity in detail in later
sections.

```{code-cell} ipython3
import numpy as np

def calculate_norms_and_cosine_similarity(vec_a, vec_b):
    l1_norm = np.sum(np.abs(vec_a - vec_b))
    l2_norm = np.sqrt(np.sum((vec_a - vec_b) ** 2))
    cosine_similarity = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return l1_norm, l2_norm, cosine_similarity

vec_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
vec_b = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])

l1_norm, l2_norm, cosine_similarity = calculate_norms_and_cosine_similarity(vec_a, vec_b)
print("L1 Norm:", l1_norm)
print("L2 Norm:", l2_norm)
print("Cosine Similarity:", cosine_similarity)

vec_c = np.array([1, 2, 3])
vec_d = np.array([2, 4, 6])

l1_norm, l2_norm, cosine_similarity = calculate_norms_and_cosine_similarity(vec_c, vec_d)
print("L1 Norm:", l1_norm)
print("L2 Norm:", l2_norm)
print("Cosine Similarity:", cosine_similarity)
```
