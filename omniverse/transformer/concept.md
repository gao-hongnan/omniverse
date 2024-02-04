# Concept

```{contents}
:local:
```

## Self-Attention

### Intuition

Let's use the sentence "Cat walks by the bank" to walk through the
self-attention mechanism with analogies and to clarify how it works step by
step. The sentence is tokenized into `["cat", "walks", "by", "the", "bank"]` and
can be represented as an input sequence $\mathbf{X}$ of $L=5$ tokens, where we
consider each word as a token.

For each token, we have a vector representation of it, which is called the
embedding of the token. We can represent the embedding of the $i$-th token as
$\mathbf{z}_i$. For example, the embedding of the token "cat" can be represented
as $\mathbf{z}_1 \in \mathbb{R}^{1 \times D}$, where $D$ is the dimension of the
embedding.

The self-attention mechanism aims to project each of our embedding
$\mathbf{z}_i$ in the input sequence into a new embedding vector, which we call
the **context vector** $\mathbf{c}_i$, which still lies in the same dimension
$D$, albeit going through some projections to different subspaces during the
process, which we will explain later.

This context vector $\mathbf{c}_i$ is a representation of the $i$-th token in
the input sequence, which is a weighted sum of all the embeddings of the tokens
in the input sequence. The weights are calculated by the attention mechanism.

More concretely, the initial embedding $\mathbf{z}_1$ of the token "cat" is only
holding semantic information about the token "cat" itself. However, after going
through the self-attention mechanism, the context vector $\mathbf{c}_1$ of the
token "cat" is a weighted sum of all the embeddings of the tokens in the input
sequence, which means that the context vector $\mathbf{c}_1$ of the token "cat"
is holding information about the token "cat" itself as well as the information
about the other tokens in the input sequence. It allows the token "cat" to have
a better understanding of itself in the context of the whole sentence (i.e.
should I, the token "cat", pay more attention to the token "bank" as a financial
institution or a river bank and pay less attention to the token "by"?).

### Analogy 1. A Generic Example

**Setting the Scene (Embedding the Sentence):** Imagine each word in the
sentence is a person at a party (our tokens). They start by telling a basic fact
about themselves (their initial embedding with semantic meaning).

**The Roles:**

-   **$Q$ (Seekers)**: Each person (word) is curious about the stories
    (contexts) of others at the party. They have their own perspective or
    question (Q vector).
-   **$K$ (Holders)**: At the same time, each person has a name tag with
    keywords that describe their story (K vector).
-   **$V$ (Retrievers)**: They also hold a bag of their experiences (V vector),
    ready to share.

**Transformations (Applying W Matrices):** We give each person a set of glasses
(the matrices $W_Q, W_K, W_V$) that changes how they see the world (the space
they project to).

-   With $W_Q$ glasses, they focus on what they want to know from others.
-   With $W_K$ glasses, they highlight their name tag details, making some
    features stand out more.
-   With $W_V$ glasses, they prepare to share the contents of their bag
    effectively.

**Attention (Calculating $Q @ K^T$):** Now, each person looks around the room
(sequence) with their $W_Q$ glasses and sees the highlighted name tags (after
$W_K$ transformation) of everyone else. They measure how similar their question
is to the others' name tags—this is the dot product $Q @ K^T$.

For "cat," let’s say it’s curious about the notion of "walks" and "bank." It
will measure the similarity (attention scores) between its curiosity and the
name tags of "walks," "by," "the," "bank."

**Normalization (Softmax):** After measuring, "cat" decides how much to focus on
each story—this is softmax. Some stories are very relevant ("walks"), some
moderately ("by," "the"), and some might be highly relevant depending on context
("bank" — is it a river bank or a financial institution?).

**Retrieval (Applying Attention to V):** Now "cat" decides to listen to the
stories in proportion to its focus. It takes pieces (weighted by attention
scores) from each person's experience bag (V vectors) and combines them into a
richer, contextual understanding of itself in the sentence. This combination
gives us the new representation of "cat," informed by the entire context of the
sentence.

In essence:

-   **Q (Query):** What does "cat" want to know?
-   **K (Key):** Who has relevant information to "cat"’s curiosity?
-   **V (Value):** What stories does "cat" gather from others, and how much does
    it take from each to understand its role in the sentence?

The output of self-attention for "cat" now encapsulates not just "cat" but its
relationship and relevance to "walks," "by," "the," "bank" in a way that no
single word could convey alone. This output then becomes the input to the next
layer, where the process can repeat, enabling the model to develop an even more
nuanced understanding.

### Analogy 2. A More Concrete Example

1. **Attention Scores**: Once you have your $ Q, K, V $ matrices (which are all
   $ L \times D $ in this simplified example), you calculate the dot product
   between queries $ Q $ and keys $ K
   $. This is essentially
   measuring how each word in the sentence relates to every other word.
   Mathematically, you'll get a matrix of shape $ L \times L $, where each
   element $ (i, j) $ represents the "affinity" between the $ i^{th} $ and
   $
   j^{th} $ words.

    **Intuition**: Imagine you're trying to understand the role of the word
    "cat" in the sentence. You calculate its dot product with every other word
    to get a set of scores. These scores tell you how much each word in the
    sentence should be "attended to" when you're focusing on "cat."

2. **Scaling and Softmax**: The attention scores are scaled down by $ \sqrt{D} $
   and then a softmax is applied. This turns the scores into probabilities
   (attention weights) and ensures that they sum to 1 for each word you're
   focusing on.

    **Intuition**: After scaling and softmax, you get a set of weights that tell
    you how to create a weighted sum of all the words in the sentence when
    you're focusing on a particular word like "cat."

3. **Context Vector**: Finally, these attention weights are used to create a
   weighted sum of the value vectors $ V $. This weighted sum is your context
   vector.

    **Intuition**: When focusing on the word "cat," you look at the attention
    weights to decide how much of each other word you should include in your
    understanding of "cat." You then sum up these weighted words to get a new
    vector, or "context," for the word "cat."

4. **Output**: The output will be another $ L \times D $ matrix, where each row
   is the new "contextualized" representation of each word in your sentence.

In your mind, you can picture it as a series of transformations: starting from
the initial $L \times D$ matrix, through an $ L \times L $ attention score
matrix and attention weights, and back to a new $ L \times D $ context matrix.
Each step refines the information content of your sentence, focusing on
different relationships between the words.

## Casual Attention/Masked Self-Attention

### Intuition

One sentence to summarize the understanding.

**Casual attention (masked self attention) in decoder reduces to self attention
for the last token in the input sequence.**

Causal attention in a decoder architecture, such as the one used in Transformer
models, effectively reduces to self-attention for the last token in the input
sequence.

1. **Causal Attention Mechanism**: In a causal attention mechanism, each token
   is allowed to attend to itself and all preceding tokens in the sequence. This
   is enforced by masking future tokens to prevent information flow from future
   tokens into the current or past tokens. This mechanism is crucial in
   generative models where the prediction for the current token should not be
   influenced by future tokens, as they are not known during inference.

2. **Self-Attention Mechanism**: In self-attention, each token computes
   attention scores with every other token in the sequence, including itself.
   These attention scores are used to create a weighted sum of the values (token
   representations), which becomes the new representation of the token.

3. **Last Token in the Sequence**: When considering the last token in the
   sequence, the causal attention mechanism's nature implies that this token has
   access to all previous tokens in the sequence, including itself. There are no
   future tokens to mask. Therefore, the attention mechanism for this token
   becomes identical to the standard self-attention mechanism where it is
   attending to all tokens up to itself.

## Perplexity

-   https://keras.io/api/keras_nlp/metrics/perplexity/
-   https://lightning.ai/docs/torchmetrics/stable/text/perplexity.html
-   https://huggingface.co/docs/transformers/perplexity

## All the Whys?

### Can you explain the Subspaces projection in Attention Mechanism?

### Why Softmax in Attention Mechanism?

See <https://www.youtube.com/watch?v=UPtG_38Oq8o> around 16 min.

### Why Scale in Attention Mechanism?

-   <https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html>

Let's look at the notes:

-   <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html>

1. **Scaling Factor Introduction**: The author introduces the concept of a
   scaling factor $\frac{1}{\sqrt{d_k}}$, which is applied during the attention
   mechanism in transformers. This scaling factor is used to maintain an
   appropriate variance of the attention scores after initialization.

2. **Initialization Goal**: The goal of initialization is to have each layer of
   the neural network maintain equal variance throughout. This is to ensure that
   the gradients are neither vanishing nor exploding as they propagate through
   the layers, which is crucial for effective learning.

3. **Variance in Dot Products**: The author then explains that when taking a dot
   product of two vectors, both sampled from normal distributions with variance
   $\sigma^2$, the resulting scalar will have a variance that is $d_k$ times
   higher, specifically $\sigma^4 \cdot d_k$. Here, $d_k$ represents the
   dimension of the key/query vectors in the attention mechanism, and $q_i$ and
   $k_i$ are the components of the query and key vectors respectively.

4. **Scaling Down the Variance**: Without scaling down the variance of the dot
   product (which is $\sigma^4 \cdot d_k$), the softmax function, which is
   applied to the attention scores to obtain the probabilities, would become
   saturated. This means that one logit (the vector of raw (non-normalized)
   predictions that a classification model generates, which is then passed to a
   normalization function) would have a very high score close to 1, while the
   rest would have scores close to 0. This saturation makes it difficult for the
   network to learn because the gradients would be close to zero for all
   elements except the one with the highest score.

5. **Maintaining Variance Close to 1**: The author notes that despite the
   multiplication by $\sigma^4$, the practice of keeping the original variance
   $\sigma^2$ close to 1 means that the scaling factor does not introduce a
   significant issue. By multiplying the dot product by $\frac{1}{\sqrt{d_k}}$,
   the variance of the product is effectively scaled back to the original level
   of $\sigma^2$, preventing the softmax function from saturating and allowing
   the model to learn effectively.

The gist is:

It is important to maintain equal variance across all layers in a neural
network, particularly in the context of the transformer model's attention
mechanism. By doing so, the model helps to ensure that the gradients are stable
during backpropagation, avoiding the vanishing or exploding gradients problem
and enabling effective learning.

In the specific context of the attention mechanism, the variance of the dot
products used to calculate attention scores is scaled down by the factor
$\frac{1}{\sqrt{d_k}}$ to prevent softmax saturation. This allows each element
to have a chance to influence the model's learning, rather than having a single
element dominate because of the variance scaling with $d_k$. This practice is
crucial for the learning process because it ensures the gradients are meaningful
and not diminished to the point where the model cannot learn from the data.

### Why do need Positional Encoding? What happens if we don't use it?

## References and Further Readings

...
