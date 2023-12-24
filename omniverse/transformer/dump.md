# Dump

```{contents}
:local:
```

# Transformer

Let's use the sentence "The cat walks by the bank" to walk through the
self-attention mechanism with analogies and to clarify how it works step by
step.

**Setting the Scene (Embedding the Sentence):** Imagine each word in the
sentence is a person at a party (our tokens). They start by telling a basic fact
about themselves (their initial embedding).

**The Roles:**

-   **Q (Seekers)**: Each person (word) is curious about the stories (contexts)
    of others at the party. They have their own perspective or question (Q
    vector).
-   **K (Holders)**: At the same time, each person has a name tag with keywords
    that describe their story (K vector).
-   **V (Retrievers)**: They also hold a bag of their experiences (V vector),
    ready to share.

**Transformations (Applying W Matrices):** We give each person a set of glasses
(the matrices $W_Q, W_K, W_V$) that changes how they see the world (the space
they project to).

-   With $W_Q$ glasses, they focus on what they want to know from others.
-   With $W_K$ glasses, they highlight their name tag details, making some
    features stand out more.
-   With $W_V$ glasses, they prepare to share the contents of their bag
    effectively.

**Attention (Calculating Q @ K.T):** Now, each person looks around the room
(sequence) with their $W_Q$ glasses and sees the highlighted name tags (after
$W_K$ transformation) of everyone else. They measure how similar their question
is to the others' name tags—this is the dot product $Q @ K^T$.

For "cat," let’s say it’s curious about the notion of "walking" and "bank." It
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

## Decoder Context

Say the input is the cat walk by the bank to find fish.

QUERY means what does cat want to know

KEY what or who words have relevant info for cat query

VALUE Now once cat got back info from QK then cat has knowledge of who is
important or not

Thisw example I realised it’s applicable to full transformer more than GPT
because I think GPT is look ahead only so the word cat is the second word
therefore it shud not have info of words ahead. But the last word fish has all
info of all words and positional info as well because GPT is autoregressive and
generate next word

## Attention

TODO: connect back later the below:

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
    "art" in the sentence. You calculate its dot product with every other word
    to get a set of scores. These scores tell you how much each word in the
    sentence should be "attended to" when you're focusing on "art."

2. **Scaling and Softmax**: The attention scores are scaled down by $ \sqrt{D} $
   and then a softmax is applied. This turns the scores into probabilities
   (attention weights) and ensures that they sum to 1 for each word you're
   focusing on.

    **Intuition**: After scaling and softmax, you get a set of weights that tell
    you how to create a weighted sum of all the words in the sentence when
    you're focusing on a particular word like "art."

3. **Context Vector**: Finally, these attention weights are used to create a
   weighted sum of the value vectors $ V $. This weighted sum is your context
   vector.

    **Intuition**: When focusing on the word "art," you look at the attention
    weights to decide how much of each other word you should include in your
    understanding of "art." You then sum up these weighted words to get a new
    vector, or "context," for the word "art."

4. **Output**: The output will be another $ L \times D $ matrix, where each row
   is the new "contextualized" representation of each word in your sentence.

In your mind, you can picture it as a series of transformations: starting from
the initial $L \times D$ matrix, through an $ L \times L $ attention score
matrix and attention weights, and back to a new $ L \times D $ context matrix.
Each step refines the information content of your sentence, focusing on
different relationships between the words.
