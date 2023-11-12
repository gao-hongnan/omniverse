# Notations

```{tableofcontents}

```

> Mostly follow Lilian's notations, except for $d_k$, $d_q$ and $d_v$, which we
> will stay consistent with the paper.

### Dimensions and Indexing

Here we list common dimensions and indexing used in the Transformer model.
Dimensions and indexing pertaining to attention will be listed in the
[Attention Notations](#attention-notations) section.

- $D$: embedding dimension. In the paper it is denoted as $d_{\text{model}}$.
  - $d$: index of an element in the embedding vector.
- $L$: sequence length.
  - $i$: index of a token in the sequence.
- $V$: vocabulary size.
  - $j$: index of a word in the vocabulary.

### General Notations

- $\mathcal{V}$: is the set of all words in the vocabulary defined as:

  $$
  \mathcal{V} = \{v_1, v_2, ..., v_V\}
  $$

  where

  - $V$: is the size of the vocabulary, also denoted as $|\mathcal{V}|$.
  - $v_j$: is a unique word in the vocabulary $\mathcal{V}$.
  - $j$: is the index of a word in the vocabulary $\mathcal{V}$.

- $\mathbf{X}$: is the input sequence defined as:

  $$
  \mathbf{X} = (x_1, x_2, ..., x_L)
  $$

  where

  - $L$: is the sequence length.
  - $x_{i}$: is a token at position $i$ in the sequence. Each $x_{i}$ is a token
    represented as an integer from the set ${0, 1, ..., V-1}$.
  - $i$: is the index of a token in the sequence $\mathbf{X}$.

- $\mathbf{O}$: one-hot representation of the input sequence $\mathbf{X}$. This
  is a $L \times V$ matrix, where each row represents a token in the sequence
  and each column corresponds to a unique word in the vocabulary $\mathcal{V}$.

  $$
  \begin{aligned}
  \mathbf{O} &= \begin{bmatrix} o_{1,1} & o_{1,2} & \cdots & o_{1,V} \\ o_{2,1} & o_{2,2} & \cdots & o_{2,V} \\ \vdots & \vdots & \ddots & \vdots \\ o_{L,1} & o_{L,2} & \cdots & o_{L,V} \end{bmatrix} \in \mathbb{R}^{L \times V} \\
  &= \begin{bmatrix} \text{---} & \mathbf{o}_{1, :} & \text{---} \\ \text{---} & \mathbf{o}_{2, :} & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{o}_{L, :} & \text{---} \end{bmatrix} \in \mathbb{R}^{L \times V}
  \end{aligned}
  $$

  where

  - $L$: is the sequence length.
  - $V$: is the vocabulary size.
  - $o_{i, j}$: is the one-hot encoded element at position $i, j$. For a given
    token $x_i$ at the $i$-th position in the sequence $\mathbf{X}$, if
    $f_{\text{stoi}}(x_i)=j$, then the element at position $j$ in the one-hot
    vector for token $x_i$ is 1, and all other elements are 0.
  - $\mathbf{o}_{i, :}$: is the one-hot encoded vector for the token $x_i$ at
    the $i$-th position in the sequence $\mathbf{X}$. This row form is more
    important than column form.

- $\mathbf{E}$: is the embedding matrix defined as:

  $$
  \mathbf{E} = \begin{bmatrix} e_{1,1} & e_{1,2} & \cdots & e_{1,D} \\ e_{2,1} & e_{2,2} & \cdots & e_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ e_{V,1} & e_{V,2} & \cdots & e_{V,D} \end{bmatrix} \in \mathbb{R}^{V \times D}
  $$

  where

  - $V$: is the vocabulary size.
  - $D$: is the embedding dimension.
  - $e_{j, d}$: is the embedding element at position $j, d$. For a word $v_j$ in
    the vocabulary $\mathcal{V}$, the corresponding row in $\mathbf{E}$ is the
    embedding vector for that word.

- $\mathbf{Z}$: is the output tensor of the embedding layer, obtained by matrix
  multiplying $\mathbf{O}$ with $\mathbf{E}$, and it is defined as:

  $$
  \mathbf{Z} = \mathbf{O} \cdot \mathbf{E}
  $$

  $$
  \begin{aligned}
  \mathbf{Z} &= \mathbf{O} \cdot \mathbf{E} \\
  &= \begin{bmatrix} z_{1,1} & z_{1,2} & \cdots & z_{1,D} \\ z_{2,1} & z_{2,2} & \cdots & z_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ z_{L,1} & z_{L,2} & \cdots & z_{L,D} \end{bmatrix} \in \mathbb{R}^{L \times D} \\
  &= \begin{bmatrix} \text{---} & \mathbf{z}_{1,:} & \text{---} \\ \text{---} & \mathbf{z}_{2,:} & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{z}_{L,:} & \text{---} \end{bmatrix} \in \mathbb{R}^{L \times D}
  \end{aligned}
  $$

  where

  - $L$: is the sequence length.
  - $D$: is the embedding dimension.
  - $z_{i, d}$: is the element at position $i, d$ in the tensor $\mathbf{Z}$.
    For a token $x_i$ at the $i$-th position in the sequence, $z_{i, :}$ is the
    $D$ dimensional embedding vector for that token.
  - $\mathbf{z}_{i, :}$: is the $D$ dimensional embedding vector for the token
    $x_i$ at the $i$-th position in the sequence.

            In this context, each token in the sequence is represented by a $D$
            dimensional vector. So, the output tensor $\mathbf{Z}$ captures the
            dense representation of the sequence. Each token in the sequence is
            replaced by its corresponding embedding vector from the embedding matrix
            $\mathbf{E}$.

            As before, the output tensor $\mathbf{Z}$ carries semantic information
            about the tokens in the sequence. The closer two vectors are in this
            embedding space, the more semantically similar they are.

- $\mathbf{P}$: is the positional encoding tensor, created with sinusoidal
  functions of different frequencies:

  Each position $i$ in the sequence has a corresponding positional encoding
  vector $p_{i, :}$ of length $D$ (the same as the embedding dimension). The
  elements of this vector are generated as follows:

  $$
  p_{i, 2i} = \sin\left(\frac{i}{10000^{2i / D}}\right)
  $$

  $$
  p_{i, 2i + 1} = \cos\left(\frac{i}{10000^{2i / D}}\right)
  $$

  for each $i$ such that $2i < D$ and $2i + 1 < D$.

  Thus, the entire tensor $\mathbf{P}$ is defined as:

  $$
  \mathbf{P} = \begin{bmatrix} p_{1,1} & p_{1,2} & \cdots & p_{1,D} \\ p_{2,1} & p_{2,2} & \cdots & p_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ p_{L,1} & p_{L,2} & \cdots & p_{L,D} \end{bmatrix} \in \mathbb{R}^{L \times D}
  $$

  where

  - $L$: is the sequence length.
  - $D$: is the embedding dimension.
  - $p_{i, d}$: is the element at position $i, d$ in the tensor $\mathbf{P}$.

- Note that $\mathbf{P}$ is independent of $\mathbf{Z}$, and it's computed based
  on the positional encoding formula used in transformers, which uses sinusoidal
  functions of different frequencies:

- OVERWRITING $\mathbf{Z}$: After computing the positional encoding tensor
  $\mathbf{P}$, we can update our original embeddings tensor $\mathbf{Z}$ to
  include positional information:

  $$
  \mathbf{Z} = \mathbf{Z} + \mathbf{P}
  $$

  This operation adds the positional encodings to the original embeddings,
  giving the final embeddings that are passed to subsequent layers in the
  Transformer model.

- Or consider using $\mathbf{Z}^{'}$?

### Attention Notations

- $H$: Number of attention heads.
  - $h$: Index of the attention head.
- $d_k = D/H$: Dimension of the keys. In the multi-head attention case, this
  would typically be $D/H$ where $D$ is the dimensionality of input embeddings
  and $H$ is the number of attention heads.
- $d_q = D/H$: Dimension of the queries. Also usually set equal to $d_k$.
- $d_v = D/H$: Dimension of the values. Usually set equal to $d_k$.
- $\mathbf{W}^q \in \mathbb{R}^{D \times H \cdot d_q = D \times D}$: The query
  weight matrix for all heads. It is used to transform the embeddings
  $\mathbf{Z}$ into query representations.

- $\mathbf{W}^k \in \mathbb{R}^{D \times H \cdot d_k = D \times D}$: The key
  weight matrix for all heads. It is used to transform the embeddings
  $\mathbf{Z}$ into key representations.

- $\mathbf{W}^v \in \mathbb{R}^{D \times H \cdot d_v = D \times D}$: The value
  weight matrix for all heads. It is used to transform the embeddings
  $\mathbf{Z}$ into value representations.
- $\mathbf{W}_{h}^{q} \in \mathbb{R}^{D \times d_q}$: The query weight matrix
  for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$ into
  query representations for the $h$-th head.
  - Important that this matrix collapses to $\mathbf{W}_{1}^q$ when $H=1$ and
    has shape $\mathbb{R}^{D \times D}$.
  - Note that this weight matrix is derived from $W^q$.
- $\mathbf{W}_{h}^{k} \in \mathbb{R}^{D \times d_k}$: The key weight matrix for
  the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$ into key
  representations for the $h$-th head.
  - Important that this matrix collapses to $\mathbf{W}_{1}^k$ when $H=1$ and
    has shape $\mathbb{R}^{D \times D}$ since $d_k = D/H = D/1 = D$.
  - Note that this weight matrix is derived from $W^k$.
- $\mathbf{W}_{h}^{v} \in \mathbb{R}^{D \times d_v}$: The value weight matrix
  for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$ into
  value representations for the $h$-th head.

  - Important that this matrix collapses to $\mathbf{W}_{1}^v$ when $H=1$ and
    has shape $\mathbb{R}^{D \times D}$.
  - Note that this weight matrix is derived from $W^v$.

- $\mathbf{Q} = \mathbf{Z} \mathbf{W}^q \in \mathbb{R}^{L \times D}$: The query
  matrix. It contains the query representations for all the tokens in the
  sequence. This is the matrix that is used to compute the attention scores.
  - Each row of the matrix $\mathbf{Q}$ is a query vector $\mathbf{q}_{i}$ for
    the token at position $i$ in the sequence.
- $\mathbf{Q}_h = \mathbf{Z} \mathbf{W}_h^q \in \mathbb{R}^{L \times d_q}$: The
  query matrix for the $h$-th head. It contains the query representations for
  all the tokens in the sequence. This is the matrix that is used to compute the
  attention scores for the $h$-th head.

- $\mathbf{K} = \mathbf{Z} \mathbf{W}^k \in \mathbb{R}^{L \times D}$: The key
  matrix. It contains the key representations for all the tokens in the
  sequence. This is the matrix that is used to compute the attention scores.

- $\mathbf{K}_h = \mathbf{Z} \mathbf{W}_h^k \in \mathbb{R}^{L \times d_k}$: The
  key matrix for the $h$-th head. It contains the key representations for all
  the tokens in the sequence. This is the matrix that is used to compute the
  attention scores for the $h$-th head.

- $\mathbf{V} = \mathbf{Z} \mathbf{W}^v \in \mathbb{R}^{L \times D}$: The value
  matrix. It contains the value representations for all the tokens in the
  sequence. This is the matrix where we apply the attention scores to compute
  the weighted average of the values.

- $\mathbf{V}_h = \mathbf{Z} \mathbf{W}_h^v \in \mathbb{R}^{L \times d_v}$: The
  value matrix for the $h$-th head. It contains the value representations for
  all the tokens in the sequence. This is the matrix where we apply the
  attention scores to compute the weighted average of the values for the $h$-th
  head.

- $\mathbf{q}_{i} = \mathbf{Q}_{i, :} \in \mathbb{R}^{d}$: The query vector for
  the token at position $i$ in the sequence.
- $\mathbf{k}_{i} = \mathbf{K}_{i, :} \in \mathbb{R}^{d}$: The key vector for
  the token at position $i$ in the sequence.
- $\mathbf{v}_{i} = \mathbf{V}_{i, :} \in \mathbb{R}^{d}$: The value vector for
  the token at position $i$ in the sequence.
- $\mathbf{A} \in \mathbb{R}^{L \times L}$: The attention matrix. It contains
  the attention scores for all the tokens in the sequence. It is computed as:

  $$
  \mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right)
  $$

  where

  - $L$: is the sequence length.
  - $\mathbf{Q} \in \mathbb{R}^{L \times D}$: is the query matrix.
  - $\mathbf{K} \in \mathbb{R}^{L \times D}$: is the key matrix.
  - $\sqrt{d_k}$: is the scaling factor.
  - $\text{softmax}(\cdot)$: is the softmax function applied row-wise.
  - More concretely, this is the **self-attention matrix** between an input
    sequence $\mathbf{X} = (x_1, x_2, ..., x_L)$ and itself. Each row in the
    matrix $\mathbf{A}$ is the attention scores for a token in the sequence. The
    attention scores are computed by comparing the query vector for a token with
    the key vectors for all the tokens in the sequence.
  - For instance, if the input sequence is "cat eat mouse", then the $L=3$, and
    the attention matrix $\mathbf{A}$'s first row is the attention scores of the
    word cat with all other words, (cat & cat, cat & eat, cat & mouse).
    Similarly, the second row is the attention scores of the word eat with all
    other words, (eat & cat, eat & eat, eat & mouse). Lastly, the third row is
    the attention scores of the word mouse with all other words, (mouse & cat,
    mouse & eat, mouse & mouse).

- $a_{i, j} \in \mathbf{A}$: The attention score between the query $i$ and the
  key $j$ in the sequence (please do not be confused with the $j$ index in
  vocabulary!). It is computed as:

  $$
  a_{i, j} = \text{softmax}\left(\frac{\mathbf{q}_{i} \mathbf{k}_{j}^T}{\sqrt{d_k}}\right)
  $$

  where

  - $\mathbf{q}_{i} \in \mathbb{R}^{d}$: is the query vector for the $i$-th
    token in the sequence.
  - $\mathbf{k}_{j} \in \mathbb{R}^{d}$: is the key vector for the $j$-th token
    in the sequence.

- $f(\cdot)$: Attention function (such as additive attention or scaled
  dot-product attention).

  - Should we find a better notation?

    The scaled dot-product attention function $f(\cdot)$ can be formulated as:

    $$
    \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) := f(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V} \in \mathbb{R}^{L \times D}
    $$

    or you can also substitute $\mathbf{A}$ to get the same result:

    $$
    \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) := f(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{A} \mathbf{V}
    $$

    For the $h$-th head, it can be represented as:

    $$
    f(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}\left( \frac{\mathbf{Q}_h \mathbf{K}_h^T}{\sqrt{d_k}} \right) \mathbf{V}_h
    $$

    In these formulas, $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ are the
    query, key, and value matrices, respectively. The function
    $\text{softmax}(\cdot)$ is applied row-wise. The division by $\sqrt{d_k}$ is
    a scaling factor that helps in training stability.

---

- $\mathbf{h}_i \in \mathbb{R}^{p_v}$: Output of the $i$-th attention head.

- $\mathbf{W}_o \in \mathbb{R}^{p_o \times h p_v}$: Output weight matrix, used
  to transform the concatenation of all head outputs.

- $p_o$: Dimension of the final output after applying the output weight matrix
  $\mathbf{W}_o$.

Let's break this down:

- $\mathbf{h}_i \in \mathbb{R}^{p_v}$: Output of the $i$-th attention head. It
  is computed as a function $f$ which applies attention (such as additive
  attention or scaled dot-product attention) to the transformed queries, keys
  and values. This function depends on the query $\mathbf{q}$, key $\mathbf{k}$,
  and value $\mathbf{v}$, and the weight matrices $\mathbf{W}_i^{(q)}$,
  $\mathbf{W}_i^{(k)}$, and $\mathbf{W}_i^{(v)}$. The dimensions $p_q$, $p_k$,
  and $p_v$ denote the output dimensions of the query, key and value
  transformations respectively, for the $i$-th head.

- $\mathbf{W}_i^{(q)} \in \mathbb{R}^{p_q \times d_q}$,
  $\mathbf{W}_i^{(k)} \in \mathbb{R}^{p_k \times d_k}$, and
  $\mathbf{W}_i^{(v)} \in \mathbb{R}^{p_v \times d_v}$: The weight matrices for
  the $i$-th attention head. These are used to transform the query, key, and
  value inputs to the dimensions suitable for the attention mechanism.

- $f(\cdot)$: This function represents the attention mechanism (like additive
  attention or scaled dot-product attention). It takes as input the transformed
  query, key, and value vectors and produces the output of the attention head.

- $\mathbf{W}_o \in \mathbb{R}^{p_o \times h p_v}$: This is the output weight
  matrix that linearly transforms the concatenation of the outputs from all
  attention heads to produce the final output of the multi-head attention
  mechanism.

- The expression
  $\mathbf{W}_o\left[\begin{array}{c}
\mathbf{h}_1 \\
\vdots \\
\mathbf{h}_h
\end{array}\right] \in \mathbb{R}^{p_o}$
  represents the final output of the multi-head attention layer. It's the result
  of applying the linear transformation defined by $\mathbf{W}_o$ to the
  concatenated outputs of all attention heads.

This notation helps us understand the inner workings of the multi-head attention
mechanism, and it provides a clear path for implementing the multi-head
attention mechanism in a neural network model.

## Attention (Show this first)

Both `torch.bmm` and `torch.matmul` can be used for matrix multiplication in
PyTorch, but their use cases and behaviors are somewhat different, especially
with higher-dimensional tensors. Let's break this down:

1. **torch.bmm**:

   - It stands for "batch matrix multiplication".
   - It expects tensors to be of rank 3: `(batch_size, rows, cols)`.
   - It performs matrix multiplication for each batch between corresponding
     matrices.

   For example, given two tensors `A` of shape `(B, M, N)` and `B` of shape
   `(B, N, P)`, the output will be of shape `(B, M, P)`.

2. **torch.matmul**:

   - It's a more general-purpose matrix multiplication function.
   - When two 3D tensors are passed, it behaves like `torch.bmm`.
   - However, it can handle tensors of rank > 3 as well. When given
     higher-dimensional tensors, it considers the last two dimensions as
     matrices to be multiplied and broadcasts over the remaining dimensions.

   Given two tensors `A` of shape `(X, Y, M, N)` and `B` of shape
   `(X, Y, N, P)`, the output will be of shape `(X, Y, M, P)`.

In the context of the provided code, both methods achieve the same result
because:

- The shape of the `queries` and transposed `keys` tensors matches the expected
  input shape for `torch.bmm` in the `DotProductAttention` class.
- In the `attention` function, the shape of `query` and transposed `key` tensors
  is also compatible with both `torch.bmm` and `torch.matmul`.

So, when used for 3D tensors, `torch.bmm` and `torch.matmul` can give the same
result. The discrepancy arises primarily with higher-dimensional tensors, where
the broadcasting behavior of `torch.matmul` distinguishes it from `torch.bmm`.

## How $W^{q}_i$ is implemented in practice?

The notation $W^{q}_i$ is used in the paper to denote the weight matrix for the
queries (Q) of the $i$-th head. However, it's essential to understand how this
is implemented in practice.

The entire process can be seen as a two-step operation:

1. **Apply Linear Transformations**: You apply linear transformations to the
   whole embeddings to create larger matrices for Q, K, V. These matrices have
   dimensions that account for all heads. In practice, this can be implemented
   using a single linear layer, such as:

   $$
   Q = \text{{embeddings}} @ \mathbf{W}^q
   $$

   where $\mathbf{W}^q$ has dimensions $D \times (h \cdot d_q)$.

2. **Reshape and Split**: After applying the linear transformations, you reshape
   and split the result into individual heads. The reshaping ensures that the
   final dimensions are $[N, H, S, d_q]$, where $N$ is the batch size, $H$ is
   the number of heads, $S$ is the sequence length, and $d_q$ is the dimension
   of queries per head.

So, while the paper uses notation like $W^{q}_i$, this doesn't mean that you
directly apply a different linear transformation to different parts of the
embeddings. Instead, you apply a single large linear transformation to the whole
embeddings and then reshape the result to obtain the individual heads.

In mathematical terms, the overall operation can be seen as:

$$
\begin{align*}
Q_{\text{{all heads}}} & = \text{{embeddings}} @ \mathbf{W}^q \\
Q_{\text{{head i}}} & = Q_{\text{{all heads}}}[:, i \cdot d_q : (i + 1) \cdot d_q]
\end{align*}
$$

Here, $Q_{\text{{all heads}}}$ is the result of applying the linear
transformation, and $Q_{\text{{head i}}}$ is the portion corresponding to the
$i$-th head, obtained by slicing along the last dimension.

## HEad is similar to kernels in CNN

The multi-head attention mechanism is similar to the convolutional layer in
convolutional neural networks. In a convolutional layer, you apply multiple
kernels to the input to obtain multiple feature maps. Similarly, in the
multi-head attention mechanism, you apply multiple attention heads to the input
to obtain multiple output vectors.

## so the catch is you do not split the embeddings in H heads, instead you split the linear transformed embeddings?

```
# Apply linear transformations to compute Q, K, V
# NOTE: here is an important misconception that if you have
# 8 heads, then you SPLIT the embeddings into 8 parts and
# then apply linear transformations to each part. This is
# WRONG. You apply linear transformations to the whole
# embeddings and then split the result into 8 parts.
```

You don't split the original embeddings into $H$ heads; instead, you apply
linear transformations to the original embeddings and then split the transformed
embeddings into $H$ heads.

Here's the step-by-step process again, highlighting this specific aspect:

1. Apply linear transformations for queries, keys, and values to the entire
   embeddings, creating matrices $Q, K,$ and $V$.
2. Split these transformed matrices into $H$ different heads, each having lower
   dimensions (e.g., if the original dimension is 512 and there are 8 heads,
   each head will have a dimension of 64).
3. Process each head through the Scaled Dot-Product Attention mechanism.
4. Concatenate the outputs from all the heads and pass through a final linear
   layer.

The split after the linear transformations allows the model to create multiple
different projections of the input and process them independently. This enables
the model to focus on different aspects of the input across the different heads,
enhancing its ability to model complex relationships.

## Confusion on Weight matrix per head

The notation and explanation in the original papers and many articles do indeed
mention separate weight matrices for each head, such as $W^{Q}_i$, but in
implementation, it's common to represent these separate weights within a single
large weight matrix. The notation might be different, but the mathematical
operation is equivalent.

Here's how the two approaches relate:

1. **Separate Weight Matrices Notation (Paper Notation):** In the theoretical
   description, you can imagine having separate weight matrices $W^{Q}_i$ for
   each head, and then you multiply the input embeddings by each weight matrix,
   applying the transformation separately for each head.

2. **Single Large Weight Matrix Implementation (Your Code):** In your
   implementation, you create one large weight matrix, $W_q$, that combines all
   the individual weight matrices for each head. When you multiply the input
   embeddings by $W_q$, you create a large transformed matrix. Then, by slicing
   this large matrix, you separate it into $H$ different heads, effectively
   applying the individual weight matrices $W^{Q}_i$ for each head.

The two approaches are mathematically equivalent. In the second approach, the
separate weight matrices for each head are not explicitly defined as learnable
parameters in the code. Instead, they are implicitly represented within the
single large weight matrix $W_q$ and separated by slicing after the linear
transformation.

This approach can be more efficient computationally and often aligns better with
hardware and library optimizations, but it may create confusion when comparing
to the paper's notation. The key is to understand that the mathematical
relationships and learning dynamics are the same, even though the notation and
coding structure might differ.

### Approach 1: Single Large Weight Matrix Implementation (Paper's Code)

In this approach, we concatenate all the individual weight matrices $W^{q}_{h}
$
into one large weight matrix $W_q$:

$$
\begin{aligned}
W_q =  \begin{bmatrix} w^{q}_{1,1} & w^{q}_{1,2} & \ldots & w^{q}_{1,D} \\
w^{q}_{2,1} & w^{q}_{2,2} & \ldots & w^{q}_{2,D} \\
\vdots & \vdots & \ddots & \vdots \\
w^{q}_{D,1} & w^{q}_{D,2} & \ldots & w^{q}_{D,D} \end{bmatrix}_{D \times D}
&= \begin{bmatrix} W^{q}_1 & W^{q}_2 & \ldots & W^{q}_H \end{bmatrix} \in \mathbb{R}^{D \times D}
\end{aligned}
$$

where each $W^{q}_h$ is a matrix with dimensions
$D \times \frac{D}{H} = D \times d_q$.

In other words, if the embedding dimension is 512 and there are 8 heads, the
original $W^q$ matrix is of size $512 \times 512$, and we can decompose it into
8 matrices of size $512 \times 64$, each forming a column of the original
matrix.

---

Side note: if users wanna see jacobian like block matrics:

We can represent the matrix in blocks by grouping its elements. Here's an
example that might suit your purpose:

$$
\begin{aligned}
W_q &=  \begin{bmatrix}
B^{q}_{1,1} & B^{q}_{1,2} & \ldots & B^{q}_{1,H} \\
B^{q}_{2,1} & B^{q}_{2,2} & \ldots & B^{q}_{2,H} \\
\vdots & \vdots & \ddots & \vdots \\
B^{q}_{H,1} & B^{q}_{H,2} & \ldots & B^{q}_{H,H} \\
\end{bmatrix}_{D \times D}
\end{aligned}
$$

Where each block $B^{q}_{i,j}$ is a sub-matrix of size $m \times m$ (assuming
$D$ is divisible by $H$ and $m = \frac{D}{H}$) and can be represented as:

$$
\begin{aligned}
B^{q}_{i,j} =  \begin{bmatrix}
w^{q}_{i \cdot m - m + 1, j \cdot m - m + 1} & \ldots & w^{q}_{i \cdot m - m + 1, j \cdot m} \\
\vdots & \ddots & \vdots \\
w^{q}_{i \cdot m, j \cdot m - m + 1} & \ldots & w^{q}_{i \cdot m, j \cdot m}
\end{bmatrix}_{m \times m}
\end{aligned}
$$

This representation can help visualize the matrix as a composition of smaller
blocks, which might be useful in certain contexts, such as when dealing with
partitioned matrices in numerical computations.

---

We then multiply the embeddings by this large weight matrix:

$$
Q = \mathbf{Z} \cdot W^{q}
$$

Then, we slice the result into $H$ parts:

$$
\begin{aligned}
Q_{h} \in \mathbb{R}^{B \times L \times d_q} &= Q\left[:, :, h \cdot \frac{D}{H} : (h+1) \cdot \frac{D}{H}\right] \\
&= Q\left[:, :, h \cdot d_q : (h+1) \cdot d_q\right] \\
&= \mathbf{Z} \cdot W^{q}_{h}
\end{aligned}
$$

where $W^{q}_{h}$ is the submatrix of $W^{q}$ that corresponds to the $h$-th
head, or in other words, let's say $W^{q}_1$, the first head, it means
subsetting the $W^q$ with rows dimension unchanged (i.e. 512), and taking the
first 64 columns, resulting in a matrix of size $512 \times 64$.

### Approach 2: Separate Weight Matrices Notation (Paper Notation)

Suppose we have $H$ heads and our embedding matrix $\mathbf{Z}$ has dimensions
$B \times L \times D$, where $B$ is the batch size, $L$ is the sequence length,
and $D$ is the embedding dimension.

For each head $h$, we have a weight matrix $W^{q}_{h}$ with dimensions
$D
\times \frac{D}{H} = D \times d_q$, and we apply this transformation to the
embeddings:

$$
Q_{h} \in \mathbb{R}^{B \times L \times d_q} = \mathbf{Z} \cdot W^{q}_{h}
$$

So the confusion arises because in the code implementation we do not see an
explicit definition of the separate weight matrices $W^{q}_{h}$, but they are
implicitly represented within the single large weight matrix $W^{q}$. But
actually you can see from approach 1, the $W^q$ is just a concatenation of all
the $W^{q}_{h}$, so it's just a different way of representing the same thing.

## is the FFN in encoder just a MLP layer

Yes, the Feed-Forward Network (FFN) in the Transformer's encoder is essentially
a Multi-Layer Perceptron (MLP) layer. It typically consists of two fully
connected layers, with a non-linear activation function (usually ReLU) applied
after the first layer.

Here's the general structure of the FFN in the Transformer's encoder:

1. **First Linear Layer:** The input is passed through a fully connected linear
   layer with weight matrix $W_1$ and bias $b_1$.
2. **Activation Function:** A non-linear activation function (such as ReLU) is
   applied to the result of the first linear layer.
3. **Second Linear Layer:** The activated output is then passed through another
   fully connected linear layer with weight matrix $W_2$ and bias $b_2$.
4. **Optional Dropout:** Some implementations might include dropout for
   regularization after one or both of the linear layers.

The mathematical expression for this process would look something like:

$$
\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2
$$

Where $x$ is the input to the FFN, and $W_1$, $W_2$, $b_1$, and $b_2$ are
learnable parameters.

So, the FFN in the Transformer's encoder is effectively a specific form of a
Multi-Layer Perceptron with two layers, with the goal of learning position-wise
transformations of the input.

## All the Whys?

<https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html>

### What are the different Subspaces?

See <https://www.youtube.com/watch?v=UPtG_38Oq8o> around 25min...

Change X to Z.

To explain the idea that the keys and queries matrices act as linear
transformations to enhance embeddings for attention, let's break it down
step-by-step.

1. **Background Context**: When working with the transformer architecture, or
   any architecture that utilizes the attention mechanism, we begin with input
   embeddings. These embeddings are vectors that represent the information
   (usually words or subwords) that we want the model to process.

1. **Assumptions**:

- Let's assume we have a sequence of tokens, represented by the embedding matrix
  $\mathbf{X} \in \mathbb{R}^{L \times D}$, where $L$ is the sequence length and
  $D$ is the embedding dimension.
- We want to obtain the query, key, and value matrices from these embeddings.
  Each of these matrices is produced by multiplying the embeddings with their
  respective weight matrices: $\mathbf{W^Q}$, $\mathbf{W^K}$, and
  $\mathbf{W^V}$.

1. **Step-by-Step Transformation**:

- **Query Matrix**: To obtain the query matrix $\mathbf{Q}$, we perform the
  following linear transformation:

  $$
  \mathbf{Q} = \mathbf{X} \mathbf{W^Q}
  $$

  Here, $\mathbf{W^Q} \in \mathbb{R}^{D \times D'}$ is the weight matrix for the
  queries, and $D'$ is the dimensionality of the query vectors. This
  multiplication transforms the embeddings in $\mathbf{X}$ to enhance them for
  the attention mechanism's querying process.

- **Key Matrix**: Similarly, to obtain the key matrix $\mathbf{K}$, we do:

  $$
  \mathbf{K} = \mathbf{X} \mathbf{W^K}
  $$

  Here, $\mathbf{W^K} \in \mathbb{R}^{D \times D'}$ is the weight matrix for the
  keys.

- **Value Matrix**: The value matrix $\mathbf{V}$ is obtained in a similar
  manner:
  $$
  \mathbf{V} = \mathbf{X} \mathbf{W^V}
  $$
  Here, $\mathbf{W^V} \in \mathbb{R}^{D \times D'}$ is the weight matrix for the
  values.

4. **Reasoning**: These linear transformations project the original embeddings
   into a space where the attention mechanism can more effectively compute
   similarities (for queries and keys) and aggregate information (for values).
   The weight matrices $\mathbf{W^Q}$, $\mathbf{W^K}$, and $\mathbf{W^V}$ are
   learned during training to optimize the attention mechanism's performance for
   the given task.

In summary, by applying these transformations, the model can focus on different
aspects of the input data when computing attention scores and aggregating
information, leading to a more powerful and flexible representation.

### Why Softmax?

See <https://www.youtube.com/watch?v=UPtG_38Oq8o> around 16 min.

### Why Scaling by $\sqrt{d_k}$?

- <https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html>
- <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html>

### Why need Positional Encoding?

## References and Further Readings

1. Stanford CS224N Course: [link](https://web.stanford.edu/class/cs224n/)
2. Aman AI Transformers Primer:
   [link](https://aman.ai/primers/ai/transformers/#transformer-core)
3. Implementing a Transformer from Scratch in PyTorch:
   [link](https://www.lesswrong.com/posts/2kyzD5NddfZZ8iuA7/implementing-a-transformer-from-scratch-in-pytorch-a-write)
4. Illustrated Transformer by Jay Alammar:
   [link](http://jalammar.github.io/illustrated-transformer/)
5. Mislav Juric's Transformer from Scratch:
   [link](https://github.com/MislavJuric/transformer-from-scratch/blob/main/layers/MultiHeadAttention.py)
6. Peter Bloem's Blog on Transformers:
   [link](https://peterbloem.nl/blog/transformers)
7. Transformer Family Explained by Lilian Weng:
   [link](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
8. D2L AI Book - Multihead Attention:
   [link](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html)
9. NLP Course at NTU:
   [link](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/self_v7.pdf)
10. Attention Is All You Need (Original Transformer Paper):
    [link](https://arxiv.org/pdf/1706.03762.pdf)
11. Harvard NLP - Attention in Transformers:
    [link](https://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking)
12. Annotated Transformer:
    [link](http://nlp.seas.harvard.edu/annotated-transformer/)
13. LabML AI - MultiHead Attention:
    [link](https://nn.labml.ai/transformers/mha.html)
14. NTU Speech and Language Processing Course:
    [link](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
15. Google Colab - Self-Attention Example:
    [link](https://colab.research.google.com/drive/1u-610KA-urqfJjDH5O0pecwfP--V9DQs?usp=sharing#scrollTo=iXZ5B0EKJGs8)
16. Self-Attention from Scratch by Sebastian Raschka:
    [link](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
17. UvA DL Course - Transformers and MHAttention:
    [link](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
18. Simple Attention-based Text Prediction Model:
    [link](https://datascience.stackexchange.com/questions/94205/a-simple-attention-based-text-prediction-model-from-scratch-using-pytorch)
19. The AI Summer - Self-Attention Explanation:
    [link](https://theaisummer.com/self-attention/#how-multi-head-attention-works-in-detail)
