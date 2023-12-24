# Notations

```{contents}
:local:
```

## Dimensions and Indexing

This section outlines the common dimensions and indexing conventions utilized in
the Transformer model. For specific notations related to attention mechanisms,
refer to the {ref}`attention-notations` section.

-   $\mathcal{B}$: The minibatch size.
-   $D$: Embedding dimension. In the original Transformer paper, this is
    represented as $d_{\text{model}}$.
    -   $d$: Index within the embedding vector, where $0 \leq d < D$.
-   $L$: Sequence length.
    -   $i$: Positional index of a token within the sequence, where
        $0 \leq i < L$.
-   $V$: Size of the vocabulary.
    -   $j$: Index of a word in the vocabulary, where $0 \leq j < V$.

## General Notations

### Vocabulary

$\mathcal{V}$: The set of all unique words in the vocabulary, defined as:

$$
\mathcal{V} = \{v_1, v_2, \ldots, v_V\}
$$

where

-   $V$ (denoted as $|\mathcal{V}|$): The size of the vocabulary.
-   $v_j$: A unique word in the vocabulary $\mathcal{V}$, where
    $v_j \in \mathcal{V}$.
-   $j$: The index of a word in $\mathcal{V}$, explicitly defined as
    $1 \leq j \leq V$.

For example, consider the following sentences in the training set:

-   "cat eat mouse"
-   "dog chase cat"
-   "mouse eat cheese"

The resulting vocabulary $\mathcal{V}$ is:

$$
\mathcal{V} = \{\text{cat}, \text{eat}, \text{mouse}, \text{dog}, \text{chase}, \text{cheese}\}
$$

where

-   $V = 6$.
-   $v_1 = \text{cat}, v_2 = \text{eat}, v_3 = \text{mouse}, v_4 = \text{dog}, v_5 = \text{chase}, v_6 = \text{cheese}$.
-   $j = 1, 2, \ldots, 6$.

Note: Depending on the transformer model, special tokens (e.g., `[PAD]`,
`[CLS]`, `[BOS]`, `[EOS]`, `[UNK]`, etc.) may also be included in $\mathcal{V}$.

### Input Sequence

$\mathbf{X}$: The input sequence, defined as:

$$
\mathbf{X} = (x_1, x_2, \ldots, x_L)
$$

where

-   $L$: The length of the sequence.
-   $x_i$: A **token** at position $i$ in the sequence, represented as an
    integer in the set $\{0, 1, \ldots, V-1\}$.
-   $i$: The index of a token in $\mathbf{X}$, where $1 \leq i \leq L$.

## Others

-   $f_{\text{stoi}}$: The function mapping a token in the sequence to its index
    in the vocabulary. For a token $x_i$, $f_{\text{stoi}}(x_i) = j$ means the
    token $x_i$ corresponds to the $j$-th word in the vocabulary $\mathcal{V}$.

-   $\mathbf{O}$: one-hot representation of the input sequence $\mathbf{X}$.
    This is a $L \times V$ matrix, where each row represents a token in the
    sequence and each column corresponds to a unique word in the vocabulary
    $\mathcal{V}$.

    $$
    \begin{aligned}
    \mathbf{O} &= \begin{bmatrix} o_{1,1} & o_{1,2} & \cdots & o_{1,V} \\ o_{2,1} & o_{2,2} & \cdots & o_{2,V} \\ \vdots & \vdots & \ddots & \vdots \\ o_{L,1} & o_{L,2} & \cdots & o_{L,V} \end{bmatrix} \in \mathbb{R}^{L \times V} \\
    &= \begin{bmatrix} \text{---} & \mathbf{o}_{1, :} & \text{---} \\ \text{---} & \mathbf{o}_{2, :} & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{o}_{L, :} & \text{---} \end{bmatrix} \in \mathbb{R}^{L \times V}
    \end{aligned}
    $$

    where

    -   $L$: is the sequence length.
    -   $V$: is the vocabulary size.
    -   $o_{i, j}$: is the one-hot encoded element at position $i, j$. For a
        given token $x_i$ at the $i$-th position in the sequence $\mathbf{X}$,
        if $f_{\text{stoi}}(x_i)=j$, then the element at position $j$ in the
        one-hot vector for token $x_i$ is 1, and all other elements are 0.
    -   $\mathbf{o}_{i, :}$: is the one-hot encoded vector for the token $x_i$
        at the $i$-th position in the sequence $\mathbf{X}$. This row form is
        more important than column form.

-   $\mathbf{E}$: is the embedding matrix defined as:

    $$
    \mathbf{E} = \begin{bmatrix} e_{1,1} & e_{1,2} & \cdots & e_{1,D} \\ e_{2,1} & e_{2,2} & \cdots & e_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ e_{V,1} & e_{V,2} & \cdots & e_{V,D} \end{bmatrix} \in \mathbb{R}^{V \times D}
    $$

    where

    -   $V$: is the vocabulary size.
    -   $D$: is the embedding dimension.
    -   $e_{j, d}$: is the embedding element at position $j, d$. For a word
        $v_j$ in the vocabulary $\mathcal{V}$, the corresponding row in
        $\mathbf{E}$ is the embedding vector for that word.

-   $\mathbf{Z}$: is the output tensor of the embedding layer, obtained by
    matrix multiplying $\mathbf{O}$ with $\mathbf{E}$, and it is defined as:

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

    -   $L$: is the sequence length.
    -   $D$: is the embedding dimension.
    -   $z_{i, d}$: is the element at position $i, d$ in the tensor
        $\mathbf{Z}$. For a token $x_i$ at the $i$-th position in the sequence,
        $z_{i, :}$ is the $D$ dimensional embedding vector for that token.
    -   $\mathbf{z}_{i, :}$: is the $D$ dimensional embedding vector for the
        token $x_i$ at the $i$-th position in the sequence.

        In this context, each token in the sequence is represented by a $D$
        dimensional vector. So, the output tensor $\mathbf{Z}$ captures the
        dense representation of the sequence. Each token in the sequence is
        replaced by its corresponding embedding vector from the embedding matrix
        $\mathbf{E}$.

        As before, the output tensor $\mathbf{Z}$ carries semantic information
        about the tokens in the sequence. The closer two vectors are in this
        embedding space, the more semantically similar they are.

-   $\mathbf{P}$: is the positional encoding tensor, created with sinusoidal
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

    -   $L$: is the sequence length.
    -   $D$: is the embedding dimension.
    -   $p_{i, d}$: is the element at position $i, d$ in the tensor
        $\mathbf{P}$.

-   Note that $\mathbf{P}$ is independent of $\mathbf{Z}$, and it's computed
    based on the positional encoding formula used in transformers, which uses
    sinusoidal functions of different frequencies:

-   OVERWRITING $\mathbf{Z}$: After computing the positional encoding tensor
    $\mathbf{P}$, we can update our original embeddings tensor $\mathbf{Z}$ to
    include positional information:

    $$
    \mathbf{Z} = \mathbf{Z} + \mathbf{P}
    $$

    This operation adds the positional encodings to the original embeddings,
    giving the final embeddings that are passed to subsequent layers in the
    Transformer model.

-   Or consider using $\mathbf{Z}^{'}$?

(attention-notations)=

## Attention Notations

-   $H$: Number of attention heads.
    -   $h$: Index of the attention head.
-   $d_k = D/H$: Dimension of the keys. In the multi-head attention case, this
    would typically be $D/H$ where $D$ is the dimensionality of input embeddings
    and $H$ is the number of attention heads.
-   $d_q = D/H$: Dimension of the queries. Also usually set equal to $d_k$.
-   $d_v = D/H$: Dimension of the values. Usually set equal to $d_k$.
-   $\mathbf{W}^q \in \mathbb{R}^{D \times H \cdot d_q = D \times D}$: The query
    weight matrix for all heads. It is used to transform the embeddings
    $\mathbf{Z}$ into query representations.

-   $\mathbf{W}^k \in \mathbb{R}^{D \times H \cdot d_k = D \times D}$: The key
    weight matrix for all heads. It is used to transform the embeddings
    $\mathbf{Z}$ into key representations.

-   $\mathbf{W}^v \in \mathbb{R}^{D \times H \cdot d_v = D \times D}$: The value
    weight matrix for all heads. It is used to transform the embeddings
    $\mathbf{Z}$ into value representations.
-   $\mathbf{W}_{h}^{q} \in \mathbb{R}^{D \times d_q}$: The query weight matrix
    for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$
    into query representations for the $h$-th head.
    -   Important that this matrix collapses to $\mathbf{W}_{1}^q$ when $H=1$
        and has shape $\mathbb{R}^{D \times D}$.
    -   Note that this weight matrix is derived from $W^q$.
-   $\mathbf{W}_{h}^{k} \in \mathbb{R}^{D \times d_k}$: The key weight matrix
    for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$
    into key representations for the $h$-th head.
    -   Important that this matrix collapses to $\mathbf{W}_{1}^k$ when $H=1$
        and has shape $\mathbb{R}^{D \times D}$ since $d_k = D/H = D/1 = D$.
    -   Note that this weight matrix is derived from $W^k$.
-   $\mathbf{W}_{h}^{v} \in \mathbb{R}^{D \times d_v}$: The value weight matrix
    for the $h$-th head. It is used to transform the embeddings $\mathbf{Z}$
    into value representations for the $h$-th head.

    -   Important that this matrix collapses to $\mathbf{W}_{1}^v$ when $H=1$
        and has shape $\mathbb{R}^{D \times D}$.
    -   Note that this weight matrix is derived from $W^v$.

-   $\mathbf{Q} = \mathbf{Z} \mathbf{W}^q \in \mathbb{R}^{L \times D}$: The
    query matrix. It contains the query representations for all the tokens in
    the sequence. This is the matrix that is used to compute the attention
    scores.
    -   Each row of the matrix $\mathbf{Q}$ is a query vector $\mathbf{q}_{i}$
        for the token at position $i$ in the sequence.
-   $\mathbf{Q}_h = \mathbf{Z} \mathbf{W}_h^q \in \mathbb{R}^{L \times d_q}$:
    The query matrix for the $h$-th head. It contains the query representations
    for all the tokens in the sequence. This is the matrix that is used to
    compute the attention scores for the $h$-th head.

-   $\mathbf{K} = \mathbf{Z} \mathbf{W}^k \in \mathbb{R}^{L \times D}$: The key
    matrix. It contains the key representations for all the tokens in the
    sequence. This is the matrix that is used to compute the attention scores.

-   $\mathbf{K}_h = \mathbf{Z} \mathbf{W}_h^k \in \mathbb{R}^{L \times d_k}$:
    The key matrix for the $h$-th head. It contains the key representations for
    all the tokens in the sequence. This is the matrix that is used to compute
    the attention scores for the $h$-th head.

-   $\mathbf{V} = \mathbf{Z} \mathbf{W}^v \in \mathbb{R}^{L \times D}$: The
    value matrix. It contains the value representations for all the tokens in
    the sequence. This is the matrix where we apply the attention scores to
    compute the weighted average of the values.

-   $\mathbf{V}_h = \mathbf{Z} \mathbf{W}_h^v \in \mathbb{R}^{L \times d_v}$:
    The value matrix for the $h$-th head. It contains the value representations
    for all the tokens in the sequence. This is the matrix where we apply the
    attention scores to compute the weighted average of the values for the
    $h$-th head.

-   $\mathbf{q}_{i} = \mathbf{Q}_{i, :} \in \mathbb{R}^{d}$: The query vector
    for the token at position $i$ in the sequence.
-   $\mathbf{k}_{i} = \mathbf{K}_{i, :} \in \mathbb{R}^{d}$: The key vector for
    the token at position $i$ in the sequence.
-   $\mathbf{v}_{i} = \mathbf{V}_{i, :} \in \mathbb{R}^{d}$: The value vector
    for the token at position $i$ in the sequence.
-   $\mathbf{A} \in \mathbb{R}^{L \times L}$: The attention matrix. It contains
    the attention scores for all the tokens in the sequence. It is computed as:

    $$
    \mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right)
    $$

    where

    -   $L$: is the sequence length.
    -   $\mathbf{Q} \in \mathbb{R}^{L \times D}$: is the query matrix.
    -   $\mathbf{K} \in \mathbb{R}^{L \times D}$: is the key matrix.
    -   $\sqrt{d_k}$: is the scaling factor.
    -   $\text{softmax}(\cdot)$: is the softmax function applied row-wise.
    -   More concretely, this is the **self-attention matrix** between an input
        sequence $\mathbf{X} = (x_1, x_2, ..., x_L)$ and itself. Each row in the
        matrix $\mathbf{A}$ is the attention scores for a token in the sequence.
        The attention scores are computed by comparing the query vector for a
        token with the key vectors for all the tokens in the sequence.
    -   For instance, if the input sequence is "cat eat mouse", then the $L=3$,
        and the attention matrix $\mathbf{A}$'s first row is the attention
        scores of the word cat with all other words, (cat & cat, cat & eat, cat
        & mouse). Similarly, the second row is the attention scores of the word
        eat with all other words, (eat & cat, eat & eat, eat & mouse). Lastly,
        the third row is the attention scores of the word mouse with all other
        words, (mouse & cat, mouse & eat, mouse & mouse).

-   $a_{i, j} \in \mathbf{A}$: The attention score between the query $i$ and the
    key $j$ in the sequence (please do not be confused with the $j$ index in
    vocabulary!). It is computed as:

    $$
    a_{i, j} = \text{softmax}\left(\frac{\mathbf{q}_{i} \mathbf{k}_{j}^T}{\sqrt{d_k}}\right)
    $$

    where

    -   $\mathbf{q}_{i} \in \mathbb{R}^{d}$: is the query vector for the $i$-th
        token in the sequence.
    -   $\mathbf{k}_{j} \in \mathbb{R}^{d}$: is the key vector for the $j$-th
        token in the sequence.

-   $f(\cdot)$: Attention function (such as additive attention or scaled
    dot-product attention).

    -   Should we find a better notation?

        The scaled dot-product attention function $f(\cdot)$ can be formulated
        as:

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
        $\text{softmax}(\cdot)$ is applied row-wise. The division by
        $\sqrt{d_k}$ is a scaling factor that helps in training stability.

---

-   $\mathbf{h}_i \in \mathbb{R}^{p_v}$: Output of the $i$-th attention head.

-   $\mathbf{W}_o \in \mathbb{R}^{p_o \times h p_v}$: Output weight matrix, used
    to transform the concatenation of all head outputs.

-   $p_o$: Dimension of the final output after applying the output weight matrix
    $\mathbf{W}_o$.

Let's break this down:

-   $\mathbf{h}_i \in \mathbb{R}^{p_v}$: Output of the $i$-th attention head. It
    is computed as a function $f$ which applies attention (such as additive
    attention or scaled dot-product attention) to the transformed queries, keys
    and values. This function depends on the query $\mathbf{q}$, key
    $\mathbf{k}$, and value $\mathbf{v}$, and the weight matrices
    $\mathbf{W}_i^{(q)}$, $\mathbf{W}_i^{(k)}$, and $\mathbf{W}_i^{(v)}$. The
    dimensions $p_q$, $p_k$, and $p_v$ denote the output dimensions of the
    query, key and value transformations respectively, for the $i$-th head.

-   $\mathbf{W}_i^{(q)} \in \mathbb{R}^{p_q \times d_q}$,
    $\mathbf{W}_i^{(k)} \in \mathbb{R}^{p_k \times d_k}$, and
    $\mathbf{W}_i^{(v)} \in \mathbb{R}^{p_v \times d_v}$: The weight matrices
    for the $i$-th attention head. These are used to transform the query, key,
    and value inputs to the dimensions suitable for the attention mechanism.

-   $f(\cdot)$: This function represents the attention mechanism (like additive
    attention or scaled dot-product attention). It takes as input the
    transformed query, key, and value vectors and produces the output of the
    attention head.

-   $\mathbf{W}_o \in \mathbb{R}^{p_o \times h p_v}$: This is the output weight
    matrix that linearly transforms the concatenation of the outputs from all
    attention heads to produce the final output of the multi-head attention
    mechanism.

-   The expression
    $\mathbf{W}_o\left[\begin{array}{c} \mathbf{h}_1 \\ \vdots \\ \mathbf{h}_h \end{array}\right] \in \mathbb{R}^{p_o}$
    represents the final output of the multi-head attention layer. It's the
    result of applying the linear transformation defined by $\mathbf{W}_o$ to
    the concatenated outputs of all attention heads.
