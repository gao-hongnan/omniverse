# Notations

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

## Dimensions and Indexing

This section outlines the common dimensions and indexing conventions utilized in
the Transformer model.

-   $\mathcal{B}$: The minibatch size.
-   $D$: Embedding dimension. In the original Transformer paper, this is
    represented as $d_{\text{model}}$.
    -   $d$: Index within the embedding vector, where $0 \leq d < D$.
-   $T$: Sequence length.
    -   $t$: Positional index of a token within the sequence, where
        $0 \leq t < T$.
-   $V$: Size of the vocabulary.
    -   $j$: Index of a word in the vocabulary, where $0 \leq j < V$.

## General Notations

### Elementwise and Vectorwise Operations

Element-wise operations like dropout or activation functions are applied to each
element of a tensor independently. For example, applying the ReLU activation
function to a tensor $\mathbf{X} \in \mathbb{R}^{B \times T \times D}$ results
in a tensor of the same shape, where the ReLU function is applied to each
element of $\mathbf{X}$ independently (i.e. you can think of it as applying the
ReLU a total of $B \times T \times D$ times).

For vector-wise operations, the operation is applied to each vector along a
specific _dimension_ or _axis_ of the tensor. For example, applying layer
normalization to a tensor $\mathbf{X} \in \mathbb{R}^{B \times T \times D}$ will
apply the normalization operation to each vector along the feature dimension $D$
independently. This means that the normalization operation is applied to each
vector of size $D$ independently across all batches and sequence positions. You
can then think of the normalization operation as being applied a total of
$B \times T$ times.

### Vocabulary

$\mathcal{V}$: The set of all unique words in the vocabulary, defined as:

$$
\mathcal{V} = \{w_1, w_2, \ldots, w_V\}
$$

where

-   $V$ (denoted as $|\mathcal{V}|$): The size of the vocabulary.
-   $w_j$: A unique word in the vocabulary $\mathcal{V}$, where
    $w_j \in \mathcal{V}$.
-   $j$: The index of a word in $\mathcal{V}$, explicitly defined as
    $0 \leq j \leq V$.

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
-   $w_1 = \text{cat}, w_2 = \text{eat}, w_3 = \text{mouse}, w_4 = \text{dog}, w_5 = \text{chase}, w_6 = \text{cheese}$.
-   $j = 1, 2, \ldots, 6$.

Note: Depending on the transformer model, special tokens (e.g., `[PAD]`,
`[CLS]`, `[BOS]`, `[EOS]`, `[UNK]`, etc.) may also be included in $\mathcal{V}$.

### Input Sequence

The input sequence $\mathbf{x}$ for a GPT model is defined as a sequence of $T$
tokens. Each token in this sequence is typically represented as an integer that
corresponds to a position in the vocabulary set $\mathcal{V}$. The sequence is
represented as:

$$
\mathbf{x} = (x_1, x_2, \ldots, x_T) \in \mathbb{Z}^{1 \times T}
$$

where

-   $T$: Total length of the sequence. It denotes the number of tokens in the
    sequence $\mathbf{x}$.
-   $x_t$: Represents a token at position $t$ within the sequence. Each token
    $x_t$ is an integer where $0 \leq x_t < V$. Here, $V$ is the size of the
    vocabulary, and each integer corresponds to a unique word or symbol in
    $\mathcal{V}$.
-   $t$: The index of a token within the sequence $\mathbf{x}$, where
    $1 \leq t \leq T$.

#### Batched Input Sequences

In practice, GPT models are often trained on batches of sequences to improve
computational efficiency. A batched input is represented as
$\mathbf{x}^{\mathcal{B}}$, where $\mathcal{B}$ denotes the batch size. The
batched input $\mathbf{x}^{\mathcal{B}}$ can be visualized as a matrix:

$$
\mathbf{x}^{\mathcal{B}} = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,T} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,T} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{\mathcal{B},1} & x_{\mathcal{B},2} & \cdots & x_{\mathcal{B},T}
\end{bmatrix} \in \mathbb{Z}^{\mathcal{B} \times T}
$$

In this matrix:

-   Each row corresponds to a sequence in the batch.
-   Each column corresponds to a token position across all sequences in the
    batch.
-   $x_{b,t}$ refers to the token at position $t$ in sequence $b$, with
    $1 \leq b \leq \mathcal{B}$ and $1 \leq t \leq T$.
-   $T$ and $V$ are as defined previously.

### Token to Index, and Index to Token Mappings

#### String-to-Index Mapping

$$
f_{\text{stoi}} : \mathcal{V} \to \{0, 1, \ldots, V-1\}
$$

-   **Function**: $f_{\text{stoi}}$
-   **Domain**: $\mathcal{V}$, the set of all tokens in the vocabulary.
-   **Codomain**: $\{0, 1, \ldots, V-1\}$, where $V$ is the size of the
    vocabulary.
-   **Purpose**: This function maps each token (word) from the vocabulary to a
    unique index. For a token $w \in \mathcal{V}$, the value
    $f_{\text{stoi}}(w) = j$ indicates that the token $w$ corresponds to the
    $j$-th position in the vocabulary $\mathcal{V}$.
-   **Example**: If $\mathcal{V} = \{\text{cat}, \text{dog}, \text{mouse}\}$ and
    $V = 3$, then $f_{\text{stoi}}(\text{cat}) = 0$,
    $f_{\text{stoi}}(\text{dog}) = 1$, and $f_{\text{stoi}}(\text{mouse}) = 2$.

#### Index-to-String Mapping

$$
f_{\text{itos}} : \{0, 1, \ldots, V-1\} \to \mathcal{V}
$$

-   **Function**: $f_{\text{itos}}$
-   **Domain**: $\{0, 1, \ldots, V-1\}$
-   **Codomain**: $\mathcal{V}$, the set of all tokens in the vocabulary.
-   **Purpose**: This function maps each index back to its corresponding token
    (word) in the vocabulary. For an index $j$, the value
    $f_{\text{itos}}(j) = w$ indicates that the index $j$ corresponds to the
    token $w$ in the vocabulary $\mathcal{V}$.
-   **Example**: Continuing the previous example,
    $f_{\text{itos}}(0) = \text{cat}$, $f_{\text{itos}}(1) = \text{dog}$, and
    $f_{\text{itos}}(2) = \text{mouse}$.

### One-Hot Representation of Input Sequence $\mathbf{x}$

The one-hot representation of the input sequence $\mathbf{x}$ is denoted as
$\mathbf{X}^{\text{ohe}}$. This representation converts each token in the
sequence to a one-hot encoded vector, where each vector has a length equal to
the size of the vocabulary $V$.

#### Definition

The one-hot encoded matrix $\mathbf{X}^{\text{ohe}}$ is defined as:

$$
\mathbf{X}^{\text{ohe}} = \begin{bmatrix}
o_{1,1} & o_{1,2} & \cdots & o_{1,V} \\
o_{2,1} & o_{2,2} & \cdots & o_{2,V} \\
\vdots  & \vdots  & \ddots & \vdots  \\
o_{T,1} & o_{T,2} & \cdots & o_{T,V}
\end{bmatrix} \in \{0, 1\}^{T \times V}
$$

where:

-   $T$: Total length of the sequence $\mathbf{x}$.
-   $V$: Size of the vocabulary $\mathcal{V}$.
-   $o_{t,j}$: Element of the one-hot encoded matrix $\mathbf{X}^{\text{ohe}}$
    at row $t$ and column $j$.

In addition, we have:

-   $\mathbf{X}^{\text{ohe}}$ is a $T \times V$ matrix.
-   Elements of $\mathbf{X}^{\text{ohe}}$ are binary, i.e., they belong to
    $\{0, 1\}$.
-   The row vector $\mathbf{o}_{t, :}$ represents the one-hot encoded vector for
    the token at position $t$ in the sequence $\mathbf{x}$.

#### One-Hot Encoding Process

For each token $x_t$ at position $t$ in the sequence $\mathbf{x}$
($1 \leq t \leq T$), the corresponding row vector $\mathbf{o}_{t, :}$ in
$\mathbf{X}^{\text{ohe}}$ is defined as:

$$
\mathbf{o}_{t, j} = \begin{cases}
1 & \text{if } f_{\text{stoi}}(x_t) = j-1\\
0 & \text{otherwise}
\end{cases}
$$

for $j = 1, 2, \ldots, V$.

Here, $f_{\text{stoi}}(x_t)$ maps the token $x_t$ to its index $j-1$ in the
vocabulary $\mathcal{V}$, the $j-1$ is because zero-based indexing used in
python (where $0 \leq j-1 < V$). Each row $\mathbf{o}_{t, :}$ in
$\mathbf{X}^{\text{ohe}}$ contains a single '1' at the column $j$ corresponding
to the vocabulary index of $x_t$, and '0's elsewhere.

```{prf:example} Example
:label: gpt-notations-one-hot-example

For example, if the vocabulary
$\mathcal{V} = \{\text{cat}, \text{dog}, \text{mouse}\}$ and the sequence
$\mathbf{x} = (\text{mouse}, \text{dog})$, then the one-hot encoded matrix
$\mathbf{X}^{\text{ohe}}$ will be:

$$
\mathbf{X}^{\text{ohe}} = \begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix} \in \{0, 1\}^{2 \times 3}
$$

In this example:

-   The sequence length $T = 2$.
-   The vocabulary size $V = 3$.
-   "mouse" corresponds to the third position in the vocabulary, and "dog" to
    the second, which is seen in their respective one-hot vectors.
```

#### Batched

The batched one-hot encoded matrix $\mathbf{X}^{\text{ohe, }\mathcal{B}}$ for a
batch of size $\mathcal{B}$ is defined as a three-dimensional tensor, where each
"slice" (or matrix) along the first dimension corresponds to the one-hot encoded
representation of a sequence in the batch:

$$
\mathbf{X}^{\text{ohe, }\mathcal{B}} = \begin{bmatrix}
\mathbf{X}^{\text{ohe}}_1 \\
\mathbf{X}^{\text{ohe}}_2 \\
\vdots \\
\mathbf{X}^{\text{ohe}}_{\mathcal{B}}
\end{bmatrix} \in \{0, 1\}^{\mathcal{B} \times T \times V}
$$

Here:

-   $\mathcal{B}$: Batch size, the number of sequences processed together.
-   $T$: Length of each sequence, assumed uniform across the batch.
-   $V$: Size of the vocabulary.
-   $\mathbf{X}^{\text{ohe}}_b$: One-hot encoded matrix of the $b^{th}$ sequence
    in the batch.

## Weights And Embeddings

### Matrix Multiplication Primer

See
[source](https://math.stackexchange.com/questions/2063241/matrix-multiplication-notation).

> If $A=(a_{ij})\in M_{mn}(\Bbb F), B=(b_{ij})\in M_{np}(\Bbb F)$ then
> $C=A\times B=(c_{ij})\in M_{mp}(\Bbb F)$. $c_{ij}=\sum_{k=1}^{n} a_{ik}b_{kj}$
> where $i=1,...m, j=1,...p$

Let's take a look at one specific element in the product $C=AB$, namely the
element on position $(i,j)$, i.e. in the $i$th row and $j$th column.

To obtain this element, you:

-   first **multiply** all elements of the _$i$th row_ of the matrix $A$
    _pairwise_ with all the elements of the _$j$th column_ of the matrix $B$;
-   and then you **add** these $n$ products.

You have to repeat this procedure for every element of $C$, but let's zoom in on
that one specific (but arbitrary) element on position $(i,j)$ for now:

$$
\begin{pmatrix}
a_{11} &\ldots  &a_{1n}\\
\vdots& \ddots &\vdots\\
\color{blue}{\mathbf{a_{i1}}} &\color{blue}{\rightarrow}  &\color{blue}{\mathbf{a_{in}}}\\
\vdots&  \ddots &\vdots\\
a_{m1} &\ldots &a_{mn}
\end{pmatrix}
\cdot
\begin{pmatrix}
b_{11}&\ldots &\color{red}{\mathbf{b_{1j}}} &\ldots &b_{1p}\\
\vdots& \ddots &\color{red}{\downarrow} &  \ddots  &\vdots\\
b_{n1}&\ldots &\color{red}{\mathbf{b_{nj}}}&\ldots &b_{np}
\end{pmatrix}
=
\begin{pmatrix}
c_{11}&\ldots& c_{1j} &\ldots &c_{1p}\\
\vdots&  \ddots & & &\vdots\\
c_{i1}& & \color{purple}{\mathbf{c_{ij}}} & &c_{ip}\\
\vdots& &  & \ddots &\vdots\\
c_{m1} &\ldots& c_{mj} &\ldots &c_{mp}
\end{pmatrix}
$$

with element $\color{purple}{\mathbf{c_{ij}}}$ equal to:

$$
\mathbf{\color{purple}{c_{ij}}  =  \color{blue}{a_{i1}} \color{red}{b_{1j}}  + \color{blue}{a_{i2}} \color{red}{b_{2j}}  +  \cdots  + \color{blue}{a_{in}} \color{red}{b_{nj}}}
$$

Now notice that in the sum above, the left outer index is always $i$ ($i$th row
of $A$) and the right outer index is always $j$ ($j$th column of $B$). The inner
indices run from $1$ to $n$ so you can introduce a summation index $k$ and write
this sum compactly using summation notation:

$$
\color{purple}{\mathbf{c_{ij}}}=\sum_{k=1}^{n} \color{blue}{\mathbf{a_{ik}}}\color{red}{\mathbf{b_{kj}}}
$$

The formule above thus gives you the element on position $(i,j)$ in the product
matrix $C=AB$ and therefore completely defines $C$ by letting $i=1,...,m$ and
$j=1,...,p$.

### $\mathbf{X}$: Output of the Embedding Layer

Once the one hot encoding representation $\mathbf{X}^{\text{ohe}}$ is well
defined, we can then pass it as input through our GPT model, in which the first
layer is a embedding lookup table. In the GPT model architecture, the first
layer typically involves mapping the one-hot encoded input vectors into a
lower-dimensional, dense embedding space using the embedding matrix
$\mathbf{W}_e$.

| Matrix Description             | Symbol                     | Dimensions            | Description                                                                              |
| ------------------------------ | -------------------------- | --------------------- | ---------------------------------------------------------------------------------------- |
| One-Hot Encoded Input Matrix   | $\mathbf{X}^{\text{ohe}}$  | $T \times V$          | Each row corresponds to a one-hot encoded vector representing a token in the sequence.   |
| Embedding Matrix               | $\mathbf{W}_e$             | $V \times D$          | Each row is the embedding vector of the corresponding token in the vocabulary.           |
| Embedded Input Matrix          | $\mathbf{X}$               | $T \times D$          | Each row is the embedding vector of the corresponding token in the input sequence.       |
| Embedding Vector for Token $t$ | $\mathbf{X}_t$             | $1 \times D$          | The embedding vector for the token at position $t$ in the input sequence.                |
| Batched Input Tensor           | $\mathbf{X}^{\mathcal{B}}$ | $B \times T \times D$ | A batched tensor containing $B$ input sequences, each sequence is of shape $T \times D$. |

More concretely, we create an embedding matrix $\mathbf{W}_{e}$ of size
$V \times D$, where $V$ is the vocabulary size, $D$ is the dimensions of the
embeddings, we would then matrix multiply $\mathbf{X}^{\text{ohe}}$ with
$\mathbf{W}_{e}$ to get the output tensor $\mathbf{X}$.

$$
\mathbf{X} = \mathbf{X}^{\text{ohe}} \cdot \mathbf{W}_{e}
$$

#### Definition

The embedding matrix $\mathbf{W}_{e}$ is structured as follows:

$$
\begin{aligned}
\mathbf{W}_e &= \left[\begin{array}{cccc}
w_{1,1} & w_{1,2} & \cdots & w_{1, D} \\
w_{2,1} & w_{2,2} & \cdots & w_{2, D} \\
\vdots & \vdots & \ddots & \vdots \\
w_{V, 1} & w_{V, 2} & \cdots & w_{V, D}
\end{array}\right] \in \mathbb{R}^{V \times D}
\end{aligned}
$$

where

-   $\mathbf{w}_j = (w_{j,1}, w_{j,2}, \ldots, w_{j,D}) \in \mathbb{R}^{1 \times D}$:
    -   Each row vector $\mathbf{w}_j$ of the matrix $\mathbf{W}_e$ represents
        the $D$-dimensional embedding vector for the $j$-th token in the
        vocabulary $\mathcal{V}$.
    -   The subscript $j$ ranges from 1 to $V$, indexing the tokens.
-   $V$ is the vocabulary size.
-   $D$ is the hidden embedding dimension.

Here is a visual representation of how each embedding vector is selected through
matrix multiplication:

$$
\begin{aligned}
\mathbf{X}^{\text{ohe}} \cdot \mathbf{W}_{e} &=
\begin{bmatrix}
0 & 1 & \cdots & 0 \\
1 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}_{T \times V}
\cdot
\begin{bmatrix}
w_{1,1} & w_{1,2} & \cdots & w_{1,D} \\
w_{2,1} & w_{2,2} & \cdots & w_{2,D} \\
\vdots & \vdots & \ddots & \vdots \\
w_{V,1} & w_{V,2} & \cdots & w_{V,D}
\end{bmatrix}_{V \times D} \\
&=
\begin{bmatrix}
w_{2,1} & w_{2,2} & \cdots & w_{2,D} \\
w_{1,1} & w_{1,2} & \cdots & w_{1,D} \\
\vdots & \vdots & \ddots & \vdots \\
w_{T,1} & w_{T,2} & \cdots & w_{T,D}
\end{bmatrix}_{T \times D}
\end{aligned}
$$

Each row in the resulting matrix $\mathbf{X}$ is the embedding of the
corresponding token in the input sequence, picked directly from $\mathbf{W}_e$
by the one-hot vectors. In other words, the matrix $\mathbf{W}_e$ can be
visualized as a table where each row corresponds to a token's embedding vector:

$$
\begin{array}{c|cccc}
\text{Token Index} & \text{Dimension 1} & \text{Dimension 2} & \cdots & \text{Dimension } D \\
\hline
1 & w_{1,1} & w_{1,2} & \cdots & w_{1,D} \\
2 & w_{2,1} & w_{2,2} & \cdots & w_{2,D} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
V & w_{V,1} & w_{V,2} & \cdots & w_{V,D} \\
\end{array}
$$

#### Lookup

When the one-hot encoded input matrix $\mathbf{X}^{\text{ohe}}$ multiplies with
the embedding matrix $\mathbf{W}_e$, each row of $\mathbf{X}^{\text{ohe}}$
effectively selects a corresponding row from $\mathbf{W}_e$. This operation
simplifies to row selection because each row of $\mathbf{X}^{\text{ohe}}$
contains exactly one '1' and the rest are '0's.

#### Semantic Representation

Now each row of the output tensor, indexed by $t$, $\mathbf{X}_{t, :}$: is the
$D$ dimensional embedding vector for the token $x_t$ at the $t$-th position in
the sequence. In this context, each token in the sequence is represented by a
$D$ dimensional vector. So, the output tensor $\mathbf{X}$ captures the dense
representation of the sequence. Each token in the sequence is replaced by its
corresponding embedding vector from the embedding matrix $\mathbf{W}_{e}$. As
before, the output tensor $\mathbf{X}$ carries semantic information about the
tokens in the sequence. The closer two vectors are in this embedding space, the
more semantically similar they are.

### $\mathbf{W}_{e}$: Embedding Matrix

The embedding matrix $\mathbf{W}_{e}$ is structured as follows:

$$
\begin{aligned}
\mathbf{W}_e &= \left[\begin{array}{cccc}
w_{1,1} & w_{1,2} & \cdots & w_{1, D} \\
w_{2,1} & w_{2,2} & \cdots & w_{2, D} \\
\vdots & \vdots & \ddots & \vdots \\
w_{V, 1} & w_{V, 2} & \cdots & w_{V, D}
\end{array}\right] \in \mathbb{R}^{V \times D}
\end{aligned}
$$

where

-   $\mathbf{w}_j = (w_{j,1}, w_{j,2}, \ldots, w_{j,D}) \in \mathbb{R}^{1 \times D}$:
    -   Each row vector $\mathbf{w}_j$ of the matrix $\mathbf{W}_e$ represents
        the $D$-dimensional embedding vector for the $j$-th token in the
        vocabulary $\mathcal{V}$.
    -   The subscript $j$ ranges from 1 to $V$, indexing the tokens.
-   $V$ is the vocabulary size.
-   $D$ is the hidden embedding dimension.

### $PE$: Positional Encoding Layer

For a given input matrix $\mathbf{X} \in \mathbb{R}^{T \times D}$, where $T$ is
the sequence length and $D$ is the embedding dimension (denoted as
$d_{\text{model}}$ in typical Transformer literature), the positional encoding
$\operatorname{PE}$ is applied to integrate sequence positional information into
the embeddings. The resultant matrix $\mathbf{X}'$ after applying positional
encoding can be expressed as follows:

$$
\mathbf{X}' = \operatorname{PE}(\mathbf{X}),
$$

where each element of $\mathbf{X}'$, denoted as $x'_{i, j}$, is calculated based
on the sinusoidal function:

$$
x'_{i, j} =
\begin{cases}
\sin\left(\frac{i}{10000^{j/D}}\right) & \text{if } j \mod 2 = 0 \\
\cos\left(\frac{i}{10000^{(j-1)/D}}\right) & \text{otherwise}
\end{cases}
$$

for $i = 1, \ldots, T$ and $j = 1, \ldots, D$.

### $\tilde{\mathbf{X}}$: Output of the Positional Encoding Layer

We can update our original embeddings tensor $\mathbf{X}$ to include positional
information:

$$
\tilde{\mathbf{X}} := \mathbf{X} + \operatorname{PE}(X)
$$

This operation adds the positional encodings to the original embeddings, giving
the final embeddings that are passed to subsequent layers in the Transformer
model.

| Matrix Description                  | Symbol                             | Dimensions            | Description                                                                                                                                                                |
| ----------------------------------- | ---------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Positional Encoding Matrix          | $\mathbf{W}_{p}$                   | $T \times D$          | Matrix with positional encoding vectors for each position in the sequence, computed using sinusoidal functions.                                                            |
| Output of Positional Encoding Layer | $\tilde{\mathbf{X}}$               | $T \times D$          | The resultant embeddings matrix after adding positional encoding $\mathbf{W}_{p}$ to the embedded input matrix $\mathbf{X}$. Each row now includes positional information. |
| Embedding Vector for Token $t$      | $\tilde{\mathbf{X}}_t$             | $1 \times D$          | The token and positional embedding vector for the token at position $t$ in the input sequence.                                                                             |
| Batched Input Tensor                | $\tilde{\mathbf{X}}^{\mathcal{B}}$ | $B \times T \times D$ | A batched tensor containing $B$ input sequences, each sequence is of shape $T \times D$.                                                                                   |

## Layer Normalization

Layer normalization modifies the activations within each layer to have zero mean
and unit variance across the features for each data point in a batch
independently, which helps in stabilizing the learning process. It then applies
a learnable affine transformation to each normalized activation, allowing the
network to scale and shift these values where beneficial.

For a given layer with inputs $\mathbf{Z} \in \mathbb{R}^{B \times T \times D}$
(where $B$ is the batch size, $T$ is the sequence length, and $D$ is the feature
dimension or hidden dimension size), the layer normalization of $\mathbf{Z}$ is
computed as follows:

1. **Mean and Variance Calculation**: Calculate the mean $\mu_t$ and variance
   $\sigma_t^2$ for each feature vector across the feature dimension $D$:

    $$
    \mu_t = \frac{1}{D} \sum_{d=1}^D \mathbf{Z}_{t, d}, \quad \sigma_t^2 = \frac{1}{D} \sum_{d=1}^D (\mathbf{Z}_{t, d} - \mu_t)^2
    $$

    - $\mu_t$ and $\sigma_t^2$ are computed for each token $t$ across all
      batches $B$ and sequence positions $T$, but independently for each batch
      and sequence position.

2. **Normalization**: Normalize the activations for each feature dimension:

    $$
    \hat{\mathbf{Z}}_{t, d} = \frac{\mathbf{Z}_{t, d} - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}
    $$

    - Where $\epsilon$ is a small constant (e.g., $10^{-5}$) added for numerical
      stability.

3. **Affine Transformation**: Apply a learnable affine transformation to each
   normalized feature:

    $$
    \overline{\mathbf{Z}}_{t, d} = \hat{\mathbf{Z}}_{t, d} \cdot \gamma_d + \beta_d
    $$

    - $\gamma_d$ and $\beta_d$ are learnable parameters that scale and shift the
      normalized feature respectively. They are of the same dimensionality $D$
      as the features and are shared across all tokens and batches.

In practice, these operations are implemented vector-wise across the feature
dimension $D$, and can be compactly expressed as:

$$
\overline{\mathbf{Z}}_t = \dfrac{\mathbf{Z}_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}} \odot \gamma + \beta
$$

-   Here, $\odot$ denotes element-wise multiplication, emphasizing that $\gamma$
    and $\beta$ scale and shift each normalized feature dimension identically
    across all tokens and batches.
-   For better understanding, we can calculate in a loop all $T$ rows of
    $\overline{\mathbf{Z}}_t$, and we stack them together to get the final
    output tensor $\overline{\mathbf{Z}}$.

| Input/Output               | Shape                                                                   | Description                                                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathbf{Z}$               | $\mathcal{B} \times T \times D$                                         | The input tensor to the layer normalization operation.                                                                                                                                       |
| $\overline{\mathbf{Z}}$    | $\mathcal{B} \times T \times D$                                         | The output tensor after applying layer normalization. The dimensionality remains the same as the input, only the scale and shift of the activations within each feature vector are adjusted. |
| $\operatorname{LayerNorm}$ | $\mathbb{R}^{B \times T \times D} \to \mathbb{R}^{B \times T \times D}$ | The layer normalization function that takes an input tensor $\mathbf{Z}$ and returns the normalized tensor $\overline{\mathbf{Z}}$ with the same shape.                                      |

## Attention Notations

### Dimensions

| Symbol      | Description                                                                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| $H$         | Number of attention heads.                                                                                                                                                           |
| $h$         | Index of the attention head.                                                                                                                                                         |
| $d_k = D/H$ | Dimension of the keys. In the multi-head attention case, this would typically be $D/H$ where $D$ is the dimensionality of input embeddings and $H$ is the number of attention heads. |
| $d_q = D/H$ | Dimension of the queries. Also usually set equal to $d_k$.                                                                                                                           |
| $d_v = D/H$ | Dimension of the values. Usually set equal to $d_k$.                                                                                                                                 |
| $L$         | Total number of decoder blocks in the GPT architecture.                                                                                                                              |
| $\ell$      | Index of the decoder block, ranging from $1$ to $L$.                                                                                                                                 |

### Query, Key and Values

| **Matrix Description**                            | **Symbol**                          | **Dimensions** | **Description**                                                                                                                                               |
| ------------------------------------------------- | ----------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Generic Query Matrix for All Heads                | $\mathbf{Q}$                        | $T \times D$   | Contains the query representations for all tokens in the sequence using combined weights of all heads.                                                        |
| Generic Key Matrix for All Heads                  | $\mathbf{K}$                        | $T \times D$   | Contains the key representations for all tokens in the sequence using combined weights of all heads.                                                          |
| Generic Value Matrix for All Heads                | $\mathbf{V}$                        | $T \times D$   | Contains the value representations for all tokens in the sequence using combined weights of all heads.                                                        |
| Query Matrix for All Heads in Layer $\ell$        | $\mathbf{Q}^{(\ell)}$               | $T \times D$   | Contains the query representations for all tokens in the sequence using combined weights of all heads in the $\ell$-th layer.                                 |
| Key Matrix for All Heads in Layer $\ell$          | $\mathbf{K}^{(\ell)}$               | $T \times D$   | Contains the key representations for all tokens in the sequence using combined weights of all heads in the $\ell$-th layer.                                   |
| Value Matrix for All Heads in Layer $\ell$        | $\mathbf{V}^{(\ell)}$               | $T \times D$   | Contains the value representations for all tokens in the sequence using combined weights of all heads in the $\ell$-th layer.                                 |
| Query Weight Matrix for All Heads in Layer $\ell$ | $\mathbf{W}^{\mathbf{Q}, (\ell)}$   | $D \times D$   | The query transformation matrix applicable to all heads in the $\ell$-th layer. It transforms the embeddings $\tilde{\mathbf{X}}$ into query representations. |
| Key Weight Matrix for All Heads in Layer $\ell$   | $\mathbf{W}^{\mathbf{K}, (\ell)}$   | $D \times D$   | The key transformation matrix applicable to all heads in the $\ell$-th layer. It transforms the embeddings $\tilde{\mathbf{X}}$ into key representations.     |
| Value Weight Matrix for All Heads in Layer $\ell$ | $\mathbf{W}^{\mathbf{V}, (\ell)}$   | $D \times D$   | The value transformation matrix applicable to all heads in the $\ell$-th layer. It transforms the embeddings $\tilde{\mathbf{X}}$ into value representations. |
| Query Matrix for Head $h$ in Layer $\ell$         | $\mathbf{Q}_h^{(\ell)}$             | $T \times d_q$ | Contains the query representations for all tokens in the sequence specific to head $h$ in the $\ell$-th layer.                                                |
| Key Matrix for Head $h$ in Layer $\ell$           | $\mathbf{K}_h^{(\ell)}$             | $T \times d_k$ | Contains the key representations for all tokens in the sequence specific to head $h$ in the $\ell$-th layer.                                                  |
| Value Matrix for Head $h$ in Layer $\ell$         | $\mathbf{V}_h^{(\ell)}$             | $T \times d_v$ | Contains the value representations for all tokens in the sequence specific to head $h$ in the $\ell$-th layer.                                                |
| Query Weight Matrix for Head $h$ in Layer $\ell$  | $\mathbf{W}_h^{\mathbf{Q}, (\ell)}$ | $D \times d_q$ | Linear transformation matrix for queries in masked attention head $h \in \{1, \ldots, H\}$ of decoder block $\ell \in \{1, \ldots, L\}$.                      |
| Key Weight Matrix for Head $h$ in Layer $\ell$    | $\mathbf{W}_h^{\mathbf{K}, (\ell)}$ | $D \times d_k$ | Linear transformation matrix for keys in masked attention head $h \in \{1, \ldots, H\}$ of decoder block $\ell \in \{1, \ldots, L\}$.                         |
| Value Weight Matrix for Head $h$ in Layer $\ell$  | $\mathbf{W}_h^{\mathbf{V}, (\ell)}$ | $D \times d_v$ | Linear transformation matrix for values in masked attention head $h \in \{1, \ldots, H\}$ of decoder block $\ell \in \{1, \ldots, L\}$.                       |
| Projection Weight Matrix for Layer $\ell$         | $\mathbf{W}^{O, (\ell)}$            | $D \times D$   | The projection matrix used to combine and transform the concatenated outputs from all heads in the $ \ell $-th layer back to the original dimension $D$.      |

We will talk about the relevant shapes in the last section.

### General Attention Mechanism

To calculate the embeddings after attention in the GPT model:

$$
\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^{\top}}{\sqrt{d_k}}\right) \mathbf{V}
$$

### Multi-Head Attention for Layer $\ell$

For the multi-head attention in layer $\ell$ of the Transformer (applicable to
both encoder and decoder in architectures that have both components, but here
tailored for GPT which primarily uses decoder stacks):

$$
\operatorname{MultiHead}^{(\ell)}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{Concat}\left(\operatorname{head}_{\ell, 1}, \operatorname{head}_{\ell, 2}, \cdots, \operatorname{head}_{\ell, H}\right) \mathbf{W}^{O, (\ell)}
$$

where

$$
\operatorname{head}_{\ell, h} = \operatorname{Attention}\left(\mathbf{Q} \mathbf{W}_{h}^{\mathbf{Q}, (\ell)}, \mathbf{K} \mathbf{W}_{h}^{\mathbf{K}, (\ell)}, \mathbf{V} \mathbf{W}_{h}^{\mathbf{V}, (\ell)}\right)
$$

-   $\mathbf{W}_{h}^{\mathbf{Q}, (\ell)}, \mathbf{W}_{h}^{\mathbf{K}, (\ell)}, \mathbf{W}_{h}^{\mathbf{V}, (\ell)}$
    are the weight matrices for queries, keys, and values for the $h$-th head in
    the $\ell$-th layer, respectively.
-   $\mathbf{W}^{O, (\ell)}$ is the output transformation matrix for the
    $\ell$-th layer.

### Masked Multi-Head Attention for Decoder Layer $\ell$

Masked multi-head attention, used in the decoder to ensure that the predictions
for position $t$ can only depend on known outputs at positions less than $t$
(auto-regressive property):

$$
\begin{aligned}
\operatorname{MaskedMultiHead}^{(\ell)}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \operatorname{Concat}\left(\operatorname{head}_{\ell, 1}^{M}, \operatorname{head}_{\ell, 2}^{M}, \cdots, \operatorname{head}_{\ell, H}^{M}\right) \mathbf{W}^{O, M, (\ell)} \\
&= \left( \operatorname{{head}_{\ell, 1}^{M}} \oplus \operatorname{{head}_{\ell, 2}^{M}} \oplus \cdots \oplus \operatorname{{head}_{\ell, H}^{M}} \right) \mathbf{W}^{O, M, (\ell)}
\end{aligned}
$$

where

$$
\operatorname{head}_{\ell, h}^{M} = \operatorname{softmax}\left(\operatorname{Mask}\left(\frac{\left(\mathbf{Q} \mathbf{W}_{h}^{\mathbf{Q}, M, (\ell)}\right)\left(\mathbf{K} \mathbf{W}_{h}^{\mathbf{K}, M, (\ell)}\right)^{\top}}{\sqrt{d_k}}\right)\right)\left(\mathbf{V} \mathbf{W}_{h}^{\mathbf{V}, M, (\ell)}\right)
$$

$$
\operatorname{Mask}(\mathbf{x})_{i, j}= \begin{cases}\mathbf{x}_{i, j} & \text{if } i \geq j \\ -\infty & \text{otherwise}\end{cases}
$$

-   $M$ denotes the masked condition.
-   $\mathbf{W}_{h}^{\mathbf{Q}, M, (\ell)}, \mathbf{W}_{h}^{\mathbf{K}, M, (\ell)}, \mathbf{W}_{h}^{\mathbf{V}, M, (\ell)}$
    are the masked weight matrices for queries, keys, and values for the $h$-th
    head in the $\ell$-th layer, specifically used under the masked condition.
-   $\mathbf{W}^{O, M, (\ell)}$ is the masked output transformation matrix for
    the $\ell$-th layer, ensuring that future tokens do not influence the
    predictions of the current token in an auto-regressive manner.

### Updated Matrix Description Table with Batch and Head Dimensions

To accurately reflect the practical shapes of the Query (Q), Key (K), and Value
(V) matrices in implementations like GPT, where batch processing and multi-head
attention are used, we should adjust the notation to include batch size $B$,
number of heads $H$, and the dimensions $d_k, d_q, d_v$ corresponding to each
head.

| Matrix Description                                                                                                        | Symbol                | Dimensions                                                                                                                                                                | Description                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Query Matrix for All Heads in Layer $\ell$                                                                                | $\mathbf{Q}^{(\ell)}$ | $\mathcal{B} \times T \times D \rightarrow B \times T \times \underset{D}{\underbrace{H \times d_q}} \xrightarrow[]{\text{transpose 1-2}} B \times H \times T \times d_q$ |
| Contains the query representations for all tokens in all sequences of a batch, separated by heads in the $\ell$-th layer. |
| Key Matrix for All Heads in Layer $\ell$                                                                                  | $\mathbf{K}^{(\ell)}$ | $\mathcal{B} \times T \times D \rightarrow B \times T \times \underset{D}{\underbrace{H \times d_k}} \xrightarrow[]{\text{transpose 1-2}} B \times H \times T \times d_k$ | Contains the key representations for all tokens in all sequences of a batch, separated by heads in the $\ell$-th layer.   |
| Value Matrix for All Heads in Layer $\ell$                                                                                | $\mathbf{V}^{(\ell)}$ | $\mathcal{B} \times T \times D \rightarrow B \times T \times \underset{D}{\underbrace{H \times d_v}} \xrightarrow[]{\text{transpose 1-2}} B \times H \times T \times d_v$ | Contains the value representations for all tokens in all sequences of a batch, separated by heads in the $\ell$-th layer. |

## Positionwise Feed-Forward Networks

The term "positionwise feed-forward network" (FFN) in the context of Transformer
models refers to a dense neural network (otherwise known as multilayer
perceptron) that operates on the output of the Multi-Head Attention mechanism.
This component is called "positionwise" because it applies the **same**
feed-forward neural network (FFN) **independently** and **identically** to each
position $t$ in the sequence of length $T$.

### Independent Processing

In the Transformer architecture, after the Multi-Head Attention mechanism
aggregates information from different positions in the sequence based on
attention scores, each element (or position) $t$ in the sequence has an updated
representation. The positionwise FFN then processes each of these updated
representations. However, rather than considering the sequence as a whole or how
elements relate to each other at this stage, the FFN operates on each position
separately. This means that for a sequence of length $T$, the same FFN is
applied $T$ times independently, and by extension, given a batch of sequences,
the FFN is applied $T \times \mathcal{B}$ times, where $\mathcal{B}$ is the
batch size.

### Identical Application

The term "using the same FFN" signifies that the same set of parameters (weights
and biases) of the feed-forward neural network is used for each position in the
sequence. The rationale is that the transformation is consistent across all
sequence positions, so each element is transformed by the same learned function.
This means the weight matrices and bias vectors of the FFN are shared across all
positions in the sequence. In other words, if a sequence has $T=3$
positions/tokens, the weight matrices and bias vectors of the FFN are the same
for all three positions.

### Definition

Typically, a positionwise FFN consists of two linear transformations with a
non-linear activation function in between. The general form can be represented
as follows.

```{prf:definition} Position-wise Feedforward Networks
:label: def-positionwise-ffn-notation

Given an input matrix $\mathbf{Z} \in \mathbb{R}^{T \times D}$, the
position-wise feedforward network computes the output matrix
$\mathbf{Z}^{\prime} \in \mathbb{R}^{T \times D}$ via the following operations:

$$
\mathbf{Z}^{\prime}=\sigma_Z\left(\mathbf{Z} \mathbf{W}^{\text{FF}}_1 + \mathbf{b}^{\text{FF}}_1\right) \mathbf{W}^{\text{FF}}_2 + \mathbf{b}^{\text{FF}}_2
$$

where:

-   $\mathbf{W}^{\text{FF}}_1 \in \mathbb{R}^{D \times d_{\text{ff}}}$ and
    $\mathbf{W}^{\text{FF}}_2 \in \mathbb{R}^{d_{\text{ff}} \times D}$ are
    learnable weight matrices.
-   $\mathbf{b}^{\text{FF}}_1 \in \mathbb{R}^{d_{\text{ff}}}$ and
    $\mathbf{b}^{\text{FF}}_2 \in \mathbb{R}^{D}$ are learnable bias vectors.
-   $\sigma_Z$ is a non-linear activation function, such as the Gaussian Error
    Linear Unit (GELU) or the Rectified Linear Unit (ReLU).
```

### Projection to a Higher Dimension Space

In the Transformer architecture, the dimensionality of the hidden layer in the
positionwise FFN, denoted as $d_{\text{ff}}$, is often chosen to be larger than
the dimensionality of the input and output embeddings, $D$. This means that the
FFN projects the input embeddings into a higher-dimensional space before
projecting them back to the original dimensionality.

The motivation behind this design choice is to allow the model to learn more
complex and expressive representations. By projecting the input embeddings into
a higher-dimensional space, the model capacity is increased, and the FFN can
capture more intricate patterns and relationships among the features. We then
project back ("unembedding") the higher-dimensional representations to the
original dimensionality to maintain the consistency of the model.

In practice, a common choice for the dimensionality of the hidden layer is to
set $d_{\text{ff}}$ to be a multiple of the input and output dimensionality $D$.
For example, in the original Transformer paper {cite}`vaswani2017attention`, the
authors used $d_{\text{ff}} = 4 \times D$.

### Gaussian Error Linear Unit (GELU)

The Gaussian Error Linear Unit (GELU) is a non-linear activation function used
in the context of neural networks, which allows the model to capture more
complex patterns in the data compared to traditional activation functions like
ReLU. The GELU activation function is defined as:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $x$ is the input to the activation function, and $\Phi(x)$ represents the
cumulative distribution function (CDF) of the standard Gaussian distribution.
The GELU function, effectively, models inputs with a non-linear transformation
that weights inputs by their value, with a probabilistic gating mechanism
derived from the Gaussian distribution.

The cumulative distribution function $\Phi(x)$ for a standard Gaussian
distribution is given by:

$$
\Phi(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

where $\text{erf}$ denotes the error function, which is a special function
integral of the Gaussian distribution. Combining these, the GELU function can be
expressed as:

$$
\text{GELU}(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

I will not pretend I have went through the entire paper and motivation of GELU,
but usually, when new and "better" activation functions are proposed, they
usually serve as an alternative to the common activation functions such as ReLU
etc, where they solve some of the problems that the common activation functions
have. From the formulation, we can see that GELU obeys the following properties:

-   **Non-linearity**: GELU introduces non-linearity to the model, a given
    requirement.
-   **Differentiability**: GELU is smooth and differentiable everywhere, which
    is beneficial for gradient-based optimization methods.
-   **Boundedness**: GELU seems to be bounded below by $-0.17$ and not upper
    bounded, but practice we can show there is an upper bound if we normalize
    the input.

```{prf:remark} Approximation of GELU
:label: remark-approx-gelu-notation

To further simplify the GELU function and enhance computational efficiency, an
approximation of the Gaussian CDF is commonly used in practice (extracted from
[Mathematical Analysis and Performance Evaluation of the GELU Activation Function in Deep Learning](https://www.hindawi.com/journals/jmath/2023/4229924/)):

$$
\Phi(\alpha x) \approx \frac{1}{2}\left(1+\tanh \left(\beta\left(\alpha x+\gamma(\alpha x)^3\right)\right)\right),
$$

where $\beta>0$ and $\gamma \in \mathbb{R}$ are constants, selected to minimize
approximation error. Substituting this approximation into the GELU function, we
arrive at the final approximate form of the GELU activation function (Figure 1):

$$
\operatorname{GELU}(x)=0.5 x\left(1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^3\right)\right)\right) .
$$
```

```{prf:definition} GELU Activation Function
:label: def-gelu-notation

For a matrix $\mathbf{Z}$ with elements $\mathbf{Z}_{t d}$ where $t$ indexes the
sequence (from 1 to $T$ ) and $d$ indexes the feature dimension (from 1 to $D$
), the GELU activation is applied **element-wise** to each element
$\mathbf{Z}_{t d}$ independently:

$$
\operatorname{GELU}\left(x_{t d}\right)=x_{t d} \cdot \frac{1}{2}\left[1+\operatorname{erf}\left(\frac{x_{t d}}{\sqrt{2}}\right)\right]
$$
```

```{admonition} References
:class: seealso

-   [Mathematical Analysis and Performance Evaluation of the GELU Activation Function in Deep Learning](https://www.hindawi.com/journals/jmath/2023/4229924/)
-   [Gaussian Error Linear Units (GELUs) ](https://arxiv.org/abs/1606.08415)
```

| **Matrix Description**                                 | **Symbol**                                     | **Dimensions**                              | **Description**                                                                                                                      |
| ------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Input to FFN in Layer $\ell$                           | $\mathbf{Z}^{(\ell)}_4$                        | $\mathcal{B} \times T \times D$             | Output from the residual connection that adds the normalized self-attention outputs to the initial input embeddings.                 |
| First Linear Transformation in FFN                     | $\mathbf{Z}^{FF, (\ell)}_1$                    | $\mathcal{B} \times T \times d_{\text{ff}}$ | Applies the first linear transformation to each position's embedding, projecting it to a higher dimensional space ($d_{\text{ff}}$). |
| Activation (e.g., GELU) Applied to First Linear Output | $\sigma\left(\mathbf{Z}^{FF, (\ell)}_1\right)$ | $\mathcal{B} \times T \times d_{\text{ff}}$ | Applies the GELU non-linear activation function to the output of the first linear transformation.                                    |
| Second Linear Transformation in FFN                    | $\mathbf{Z}^{(\ell)}_5$                        | $\mathcal{B} \times T \times D$             | Transforms the activated output back down to the original dimensionality $D$ of the embeddings.                                      |
| **Weights and Biases**                                 |                                                |                                             |                                                                                                                                      |
| Weights for First Linear Transformation                | $\mathbf{W}^{FF, (\ell)}_1$                    | $D \times d_{\text{ff}}$                    | Weights used to transform the input embeddings from dimension $D$ to $d_{\text{ff}}$.                                                |
| Biases for First Linear Transformation                 | $\mathbf{b}^{FF, (\ell)}_1$                    | $d_{\text{ff}}$                             | Biases added to the linearly transformed embeddings in the first FFN layer.                                                          |
| Weights for Second Linear Transformation               | $\mathbf{W}^{FF, (\ell)}_2$                    | $d_{\text{ff}} \times D$                    | Weights used to project the activated embeddings from dimension $d_{\text{ff}}$ back to $D$.                                         |
| Biases for Second Linear Transformation                | $\mathbf{b}^{FF, (\ell)}_2$                    | $D$                                         | Biases added to the output of the second linear transformation in the FFN, shaping it back to the original embedding dimension.      |

## The Training Phase

### Autoregressive Self-Supervised Learning Paradigm

Let $\mathcal{D}$ be the true but unknown distribution of the natural language
space. In the context of unsupervised learning with self-supervision, such as
language modeling, we consider both the inputs and the implicit labels derived
from the same data sequence. Thus, while traditionally we might decompose the
distribution $\mathcal{D}$ of a supervised learning task into input space
$\mathcal{X}$ and label space $\mathcal{Y}$, in this scenario, $\mathcal{X}$ and
$\mathcal{Y}$ are intrinsically linked, because $\mathcal{Y}$ is a shifted
version of $\mathcal{X}$, and so we can consider $\mathcal{D}$ as a distribution
over $\mathcal{X}$ only.

Since $\mathcal{D}$ is a distribution, we also define it as a probability
distribution over $\mathcal{X}$, and we can write it as:

$$
\begin{aligned}
\mathcal{D} &= \mathbb{P}(\mathcal{X} ; \boldsymbol{\Theta}) \\
            &= \mathbb{P}_{\{\mathcal{X} ; \boldsymbol{\Theta}\}}(\mathbf{x})
\end{aligned}
$$

where $\boldsymbol{\Theta}$ is the parameter space that defines the distribution
$\mathbb{P}(\mathcal{X} ; \boldsymbol{\Theta})$ and $\mathbf{x}$ is a sample
from $\mathcal{X}$ generated by the distribution $\mathcal{D}$. It is common to
treat $\mathbf{x}$ as a sequence of tokens (i.e. a sentence is a sequence of
tokens), and we can write $\mathbf{x} = \left(x_1, x_2, \ldots, x_T\right)$,
where $T$ is the length of the sequence.

Given such a sequence $\mathbf{x}$, the joint probability of the sequence can be
factorized into the product of the conditional probabilities of each token in
the sequence via the
[chain rule of probability](<https://en.wikipedia.org/wiki/Chain_rule_(probability)>):

$$
\mathbb{P}(\mathbf{x} ; \boldsymbol{\Theta}) = \prod_{t=1}^T \mathbb{P}(x_t \mid x_1, x_2, \ldots, x_{t-1} ; \boldsymbol{\Theta})
$$

We can do this because natural language are _inherently ordered_. Such
decomposition allows for _tractable sampling_ from and _estimation_ of the
distribution $\mathbb{P}(\mathbf{x} ; \boldsymbol{\Theta})$ as well as any
conditionals in the form of
$\mathbb{P}(x_{t-k}, x_{t-k+1}, \ldots, x_{t} \mid x_{1}, x_{2}, \ldots, x_{t-k-1} ; \boldsymbol{\Theta})$
{cite}`radford2019language`.

To this end, consider a corpus $\mathcal{S}$ with $N$ sequences
$\left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{N}\right\}$,

$$
\mathcal{S} = \left\{\mathbf{x}_{1}, \mathbf{x}_{2}, \ldots, \mathbf{x}_{N}\right\} \underset{\text{i.i.d.}}{\sim} \mathcal{D}
$$

where each sequence $\mathbf{x}_{n}$ is a sequence of tokens that are sampled
$\text{i.i.d.}$ from the distribution $\mathcal{D}$.

Then, we can frame the
[likelihood function](https://gao-hongnan.github.io/gaohn-galaxy/probability_theory/08_estimation_theory/maximum_likelihood_estimation/concept.html)
$\hat{\mathcal{L}}(\cdot)$ as the likelihood of observing the sequences in the
corpus $\mathcal{S}$,

$$
\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right) = \prod_{n=1}^N \mathbb{P}(\mathbf{x}_{n} ; \hat{\boldsymbol{\Theta}})
$$

where $\hat{\boldsymbol{\Theta}}$ is the estimated parameter space that
approximates the true parameter space $\boldsymbol{\Theta}$.

Subsequently, the objective function is now well-defined, to be the maximization
of the likelihood of the sequences in the corpus $\mathcal{S}$,

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}^{*} &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right) \\
                              &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \prod_{n=1}^N \mathbb{P}(\mathbf{x}_{n} ; \hat{\boldsymbol{\Theta}}) \\
                              &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \prod_{n=1}^N \prod_{t=1}^{T_n} \mathbb{P}(x_{n, t} \mid x_{n, 1}, x_{n, 2}, \ldots, x_{n, t-1} ; \hat{\boldsymbol{\Theta}}) \\
\end{aligned}
$$

where $T_n$ is the length of the sequence $\mathbf{x}_{n}$.

Owing to the fact that multiplying many probabilities together can lead to
[numerical instability](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html#numerical-optimization-and-the-negative-log-likelihood)
because the product of many probabilities can be very small, it is common and
necessary to use the log-likelihood as the objective function, because it can be
proven that maximizing the log-likelihood is equivalent to maximizing the
likelihood itself.

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}^{*} &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \log\left(\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)\right) \\
&= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmax}} \sum_{n=1}^N \sum_{t=1}^{T_n} \log \mathbb{P}(x_{n, t} \mid x_{n, 1}, x_{n, 2}, \ldots, x_{n, t-1} ; \hat{\boldsymbol{\Theta}}) \\
\end{aligned}
$$

Furthermore, since we are treating the the loss function as a form of
minimization, we can simply negate the log-likelihood to obtain the negative
log-likelihood as the objective function to be minimized,

$$
\begin{aligned}
\hat{\boldsymbol{\theta}}^{*} &= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmin}} \left(-\log\left(\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)\right)\right) \\
&= \underset{\hat{\boldsymbol{\theta}} \in \boldsymbol{\Theta}}{\text{argmin}} \left(-\sum_{n=1}^N \sum_{t=1}^{T_n} \log \mathbb{P}(x_{n, t} \mid x_{n, 1}, x_{n, 2}, \ldots, x_{n, t-1} ; \hat{\boldsymbol{\Theta}})\right) \\
\end{aligned}
$$

It is worth noting that the objective function is a function of the parameter
space $\hat{\boldsymbol{\Theta}}$, and not the data $\mathcal{S}$, so all
analysis such as convergence and consistency will be with respect to the
parameter space $\hat{\boldsymbol{\Theta}}$.

To this end, we denote the GPT model $\mathcal{G}$ to be an _autoregressive_ and
_self-supervised learning_ model that is trained to maximize the likelihood of
observing all data points $\mathbf{x} \in \mathcal{S}$ via the objective
function $\hat{\mathcal{L}}\left(\mathcal{S} ; \hat{\boldsymbol{\Theta}}\right)$
by learning the conditional probability distribution
$\mathbb{P}(x_t \mid x_{<t} ; \hat{\boldsymbol{\Theta}})$ over the vocabulary
$\mathcal{V}$ of tokens, conditioned on the contextual preciding tokens
$x_{<t} = \left(x_1, x_2, \ldots, x_{t-1}\right)$. We are clear that although
the goal is to model the joint probability distribution of the token sequences,
we can do so by estimating the joint probability distribution via the
conditional probability distributions.

### Corpus and Tokenization

```{admonition} Step 1. Corpus
:class: note

Consider a corpus $\mathcal{S}$ consisting of $N$ sequences, denoted as
${\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N}$, where each sequence
$\mathbf{x} = (x_1, x_2, \ldots, x_T) \in \mathcal{S}$ is a sequence of $T$
tokens. These tokens are sampled i.i.d. from a true, unknown distribution
$\mathcal{D}$:

$$
\mathcal{S}=\left\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\right\} \underset{\text { i.i.d. }}{\sim} \mathcal{D}
$$

Each sequence $\mathbf{x} \in \mathcal{S}$ represents a collection of tokenized
elements (e.g., words or characters), where each token $x_t$ comes from a finite
vocabulary $\mathcal{V}$.
```

```{admonition} Step 2. Vocabulary and Tokenization
:class: note

Let $\mathcal{V} = \{w_1, w_2, \ldots, w_V\}$ be the vocabulary set, where $w_j$
is the $j$-th token in the vocabulary and $V = |\mathcal{V}|$ is the size of the
vocabulary. It is worth noting that it is common to train one's own vocabulary
and tokenizer on the corpus $\mathcal{S}$, but for simplicity, we assume that
the vocabulary set $\mathcal{V}$ is predefined.

Let $\mathcal{X}$ be the set of all possible sequences that can be formed by
concatenating tokens from the vocabulary set $\mathcal{V}$. Each sequence
$\mathbf{x} \in \mathcal{X}$ is a finite sequence of tokens, and the length of
each sequence is denoted by $\tau$. Formally:

$$
\mathcal{X} = \bigcup_{\tau=1}^{T} \mathcal{V}^{\tau}
$$

where $\mathcal{V}^\tau$ represents the set of all sequences of length $\tau$
formed by concatenating tokens from $\mathcal{V}$, and $T$ is the maximum
sequence length.

Now, let
$\mathcal{S} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\} \subset \mathcal{X}$
be a corpus of $N$ sequences, where each sequence $\mathbf{x}_n \in \mathcal{X}$
is a finite sequence of tokens from the vocabulary set $\mathcal{V}$.

The tokenizer algorithm $\mathcal{T}$ is a function that operates on individual
sequences $\mathbf{x}_n$ from the corpus $\mathcal{S}$ and maps the tokens to
their corresponding integer indices using the vocabulary set $\mathcal{V}$:

$$
\mathcal{T}: \mathcal{X} \rightarrow \mathbb{N}^{\leq T}
$$

where $\mathbb{N}^{\leq T}$ represents the set of all finite sequences of
natural numbers (non-negative integers) with lengths up to $T$. The output of
$\mathcal{T}$ is a tokenized sequence, which is a finite sequence of integer
indices corresponding to the tokens in the input sequence.

To map the tokens to their corresponding integer indices, we define a bijective
mapping function $f: \mathcal{V} \rightarrow \{1, 2, \ldots, V\}$ such that:

$$
f(w_j) = j, \quad \forall j \in \{1, 2, \ldots, V\}
$$

where $f(w_j)$ represents the integer index assigned to the token
$w_j \in \mathcal{V}$.

Given a sequence $\mathbf{x} = (x_1, x_2, \ldots, x_\tau) \in \mathcal{X}$,
where $\tau \leq T$ is the length of the sequence, the tokenizer algorithm
$\mathcal{T}$ maps each token $x_t$ to its corresponding integer index using the
bijective mapping function $f$. The tokenized representation of the sequence
$\mathbf{x}$ can be defined as:

$$
\mathcal{T}(\mathbf{x}) = \left(f(x_1), f(x_2), \ldots, f(x_\tau)\right)
$$

where $f(x_t)$ is the integer index assigned to the token $x_t$ based on $f$.

In the case where a token $x_t$ is not present in the vocabulary set
$\mathcal{V}$, a special token index, such as $f(\text{<UNK>})$, can be assigned
to represent an unknown token.

The tokenizer algorithm $\mathcal{T}$ can be applied to each sequence
$\mathbf{x}_n$ in the corpus $\mathcal{S}$ to obtain the tokenized corpus
$\mathcal{S}^{\mathcal{T}}$:

$$
\mathcal{S}^{\mathcal{T}} = \left\{\mathcal{T}(\mathbf{x}_1), \mathcal{T}(\mathbf{x}_2), \ldots, \mathcal{T}(\mathbf{x}_N)\right\} \subset \mathbb{N}^{\leq T}
$$

where $\mathcal{T}(\mathbf{x}_n)$ is the tokenized representation of the
sequence $\mathbf{x}_n \in \mathcal{S}$.

The tokenized corpus $\mathcal{S}^{\mathcal{T}}$ is a set of sequences, where
each sequence is a finite sequence of integer indices representing the tokens in
the original sequences from the corpus $\mathcal{S}$.
```

### Token Embedding and Positional Encoding

```{admonition} Step 3. One Hot Encoding
:class: note

For each sequence $\mathbf{x} \in \mathcal{S}^{\mathcal{T}}$ in the corpus, we
would apply one hot encoding so that each sample/sequence is transformed to
$\mathbf{X}^{\text{ohe}} \in \{0, 1\}^{T \times V}$ where $V$ is the vocabulary
size and $T$ the pre-defined context window size.

$$
\mathbf{X}^{\text{ohe}} = \begin{bmatrix}
o_{1,1} & o_{1,2} & \cdots & o_{1,V} \\
o_{2,1} & o_{2,2} & \cdots & o_{2,V} \\
\vdots  & \vdots  & \ddots & \vdots  \\
o_{T,1} & o_{T,2} & \cdots & o_{T,V}
\end{bmatrix} \in \{0, 1\}^{T \times V}
$$

Each row $\mathbf{X}^{\text{ohe}}_{t} \in \mathbb{R}^{1 \times V}$ represents
the one-hot encoded representation of the token at position $t$ in the sequence.
```

```{admonition} Step 4. Token Embedding
:class: note

Given the one-hot encoded input
$\mathbf{X}^{\text{ohe}} \in \{0, 1\}^{T \times |\mathcal{V}|}$, where $T$ is
the sequence length and $V = |\mathcal{V}|$ is the vocabulary size, we obtain
the token embedding matrix $\mathbf{X} \in \mathbb{R}^{T \times D}$ by matrix
multiplying $\mathbf{X}^{\text{ohe}}$ with the token embedding weight matrix
$\mathbf{W}_e \in \mathbb{R}^{V \times D}$, where $D$ is the embedding
dimension:

$$
\begin{aligned}
\mathbf{X} &= \mathbf{X}^{\text{ohe}} \operatorname{@} \mathbf{W}_{e} \\
T \times D                      &\leftarrow T \times V \operatorname{@} V \times D  \\
\mathcal{B} \times T \times D   &\leftarrow\mathcal{B} \times T \times V \operatorname{@} V \times D
\end{aligned}
$$
```

```{admonition} Weight Sharing
:class: tip

Note carefully that with the addition of batch dimension $\mathcal{B}$ the
matrix multiplication is still well-defined for such tensor in PyTorch because
we are essentially just performing matrix multiplication in $T \times D$ for
each sequence $\mathbf{X}_b \in \mathbf{X}^{\mathcal{B}}$ with the same weight
matrix $\mathbf{W}_{e}$.

The token embedding weight matrix $\mathbf{W}_e$ with dimensions $V \times D$ is
shared across all sequences in the batch. Each sequence $\mathbf{X}^{(b)}$ in the
batched input tensor $\mathbf{X}^{\mathcal{B}}$ undergoes the same matrix
multiplication with $\mathbf{W}_e$ to obtain the corresponding embedded sequence
representation.

The idea of weight sharing is that the same set of parameters (in this case, the
embedding weights) is used for processing multiple instances of the input
(sequences in the batch). Instead of having separate embedding weights for each
sequence, the same embedding matrix is applied to all sequences. This parameter
sharing allows the model to learn a common representation for the tokens across
different sequences.
```

```{admonition} Step 5. Positional Embedding
:class: note

In addition to the token embeddings, we incorporate positional information into
the input representation to capture the sequential nature of the input
sequences. Let $\operatorname{PE}(\cdot)$ denote the positional encoding
function that maps the token positions to their corresponding positional
embeddings.

Given the token embedding matrix $\mathbf{X} \in \mathbb{R}^{T \times D}$, where
$T$ is the sequence length and $D$ is the embedding dimension, we add the
positional embeddings to obtain the position-aware input representation
$\tilde{\mathbf{X}} \in \mathbb{R}^{T \times D}$:

$$
\begin{aligned}
\tilde{\mathbf{X}} &= \operatorname{PE}(\mathbf{X}) + \mathbf{X} \\
T \times D                      &\leftarrow T \times D \operatorname{+} T \times D  \\
\mathcal{B} \times T \times D   &\leftarrow \mathcal{B} \times T \times D \operatorname{+} \mathcal{B} \times T \times D
\end{aligned}
$$

The positional encoding function $\operatorname{PE}(\cdot)$ can be implemented
in various ways, such as using fixed sinusoidal functions or learned positional
embeddings. For the latter, we can easily replace $\operatorname{PE}(\cdot)$
with a learnable positional embedding layer in the model architecture
($\mathbf{W}_{p}$).
```

```{admonition} Dropout And Elementwise Operation
:class: tip

At this stage, it is common practice to apply a dropout layer
$\operatorname{Dropout}(\cdot)$ to the position-aware input representation
$\tilde{\mathbf{X}}$ (or $\tilde{\mathbf{X}}_{\text{batch}}$ in the case of a
batch). Dropout is a regularization technique that randomly sets a fraction of
the elements in the input tensor to zero during training and is an
**_element-wise_** operation that acts **_independently_** on each element in
the tensor. This means that each element has a fixed probability (usually
denoted as $p$) of being set to zero, regardless of its position or the values
of other elements in the tensor.

Mathematically, for an input tensor $\mathbf{X} \in \mathbb{R}^{T \times D}$,
elementwise dropout can be expressed as:

$$
\begin{aligned}
\mathbf{X}^{\text{dropout}} &= \mathbf{X} \odot \mathbf{M} \\
T \times D &= T \times D \odot T \times D
\end{aligned}
$$

where $\odot$ denotes the elementwise (Hadamard) product, and
$\mathbf{M} \in \{0, 1\}^{T \times D}$ is a binary mask tensor of the same shape
as $\mathbf{X}$. Each element in $\mathbf{M}$ is independently sampled from a
Bernoulli distribution with probability $p$ of being 0 (i.e., dropped) and
probability $1-p$ of being 1 (i.e., retained).
```

### Backbone Architecture

````{admonition} Step 6. Pre-Layer Normalization For Masked Multi-Head Attention
:class: note

Before passing the input through the Multi-Head Attention (MHA) layer, we apply
Layer Normalization to the positionally encoded embeddings $\tilde{\mathbf{X}}$.
This is known as pre-layer Normalization in the more modern GPT architecture (as
opposed to post-layer Normalization, which is applied after the MHA layer).

The Layer Normalization function $\operatorname{LayerNorm}(\cdot)$ is a
**_vectorwise_** operation that operates on the feature dimension $D$ of the
input tensor. It normalizes the activations to have zero mean and unit variance
across the features for each token independently. The vectorwise nature of Layer
Normalization arises from the fact that it computes the mean and standard
deviation along the feature dimension, requiring **aggregation** of information
across the entire feature vector for each token.

Mathematically, for an input tensor $\mathbf{X} \in \mathbb{R}^{T \times D}$,
Layer Normalization is applied independently to each row
$\mathbf{x}_t \in \mathbb{R}^{1 \times D}$, where $t \in \{1, 2, \ldots, T\}$.
The normalization is performed using the following formula:

$$
\operatorname{LayerNorm}(\mathbf{x}_t) = \frac{\mathbf{x}_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}} \odot \gamma + \beta
$$

where $\mu_t \in \mathbb{R}$ and $\sigma_t^2 \in \mathbb{R}$ are the mean and
variance of the features in $\mathbf{x}_t$ (broadcasted), respectively,
$\epsilon$ is a small constant for numerical stability,
$\gamma \in \mathbb{R}^D$ and $\beta \in \mathbb{R}^D$ are learnable affine
parameters (scale and shift), and $\odot$ denotes the elementwise product.

Applying Layer Normalization to the positionally encoded embeddings
$\tilde{\mathbf{X}}$ at layer $\ell$ results in the normalized embeddings
$\mathbf{Z}^{(\ell)}_1$:

$$
\begin{aligned}
\mathbf{Z}^{(\ell)}_1 &= \operatorname{LayerNorm}\left(\tilde{\mathbf{X}}\right) \\
T \times D &\leftarrow T \times D \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

Here, $\mathbf{Z}^{(\ell)}_1$ represents the normalized embeddings at layer
$\ell$, and the index $1$ refers to the first sub-layer/sub-step in the decoder
block.

For the first layer ($\ell = 1$), $\tilde{\mathbf{X}}$ is the output from Step 4
(Positional Embedding). So we have:

$$
\begin{aligned}
\mathbf{Z}^{(1)}_1 &= \operatorname{LayerNorm}\left(\tilde{\mathbf{X}}\right) \\
T \times D &\leftarrow T \times D \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

In code we have:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)  # [1]
    std = x.std(dim=-1, keepdim=True, unbiased=False)  # [2]
    if self.elementwise_affine:
        return self.gamma * (x - mean) / (std + self.eps) + self.beta  # [3]
    return (x - mean) / (std + self.eps)  # [4]
```
````

| **Line** | **Code**                                                        | **Operation Description**                                                                        | **Input Shape**                 | **Output Shape**                | **Notes**                                                                                                                                                                         |
| -------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [1]      | `mean = x.mean(dim=-1, keepdim=True)`                           | Computes the mean of `x` along the last dimension.                                               | $\mathcal{B} \times T \times D$ | $\mathcal{B} \times T \times 1$ | `keepdim=True` ensures the number of dimensions is preserved, facilitating broadcasting in subsequent operations. Mean is computed for each feature vector.                       |
| [2]      | `std = x.std(dim=-1, keepdim=True, unbiased=False)`             | Computes the standard deviation along the last dimension.                                        | $\mathcal{B} \times T \times D$ | $\mathcal{B} \times T \times 1$ | Similar to the mean, `std` is computed per feature vector with unbiased variance estimation disabled (appropriate for normalization purposes).                                    |
| [3]      | `return self.gamma * (x - mean) / (std + self.eps) + self.beta` | Applies the normalization formula with learnable parameters gamma ($\gamma$) and beta ($\beta$). | $\mathcal{B} \times T \times D$ | $\mathcal{B} \times T \times D$ | Element-wise operations are used. $\gamma$ and $\beta$ are of shape $D$, and are broadcasted to match the input shape. This line only executes if `elementwise_affine` is `True`. |
| [4]      | `return (x - mean) / (std + self.eps)`                          | Applies the normalization formula without learnable parameters.                                  | $\mathcal{B} \times T \times D$ | $\mathcal{B} \times T \times D$ | Simple normalization where each element in the feature vector $x$ is normalized by the corresponding mean and standard deviation.                                                 |

```{admonition} Step 7. Masked Multi-Head Self-Attention
:class: note

Given the normalized input embeddings
$\mathbf{Z}^{(\ell)}_1 \in \mathbb{R}^{\mathcal{B} \times T \times D}$ from Step
6 (Pre-Layer Normalization), we apply the masked multi-head self-attention
mechanism to compute the output embeddings $\mathbf{Z}^{(\ell)}_2$, where the
index $2$ denotes the second sub-layer within the $\ell$-th decoder layer
(multi-head attention).

Let $\operatorname{MaskedMultiHead}^{(\ell)}(\cdot)$ denote the masked
multi-head self-attention function at layer $\ell$. The masked multi-head
self-attention operation takes the normalized input embeddings
$\mathbf{Z}^{(\ell)}_1$ as the query, key, and value matrices, and produces the
output embeddings $\mathbf{Z}^{(\ell)}_2$.

For the first layer ($\ell = 1$), the masked multi-head self-attention operation
can be expressed as:

$$
\begin{aligned}
\mathbf{Z}^{(1)}_2 &= \operatorname{MaskedMultiHead}^{(1)}\left(\mathbf{Z}^{(1)}_1, \mathbf{Z}^{(1)}_1, \mathbf{Z}^{(1)}_1\right) \\
\mathcal{B} \times T \times D &\leftarrow \operatorname{MaskedMultiHead}^{(1)}\left(\mathcal{B} \times T \times D, \mathcal{B} \times T \times D, \mathcal{B} \times T \times D\right) \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

Here, $\mathbf{Z}^{(1)}_2 \in \mathbb{R}^{\mathcal{B} \times T \times D}$
represents the output embeddings of the masked multi-head self-attention
operation at layer $1$, and
$\mathbf{Z}^{(1)}_1 \in \mathbb{R}^{\mathcal{B} \times T \times D}$ represents
the normalized input embeddings from Step 6.

The $\operatorname{MaskedMultiHead}^{(\ell)}(\cdot)$ function internally
performs the following steps:

1. Linearly projects the input embeddings $\mathbf{Z}^{(\ell)}_1$ into query,
   key, and value matrices for each attention head.
2. Computes the scaled dot-product attention scores between the query and key
   matrices, and applies the attention mask to prevent attending to future
   tokens.
3. Applies the softmax function to the masked attention scores to obtain the
   attention weights.
4. Multiplies the attention weights with the value matrices to produce the
   output embeddings for each attention head.
5. Concatenates the output embeddings from all attention heads and linearly
   projects them to obtain the final output embeddings $\mathbf{Z}^{(\ell)}_2$.

The specifics of the scaled dot-product attention mechanism and the multi-head
attention operation will be discussed in the next few steps.
```

````{admonition} Step 7.1. Linear Projections, Query, Key, and Value Matrices
:class: note

In the masked multi-head self-attention mechanism, the first step is to linearly
project the normalized input embeddings $\mathbf{Z}^{(\ell)}_1$ into query, key,
and value matrices for each attention head. This step is performed using
learnable weight matrices $\mathbf{W}^{Q, (\ell)}$, $\mathbf{W}^{K, (\ell)}$,
and $\mathbf{W}^{V, (\ell)}$.

Mathematically, the linear projections can be expressed as:

$$
\begin{aligned}
\mathbf{Q}^{(\ell)} &= \mathbf{Z}^{(\ell)}_1 \mathbf{W}^{Q, (\ell)} ,\quad \mathcal{B} \times T \times D \leftarrow \mathcal{B} \times T \times D \times D \\
\mathbf{K}^{(\ell)} &= \mathbf{Z}^{(\ell)}_1 \mathbf{W}^{K, (\ell)} ,\quad \mathcal{B} \times T \times D \leftarrow \mathcal{B} \times T \times D \times D \\
\mathbf{V}^{(\ell)} &= \mathbf{Z}^{(\ell)}_1 \mathbf{W}^{V, (\ell)} ,\quad \mathcal{B} \times T \times D \leftarrow \mathcal{B} \times T \times D \times D
\end{aligned}
$$

where:

-   $\mathbf{Q}^{(\ell)} \in \mathbb{R}^{\mathcal{B} \times T \times D}$ is the
    query matrix for the $\ell$-th decoder layer.
-   $\mathbf{K}^{(\ell)} \in \mathbb{R}^{\mathcal{B} \times T \times D}$ is the
    key matrix for the $\ell$-th decoder layer.
-   $\mathbf{V}^{(\ell)} \in \mathbb{R}^{\mathcal{B} \times T \times D}$ is the
    value matrix for the $\ell$-th decoder layer.
-   $\mathbf{W}^{Q, (\ell)} \in \mathbb{R}^{D \times D}$,
    $\mathbf{W}^{K, (\ell)} \in \mathbb{R}^{D \times D}$, and
    $\mathbf{W}^{V, (\ell)} \in \mathbb{R}^{D \times D}$ are the learnable
    weight matrices that transform the normalized embeddings into queries, keys,
    and values, respectively.
-   Again notice that we are using the same weight matrices for all heads,
    weight/parameters sharing.

The linear projections are performed using matrix multiplication between the
normalized input embeddings $\mathbf{Z}^{(\ell)}_1$ and the corresponding weight
matrices. The resulting query, key, and value matrices have the same shape as
the input embeddings: $\mathcal{B} \times T \times D$.

In the provided code snippet, the linear projections are implemented using the
`torch.nn.Linear` modules `self.W_Q`, `self.W_K`, and `self.W_V`:

```python
Q: torch.Tensor = self.W_Q(z).contiguous()  # Z @ W_Q = [B, T, D] @ [D, D] = [B, T, D]
K: torch.Tensor = self.W_K(z).contiguous()  # Z @ W_K = [B, T, D] @ [D, D] = [B, T, D]
V: torch.Tensor = self.W_V(z).contiguous()  # Z @ W_V = [B, T, D] @ [D, D] = [B, T, D]
```
````

````{admonition} Step 7.2. Reshaping and Transposing Query, Key, and Value Matrices
:class: note

Subsequently, we have already known that instead of for loop to compute each
head, we can compute all heads in parallel using matrix operations. The query,
key, and value matrices are split into $H$ heads, and the attention scores are
computed in parallel. So our aim is simple, we want to reshape the query, key,
and value matrices to include the head dimension, basically splitting the $D$
dimension into $H$ heads. We can denote the reshaping and transposition
operation using tensor index notation which makes it explicit how indices are
permuted and combined:

$$
\begin{aligned}
\mathbf{Q}_{b,t,d} & \rightarrow \mathbf{Q}_{b,t,h,d_q} \quad \text{where } d = h \cdot (D // H) + d_q, \text{ for } h \in [0, H-1] \text{ and } d_q \in [0, D//H-1] \\
\mathbf{Q}_{b,t,h,d_q} & \rightarrow \mathbf{Q}_{b,h,t,d_q} \quad \text{(transpose dimensions)}
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{K}_{b,t,d} & \rightarrow \mathbf{K}_{b,t,h,d_k} \quad \text{where } d = h \cdot (D // H) + d_k, \text{ for } h \in [0, H-1] \text{ and } d_k \in [0, D//H-1] \\
\mathbf{K}_{b,t,h,d_k} & \rightarrow \mathbf{K}_{b,h,t,d_k} \quad \text{(transpose dimensions)}
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{V}_{b,t,d} & \rightarrow \mathbf{V}_{b,t,h,d_v} \quad \text{where } d = h \cdot (D // H) + d_v, \text{ for } h \in [0, H-1] \text{ and } d_v \in [0, D//H-1] \\
\mathbf{V}_{b,t,h,d_v} & \rightarrow \mathbf{V}_{b,h,t,d_v} \quad \text{(transpose dimensions)}
\end{aligned}
$$

To this end, we have reshaped and transposed the query, key, and value matrices
as follows:

$$
\begin{aligned}
\mathbf{Q}^{(\ell)} &\in \mathbb{R}^{\mathcal{B} \times T \times D} \rightarrow \mathbf{Q}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times T \times H \times D // H} \rightarrow \mathbf{Q}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times H \times T \times D // H} \\
\mathbf{K}^{(\ell)}  &\in \mathbb{R}^{\mathcal{B} \times T \times D} \rightarrow \mathbf{K}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times T \times H \times D // H} \rightarrow \mathbf{K}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times H \times T \times D // H} \\
\mathbf{V}^{(\ell)}  &\in \mathbb{R}^{\mathcal{B} \times T \times D} \rightarrow \mathbf{V}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times T \times H \times D // H} \rightarrow \mathbf{V}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times H \times T \times D // H}
\end{aligned}
$$

In code, the reshaping and transposition operations are performed as follows:

```python
Q = Q.view(B, T, self.H, D // self.H).transpose(dim0=1, dim1=2) # [B, T, D] -> [B, T, H, D // H] -> [B, H, T, D//H]
K = K.view(B, T, self.H, D // self.H).transpose(dim0=1, dim1=2)
V = V.view(B, T, self.H, D // self.H).transpose(dim0=1, dim1=2)
```

The `view` operation reshapes the matrices to include the head dimension, and
the `transpose` operation swaps the sequence and head dimensions to obtain the
desired ordering of dimensions.
````

```{admonition} Step 7.3. Scaled Dot-Product Attention and Masking
:class: note

The attention mask matrix $\mathbf{M} \in \{0, -\infty\}^{T \times T}$ is
initially constructed as a lower triangular matrix of ones and zeros:

$$
\mathbf{M} = \begin{bmatrix}
1 & 0 & 0 & \cdots & 0 \\
1 & 1 & 0 & \cdots & 0 \\
1 & 1 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & 1 & 1 & \cdots & 1
\end{bmatrix}
$$

In this matrix, "1" (conceptually) allows attention, and "0" blocks it. This
mask is broadcastable to the attention scores tensor
$\mathcal{B} \times H \times T \times T$ since this is a typical shape for
multi-head self-attention.

However, before applying the mask, the ones in $\mathbf{M}$ are typically
replaced with zeros, and the zeros are replaced with a large negative value
(e.g., $-\infty$) to effectively set the attention weights of the masked
positions to zero after the softmax operation (which is to mimic the
`masked_fill` operation in the code). So, the actual mask matrix used in the
attention mechanism looks like this:

$$
\mathbf{M}_{ij} = \left\{\begin{array}{ll}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{array} \right\} \quad \text{forms a triangular matrix:} \quad \begin{bmatrix}
0 & -\infty & -\infty & \cdots & -\infty \\
0 & 0 & -\infty & \cdots & -\infty \\
0 & 0 & 0 & \cdots & -\infty \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 0
\end{bmatrix}
$$

The masking operation will then be applied elementwise to the attention scores
tensor $\mathbf{A}_{s}^{(\ell)}$, which is first obtained by computing the
scaled dot product between the query matrix $\mathbf{Q}^{(\ell)}$ and the key
matrix $\mathbf{K}^{(\ell)}$:

$$
\begin{aligned}
\mathbf{A}_{s}^{(\ell)} &= \frac{\mathbf{Q}^{(\ell)} (\mathbf{K}^{(\ell)})^T}{\sqrt{D//H}} \\
\mathcal{B} \times H \times T \times T &\leftarrow \frac{\mathcal{B} \times H \times T \times D//H \operatorname{@} \mathcal{B} \times H \times D//H \times T}{\sqrt{D//H}}
\end{aligned}
$$

The masking operation is performed using the elementwise sum ($\oplus$) between
the attention scores tensor
$\mathbf{A}_{s}^{(\ell)} \in \mathbb{R}^{\mathcal{B} \times H \times T \times T}$
and the **_broadcasted_** mask matrix
$\mathbf{M} \in \{0, -\infty\}^{\mathcal{B} \times H \times T \times T}$:

$$
\begin{aligned}
\mathbf{A}_{s}^{M, (\ell)} &= \mathbf{A}_{s}^{(\ell)} \oplus \mathbf{M} \\
\mathcal{B} \times H \times T \times T &\leftarrow \mathcal{B} \times H \times T \times T \oplus \mathcal{B} \times H \times T \times T
\end{aligned}
$$

The elementwise sum ensures that the attention scores corresponding to future
tokens (positions above the diagonal) are added by $-\infty$ to effectively
block attention to those positions and added by $0$ for the rest of the
positions that are allowed to attend to.

Finally, the masked attention scores $\mathbf{A}_{s}^{M, (\ell)}$ are passed
through the softmax function to obtain the attention weights
$\mathbf{A}_{w}^{(\ell)}$:

$$
\begin{aligned}
\mathbf{A}_{w}^{(\ell)} &= \operatorname{softmax}\left(\mathbf{A}_{s}^{M, (\ell)}\right) \\
\mathcal{B} \times H \times T \times T &\leftarrow \operatorname{softmax}\left(\mathcal{B} \times H \times T \times T\right) \\
\mathcal{B} \times H \times T \times T &\leftarrow \mathcal{B} \times H \times T \times T
\end{aligned}
$$

The softmax function is then applied to the masked attention scores tensor in a
**vectorwise** manner. Specifically, the softmax operation is applied
**independently** to each row of the last two dimensions ($T \times T$) for each
batch and head. This ensures that the attention weights for each token position
across the sequence length sum up to 1. The large negative values in the masked
positions ensure that the corresponding attention weights become close to zero
after the softmax operation.

Finally, the context matrix $\mathbf{C}^{(\ell)}$ is obtained by multiplying the
attention weights $\mathbf{A}_{w}^{(\ell)}$ with the value matrix
$\mathbf{V}^{(\ell)}$:

$$
\begin{aligned}
\mathbf{C}^{(\ell)} &= \mathbf{A}_{w}^{(\ell)} \mathbf{V}^{(\ell)} \\
\mathcal{B} \times H \times T \times D//H &\leftarrow \mathcal{B} \times H \times T \times T \times \mathcal{B} \times H \times T \times D//H
\end{aligned}
$$

The resulting context matrix $\mathbf{C}^{(\ell)}$ contains the attended values
for each head in layer $\ell$.

Note $\mathbf{C}^{(\ell)}$ is the context matrix which is the output of the
self-attention mechanism and it contains $\operatorname{head}_{\ell, h}^{M}$ for
each head $h$ in the layer $\ell$.
```

```python
d_q = query.size(dim=-1)

attention_scores = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / torch.sqrt(torch.tensor(d_q).float())         # [B, H, T, d_q] @ [B, H, d_q, T] = [B, H, T, T]
attention_scores = attention_scores.masked_fill(mask == 0, float("-inf")) if mask is not None else attention_scores     # [B, H, T, T]

attention_weights = attention_scores.softmax(dim=-1)        # [B, H, T, T]
attention_weights = self.dropout(attention_weights)         # [B, H, T, T]

context_vector = torch.matmul(attention_weights, value)     # [B, H, T, T] @ [B, H, T, d_v] = [B, H, T, d_v]
```

| **Line** | **Code**                                                                                                              | **Operation Description**                                                                                                                                      | **Input Shape**                                                                                                  | **Output Shape**                           | **Notes**                                                                                                                                                                                                                                                      |
| -------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [1]      | `d_q = query.size(dim=-1)`                                                                                            | Retrieves the dimension of the query vectors.                                                                                                                  | $\mathcal{B} \times H \times T \times d_q$                                                                       | Scalar value                               | The last dimension of the query tensor represents the query vector dimension.                                                                                                                                                                                  |
| [2]      | `attention_scores = torch.matmul(query, key.transpose(dim0=-2, dim1=-1)) / torch.sqrt(torch.tensor(d_q).float())`     | Computes the scaled dot-product attention scores by matrix multiplying the query and key matrices and scaling by the square root of the query dimension.       | Query: $\mathcal{B} \times H \times T \times d_q$<br>Key: $\mathcal{B} \times H \times T \times d_k$             | $\mathcal{B} \times H \times T \times T$   | The key matrix is transposed to align the dimensions for matrix multiplication. The scaling factor helps stabilize gradients during training.                                                                                                                  |
| [3]      | `attention_scores = attention_scores.masked_fill(mask == 0, float("-inf")) if mask is not None else attention_scores` | Applies the attention mask to the attention scores. Positions where the mask is 0 are filled with $-\infty$ to effectively block attention to those positions. | $\mathcal{B} \times H \times T \times T$                                                                         | $\mathcal{B} \times H \times T \times T$   | The `masked_fill` operation is an elementwise operation that replaces the attention scores at masked positions with $-\infty$. This step is skipped if no mask is provided.                                                                                    |
| [4]      | `attention_weights = attention_scores.softmax(dim=-1)`                                                                | Applies the softmax function to the masked attention scores along the last dimension to obtain the attention weights.                                          | $\mathcal{B} \times H \times T \times T$                                                                         | $\mathcal{B} \times H \times T \times T$   | The softmax operation is applied in a vectorwise manner, independently for each row of the last two dimensions ($T \times T$) for each batch and head. This ensures that the attention weights for each token position across the sequence length sum up to 1. |
| [5]      | `attention_weights = self.dropout(attention_weights)`                                                                 | Applies dropout regularization to the attention weights to prevent overfitting.                                                                                | $\mathcal{B} \times H \times T \times T$                                                                         | $\mathcal{B} \times H \times T \times T$   | Dropout randomly sets a fraction of the attention weights to zero during training, which helps improve generalization. This is element-wise operation.                                                                                                         |
| [6]      | `context_vector = torch.matmul(attention_weights, value)`                                                             | Computes the context vector by matrix multiplying the attention weights with the value matrix.                                                                 | Attention Weights: $\mathcal{B} \times H \times T \times T$<br>Value: $\mathcal{B} \times H \times T \times d_v$ | $\mathcal{B} \times H \times T \times d_v$ | The attention weights are used to weight the importance of each token's value vector. The resulting context vector captures the attended information from the input sequence.                                                                                  |

````{admonition} Step 7.4. Concatenation and Projection
:class: note

Recall that the output from the masked multi-head self-attention operation,
denoted as $\mathbf{C}^{(\ell)}$, has a shape of
$\mathcal{B} \times H \times T \times D//H$, where $\mathcal{B}$ is the batch
size, $H$ is the number of attention heads, $T$ is the sequence length, and
$D//H$ is the dimension of each head.

To concatenate the heads and obtain a tensor of shape
$\mathcal{B} \times T \times D$, we first need to transpose the dimensions of
$\mathbf{C}^{(\ell)}$ from $\mathcal{B} \times H \times T \times D//H$ to
$\mathcal{B} \times T \times H \times D//H$ - necessary to concatenate the heads
along the last dimension (feature dimension).

Using tensor index notation or semi einsum notation, we can denote the
transposition operation as follows:

$$
\begin{aligned}
\mathbf{C}^{(\ell)}_{b,h,t,d} & \rightarrow \mathbf{C}^{(\ell)}_{b,t,h,d} \quad \text{(transpose dimensions)}
\end{aligned}
$$

After transposition, the heads are concatenated along the last dimension
(feature dimension) to obtain a tensor of shape $\mathcal{B} \times T \times D$.
The concatenation operation can be expressed using the direct sum notation as
follows:

$$
\begin{aligned}
\mathbf{C}^{(\ell)}_{\text{concat}} &= \bigoplus_{h=0}^{H-1} \mathbf{C}^{(\ell)}_{b,t,h,:} \\
&= \mathbf{C}^{(\ell)}_{b,t,0,:} \oplus \mathbf{C}^{(\ell)}_{b,t,1,:} \oplus \cdots \oplus \mathbf{C}^{(\ell)}_{b,t,H-1,:} \\
&= \operatorname{head}^{\ell}_1 \oplus \operatorname{head}^{\ell}_2 \oplus \cdots \oplus \operatorname{head}^{\ell}_H \\
\end{aligned}
$$

where $\mathbf{C}^{(\ell)}_{b,t,h,:}$ represents the tensor slice corresponding
to the $h$-th head at batch index $b$ and time step $t$, and $\oplus$ denotes
the concatenation operation along the feature dimension.

```{admonition} Direct Sum Notation Is Concatenation
:class: dropdown

In the concatenation of attention heads, the direct sum notation ($\oplus$) is
used to represent the concatenation operation, not elementwise addition. The
direct sum combines the output tensors from each attention head along a specific
dimension (usually the feature dimension). It is a vectorwise operation that
stacks the tensors along the specified dimension, creating a new tensor with an
increased dimension size.
```

The concatenation operation can be summarized as:

$$
\begin{aligned}
\mathbf{C}^{(\ell)} &\in \mathbb{R}^{\mathcal{B} \times T \times H \times D//H} \xrightarrow{\text{concatenate}} \mathbf{C}^{(\ell)}_{\text{concat}} \in \mathbb{R}^{\mathcal{B} \times T \times D}
\end{aligned}
$$

To summarize, the transposition and concatenation operations can be represented
as:

$$
\begin{aligned}
\mathbf{C}^{(\ell)} &\in \mathbb{R}^{\mathcal{B} \times H \times T \times D//H} \rightarrow \mathbf{C}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times T \times H \times D//H} \rightarrow \mathbf{C}^{(\ell)}  \in \mathbb{R}^{\mathcal{B} \times T \times D}
\end{aligned}
$$

Finally, the concatenated tensor is linearly transformed using the projection
matrix $\mathbf{W}^{O, (\ell)}$ to obtain the output $\mathbf{Z}^{(\ell)}_2$:

$$
\begin{aligned}
\mathbf{Z}^{(\ell)}_2 &= \mathbf{C}^{(\ell)}_{\text{concat}} \mathbf{W}^{O, (\ell)} \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D \operatorname{@} D \times D \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

In code, these operations can be implemented as follows:

```python
self.context_vector = self.context_vector.transpose(dim0=1, dim1=2).contiguous().view(B, T, D)  # merge all heads together

projected_context_vector: torch.Tensor = self.resid_dropout(
    self.context_projection(self.context_vector)  # [B, T, D] @ [D, D] = [B, T, D]
)
```

To this end, for layer $\ell=1$, we would have the output tensor
$\mathbf{Z}^{(1)}_2$ - which becomes the input to the next sub-layer within the
same block $\ell$. Optionally, we can apply a projection dropout to the output
tensor $\mathbf{Z}^{(\ell)}_2$ before passing it to the next sub-layer.
````

````{admonition} Step 8. Residual Connection
:class: note

In the Transformer/GPT architecture, residual connections are used to facilitate
the flow of information and gradients throughout the network. The residual
connection in the decoder block is added between the input to the block and the
output of the Masked Multi-Head Attention layer.

For the first decoder block ($\ell = 1$), the residual connection is added
between the positionally encoded embeddings $\tilde{\mathbf{X}}$ and the output
of the Masked Multi-Head Attention layer $\mathbf{Z}^{(1)}_2$:

$$
\begin{aligned}
\mathbf{Z}^{(1)}_{3} &= \tilde{\mathbf{X}} + \text{MaskedMultiHead}^{(1)}\left(\text{LayerNorm}\left(\tilde{\mathbf{X}}\right), \text{LayerNorm}\left(\tilde{\mathbf{X}}\right), \text{LayerNorm}\left(\tilde{\mathbf{X}}\right)\right) \\
&= \tilde{\mathbf{X}} + \text{MaskedMultiHead}^{(1)}\left(\mathbf{Z}^{(1)}_{1}, \mathbf{Z}^{(1)}_{1}, \mathbf{Z}^{(1)}_{1}\right) \\
&= \tilde{\mathbf{X}} + \mathbf{Z}^{(1)}_{2} \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D + \mathcal{B} \times T \times D \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

where $\mathbf{Z}^{(1)}_{3}$ represents the output of the residual connection
for the first decoder block, $\tilde{\mathbf{X}}$ is the positionally encoded
embeddings, and $\mathbf{Z}^{(1)}_{2}$ is the output of the Masked Multi-Head
Attention layer.

For subsequent decoder blocks ($\ell > 1$), the residual connection is added
between the output of the previous decoder block
$\mathbf{Z}^{(\ell-1)}_{\text{out}}$ and the output of the Masked Multi-Head
Attention layer $\mathbf{Z}^{(\ell)}_2$ of the current block:

$$
\mathbf{Z}^{(\ell)}_3 = \mathbf{Z}^{(\ell-1)}_{\text{out}} + \mathbf{Z}^{(\ell)}_2
$$

where $\mathbf{Z}^{(\ell)}_3$ represents the output of the residual connection
for the $\ell$-th decoder block, $\mathbf{Z}^{(\ell-1)}_{\text{out}}$ is the
output of the previous decoder block after the Position-wise Feed-Forward
Network and the second residual connection, and $\mathbf{Z}^{(\ell)}_2$ is the
output of the Masked Multi-Head Attention layer of the current block.

The complete set of equations for the $\ell$-th decoder block ($\ell > 1$) can
be summarized as follows:

$$
\begin{aligned}
\mathbf{Z}^{(\ell)}_1 &= \text{LayerNorm}\left(\mathbf{Z}^{(\ell-1)}_{\text{out}}\right) \\
\mathbf{Z}^{(\ell)}_2 &= \text{MaskedMultiHead}^{(\ell)}\left(\mathbf{Z}^{(\ell)}_1, \mathbf{Z}^{(\ell)}_1, \mathbf{Z}^{(\ell)}_1\right) \\
\mathbf{Z}^{(\ell)}_3 &= \mathbf{Z}^{(\ell-1)}_{\text{out}} + \mathbf{Z}^{(\ell)}_2
\end{aligned}
$$

where $\mathbf{Z}^{(\ell)}_1$ represents the output of the Layer Normalization
step, $\mathbf{Z}^{(\ell)}_2$ represents the output of the Masked Multi-Head
Attention layer, and $\mathbf{Z}^{(\ell)}_3$ represents the output of the
residual connection.

The residual connection allows the model to learn the identity function more
easily, enabling the flow of information and gradients across multiple layers.
By adding the input of the block to the output of the Masked Multi-Head
Attention layer, the model can choose to either learn new information from the
attention mechanism or retain the original input information if it is already
sufficient.

In code, the whole series of operation up till now is simply:

```python
z = z + self.attn(z=self.ln_1(z))
```
````

```{admonition} Step 9. Pre-Layer Normalization For Position-wise Feed-Forward Network
:class: note

After the masked multi-head attention block and the residual connection, the
next step is to apply the position-wise feed-forward network (FFN) to the output
of the self-attention mechanism and the residual block $\mathbf{Z}^{(\ell)}_3$.
However, before applying the FFN, we perform pre-layer normalization on the
input to the FFN.

Mathematically, the pre-layer normalization step for the FFN can be expressed
as:

$$
\begin{aligned}
\mathbf{Z}^{(\ell)}_4 &= \operatorname{LayerNorm}\left(\mathbf{Z}^{(\ell)}_3\right) \\
\mathcal{B} \times T \times D &\leftarrow \operatorname{LayerNorm}\left(\mathcal{B} \times T \times D\right) \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

where $\mathbf{Z}^{(\ell)}_4$ represents the normalized input to the FFN at
layer $\ell$, and $\mathbf{Z}^{(\ell)}_3$ is the output of the residual
connection from the previous step.

As discussed in Step 6, the Layer Normalization function
$\operatorname{LayerNorm}(\cdot)$ is a vectorwise operation that operates on the
feature dimension $D$ of the input tensor. It normalizes the activations to have
zero mean and unit variance across the features for each token independently.
It's important to note that the pre-layer normalization step is applied
independently to each token
$\mathbf{Z}_{3, t}^{(\ell)} \in \mathbf{Z}_3^{(\ell)}$ in the sequence with
shape $T \times D$, where $T$ is the sequence length and $D$ is the feature
dimension, normalizing the features across the feature dimension $D$.

For the first layer ($\ell = 1$), the pre-layer normalization step can be
written as:

$$
\mathbf{Z}^{(1)}_4 = \operatorname{LayerNorm}\left(\mathbf{Z}^{(1)}_3\right)
$$
```

```{admonition} Step 10. Position-wise Feed-Forward Network
:class: note

Given the normalized input $\mathbf{Z}^{(\ell)}_4$ to the Position-wise
Feed-Forward Network (FFN) in layer $\ell$, the FFN applies two linear
transformations with a GELU activation function in between. The operations
within the FFN can be mathematically represented as follows:

$$
\begin{aligned}
\mathbf{Z}^{FF, (\ell)}_1 &= \text{GELU}\left(\mathbf{Z}^{(\ell)}_4 \mathbf{W}^{FF, (\ell)}_1 + \mathbf{b}^{FF, (\ell)}_1\right) \\
\mathcal{B} \times T \times d_{\text{ff}} &\leftarrow \text{GELU}\left(\mathcal{B} \times T \times D \operatorname{@} D \times d_{\text{ff}} + d_{\text{ff}}\right) \\
\mathcal{B} \times T \times d_{\text{ff}} &\leftarrow \mathcal{B} \times T \times d_{\text{ff}} \\
\mathbf{Z}^{(\ell)}_5 &= \mathbf{Z}^{FF, (\ell)}_1 \mathbf{W}^{FF, (\ell)}_2 + \mathbf{b}^{FF, (\ell)}_2 \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times d_{\text{ff}} \operatorname{@} d_{\text{ff}} \times D + D \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

where:

-   $\mathbf{W}^{FF, (\ell)}_1 \in \mathbb{R}^{D \times d_{\text{ff}}}$ and
    $\mathbf{b}^{FF, (\ell)}_1 \in \mathbb{R}^{d_{\text{ff}}}$ are the weights
    and biases of the first linear transformation, respectively.
-   $\mathbf{W}^{FF, (\ell)}_2 \in \mathbb{R}^{d_{\text{ff}} \times D}$ and
    $\mathbf{b}^{FF, (\ell)}_2 \in \mathbb{R}^{D}$ are the weights and biases of
    the second linear transformation, respectively.
-   $d_{\text{ff}}$ is the dimensionality of the hidden layer in the FFN, which
    is typically larger than the input dimensionality $D$.
-   $\operatorname{GELU}(\cdot)$ denotes the Gaussian Error Linear Unit
    activation function.

Note the slight abuse of notation where $\mathbf{Z}^{FF, (\ell)}_1$ is used to
denote the intermediate output of the first linear transformation in the FFN.
This should not be confused with the earlier notation $\mathbf{Z}^{(\ell)}_1$.

For the first layer ($\ell = 1$), the FFN operations can be written as:

$$
\begin{aligned}
\mathbf{Z}^{FF, (1)}_1 &= \operatorname{GELU}\left(\mathbf{Z}^{(1)}_4 \mathbf{W}^{FF, (1)}_1 + \mathbf{b}^{FF, (1)}_1\right) \\
\mathbf{Z}^{(1)}_5 &= \mathbf{Z}^{FF, (1)}_1 \mathbf{W}^{FF, (1)}_2 + \mathbf{b}^{FF, (1)}_2
\end{aligned}
$$
```

```{admonition} Step 11. Residual Connection
:class: note

After obtaining the output $\mathbf{Z}^{(\ell)}_5$ from the Position-wise
Feed-Forward Network (FFN) in layer $\ell$, the final step in the decoder block
is to apply a residual connection.

Mathematically, this step can be represented as follows:

$$
\begin{aligned}
\mathbf{Z}^{(\ell)}_{\text{out}} &= \mathbf{Z}^{(\ell)}_3 + \mathbf{Z}^{(\ell)}_5 \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D + \mathcal{B} \times T \times D \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

where:

-   $\mathbf{Z}^{(\ell)}_3 \in \mathbb{R}^{\mathcal{B} \times T \times D}$ is
    the output of the residual connection from the Masked Multi-Head Attention
    block (Step 8).
-   $\mathbf{Z}^{(\ell)}_5 \in \mathbb{R}^{\mathcal{B} \times T \times D}$ is
    the output from the FFN (Step 10).
-   $\mathbf{Z}^{(\ell)}_{\text{out}} \in \mathbb{R}^{\mathcal{B} \times T \times D}$
    is the output of the decoder block at layer $\ell$ and serves as the input
    to the next decoder block ($\ell + 1$).

The residual connection is performed by adding the output of the Masked
Multi-Head Attention block $\left(\mathbf{Z}^{(\ell)}_3\right)$ and the output
of the FFN $\left(\mathbf{Z}^{(\ell)}_5\right)$. This addition operation is
element-wise, where corresponding elements of the two tensors are added
together.

For the first layer ($\ell = 1$), the residual connection step can be written
as:

$$
\mathbf{Z}^{(1)}_{\text{out}} = \mathbf{Z}^{(1)}_3 + \mathbf{Z}^{(1)}_5
$$

The output of this step, $\mathbf{Z}^{(\ell)}_{\text{out}}$, becomes the input
to the next decoder block.
```

### Iterative Process Through L Decoder Blocks

Let $\mathbf{Z}^{(1)}_{\text{out}}$ be the output of the first decoder block.
This output becomes the input to the next decoder block, and the process
continues iteratively. Each decoder block builds upon the output of the previous
block through a series of mathematical transformations. The subscript notation
$\mathbf{Z}^{(\ell)}_i$ indicates the i-th step output of the $\ell$-th decoder
block.

#### First Decoder Block ($\ell = 1$)

$$
\begin{aligned}
\mathbf{Z}^{(1)}_1 &= \operatorname{LayerNorm}\left(\tilde{\mathbf{X}}\right) & \text{(Initial normalization of inputs)} \\
\mathbf{Z}^{(1)}_2 &= \operatorname{MaskedMultiHead}\left(\mathbf{Z}^{(1)}_1, \mathbf{Z}^{(1)}_1, \mathbf{Z}^{(1)}_1\right) & \text{(Self-attention mechanism)} \\
\mathbf{Z}^{(1)}_3 &= \tilde{\mathbf{X}} + \mathbf{Z}^{(1)}_2 & \text{(Addition of the first residual connection)} \\
\mathbf{Z}^{(1)}_4 &= \operatorname{LayerNorm}\left(\mathbf{Z}^{(1)}_3\right) & \text{(Normalization before FFN)}\\
\mathbf{Z}^{(1)}_5 &= \operatorname{FFN}\left(\mathbf{Z}^{(1)}_4\right) & \text{(Feed-forward network)}\\
\mathbf{Z}^{(1)}_{\text{out}} &= \mathbf{Z}^{(1)}_3 + \mathbf{Z}^{(1)}_5 & \text{(Second residual connection)}
\end{aligned}
$$

#### Subsequent Decoder Blocks ($\ell > 1$)

For each subsequent decoder block $\ell$, the output of the previous block's
final step $\mathbf{Z}^{(\ell-1)}_{\text{out}}$ serves as the input for the
current block's operations.

$$
\begin{aligned}
\mathbf{Z}^{(\ell)}_1 &= \operatorname{LayerNorm}\left(\mathbf{Z}^{(\ell-1)}_{\text{out}}\right) & \text{(Normalization of previous block's output)} \\
\mathbf{Z}^{(\ell)}_2 &= \operatorname{MaskedMultiHead}\left(\mathbf{Z}^{(\ell)}_1, \mathbf{Z}^{(\ell)}_1, \mathbf{Z}^{(\ell)}_1\right) & \text{(Self-attention mechanism)} \\
\mathbf{Z}^{(\ell)}_3 &= \mathbf{Z}^{(\ell-1)}_{\text{out}} + \mathbf{Z}^{(\ell)}_2 & \text{(First residual connection post self-attention)} \\
\mathbf{Z}^{(\ell)}_4 &= \operatorname{LayerNorm}\left(\mathbf{Z}^{(\ell)}_3\right) & \text{(Normalization before FFN)}\\
\mathbf{Z}^{(\ell)}_5 &= \operatorname{FFN}\left(\mathbf{Z}^{(\ell)}_4\right) & \text{(Feed-forward network)}\\
\mathbf{Z}^{(\ell)}_{\text{out}} &= \mathbf{Z}^{(\ell)}_3 + \mathbf{Z}^{(\ell)}_5 & \text{(Second residual connection post FFN)}
\end{aligned}
$$

After processing the input through a total of $L$ decoder blocks, the final
output is denoted as $\mathbf{Z}^{(L)}_{\text{out}}$, which is the output of the
last decoder block. The shape of $\mathbf{Z}^{(L)}_{\text{out}}$ is
$\mathcal{B} \times T \times D$, where $\mathcal{B}$ is the batch size, $T$ is
the sequence length, and $D$ is the hidden dimension.

### Layer Normalization Before Projection

```{admonition} Step 10. Layer Normalization Before Projection
:class: note

The final output of the decoder block $\mathbf{Z}^{(L)}_{\text{out}}$ undergoes
a layer normalization step before being projected to the vocabulary space. This
step is commonly referred to as the "pre-projection layer normalization" or
"final layer normalization" in the context of the Transformer/GPT architecture.

The pre-projection/head layer normalization can be represented as follows:

$$
\begin{aligned}
\mathbf{Z}_{\text{pre-head}} &= \operatorname{LayerNorm}\left(\mathbf{Z}^{(L)}_{\text{out}}\right) \\
\mathcal{B} \times T \times D &\leftarrow \operatorname{LayerNorm}\left(\mathcal{B} \times T \times D\right) \\
\mathcal{B} \times T \times D &\leftarrow \mathcal{B} \times T \times D
\end{aligned}
$$

where:

-   $\mathbf{Z}^{(L)}_{\text{out}} \in \mathbb{R}^{\mathcal{B} \times T \times D}$
    is the output of the last decoder block ($\ell = L$).
-   $\operatorname{LayerNorm}(\cdot)$ denotes the Layer Normalization operation,
    which normalizes the activations across the feature dimension $D$ for each
    token independently.
-   $\mathbf{Z}_{\text{pre-proj}} \in \mathbb{R}^{\mathcal{B} \times T \times D}$
    is the normalized output, which is ready to be projected to the vocabulary
    space.

The usual purpose of applying layer normalization before the projection step is
to stabilize the activations and improve training stability.
```

### Head

The final step in the GPT architecture is to project the normalized output of
the last decoder block, $\mathbf{Z}_{\text{pre-head}}$, to the vocabulary space.
This projection is performed using a linear transformation, where the weights of
the projection layer are denoted as
$\mathbf{W}_{s} \in \mathbb{R}^{D \times V}$. The subscript $s$ indicates that
this is the projection layer before the softmax operation, and $V$ represents
the size of the vocabulary.

Mathematically, the projection operation can be expressed as follows:

$$
\begin{aligned}
\mathbf{Z} &= \mathbf{Z}_{\text{pre-head}} \mathbf{W}_{s} \\
\mathcal{B} \times T \times V &\leftarrow \mathcal{B} \times T \times D \times \operatorname{@} D \times V \\
\mathcal{B} \times T \times V &\leftarrow \mathcal{B} \times T \times V
\end{aligned}
$$

where:

-   $\mathbf{Z}_{\text{pre-head}} \in \mathbb{R}^{\mathcal{B} \times T \times D}$
    is the normalized output from the pre-projection layer normalization step
    (Step 12).
-   $\mathbf{W}_{s} \in \mathbb{R}^{D \times V}$ is the weight matrix of the
    projection layer, which maps the hidden dimension $D$ to the vocabulary size
    $V$.
-   $\mathbf{Z} \in \mathbb{R}^{\mathcal{B} \times T \times V}$ is the resulting
    logits tensor, representing the unnormalized scores for each token in the
    vocabulary at each position in the sequence.

The purpose of the projection layer is to map the hidden representations from
the decoder to the vocabulary space, allowing the model to generate probability
distributions over the vocabulary for each token position. The logits tensor
$\mathbf{Z}$ can be further processed by applying a softmax function to obtain
the final probability distribution for token prediction.

## Table

| Matrix Description                                     | Symbol                             | Dimensions            | Description                                                                                                                                                                          |
| ------------------------------------------------------ | ---------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| One-Hot Encoded Input Matrix                           | $\mathbf{X}^{\text{ohe}}$          | $T \times V$          | Each row corresponds to a one-hot encoded vector representing a token in the sequence.                                                                                               |
| Embedding Matrix                                       | $\mathbf{W}_e$                     | $V \times D$          | Each row is the embedding vector of the corresponding token in the vocabulary.                                                                                                       |
| Embedded Input Matrix                                  | $\mathbf{X}$                       | $T \times D$          | Each row is the embedding vector of the corresponding token in the input sequence.                                                                                                   |
| Embedding Vector for Token $t$                         | $\mathbf{X}_t$                     | $1 \times D$          | The embedding vector for the token at position $t$ in the input sequence.                                                                                                            |
| Batched Input Tensor                                   | $\mathbf{X}^{\mathcal{B}}$         | $B \times T \times D$ | A batched tensor containing $B$ input sequences, each sequence is of shape $T \times D$.                                                                                             |
| Positional Encoding Matrix                             | $\mathbf{W}_{p}$                   | $T \times D$          | Matrix with positional encoding vectors for each position in the sequence, computed using sinusoidal functions.                                                                      |
| Output of Positional Encoding Layer                    | $\tilde{\mathbf{X}}$               | $T \times D$          | The resultant embeddings matrix after adding positional encoding $\mathbf{W}_{p}$ to the embedded input matrix $\mathbf{X}$. Each row now includes positional information.           |
| Embedding Vector for Token $t$                         | $\tilde{\mathbf{X}}_t$             | $1 \times D$          | The token and positional embedding vector for the token at position $t$ in the input sequence.                                                                                       |
| Batched Input Tensor                                   | $\tilde{\mathbf{X}}^{\mathcal{B}}$ | $B \times T \times D$ | A batched tensor containing $B$ input sequences, each sequence is of shape $T \times D$.                                                                                             |
| Batched Embedding Matrix Sequence $b$                  | $\mathbf{X}^{(b)}$                 | $T \times D$          | The token and positional embedding vector for the $b$-th input sequence of the batch.                                                                                                |
| Batched Embedding Vector for Token $t$ in Sequence $b$ | $\mathbf{X}^{(b)}_t$               | $1 \times D$          | The token and positional embedding vector for the token at position $t$ in the $b$-th input sequence of the batch.                                                                   |
| First Layer Normalized Input                           | $\mathbf{Z}^{(1)}_1$               | $B \times T \times D$ | The output of the initial layer normalization applied to $\tilde{\mathbf{X}}^{\mathcal{B}}$, serving as the input to the first decoder block's self-attention mechanism.             |
| First Self-Attention Output                            | $\mathbf{Z}^{(1)}_2$               | $B \times T \times D$ | Output from the first block's self-attention mechanism; processes $\mathbf{Z}^{(1)}_1$ with respect to itself to refine token representations.                                       |
| Output After First Residual Connection                 | $\mathbf{Z}^{(1)}_3$               | $B \times T \times D$ | Resultant tensor after adding the self-attention outputs back to the initial normalized inputs ($\mathbf{Z}^{(1)}_1$), i.e., the input to the first feed-forward network.            |
| Normalized Before Feed-Forward Network (FFN)           | $\mathbf{Z}^{(1)}_4$               | $B \times T \times D$ | Output of applying layer normalization to $\mathbf{Z}^{(1)}_3$, prepping it for processing through the FFN.                                                                          |
| Output of First Feed-Forward Network                   | $\mathbf{Z}^{(1)}_5$               | $B \times T \times D$ | The result of applying the first FFN to $\mathbf{Z}^{(1)}_4$, which involves two linear transformations and a non-linear activation (typically GELU).                                |
| Output After Second Residual Connection                | $\mathbf{Z}^{(1)}_6$               | $B \times T \times D$ | Final output of the first decoder block, which is the sum of $\mathbf{Z}^{(1)}_3$ and $\mathbf{Z}^{(1)}_5$. This output is used as the input to the next decoder block ($\ell = 2$). |
| Subsequent Block Input (Normalized)                    | $\mathbf{Z}^{(\ell)}_1$            | $B \times T \times D$ | For $\ell > 1$, $\mathbf{Z}^{(\ell)}_1$ is the layer normalized output of $\mathbf{Z}^{(\ell-1)}_6$, serving as the input to the self-attention of block $\ell$.                     |
| Subsequent Self-Attention Output                       | $\mathbf{Z}^{(\ell)}_2$            | $B \times T \times D$ | Self-attention output for block $\ell$, refining the input based on the learned attention mechanisms within the block.                                                               |
| Output After First Residual Connection (Block $\ell$)  | $\mathbf{Z}^{(\ell)}_3$            | $B \times T \times D$ | Resultant tensor after adding the self-attention output ($\mathbf{Z}^{(\ell)}_2$) to the normalized input from the previous block's output ($\mathbf{Z}^{(\ell)}_1$).                |
| Normalized Before FFN (Block $\ell$)                   | $\mathbf{Z}^{(\ell)}_4$            | $B \times T \times D$ | Output from applying layer normalization to $\mathbf{Z}^{(\ell)}_3$, which is the input to the FFN of block $\ell$.                                                                  |
| Output of FFN (Block $\ell$)                           | $\mathbf{Z}^{(\ell)}_5$            | $B \times T \times D$ | Output from the FFN for block $\ell$, which includes non-linear processing through two layers of linear transformations and GELU activation.                                         |
| Output After Second Residual Connection (Block $\ell$) | $\mathbf{Z}^{(\ell)}_6$            | $B \times T \times D$ | Final output of block $\ell$, being the addition of $\mathbf{Z}^{(\ell)}_3$ and $\mathbf{Z}^{(\ell)}_5$. This output is used as the input to the next block or as the final output.  |

## References

-   [The Transformer Family v2.0 - Lilian Weng, OpenAI](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
-   [Transformer Explained - Lei Mao, NVIDIA](https://leimao.github.io/blog/Transformer-Explained/)
