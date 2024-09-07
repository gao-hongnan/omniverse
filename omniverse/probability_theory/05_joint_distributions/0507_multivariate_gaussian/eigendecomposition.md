# Eigendecomposition and Covariance Matrix

```{prf:definition} Eigenvalue and Eigenvector
:label: def:eigenvalue

Assume that

$$
\lambda \in \mathbb{F} .
$$

Let $V \in \mathbb{R}^{D}$ be a vector space over a field $\mathbb{F}$. Let

$$
T: V \rightarrow V
$$

be a linear operator. A nonzero vector $\mathbf{v}$ in $V$ is called an eigenvector of $T$ corresponding to the eigenvalue $\lambda \in \mathbb{F}$ of $T$ if:

$$
T(\mathbf{v}) \in \operatorname{span}\{\mathbf{v}\} \Rightarrow T(\mathbf{v})=\lambda \mathbf{v}
$$

In linear algebra, an **eigenvector** or characteristic vector of a linear transformation is a non-zero vector that only changes by an overall scale when that linear transformation is applied to it. More formally, if $T$ is a linear transformation from a vector space $V$ over a field $\mathbb{F}$ into itself and $\mathbf{v}$ is a vector in $V$ that is not the zero vector, then $\mathbf{v}$ is an eigenvector of $T$ if $T(\mathbf{v})$ is a scalar multiple of $\mathbf{v}$.

In matrix form, for an $D \times D$ matrix $\boldsymbol{A}$ in $M_D(\mathbb{F})$, a nonzero column vector $\mathbf{v}$ in $F_c^D$ is called an eigenvector of $\boldsymbol{A}$ corresponding to the eigenvalue $\lambda \in \mathbb{F}$ of $\boldsymbol{A}$ if:

$$
\boldsymbol{A} \mathbf{v}=\lambda \mathbf{v}
$$
```

```{prf:theorem} Equivalent Conditions for $\lambda$ to be an Eigenvalue
:label: thm:equivalent_conditions_for_lambda_to_be_an_eigenvalue

There are a number of equivalent conditions for $\lambda$ to be an eigenvalue:
- There exists $\boldsymbol{u} \neq 0$ such that $\boldsymbol{A} \boldsymbol{u}=\lambda \boldsymbol{u}$;
- There exists $\boldsymbol{u} \neq 0$ such that $(\boldsymbol{A}-\lambda \boldsymbol{I}) \boldsymbol{u}=\mathbf{0}$
- $(\boldsymbol{A}-\lambda \boldsymbol{I})$ is not invertible;
- $\operatorname{det}(\boldsymbol{A}-\lambda \boldsymbol{I})=0$.
```

```{prf:theorem} Eigenvalues of a Symmetric Matrix
:label: thm:eigenvalues_of_a_symmetric_matrix

If $\boldsymbol{A}$ is symmetric, then all the eigenvalues are real.

Note in the context of probability theory, the covariance matrix is always symmetric!
```

```{prf:definition} Matrix Representation of a Eigenvalue and Eigenvector
:label: def:matrix_representation_of_a_eigenvalue_and_eigenvector

Given a matrix $\boldsymbol{A} \in \mathbb{R}^{D \times D}$, let $\boldsymbol{u} \in \mathbb{R}^{D}$ be an eigenvector of $\boldsymbol{A}$ corresponding to the eigenvalue $\lambda \in \mathbb{R}$.

If we construct a matrix $\boldsymbol{U} \in \mathbb{R}^{D \times D}$ whose columns are the eigenvectors of $\boldsymbol{A}$, then $\boldsymbol{U}^T \boldsymbol{A} \boldsymbol{U}=\boldsymbol{\Lambda}$, and a matrix $\boldsymbol{\Lambda} \in \mathbb{R}^{D \times D}$ whose diagonal elements are the eigenvalues of $\boldsymbol{A}$.

Then we can write

$$
\underbrace{\begin{bmatrix} \boldsymbol{a}_1 & \cdots & \boldsymbol{a}_D \end{bmatrix}}_{\boldsymbol{A}} \underbrace{\begin{bmatrix} \boldsymbol{u}_1 & \cdots & \boldsymbol{u}_D \end{bmatrix}}_{\boldsymbol{U}} =  \underbrace{\begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_D \end{bmatrix}}_{\boldsymbol{\Lambda}} \underbrace{\begin{bmatrix} \boldsymbol{u}_1 & \cdots & \boldsymbol{u}_D \end{bmatrix}}_{\boldsymbol{U}}
$$

Note by the equivalent definitions of eigenvalues and eigenvectors, we see that if all the eigenvalues are distinct,
then the matrix $\boldsymbol{U}$ is invertible.

So,

$$
\boldsymbol{A}=\boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^{-1}
$$
```

```{prf:definition} Characteristic Polynomial
:label: def:characteristic_polynomial

Let $\boldsymbol{A} \in \mathbb{R}^{D \times D}$ be a square matrix. The **characteristic polynomial** of $\boldsymbol{A}$ is defined as:

$$
\operatorname{char}(\boldsymbol{A})=\det(\boldsymbol{A}-\lambda \boldsymbol{I})
$$

where $\boldsymbol{I}$ is the identity matrix. The eigenvalues of $\boldsymbol{A}$ are the roots of the characteristic polynomial.
```

```{prf:theorem} Eigenvalues are roots of the characteristic polynomial
:label: thm:eigenvalues_are_roots_of_the_characteristic_polynomial

Let $\boldsymbol{A} \in \mathbb{R}^{D \times D}$ be a square matrix. The eigenvalues of $\boldsymbol{A}$ are the roots of the characteristic polynomial.

In other words, let the characteristic polynomial of $\boldsymbol{A}$ be:

$$
f(\lambda)=\det(\boldsymbol{A}-\lambda \boldsymbol{I})
$$

Then $f(\lambda)=0$ has $D$ roots, which are the eigenvalues of $\boldsymbol{A}$.

Note, however, the eigenvalues are not necessarily distinct, and therefore the
roots of the characteristic polynomial are not necessarily distinct.
```

```{prf:definition} Diagonalizable Matrix
:label: def:diagonalizable_matrix

1. Let $V$ be an $D$-dimensional vector
    space over a field $\mathbb{F}$. A linear operator

    $$
    \begin{equation*}
    T : V \rightarrow V
    \end{equation*}
    $$

    is **diagonalizable** if the representation matrix
    $[T]_B$ relative to some basis $B$ of $V$ is a diagonal matrix in $M_D(\mathbb{F})$.


2. A square matrix $\boldsymbol{A} \in \mathbb{F}^{D \times D}$ is **diagonalizable** if $\boldsymbol{A}$ is similar to a diagonal matrix in $\mathbb{F}^{D \times D}$.
```


```{prf:definition} Eigendecomposition
:label: def:eigendecomposition

If $\boldsymbol{A} \in \mathbb{R}^{D \times D}$ is **symmetric**, then there exists a diagonal matrix $\boldsymbol{\Lambda} \in \mathbb{R}^{D \times D}$ whose diagonal elements are the eigenvalues of $\boldsymbol{A}$, and there exists $\boldsymbol{U} \in \mathbb{R}^{D \times D}$ such that $\boldsymbol{U}^T \boldsymbol{U}=\boldsymbol{I}$ and $\boldsymbol{A}=\boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^T$.

Recall that

$$
\boldsymbol{A}=\boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^{-1}
$$

and since $\boldsymbol{U}^T \boldsymbol{U}=\boldsymbol{I}$, we have $\boldsymbol{U}^{-1}=\boldsymbol{U}^T$.

$$
\boldsymbol{A}=\boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^T
$$

All the goodies is because of the assumption that $\boldsymbol{A}$ is symmetric.

Since now $\boldsymbol{U}$ is orthogonal, we can replace the symbol to $\boldsymbol{Q}$ to indicate that $\boldsymbol{Q}$ is an orthogonal matrix.

$$
\boldsymbol{A}=\boldsymbol{Q} \boldsymbol{\Lambda} \boldsymbol{Q}^T
$$

$$
\underbrace{\begin{bmatrix} \boldsymbol{a}_1 & \boldsymbol{a}_2 & \cdots & \boldsymbol{a}_D \end{bmatrix}}_{\boldsymbol{A}} = \underbrace{\begin{bmatrix} \boldsymbol{q}_1 & \boldsymbol{q}_2 & \cdots & \boldsymbol{q}_D \end{bmatrix}}_{\boldsymbol{Q}} \underbrace{\begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_D \end{bmatrix}}_{\boldsymbol{\Lambda}} \underbrace{\begin{bmatrix} \boldsymbol{q}_1^T \\ \boldsymbol{q}_2^T \\ \vdots \\ \boldsymbol{q}_D^T \end{bmatrix}}_{\boldsymbol{Q}^T}
$$
```

```{prf:theorem} Orthornormal Basis
:label: thm:orthornormal_basis

Given a set of vectors $\{\boldsymbol{q}_1, \boldsymbol{q}_2, \cdots, \boldsymbol{q}_D\}$, if $\boldsymbol{q}_i^T \boldsymbol{q}_j=0$ for $i \neq j$, and $\boldsymbol{q}_i^T \boldsymbol{q}_i=1$ for $i=1,2,\cdots,D$, then $\{\boldsymbol{q}_1, \boldsymbol{q}_2, \cdots, \boldsymbol{q}_D\}$ is an **orthornormal basis**.

In other words, $\{\boldsymbol{q}_d\}_{d=1}^D$ is orthonormal means it is a basis of any vector $\boldsymbol{x} \in \mathbb{R}^D$

$$
\boldsymbol{x} = \sum_{d=1}^D \alpha_d \boldsymbol{q}_d
$$

where $\alpha_d$ is the **basis coefficient** of $\boldsymbol{q}_d$.
```

