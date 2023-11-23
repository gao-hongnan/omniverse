## Precursor to Linear Combinations

### Building from Vectors and Scalar Multiplication

Before delving into the concept of linear combinations, it's crucial to
reinforce our understanding of vectors in $\mathbb{R}^D$ and the operation of
scalar-vector multiplication. Recall that a vector $\mathbf{v} \in \mathbb{R}^D$
can be scaled by a scalar $\lambda \in \mathbb{R}$, resulting in a new vector
$\lambda \mathbf{v}$, as discussed in the previous sections.

### Introduction to Vector Spaces and Span

The idea of a vector space is fundamental in linear algebra. A vector space over
a field (such as the real numbers, $\mathbb{R}$) is a collection of objects
(vectors) that can be added together and multiplied by scalars from that field.
An essential concept within vector spaces is the **span** of a set of vectors.
Informally, the span of a set of vectors is the set of all possible vectors that
can be formed by scalar multiplication and vector addition of those vectors.

### Visualizing Span in 2D and 3D

To visualize the concept of span, consider vectors in 2D or 3D spaces. For
instance, two non-parallel vectors in $\mathbb{R}^2$ (the 2D plane) span the
entire plane because any point on the plane can be reached by scaling and adding
these vectors. Similarly, in $\mathbb{R}^3$, three non-coplanar vectors span the
entire 3D space.

### Scalar Multiplication and Span

Scalar multiplication plays a pivotal role in spanning a space with vectors. For
example, a single vector $\mathbf{u} \in \mathbb{R}^2$ can only span a line,
which is all scalar multiples of $\mathbf{u}$. However, adding another
non-collinear vector $\mathbf{v}$ expands the span to cover the entire plane.
Graphical representations of these vectors and their scaled versions can
effectively demonstrate how they span different spaces.

### Toward Linear Combinations

The concepts we've discussed set the stage for understanding linear
combinations. A linear combination of a set of vectors involves forming new
vectors by adding scaled versions of these vectors. Specifically, for vectors
$\mathbf{u}, \mathbf{v} \in \mathbb{R}^D$ and scalars
$\alpha, \beta \in \mathbb{R}$, a linear combination is expressed as
$\alpha\mathbf{u} + \beta\mathbf{v}$. This is a natural extension of what we
have already explored with vector addition and scalar multiplication. In the
upcoming sections, we will formalize this concept and delve into its deeper
implications in linear algebra.
