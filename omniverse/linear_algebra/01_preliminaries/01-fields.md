# Fields

```{contents}
:local:
```

## Learning Objectives

- Form intuition for the concept of a field and its relevance in linear algebra.
- Comprehend the definition and properties of a field in mathematics.
- Recognize different examples of fields and their relevance in various
  contexts.
- Apply the concept of fields to understand the foundation of vector spaces.
- Bridge the gap between abstract mathematical concepts and their practical
  applications in machine learning.

## Introduction

Why is an understanding of fields essential in the study of linear algebra,
especially for machine learning? Fields are not just sets of numbers; they are
sets where operations like addition, subtraction, multiplication, and division
adhere to specific, well-defined rules. These rules are crucial because they
provide a consistent framework in which linear algebra operates. Without them,
linear algebra would be a collection of arbitrary rules and procedures, making
gradient descent, linear regression, embeddings, and other machine learning
algorithms impossible.

In linear algebra, a field can be thought of as a structured "space" — for
instance, the set of real numbers $\mathbb{R}$ — where operations such as
addition and multiplication are not just possible but follow specific,
well-defined rules. These rules ensure mathematical consistency and logic in the
operations performed within this space.

The elements within a field, known as scalars, must adhere to these rules. This
adherence is what distinguishes a field from a mere collection of numbers.

## Definition

A **[field](<https://en.wikipedia.org/wiki/Field_(mathematics)>)** is an ordered
pair of triplets, defined as

$$
( (\mathbb{F}, \oplus, \mathbf{0}), (\mathbb{F}, \otimes, \mathbf{1}) ),
$$

where:

1. $\mathbb{F}$ is a set,
2. $\oplus$ and $\otimes$ are binary operations defined on $\mathbb{F}$ such
   that

   $$
   \begin{aligned}
   & \oplus \colon \mathbb{F}  \otimes \mathbb{F} \to \mathbb{F} \\
   & \otimes \colon \mathbb{F} \otimes \mathbb{F} \to \mathbb{F}
   \end{aligned}
   $$

3. These operations are
   [**well-defined**](<https://en.wikipedia.org/wiki/Closure_(mathematics)>)[^well-defined]
   and satisfy the field axioms (detailed below).

In this representation:

- The first triplet $(\mathbb{F}, \oplus, \mathbf{0})$ encapsulates the addition
  operation in the field, where $\oplus$ is the addition operation, and $0$ is
  the additive identity in $\mathbb{F}$.
- The second triplet $(\mathbb{F}, \otimes, \mathbf{1})$ encapsulates the
  multiplication operation, where $\otimes$ is the multiplication operation, and
  $1$ is the multiplicative identity.

````{prf:definition} Field
:label: linear-algebra-01-preliminaries-field

$\mathbb{F} := \left\{ \mathbb{F}, \oplus, \otimes \right\}$ is a **field**
if and only if the following axioms hold:

```{list-table} Field Axioms
:header-rows: 1
:name: linear-algebra-preliminaries-definition-of-a-field

* - **Property**
  - **Description**
* - **Well Defined (Closure)**
  - For all $a, b \in \mathbb{F}$, we have $a \oplus b \in \mathbb{F}$ and
    $a \otimes b \in \mathbb{F}$.
* - **Commutative Law for Addition**
  - For all $a, b \in \mathbb{F}$, $a \oplus b = b \oplus a$.
* - **Associative Law for Addition**
  - For all $a, b, c \in \mathbb{F}$,
    $(a \oplus b) \oplus c = a \oplus (b \oplus c)$.
* - **Existence of the Additive Identity**
  - There exists $\mathbf{0} \in \mathbb{F}$ such that for all
    $a \in \mathbb{F}$, $\mathbf{0} \oplus a = a$.

    In other words, an additive identity for the set $\mathbb{F}$ is any element
    $e$ such that for any element $x \in \mathbb{F}$, we have
    $e \oplus x = x = x \oplus e$.
    In familiar fields like the Real Numbers $\mathbb{R}$, the additive identity
    is $\mathbf{0}$.
* - **Existence of Additive Inverse**
  - For every $a \in \mathbb{F}$, there exists $b \in \mathbb{F}$ such that
    $a \oplus b = 0$. We call $b$ the additive
    inverse and denote $b$ by $-a$.
* - **Commutative Law for Multiplication**
  - For all $a, b \in \mathbb{F}$, $ab = ba$.
* - **Associative Law for Multiplication**
  - For all $a, b, c \in \mathbb{F}$, $(ab)c = a(bc)$.
* - **Existence of the Multiplicative Identity**
  - There exists $\mathbf{1}\in \mathbb{F}$ such that
    $\mathbf{1} \otimes a = a\otimes \mathbf{1} = a$ for all $a\in \mathbb{F}$.
* - **Existence of Multiplicative Inverse**
  - For every non-zero $a \in \mathbb{F}$, there exists $b \in \mathbb{F}$ such
     that $a \otimes b = 1$. We denote $b$ as $a^{-1}$.
* - **Distributive Law**
  - - $\otimes$ distributes over $\oplus$ on the left: for all $a,b,c\in \mathbb{F}$,
    $a \otimes (b \oplus c) = (a \otimes b) \oplus (a \otimes c)$.
    - $\otimes$ distributes over $\oplus$ on the right: for all $a,b,c\in \mathbb{F}$,
    $(b \oplus c) \otimes a = (b \otimes a) \oplus (c \otimes a)$.
```
````

## Examples

Let's look at some examples of fields as well as non-fields.

### Fields

Fields can consist of:

- The real numbers $\mathbb{R}$.
- The complex numbers $\mathbb{C}$.
- The rational numbers $\mathbb{Q}$.

```{prf:example} Fields on the Real Numbers
:label: linear-algebra-preliminaries-fields-on-the-real-numbers

Here we replace $\oplus$ with $+$ and $\otimes$ with $\times$.

1. **Closure**:

   - **Addition**: If you take any two real numbers, say $2.3$ and $3.7$, their
     sum ($6.0$) is also a real number.
   - **Multiplication**: Similarly, if you multiply these numbers
     ($2.3 \times 3.7$), the product ($8.51$) is also a real number.

2. **Commutativity**:

   - **Addition**: $2.3 + 3.7$ equals $3.7 + 2.3$.
   - **Multiplication**: $2.3 \times 3.7$ equals $3.7 \times 2.3$.

3. **Associativity**:

   - **Addition**: $(2.3 + 3.7) + 1.5$ equals $2.3 + (3.7 + 1.5)$.
   - **Multiplication**: $(2.3 \times 3.7) \times 1.5$ equals
     $2.3 \times (3.7
     \times 1.5)$.

4. **Additive Identity**: The additive identity in $\mathbb{R}$ is $0$. For any
   real number $a$, $a + 0 = a$. For example, $2.3 + 0 = 2.3$.

5. **Multiplicative Identity**: The multiplicative identity in $\mathbb{R}$ is
   $1$. For any real number $a$, $a \times 1 = a$. For example,
   $2.3 \times 1 =
   2.3$.

6. **Additive Inverse**: For every real number, there is an additive inverse (or
   opposite). For $2.3$, the additive inverse is $-2.3$, because
   $2.3 + (-2.3) = 0$.

7. **Multiplicative Inverse**: For every non-zero real number, there is a
   multiplicative inverse (or reciprocal). The multiplicative inverse of $2.3$
   is $\frac{1}{2.3}$ because $2.3 \times \frac{1}{2.3}$ = 1.

8. **Distributive Law**: Multiplication distributes over addition. For example,
   $2 \times (3 + 4)$ equals $(2 \times 3) + (2 \times 4)$.
```

```{admonition} Example is not a Proof
:class: warning

Do not mistake the example above as a form of proof that the real number
system $\mathbb{R}$ is a field. The example only shows that a particular
set of real numbers satisfies the field axioms. To prove that $\mathbb{R}$
is a field, you must show that all real numbers satisfy the field axioms.
```

### Non-Fields

Non-fields include:

- The natural numbers $\mathbb{N}$.
- The integers $\mathbb{Z}$.

Why are these non-fields? They lack **additive** and/or **multiplicative**
inverses.

For example:

- The natural numbers $\mathbb{N}$ do not have additive inverses. For any
  $n \in \mathbb{N}$, there is no $m \in \mathbb{N}$ such that $n + m = 0$.
- The integers $\mathbb{Z}$ do not have multiplicative inverses. For any
  $n \in \mathbb{Z}$, there is no $m \in \mathbb{Z}$ such that $n \times m = 1$
  (besides $n = 1$ and $m = 1$).

### Binary Field $\mathbb{F}_2$

A special field is the binary field $\mathbb{F}_2$. It consists of the elements
$\{0, 1\}$, where $0$ and $1$ are the additive and multiplicative identities,
respectively. The binary field is fundamental in digital logic and computer
science.

## The Importance of a Field in Vector Spaces and Deep Learning

Vector spaces, their subspaces, and the linear transformations within these
spaces are foundational concepts in deep learning. A prominent example is the
attention mechanism, widely recognized in deep learning. This mechanism encodes
sequences of words into a vector space or subspace, with each word represented
as a vector in a $D$-dimensional space, denoted as $\mathbb{R}^D$. These
vectors, known as embeddings, encapsulate both the semantic and contextual
information about words in a sequence. The effectiveness of this representation
hinges on the vector space being well-defined, a state that is contingent on the
precise definition and consistent application of operations within the space.
This is where the role of a field becomes crucial.

A field provides a structured set of scalars with which vectors can be scaled
and combined. The operations of vector addition and scalar multiplication,
fundamental to any vector space, rely on the properties of a field—such as
closure, associativity, and distributivity—to ensure their consistency and
reliability. These properties guarantee that any combination or transformation
of vectors results in another vector within the same space, a necessity for
maintaining the integrity of operations in machine learning and deep learning
algorithms. We can say with certainty that the concept of vector spaces and its
properites underlie the foundation of all machine learning and deep learning
algorithms {cite}`deisenroth2020mathematics`.

## Summary

The content provides a comprehensive overview of fields in mathematics,
emphasizing their significance in linear algebra and machine learning. It
defines a field as a structured set with specific rules for operations like
addition and multiplication, essential for consistency in mathematical
procedures. Examples of fields, such as real numbers, complex numbers, and
rational numbers, are discussed to illustrate their properties and applications.
In contrast, non-fields like natural numbers and integers are highlighted for
lacking certain field properties.

In what follows, we will focus on the field of real numbers $\mathbb{R}^{D}$,
where $D$ is the dimension of the vector space. This is because most machine
learning algorithms are defined over the real numbers
{cite}`deisenroth2020mathematics`.

## References and Further Readings

- Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). _Mathematics for
  Machine Learning_. Cambridge University Press. (pp. 17-18).

[^well-defined]:
    Well defined means closure for addition and multiplication is satisfied.
