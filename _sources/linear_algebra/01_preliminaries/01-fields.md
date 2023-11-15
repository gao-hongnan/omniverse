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
provide a consistent framework in which linear algebra operates.

In linear algebra, a field can be thought of as a structured "space" — for
instance, the set of real numbers ($\mathbb{R}$) — where operations such as
addition and multiplication are not just possible but follow specific,
well-defined rules. These rules ensure mathematical consistency and logic in the
operations performed within this space.

The elements within a field, known as scalars, must adhere to these rules. This
adherence is what distinguishes a field from a mere collection of numbers.

## Definition

A **[field](<https://en.wikipedia.org/wiki/Field_(mathematics)>)** is an ordered
pair of triplets, defined as

$$
( (\mathbb{F}, +, 0), (\mathbb{F}, \times, 1) ),
$$

where:

1. $\mathbb{F}$ is a set,
2. $+$ and $\times$ are binary operations defined on $\mathbb{F}$ such that
   $+\colon \mathbb{F}\times \mathbb{F} \to \mathbb{F}$ and
   $\times\colon \mathbb{F} \times \mathbb{F}\to \mathbb{F}$,
3. These operations are
   [**well-defined**](<https://en.wikipedia.org/wiki/Closure_(mathematics)>)[^well-defined]
   and satisfy the field axioms (detailed below).

In this representation:

- The first triplet $(\mathbb{F}, +, 0)$ encapsulates the addition operation in
  the field, where $+$ is the addition operation, and $0$ is the additive
  identity in $\mathbb{F}$.
- The second triplet $(\mathbb{F}, \times, 1)$ encapsulates the multiplication
  operation, where $\times$ is the multiplication operation, and $1$ is the
  multiplicative identity.

```{list-table} Definition of a Field
:header-rows: 1
:name: linear-algebra-preliminaries-definition-of-a-field

* - **Property**
  - **Description**
* - **Well Defined (Closure)**
  - For all $a, b \in \mathbb{F}$, we have $a + b \in \mathbb{F}$ and
    $a \times b \in \mathbb{F}$.
* - **Commutative Law for Addition**
  - For all $a, b \in \mathbb{F}$, $a + b = b + a$.
* - **Associative Law for Addition**
  - For all $a, b, c \in \mathbb{F}$, $(a + b) + c = a + (b + c)$.
* - **Existence of the Additive Identity**
  - There exists $\mathbf{0} \in \mathbb{F}$ such that for all
    $a \in \mathbb{F}$, $\mathbf{0} + a = a$.

    In other words, an additive identity for the set $\mathbb{F}$ is any element
    $e$ such that for any element $x \in \mathbb{F}$, we have $e + x = x = x + e$.
    In familiar fields like the Real Numbers $\mathbb{R}$, the additive identity
    is $\mathbf{0}$.
* - **Existence of Additive Inverse**
  - For every $a \in \mathbb{F}$, there exists $b \in \mathbb{F}$ such that
    $a + b = 0$. We call $b$ the additive
    inverse and denote $b$ by $-a$.
* - **Commutative Law for Multiplication**
  - For all $a, b \in \mathbb{F}$, $ab = ba$.
* - **Associative Law for Multiplication**
  - For all $a, b, c \in \mathbb{F}$, $(ab)c = a(bc)$.
* - **Existence of the Multiplicative Identity**
  - There exists $\mathbf{1}\in \mathbb{F}$ such that
    $\mathbf{1} \times a = a\times \mathbf{1} = a$ for all $a\in \mathbb{F}$.
* - **Existence of Multiplicative Inverse**
  - For every non-zero $a \in \mathbb{F}$, there exists $b \in \mathbb{F}$ such
     that $a \times b = 1$. Denoted as $a^{-1}$.
* - **Distributive Law**
  - - $\times$ distributes over $+$ on the left: for all $a,b,c\in \mathbb{F}$,
    $a\times(b+c) = (a\times b)+(a\times c)$.
    - $\times$ distributes over $+$ on the right: for all $a,b,c\in \mathbb{F}$,
    $(b+c)\times a = (b\times a)+(c\times a)$.
```

## Examples

Let's look at some examples of fields as well as non-fields.

### Fields

Fields can consist of:

- The real numbers $\mathbb{R}$.
- The complex numbers $\mathbb{C}$.
- The rational numbers $\mathbb{Q}$.

```{prf:example} Fields on the Real Numbers
:label: linear-algebra-preliminaries-fields-on-the-real-numbers

1. **Closure**:

   - **Addition**: If you take any two real numbers, say 2.3 and 3.7, their sum
     (6.0) is also a real number.
   - **Multiplication**: Similarly, if you multiply these numbers (2.3 × 3.7),
     the product (8.51) is also a real number.

2. **Commutativity**:

   - **Addition**: 2.3 + 3.7 equals 3.7 + 2.3.
   - **Multiplication**: 2.3 × 3.7 equals 3.7 × 2.3.

3. **Associativity**:

   - **Addition**: (2.3 + 3.7) + 1.5 equals 2.3 + (3.7 + 1.5).
   - **Multiplication**: (2.3 × 3.7) × 1.5 equals 2.3 × (3.7 × 1.5).

4. **Additive Identity**: The additive identity in $\mathbb{R}$ is 0. For any
   real number $a$, $a + 0 = a$. For example, 2.3 + 0 = 2.3.

5. **Multiplicative Identity**: The multiplicative identity in $\mathbb{R}$
   is 1. For any real number $a$, $a × 1 = a$. For example, 2.3 × 1 = 2.3.

6. **Additive Inverse**: For every real number, there is an additive inverse (or
   opposite). For 2.3, the additive inverse is -2.3, because 2.3 + (-2.3) = 0.

7. **Multiplicative Inverse**: For every non-zero real number, there is a
   multiplicative inverse (or reciprocal). The multiplicative inverse of 2.3 is
   $\frac{1}{2.3}$ because 2.3 × $\frac{1}{2.3}$ = 1.

8. **Distributive Law**: Multiplication distributes over addition. For example,
   2 × (3 + 4) equals (2 × 3) + (2 × 4).
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

## Summary

The content provides a comprehensive overview of fields in mathematics,
emphasizing their significance in linear algebra and machine learning. It
defines a field as a structured set with specific rules for operations like
addition and multiplication, essential for consistency in mathematical
procedures. Examples of fields, such as real numbers, complex numbers, and
rational numbers, are discussed to illustrate their properties and applications.
In contrast, non-fields like natural numbers and integers are highlighted for
lacking certain field properties. Additionally, the binary field
$\mathbb{F}\_2$, crucial in digital logic and computer science, is introduced.
The summary underscores the importance of fields in ensuring logical and
consistent mathematical operations, fundamental to various scientific and
technological disciplines.

[^well-defined]:
    Well defined means closure for addition and multiplication is satisfied.
