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
Keeping in mind the concept of scalars is crucial as we later explore vectors,
which are fundamentally different from scalars yet interact with them within the
framework of a field in linear algebra.

## Definition

A **[field](<https://en.wikipedia.org/wiki/Field_(mathematics)>)** is a
[**quintuple**](https://www.merriam-webster.com/dictionary/quintuple),
$(\mathbb{F},+,0;\times,1)$, where $\mathbb{F}$ is a set,
$+\colon \mathbb{F}\times \mathbb{F} \to \mathbb{F}$ and
$\times\colon \mathbb{F} \times \mathbb{F}\to \mathbb{F}$ are
[**well defined**](<https://en.wikipedia.org/wiki/Closure_(mathematics)>) binary
operations[^well-defined] such that:

A **field** is an ordered pair of triplets, defined as

$$
( (\mathbb{F}, +, 0), (\mathbb{F}, \times, 1) ),
$$

where:

1. $\mathbb{F}$ is a set,
2. $+$ and $\times$ are binary operations defined on $\mathbb{F}$ such that
   $+\colon \mathbb{F}\times \mathbb{F} \to \mathbb{F}$ and
   $\times\colon \mathbb{F} \times \mathbb{F}\to \mathbb{F}$,
3. These operations are well-defined and satisfy the field axioms.

In this representation:

- The first triple $(\mathbb{F}, +, 0)$ encapsulates the addition operation in
  the field, where $+$ is the addition operation, and $0$ is the additive
  identity in $\mathbb{F}$.
- The second triple $(\mathbb{F}, \times, 1)$ encapsulates the multiplication
  operation, where $\times$ is the multiplication operation, and $1$ is the
  multiplicative identity.

This format clearly distinguishes the two fundamental operations (addition and
multiplication) and their respective identities, encapsulating the essence of a
field in a more mathematically conventional and rigorous way.

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

## Notations

In the following sections and chapters, we will use the notation $\mathbb{F}$ to
denote a field. In addition, we can use interchangeably the symbols $\mathbb{R}$
and $\mathbb{C}$ to denote the fields of real and complex numbers, respectively.

## Intuition

A field in mathematics is like a rulebook for numbers. It dictates how we can
add, subtract, multiply, and divide numbers in a way that makes sense and keeps
the universe of mathematics harmonious. In machine learning, fields underpin the
operations we perform on data, influencing everything from simple linear
regression to complex neural networks.

## Examples

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

### Non-Fields

Non-fields include:

- The natural numbers $\mathbb{N}$.
- The integers $\mathbb{Z}$.

Why are these non-fields? They lack additive inverses. For example, the natural
numbers $\mathbb{N}$ do not have additive inverses. For any $n \in \mathbb{N}$,
there is no $m \in \mathbb{N}$ such that $n + m = 0$. The same is true for the
integers $\mathbb{Z}$.

### Binary Field $\mathbb{F}_2$

A special field is the binary field $\mathbb{F}_2$. It consists of the elements
$\{0, 1\}$, where $0$ and $1$ are the additive and multiplicative identities,
respectively. The binary field is fundamental in digital logic and computer
science.

## Summary

The text discusses fields in mathematics, emphasizing their importance in linear
algebra and machine learning. Fields are structured sets with specific rules for
addition, subtraction, multiplication, and division, ensuring consistency in
mathematical operations. Key aspects include understanding field properties,
their relevance in linear algebra, and practical applications in machine
learning. Fields like real numbers, complex numbers, and rational numbers are
explored, while non-fields like natural numbers and integers are distinguished
by their lack of certain properties. A special mention is given to the binary
field, significant in digital logic and computer science.

[^well-defined]:
    Well defined means closure for addition and multiplication is satisfied.
