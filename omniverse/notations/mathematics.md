# Mathematical Notations

We largely follow the
[Machine Learning: The Basics](https://link.springer.com/book/10.1007/978-981-16-8193-6)
book in terms of notations.

## Set Notation

```{list-table} Set Notations
:header-rows: 1
:name: set-notations

* - Notation
  - Description
* - $a \in \mathcal{A}$
  - This statement indicates that the object $a$ is an element of the set $\mathcal{A}$.
* - $a:=b$
  - This statement defines $a$ to be shorthand for $b$.
* - $|\mathcal{A}|$
  - The cardinality (number of elements) of a finite set $\mathcal{A}$.
* - $\mathcal{A} \subseteq \mathcal{B}$
  - $\mathcal{A}$ is a subset of $\mathcal{B}$.
* - $\mathcal{A} \subset \mathcal{B}$
  - $\mathcal{A}$ is a strict subset of $\mathcal{B}$.
* - $\mathbb{N}$
  - The set of natural numbers $1,2,\ldots$.
* - $\mathbb{R}$
  - The set of real numbers $x$.
* - $\mathbb{R}_{+}$
  - The set of non-negative real numbers $x \geq 0$.
* - $\{0,1\}$
  - The set consisting of two real-number 0 and 1.
* - $[0,1]$
  - The closed interval of real numbers $x$ with $0 \leq x \leq 1$.
* - $f(\cdot), h(\cdot)$
  - A function or map $f(\cdot)$ that accepts any element $a \in \mathcal{A}$ from a set $\mathcal{A}$ as input and delivers a well-defined element $f(a) \in \mathcal{B}$ of a set $\mathcal{B}$. The set $\mathcal{A}$ is the domain of the function $f$ and the set $\mathcal{B}$ is the codomain of $f$. Machine Learning revolves around finding (or learning) a function $h$ (which we call hypothesis) that reads in the features $\mathbf{x}$ of a data point and delivers a prediction $h(\mathbf{x})$ for the label $y$ of the data point.
```
