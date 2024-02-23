---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Subsumption

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-MaybeWrong-red)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple
from rich.pretty import pprint
```

In this post, we will discuss the concept of subsumption in type theory.
Previously, we explored the avenues through which a subtype relationship can be
established, specifically through nominal and structural subtyping. However,
there exists a need for a systematic method to assess and confirm the legitimacy
of a subtype's status in relation to a supertype.

The concept of subsumption provides a formal framework for evaluating the
validity of subtype relationships. Type checkers use subsumption to verify that
the properties and behaviors of a subtype align with those of its supertype
irregardless of the specific type system (nominal or structural) in use.

## Liskov Substitution Principle

The
[Liskov Substitution Principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle)
(LSP) is a key concept in understanding subtype relationships and type safety.
Formulated by Barbara Liskov, the principle states that objects of a superclass
shall be replaceable with objects of its subclasses without affecting the
correctness of the program. This principle is crucial for ensuring that subtype
polymorphism is used correctly in object-oriented programming.

```{prf:theorem} Liskov Substitution Principle
:label: type-theory-liskov-substitution-principle

If $S$ is a subtype of $T$, then objects of type $T$ in a program may be
replaced with objects of type $S$ (i.e., objects of the subtype) without
altering any of the desirable properties of that program (correctness, task
performed, etc.).
```

The LSP emphasizes that a subtype should not only fulfill the structural
requirements of its supertype but also adhere to its behavioral expectations.
This means that the subtype should meet the following conditions:

1. **Signature Matching**: The subtype should offer all methods and properties
   that the supertype offers, maintaining the same signatures.

2. **Behavioral Compatibility**: The behavior of these methods and properties in
   the subtype should conform to the behavior expected by the supertype. This
   includes respecting invariants, postconditions, and preconditions defined by
   the supertype.

The LSP is a foundational guideline in object-oriented design, ensuring that
subclasses extend superclasses in a manner that does not compromise the
functionality and integrity of the superclass.

Understanding (at least the definition) of the LSP serves as a precursor to
understanding subsumption in type theory.

## Subsumption

In type theory the concept of subsumption is used to define or evaluate whether
a type $S$ is a subtype of type $T$.

Now we will formalize the subtype relationship via
[**mathematical relation**](<https://en.wikipedia.org/wiki/Relation_(mathematics)>).

### Criterion for Subtype Relationships

```{prf:criterion} Subtype Criterion
:label: type-theory-subtype-criterion

Let $\mathcal{T}_1$ and $\mathcal{T}_2$ represent two types in a type system. We
say that $\mathcal{T}_2$ is a subtype of $\mathcal{T}_1$ (denoted as
$\mathcal{T}_2 <: \mathcal{T}_1$) if and only if the following criteria are met:

1. **Value Inclusion (Set Membership)**:
   Every value of the subtype is also a value of the supertype, formally
   expressed as:

    $$
    \forall v \in \mathcal{T}_2, v \in \mathcal{T}_1
    $$

    This criterion ensures that instances of the subtype can be used in any
    context where instances of the supertype are expected.

2. **Function Applicability (Behavioral Compatibility and Preservation of
   Semantics)**:
   This criterion ensures that not only are the operations or functions
   applicable to both the supertype and the subtype, but they also preserve
   their semantics and invariants across these types. This can be formally
   expressed in two parts:

    - **Applicability**: Every function or operation applicable to instances of
      the supertype must also be applicable to instances of the subtype.
      Formally:

        $$
        \forall f, \left( f: \mathcal{T}_1 \rightarrow \mathcal{Y} \right) \Rightarrow \left( f: \mathcal{T}_2 \rightarrow \mathcal{Y} \right)
        $$

        where $f$ is a function from type $\mathcal{T}_1$ (or $\mathcal{T}_2$)
        to some type $\mathcal{Y}$.

    - **Semantic Preservation**: The behavior and semantics of the functions or
      operations must be consistent when applied to either the subtype or the
      supertype. This includes maintaining the expected results, side effects,
      and invariants. Formally, for every function $f$ and for all values
      $v_1 \in \mathcal{T}_1$ and $v_2 \in \mathcal{T}_2$, if $v_1$ and $v_2$
      are considered equivalent in the context of $f$, then $f(v_1)$ and
      $f(v_2)$ must yield equivalent results. This can be represented as:

        $$
        \forall f, \forall v_1 \in \mathcal{T}_1, \forall v_2 \in \mathcal{T}_2, \left( v_1 \sim v_2 \right) \Rightarrow \left( f(v_1) \sim f(v_2) \right)
        $$

        where $\sim$ denotes an equivalence relation appropriate to the context
        of $f$.

3. **Property Preservation (Invariant Maintenance)**:
   All invariants or properties that are true for the supertype must also be
   true for the subtype. This can be formally represented using predicate logic:

    $$
    \forall P, \left( P(\mathcal{T}_1) \right) \Rightarrow \left( P(\mathcal{T}_2) \right)
    $$

    where $P$ is a predicate representing a property or invariant of the type.

    For instance, if $\mathcal{T}_1$ has a property P, then $\mathcal{T}_2$ should
    also exhibit property P.
```

A simpler way to understand **function applicability**:

Given a function $f$, if $f$ is applicable to $\mathcal{T}_1$, then it should
also be applicable to $\mathcal{T}_2$.

$$
\forall f, \left(f: \mathcal{T}_1 \rightarrow \dots \right) \Rightarrow \left(f: \mathcal{T}_2 \rightarrow \dots \right)
$$

In other words, consider without loss of generality that $\mathcal{T}_1$ has $N$
functionalities (methods), $f_1, f_2, \ldots, f_N$, then $\mathcal{T}_2$ also
have these $N$ functionalities (methods), $f_1, f_2, \ldots, f_N$ or more.

### Connection to Liskov Substitution Principle

The subsumption criterion is closely related to the Liskov Substitution
Principle (LSP). The LSP emphasizes that a subtype should not only fulfill the
structural requirements of its supertype but also adhere to its behavioral
expectations.

The LSP, by requiring that objects of a superclass be replaceable with objects
of its subclasses without affecting the correctness of the program, inherently
sets the stage for the behavioral compatibility and preservation of semantics
criterion in the subsumption criterion.

### Reflexivity, Transivity and Antisymmetry

A subtype relation is transitive, reflexive and obeys antisymmetry. These are
fundamental properties of the subtype relationship in type theory:

1. **Reflexive**: This means every type is a subtype of itself. Formally, for
   any type $T$, $T <: T$. This reflexivity indicates that you can always use a
   type in the same context as itself, which is a basic and necessary property
   for any type system.

2. **Transitive**: This property means that if type $S$ is a subtype of type
   $T$, and type $T$ is a subtype of type $U$, then type $S$ is also a subtype
   of type $U$. In formal notation, if $S <: T$ and $T <: U$, then $S <: U$. The
   transitive property is crucial for maintaining consistent type relationships
   across a hierarchy or chain of types.

3. **Antisymmetry**: This property states that if type $S$ is a subtype of type
   $T$, and type $T$ is a subtype of type $S$, then $S$ and $T$ are the same
   type. In formal notation, if $S <: T$ and $T <: S$, then $S = T$.
   Antisymmetry ensures that the subtype relationship helps define a clear
   hierarchy or ordering of types, preventing circular relationships where two
   distinct types are each other's subtypes.

### Narrowing Values, Widening Functions

In the subtype process, the set of values of $S$ is a subset (or equal to) of
$T$, and the set of functions applicable to $S$ is a superset (or equal to) of
those applicable to $T$. Stated in points for clarity:

-   The set of values for which $S$ can take becomes smaller in the process of
    subtyping;
-   The set of functions which is applicable to $S$ becomes larger in the
    process of subtyping.

## Integers as a Subtype of Real Numbers

Consider the real number system $\mathbb{R}$, then we say that the integers
(whole numbers) $\mathbb{Z}$ is a **_subtype_** of $\mathbb{R}$ because it
fulfills the criteria for subsumption {prf:ref}`type-theory-subtype-criterion`.

By demonstrating that integers $\mathbb{Z}$ meet all three criteria under the
subtype definition, we can formally argue that $\mathbb{Z}$ is indeed a subtype
of $\mathbb{R}$. The fact that integers can have additional operations that are
not defined for all real numbers (like the bitshift operation) does not violate
any of the subtype criteria; it simply means that the subtype $\mathbb{Z}$ has
more operations than its supertype $\mathbb{R}$, which is permissible in
subtyping.

### Fulfilling the Subtype Criterion

1. **Value Inclusion (Set Membership)**:

    - Every integer is also a real number by definition, since the set of
      integers $\mathbb{Z}$ is included within the set of real numbers
      $\mathbb{R}$. This satisfies the value inclusion criterion:
      $$\forall v \in \mathbb{Z}, v \in \mathbb{R}$$

2. **Function Applicability (Behavioral Compatibility and Preservation of
   Semantics)**:

    - All basic arithmetic operations (addition, subtraction, multiplication,
      division) that are defined for real numbers are also defined for integers.
      This satisfies the applicability part of the function applicability
      criterion:

        $$
        \forall f, \left( f: \mathbb{R} \rightarrow \mathcal{Y} \right) \Rightarrow \left( f: \mathbb{Z} \rightarrow \mathcal{Y} \right)
        $$

    - The results of these operations when applied to integers, as a subset of
      real numbers, yield outcomes that are consistent with their application to
      real numbers (with the understanding that division by an integer may need
      to be treated with care, as it can result in a real number that is not an
      integer). This satisfies the semantic preservation part:

        $$
        \forall f, \forall v_1 \in \mathbb{R}, \forall v_2 \in \mathbb{Z}, \left( v_1 \sim v_2 \right) \Rightarrow \left( f(v_1) \sim f(v_2) \right)
        $$

        Here $\sim$ can be interpreted as numerical equality when $v_2$ is
        treated as a real number.

3. **Property Preservation (Invariant Maintenance)**:

    - Invariants that hold for real numbers, such as associative and commutative
      properties of addition and multiplication, also hold for integers. This
      satisfies the property preservation criterion:

        $$
        \forall P, \left( P(\mathbb{R}) \right) \Rightarrow \left( P(\mathbb{Z}) \right)
        $$

### Satisfies Narrowing Values, Widening Functions

Indeed, if we consider the integer number system as a subtype of the real number
system, then the set of integers values is indeed a subset of real numbers
(narrowing values), and the set of operations/functions applicable by the set of
integers widens (widening functions) since integers **can do bitshift** but real
numbers **cannot**.

In python, the $\mathbb{R}$ can be denoted as type `float` and $\mathbb{Z}$ as
`int` (ignoring the fact that not all real numbers can be represented precisely
in computer systems).

## Circle as a Subtype of Shape in 2D Euclidean Geometry

### Criteria 1: Value Inclusion (Set Membership)

Consider the class `Shape` to represent all 2 dimensional Euclidean Geometry. We
will show how to use the criterion to deduce that the `Circle` class is not only
a subclass of `Shape`, but is also a subtype. We denote $\mathcal{T}_1$ as
`Circle` and $\mathcal{T}_2$ as `Shape`.

```{code-cell} ipython3
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    """Assume `Shape` is 2D and obeys Euclidean Geometry."""

    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError("All 2D Shapes must implement area method")

    @abstractmethod
    def perimeter(self) -> float:
        raise NotImplementedError("All 2D Shapes must implement area method")


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius**2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius
```

In the given example, `Circle` is a subclass of `Shape`. According to the Value
Inclusion criterion, every instance of `Circle` should also be considered an
instance of `Shape`. This relationship is demonstrated through inheritance in
Python, where `Circle` inherits from the abstract class `Shape`.

This is semantically true in the Euclidean space since circle is definitely a
subset of the 2 dimensional shapes. In python, to illustrate this, we create
instances of `Circle` and verify that they are indeed instances of `Shape`.

```{code-cell} ipython3
circle = Circle(radius=10)
isinstance(circle, Shape)
```

### Criteria 2: Function Applicability

This criterion ensures that all methods/functions defined in the supertype
(`Shape`) are also implemented in the subtype (`Circle`), with the same
signature and expected behavior.

Consider $f$ be all the methods that `Shape` has (i.e. $f_1$ and $f_2$
corresponding to `area` and `perimeter` respectively), then, according to the
criterion:

$$
\forall f, \left( f: \mathcal{T}_1 \rightarrow \mathcal{Y} \right) \Rightarrow \left( f: \mathcal{T}_2 \rightarrow \mathcal{Y} \right)
$$

means that for all functions defined in the supertype `Shape` ($\mathcal{T}_1$)
with return type $\mathcal{Y}$, we **_must_** have the subtype `Circle`
($\mathcal{T}_2$) to have the same method/signature.

-   **Applicability**: `Circle` implements the `area` and `perimeter` method
    with the same input (none in this case) and output signature (returns a
    float), satisfying the applicability criterion.
-   **Preserve Behavior and Semantics**: The `area` and `perimeter` method in
    `Circle` correctly calculates the area and perimeter of a circle, which is
    consistent with the expected behavior of an `area` and `perimeter` method in
    a 2D shape. This can be checked by verifying that the method returns correct
    values for known inputs.

```{code-cell} ipython3
circle = Circle(radius=5)
assert math.isclose(circle.area(), math.pi * 5**2), "Area method in Circle does not behave as expected."
```

### Criteria 3: Property Preservation (Invariant Maintenance)

This part is usually associated with the inherent invariance of the parent
class.

An invariant in the context of the `Shape` class could be a property that is
universally true for all shapes. Since `Shape` is an abstract class representing
2D Euclidean shapes, a possible invariant might be:

-   **Non-Negative Area**: All shapes must have a non-negative area.

This property is inherent to the concept of a 2D shape in Euclidean geometry.
The area, being a measure of the extent of a shape in a plane, cannot be
negative.

### Circle is a Subtype of Shape

Thus, we have shown `Circle` to obey all 3 criterion, and is therefore a subtype
of `Shape`.

## Demonstrating Reflexivity, Transivity and Antisymmetry

We will use the `Dog` and `Animal` to illustrate these 3 points, but this is a
very loose demonstration and may not cover all nuances.

```{code-cell} ipython3
class Animal:
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> str:
        return "Generic Animal Sound!"

class Dog(Animal):
    def make_sound(self) -> str:
        return "Woof!"

    def fetch(self) -> str:
        return "Happily fetching balls!"

class PoliceDog(Dog):
    def search(self) -> str:
        print("Found something!")
```

**Reflexivity** in type theory can be roughly illustrated in Python using the
`isinstance` function. Reflexivity states that a type is a subtype of itself.

In Python, this can be demonstrated by showing that an instance of a class
(e.g., `generic_dog`) is indeed an instance of that same class (e.g., `Dog`).
This is a practical demonstration of the concept that a type (or class) is
compatible with itself.

```{code-cell} ipython3
generic_dog = Dog()
isinstance(generic_dog, Dog)
```

This code checks whether `generic_dog` is an instance of `Dog`, which will
return `True`. This result aligns with the principle of reflexivity, showing
that an object of a type is always an instance of its own type.

**Transitivity** in type theory states that if Type A is a subtype of Type B,
and Type B is a subtype of Type C, then Type A is also a subtype of Type C. In
the context of our example:

1. `PoliceDog` is a subtype of `Dog`.
2. `Dog` is a subtype of `Animal`.
3. Therefore, `PoliceDog` should be a subtype of `Animal`.

```{code-cell} ipython3
police_dog = PoliceDog()

is_police_dog_instance_of_dog = isinstance(police_dog, Dog)  # Expected to be True
is_police_dog_instance_of_animal = isinstance(police_dog, Animal)  # Expected to be True

print(is_police_dog_instance_of_dog, is_police_dog_instance_of_animal)
```

Anti-symmetry is hard to show in practice as it is more of a theoretical
property.

## References and Further Readings

-   [Liskov Substitution Principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle)
-   [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/)
-   [Unit 2: Variable and Type - CS2030S](https://nus-cs2030s.github.io/2021-s2/02-type.html)
-   [Subtype Subsumption - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subsumption)
-   [Subtyping - eduNitas](https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html)
