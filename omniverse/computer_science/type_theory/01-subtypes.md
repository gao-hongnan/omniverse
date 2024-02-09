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

# Subtypes

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

In programming language theory, subtyping (also called subtype polymorphism or
inclusion polymorphism) is a form of type polymorphism. A subtype is a datatype
that is related to another datatype (the supertype) by some notion of
substitutability, meaning that program elements (typically subroutines or
functions), written to operate on elements of the supertype, can also operate on
elements of the subtype [^subtype-wikipedia].

In what follows, we will discuss type theory in more details. But one thing you
need to remember is that "**A type is a set of values**"
[^subtype-subsumption-wikipedia].

## Types are Sets

For people coming from a mathematical background, it may be useful to think of
types as sets. Indeed, a type in the context of type theory, is a set of values
[^subtype-subsumption-wikipedia].

## Nominal vs. Structural Subtyping

In the realm of type theory, a crucial distinction is made between two primary
subtyping schemes: **nominal subtyping** and **structural subtyping**. This
distinction is fundamental in understanding how different programming languages
conceptualize and implement subtype relationships. Nominal subtyping bases the
subtype relationship on explicit declarations (like class inheritance), while
structural subtyping determines it based on the actual structure (methods and
properties) of the types.

This distinction is particularly important for static type checkers, which rely
on it to determine if one type, $\mathcal{S}$, is a subtype of another type,
$\mathcal{T}$. In nominal subtyping, the checker looks for explicit declarations
in the code, whereas, in structural subtyping, it analyzes the type's structure
to make this determination. Therefore, understanding nominal versus structural
subtyping aids in designing and interpreting type systems in a way that aligns
with a language's approach to type safety, polymorphism, and inheritance.

### Nominal Subtyping - Class Hierarchy Determines Subtypes

-   **Description**: Nominal subtyping is based on explicit declarations. A type
    is a subtype of another only if it is explicitly declared as such. This is
    typically seen in class-based object-oriented programming languages.
-   **Characteristics**: In languages that use nominal subtyping, subclassing or
    interface implementation are used to establish subtype relationships. A
    classic example is Java, where a class must explicitly extend another class
    or implement an interface to be considered its subtype.
-   **Implication**: The subtype relationship is determined by the lineage of
    the type declarations rather than the actual structure or content of the
    types themselves.
-   **Example**: In Java, if `class Dog extends Animal`, `Dog` is a nominal
    subtype of `Animal` because it explicitly extends `Animal`.

This approach allows for controlled polymorphism where the relationships between
types are well-defined and restricted according to the design of the class
hierarchy.

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


class Cat(Animal):
    def make_sound(self) -> str:
        return "Meow"

    def how_many_lives(self) -> str:
        return "I have 9 lives!"

# Testing nominal subtyping
dog = Dog()
print(isinstance(dog, Animal))  # True, Dog is a nominal subtype of Animal
```

In this example, `Dog` and `Cat` are nominal subtypes of `Animal` because they
explicitly inherit from the `Animal` class. Note that python allows unsafe
overriding of attributes and methods, so we really want static type checker to
ensure we do not violate any rules such as
[Liskov Substitution Principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle).

### Structural Subtyping

-   **Description**: Structural subtyping, in contrast, is based on the
    structure of types. A type is considered a subtype of another if it has at
    least all the members (properties and methods) of the supertype.
-   **Characteristics**: This scheme is common in dynamically typed languages,
    such as Python, where subtyping is determined by the actual capabilities
    (methods and properties) of the objects, not their explicit type
    declarations. This is often referred to as "duck typing" - if it looks like
    a duck and quacks like a duck, it is treated as a duck.
-   **Example**: In a structurally-typed language, if type `A` defines all the
    methods of type `B`, then `A` is a subtype of `B`, irrespective of their
    inheritance relationship.

Structural subtyping in Python can be illustrated using the `Protocol` class
from the `typing` module. A structural subtype relationship is established based
on the structure (methods and attributes) of the type, not its explicit
inheritance hierarchy.

```{code-cell} ipython3
from typing import Protocol, runtime_checkable

@runtime_checkable
class Flyable(Protocol):
    def fly(self) -> str:
        ...

class Bird:
    def fly(self) -> str:
        return "Bird flying"

class Airplane:
    def fly(self) -> str:
        return "Airplane flying"

# Testing structural subtyping
bird = Bird()
airplane = Airplane()
print(isinstance(bird, Flyable))  # True, Bird is a structural subtype of Flyable
print(isinstance(airplane, Flyable))  # True, Airplane is a structural subtype of Flyable
```

Here, both `Bird` and `Airplane` are considered structural subtypes of the
`Flyable` protocol because they implement the required `fly` method, even though
they don't explicitly inherit from `Flyable`.

In summary:

-   In the nominal subtyping example, the subtype relationship is established
    through explicit class inheritance.
-   In the structural subtyping example, the subtype relationship is based on
    the implementation of a specific interface (defined by a `Protocol`),
    regardless of the inheritance relationship.

## Subtyping Schemes

Implementations of programming languages with subtyping fall into two general
classes, inclusive and coercive [^subtype-schemes-wikipedia].

### Inclusive Implementations

-   **Basic Idea**: In inclusive implementations, the internal representation of
    a value from a subtype is compatible with that of its supertype. This means
    that the value itself doesn't need to change or be converted when treated as
    a value of the supertype.

-   **How It Works in Subtyping**: If you have a subtype `A` and a supertype `B`
    (`A <: B`), any value that belongs to `A` is also valid as a value of `B`.
    The representation of the value in `A` is sufficient to represent it in `B`
    as well.

-   **Example in Object-Oriented Languages**: Consider a class hierarchy where
    `Dog` extends `Animal`. A `Dog` object has all the characteristics (fields,
    methods) of an `Animal` and possibly more. When you treat a `Dog` object as
    an `Animal` (for example, passing it to a function that expects an
    `Animal`), no conversion is needed - the `Dog` object 'fits' into the
    `Animal` type because it already includes all aspects of an `Animal`.

### Coercive Implementations

-   **Basic Idea**: Coercive implementations involve automatically converting a
    value from one type to another when necessary. This conversion is typically
    needed when the internal representations of the types are different.

-   **How It Works in Subtyping**: If you have a subtype `A` and a supertype
    `B`, and `A <: B`, coercive implementations might require converting a value
    from `A` into a compatible format for `B`. This is often seen in types that
    are conceptually related but have different internal representations.

-   **Example with Primitive Types**: A common example is numeric types, like
    integers and floating-point numbers. In many languages, an integer can be
    used where a floating-point number is expected. The integer (type `A`) is
    automatically converted (coerced) into a floating-point number (type `B`).
    For instance, in a language like Python, if you have a function that expects
    a float but you pass an integer, the integer will be automatically converted
    to a float.

## References and Further Readings

-   [mypy - Protocols](https://mypy.readthedocs.io/en/stable/protocols.html)
-   [Subtyping schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)
-   [Type Systems: Structural vs. Nominal Typing Explained - Medium](https://medium.com/@thejameskyle/type-systems-structural-vs-nominal-typing-explained-56511dd969f4)
-   [Subtyping - eduNitas](https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html)

[^subtype-wikipedia]:
    [Subtyping - Wikipedia](https://en.wikipedia.org/wiki/Subtyping)

[^subtype-schemes-wikipedia]:
    [Subtyping Schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)

[^subtype-subsumption-wikipedia]:
    [Subtype Subsumption - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subsumption)
