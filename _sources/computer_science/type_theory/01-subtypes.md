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

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-Maybe_Chaotic-orange)

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

In
[programming language theory](https://en.wikipedia.org/wiki/Programming_language_theory),
[**subtyping**](https://en.wikipedia.org/wiki/Subtyping) (also called subtype
polymorphism or inclusion polymorphism) is a form of
[type polymorphism](<https://en.wikipedia.org/wiki/Polymorphism_(computer_science)>).
A subtype is a datatype that is related to another datatype (the supertype) by
some notion of
[substitutability](https://en.wikipedia.org/wiki/Substitutability) (read: Liskov
substitution principle), meaning that program elements (typically
[subroutines](https://en.wikipedia.org/wiki/Subroutines) or
[functions](<https://en.wikipedia.org/wiki/Function_(computer_programming)>)),
written to operate on elements of the supertype, can also operate on elements of
the subtype[^subtype-wikipedia].

## Types are Sets

For people coming from a mathematical background, it may be useful to think of
types as [sets](<https://en.wikipedia.org/wiki/Set_(mathematics)>). Indeed, a
type in the context of type theory, _is a_ set of
values[^subtype-subsumption-wikipedia]. In essence, a type defines a
collection—or set—of values that share certain characteristics.

```{prf:example} Integer Type as a Set
:label: type-theory-01-subtypes-example-int-type-as-set

To illustrate, consider the **Integer** (`int`) type in many programming
languages. You can think of this type as a set that includes all whole numbers
from negative infinity to positive infinity. Each number in this set ranging
from $-\infty$ to $\infty$ is an **element** of the **Integer** type.
```

## Nominal vs. Structural Subtyping

In type theory, a crucial distinction is made between two primary subtyping
schemes:
[**_nominal subtyping_**](https://en.wikipedia.org/wiki/Nominal_type_system) and
[**_structural subtyping_**](https://en.wikipedia.org/wiki/Structural_type_system).
This distinction is fundamental in understanding how different programming
languages conceptualize and implement **subtype relationships**. **_Nominal
subtyping_** bases the subtype relationship on **explicit declarations** (like
class inheritance), while **_structural subtyping_** determines it based on the
actual **structure** (methods and properties) of the types.

This distinction is particularly important for static type checkers, which
checks the types at
[**compile time**](https://stackoverflow.com/questions/71616237/in-general-static-languages-are-type-checked-at-compile-time-is-typescript-als),
and rely on the subtyping schemes to determine if one type, $\mathcal{A}$, is a
subtype of another type, $\mathcal{B}$.

In **nominal subtyping**, the static type checker searches for **_explicit
declarations of inheritance_** (e.g., class `A` extends `B`), clearly indicating
that `A` is a subtype of `B`. This establishes a formal, name-based relationship
between types at the time of declaration which means that this schema relies
more on the declared hierarchy and naming of the types rather than their
inherent structure or functionalities. Conversely, **structural subtyping**
involves the checker assessing whether a potential subtype possesses all
necessary **_structural features, such as methods and properties_**, to fulfill
the requirements of its supertype, _without_ requiring any explicit declaration
of this relationship. For instance, the checker would examine if the subtype
implements all the _methods_ present in the supertype, ensuring _compatibility_
based solely on structural _characteristics_.

```{admonition} Declaration, Compile, and Run Time
:class: note

Nominal subtype relationships are established at declaration time (i.e., when a
new subclass is declared), and checked at compile time, whereas structural
subtype relationships are established at the point of use, and checked at run
time (though in the context of statically typed languages that support
structural subtyping, the checks may be implemented at compile time).
```

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

For pedagogical purposes, we can illustrate structural subtyping by implementing
it manually. Our `is_flyable` function checks if an object has a `fly`
attribute, and if that attribute is callable so we know that this attribute is a
method or function, and not a data attribute.

```{code-cell} ipython3
def is_flyable(obj: Any) -> bool:
    return hasattr(obj, "fly") and callable(obj.fly)

class Bird:
    def fly(self) -> str:
        return "Bird flying"

class Airplane:
    def fly(self) -> str:
        return "Airplane flying"

class Car:
    def drive(self) -> str:
        return "Car driving"

print(is_flyable(Bird()))       # True, because Bird implements a callable fly method
print(is_flyable(Airplane()))   # True, Airplane also implements a callable fly method
print(is_flyable(Car()))        # False, Car does not implement a callable fly method

objects = [Bird(), Airplane(), Car()]
for obj in objects:
    if is_flyable(obj):
        print(f"{obj.__class__.__name__} can fly: {obj.fly()}")
    else:
        print(f"{obj.__class__.__name__} cannot fly.")
```

While manual checks like the one above illustrate the core idea of structural
subtyping, Python offers a more streamlined approach through the `typing`
module. By defining a [protocol](https://peps.python.org/pep-0544/) via the
`Protocol` class, you can specify the required methods and properties for a
type, and use `runtime_checkable` to enable runtime instance checks:

```{code-cell} ipython3
from typing import Protocol, runtime_checkable

@runtime_checkable
class Flyable(Protocol):
    def fly(self) -> str:
        ...

# Testing structural subtyping
bird = Bird()
airplane = Airplane()
car = Car()
print(isinstance(bird, Flyable))        # True, Bird is a structural subtype of Flyable
print(isinstance(airplane, Flyable))    # True, Airplane is a structural subtype of Flyable
print(isinstance(car, Flyable))         # False, Car is not a structural subtype of Flyable
```

Here, both `Bird` and `Airplane` are considered structural subtypes of the
`Flyable` protocol because they implement the required `fly` method, even though
they don't explicitly inherit from `Flyable`. The `Car` class, on the other
hand, does not implement the `fly` method and is not considered a structural
subtype of `Flyable`.

### Pros and Cons of Nominal and Structural Subtyping

In summary:

-   In the nominal subtyping example, the subtype relationship is established
    through explicit class inheritance.
-   In the structural subtyping example, the subtype relationship is based on
    the implementation of a specific interface (defined by a `Protocol`),
    regardless of the inheritance relationship.

## Subtyping Schemes

Implementations of programming languages with subtyping fall into two general
classes, inclusive and coercive[^subtype-schemes-wikipedia].

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

```{admonition} References
:class: seealso

-   [mypy - Protocols](https://mypy.readthedocs.io/en/stable/protocols.html)
-   [Subtyping schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)
-   [Type Systems: Structural vs. Nominal Typing Explained - Medium](https://medium.com/@thejameskyle/type-systems-structural-vs-nominal-typing-explained-56511dd969f4)
-   [Subtyping - eduNitas](https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html)
-   [PEP 544 – Protocols: Structural subtyping (static duck typing)](https://peps.python.org/pep-0544/)
```

[^subtype-wikipedia]:
    [Subtyping - Wikipedia](https://en.wikipedia.org/wiki/Subtyping)

[^subtype-schemes-wikipedia]:
    [Subtyping Schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)

[^subtype-subsumption-wikipedia]:
    [Subtype Subsumption - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subsumption)
