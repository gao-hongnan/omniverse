---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Subtypes

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Structured_Musings-purple)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-Vetted-green)

```{contents}
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from io import TextIOBase, TextIOWrapper
from typing import Any, BinaryIO, Dict, Generator, Generic, List, Literal, Sized, Tuple, TypeVar, Union, Sequence, Type
from rich.pretty import pprint
```

In
[programming language theory](https://en.wikipedia.org/wiki/Programming_language_theory),
[**subtyping**](https://en.wikipedia.org/wiki/Subtyping) (also called subtype
polymorphism or inclusion polymorphism) is a form of
[type polymorphism](<https://en.wikipedia.org/wiki/Polymorphism_(computer_science)>).
A subtype is a datatype that is related to another datatype (the supertype) by
some notion of
[substitutability](https://en.wikipedia.org/wiki/Substitutability) (read:
[Liskov substitution principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle)),
meaning that program elements (typically
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

**Nominal subtype relationships** are established at **declaration time** (i.e.,
when a new subclass is declared), and checked at **compile time**, whereas
**structural subtype relationships** are established at the **point of use**,
and checked at **runtime**. However, when defining via `mypy`'s `Protocol`,
the structural subtyping is actually checked at **compile time**. We will see
the difference later.
```

### Nominal Subtyping - Class Hierarchy Determines Subtypes

Given the backdrop in the previous section, we would condense out the key
concepts of nominal subtyping below, and end it off with a python example.

#### What is Nominal Subtyping?

**_Nominal subtyping_** is a type system concept where a type is considered a
subtype of another **only if it is explicitly declared as such**. This mechanism
is rooted in **explicit declarations** of type relationships, typically through
class inheritance in object-oriented programming languages.

#### Why Nominal Subtyping?

Nominal subtyping provides a **controlled environment for polymorphism**, where
the relationships between types are **well-defined** and **restricted according
to the class hierarchy**. Consequently, the _explicitness_ of such declaration
provides clarity to developers. Furthermore, nominal subtype relationships need
to be planned in advance, and hence it might be easier to ensure that certain
principles (e.g, the Liskov substitution principle) hold for subtypes.

#### How to Implement Nominal Subtyping?

In languages that utilize **_nominal subtyping_**, **subclassing** or
**interface implementation** are the primary means to establish subtype
relationships. For instance, a class must **explicitly extend another class** or
**implement an interface** to be considered its subtype. This approach relies on
the **lineage of type declarations** to determine subtype relationships,
focusing on **names and declarations** rather than the structural content of the
types.

In Java for instance, if `class Dog extends Animal`, **_Dog_** is a **_nominal
subtype_** of **_Animal_** because it **explicitly extends** `Animal`. We see a
similar implementation in Python below, detailing how `Dog` and `Cat` are both
subtypes of their parent class `Animal` through inheritance.

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

class Robot:
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> str:
        return "Generic Robot Sound!"

cat = Cat()
dog = Dog()
rob = Robot()
print(isinstance(cat, Animal))  # True,  Cat is a nominal subtype of Animal
print(isinstance(dog, Animal))  # True,  Dog is a nominal subtype of Animal
print(isinstance(rob, Animal))  # False, Robit is not a nominal subtype of Animal
```

In this example, `Dog` and `Cat` are nominal subtypes of `Animal` because they
explicitly inherit from the `Animal` class. However, `Robot` which has the exact
same methods as `Animal`, is not a subclass of `Animal` and therefore do not
qualify as a subtype of `Animal` under the nominal subtyping framework. Note
that python allows unsafe overriding of attributes and methods, so we really
want static type checker to ensure we do not violate any rules such as
[Liskov Substitution Principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle).

### Structural Subtyping

#### What is Structural Subtyping?

**_Structural subtyping_** is a type system strategy where a type is considered
a subtype of another based on its **structure** — specifically, if it possesses
all the **members** (properties and methods) required by the supertype. This
approach contrasts with nominal subtyping by focusing on the capabilities of
types rather than their explicit declarations or lineage. It aligns with the
concept of "[duck typing](https://en.wikipedia.org/wiki/Duck_typing)" in
dynamically typed languages: if an object behaves like a duck (implements all
the duck behaviors), it can be treated as a duck

#### Why Structural Subtyping?

The flexibility of structural subtyping allows for **novel and unintended uses**
of existing code by enabling objects that do not share a common inheritance path
to interact seamlessly as long as they fulfill the structural criteria.
Sometimes you would like to enable loose coupling and subclass (nominal) may
just add unwanted complexity.

Consider a toy example below, where we construct `Dataset` to hold a `Sequence`
containing elements of type `T`. The current implementation does not have any
subtyping schemes to it, and therefore, if we try to check if this `Dataset` is
an instance of
[`Sized`](https://github.com/python/cpython/blob/15309329b65a285cb7b3071f0f08ac964b61411b/Lib/_collections_abc.py#L399),
we would get `False`.

```{code-cell} ipython3
T = TypeVar("T")


class Dataset:
    def __init__(self, elements: Sequence[T]) -> None:
        self.elements = elements

dataset = Dataset([1, 2, 3, 4, 5])
print(isinstance(dataset, Sized))
```

However, once we add `__len__` to the example, then `Dataset` is now an instance
of the `Sized`. The Sized protocol requires just one thing: a `__len__` method
that returns the size of the container. Despite Dataset not inheriting from any
specific class that implements Sized, the mere presence of the said method
adheres to the structural expectations of being "sizable".

```{code-cell} ipython3
class Dataset:
    def __init__(self, elements: Sequence[T]) -> None:
        self.elements = elements

    def __len__(self) -> int:
        """Returns the number of elements in the collection."""
        return len(self.elements)

dataset = Dataset([1, 2, 3, 4, 5])
print(isinstance(dataset, Sized))
```

It is worth noting that the `Sized` protocol is not really the `Protocol` we
know of, instead they use `__subclasshook__` for the **_structural typing dark
magic_** to happen.

```{code-block} md
---
linenos: true
emphasize-lines: 9-13
---
class Sized(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __len__(self):
        return 0

    @classmethod
    def __subclasshook__(cls: Type[Sized], C: Type) -> bool:
        if cls is Sized:
            return _check_methods(C, "__len__")
        return NotImplemented
```

To this end, the `Dataset` class is now a structural subtype of the `Sized`
class, as it implements the `__len__` method required by the `Sized` "protocol".
The check is done at **runtime** via the `__subclasshook__` method, which
verifies if the class implements the necessary methods for the protocol.

#### How to Implement Structural Subtyping?

In languages supporting **_structural subtyping_**, subtype relationships are
established through the implementation of the required members, without the need
for explicit inheritance or interface implementation. This method focuses on the
actual implementation of the required properties and methods. More concretely,
if type `A` defines all the methods of type `B` (and `B` is usually a
`Protocol`), then `A` is a subtype of `B`, irrespective of their inheritance
relationship.

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
type,

```{code-cell} ipython3
from typing import Protocol

class Flyable(Protocol):
    def fly(self) -> str:
        ...

def can_we_fly(obj: Flyable) -> None:
    ...

bird = Bird()
airplane = Airplane()
car = Car()

can_we_fly(bird)       # No error, Bird is a structural subtype of Flyable
can_we_fly(airplane)   # No error, Airplane is a structural subtype of Flyable
can_we_fly(car)        # Error, Car is not a structural subtype of Flyable
```

Here, both `Bird` and `Airplane` are considered structural subtypes of the
`Flyable` protocol because they implement the required `fly` method, even though
they don't explicitly inherit from `Flyable`. The `Car` class, on the other
hand, does not implement the `fly` method and is not considered a structural
subtype of `Flyable`.

It is worth noting that `mypy` is a static type checker, and hence if you run
`mypy` on the above code, the code is checked at **compile time** to ensure that
the `Bird` and `Airplane` classes are structural subtypes of the `Flyable`
protocol and that the `Car` class is not.

If you want to ensure that the check is done at runtime with `isinstance`, you
can use the decorator `runtime_checkable` to enable runtime instance
checks[^runtime-checkable] (you cannot call `isinstance` on `Flyable` without
this decorator):

```{code-cell} ipython3
from typing import Protocol, runtime_checkable

@runtime_checkable
class Flyable(Protocol):
    def fly(self) -> str:
        ...

print(isinstance(bird, Flyable))        # True, Bird is a structural subtype of Flyable
print(isinstance(airplane, Flyable))    # True, Airplane is a structural subtype of Flyable
print(isinstance(car, Flyable))         # False, Car is not a structural subtype of Flyable
```

### Pros and Cons of Nominal and Structural Subtyping

In the nominal subtyping example, the subtype relationship is established
through explicit class inheritance. In the structural subtyping example, the
subtype relationship is based on the implementation of a specific interface
(defined by a `Protocol`), regardless of the inheritance relationship.

In the context of structural subtyping, a _nuanced_ issue arises from the
application of the Liskov Substitution Principle (LSP). The LSP _asserts that
objects of a superclass should be replaceable with objects of a subclass without
affecting the correctness of the program_. Structural subtyping, however,
_evaluates_ type compatibility based on the _presence and signature of methods_,
_not_ on the **inherent relationship** or **semantic compatibility** between the
types. This leads to scenarios where a class _might_ unintentionally become a
subtype of another by merely implementing the **same** method signatures,
potentially violating the LSP due to semantic discrepancies.

Consider the same example from nominal subtyping, but with an added
`__subclasshook__` method to the `Animal` class. This method is used to check if
a class is a structural subtype of `Animal` by checking if it implements the
`describe` and `make_sound` methods.

```{code-block} md
---
linenos: true
emphasize-lines: 14,17,42,45
---

def _check_methods(C: Type, *methods: str) -> bool:
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True

class Animal:
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> str:
        return "Generic Animal Sound!"

    @classmethod
    def __subclasshook__(cls: Type[Animal], C: Type) -> bool:
        if cls is Animal:
            return _check_methods(C, "describe", "make_sound")
        return NotImplemented

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

class Robot:
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> str:
        return "Generic Robot Sound!"
```

In this code, `Robot` implements the `make_sound` method, which according to the
`__subclasshook__` in `Animal`, qualifies it as a subtype of `Animal` from a
structural subtyping perspective. However, from a semantic standpoint,
classifying a `Robot` as a subtype of `Animal` is incorrect because they belong
to fundamentally different categories of entities.

In practice, this can be avoided by adhering to good design patterns for your
type protocols or interfaces. Golang is a famous language that relies almost
exclusively on structural subtyping, here's a good
[post](https://appmaster.io/blog/interface-implementation-go) that summarizes
some of thees rules.

## Inclusive vs. Coercive Implementations

While nominal and structural subtyping focus on _how_ type relationships are
defined, **inclusive** and **coercive** implementations concern themselves with
_what happens_ when types **interact** in a program, specifically how values of
one type are treated or converted to another type within the subtype
relationship[^subtype-schemes-wikipedia].

### Inclusive Implementations

The **basic idea** of _inclusive implementations_ is that the internal
representation of a value from a subtype is compatible with that of its
supertype. This means that the value itself doesn't need to change or be
converted when treated as a value of the supertype.

Think of this as direct **"plug-and-play"**. A subtype is a special case of the
supertype and can directly be used in any context where the supertype is
expected. No transformation or special handling is needed; the system inherently
understands that the subtype fits because its representation **includes all
necessary aspects** of the supertype.

In other words, if you have a subtype $\mathcal{A}$ and a supertype
$\mathcal{B}$, any value that belongs to $\mathcal{A}$ is also valid as a value
of $\mathcal{B}$. The representation of the value in $\mathcal{A}$ is sufficient
to represent it in $\mathcal{B}$ as well.

```{prf:example} Inclusive Implementations in Object-Oriented Languages
:label: type-theory-01-subtypes-example-inclusive-impl-oop

Consider a class hierarchy where `Dog` extends `Animal`. A `Dog` object has all
the characteristics (fields, methods) of an `Animal` and possibly more. When you
treat a `Dog` object as an `Animal` (for example, passing it to a function that
expects an `Animal`), no conversion is needed - the `Dog` object 'fits' into the
`Animal` type because it already includes all aspects of an `Animal`. This very
much sounds like the nominal subtyping we discussed earlier.
```

### Coercive Implementations

On the other hand, coercive implementations involve automatically converting a
value from one type to another when necessary. This conversion is typically
needed when the internal representations of the types are different. Imagine
this as needing an "adapter" to make one type fit into the context of another.
The system recognizes that while the two types are not the same, there's a known
way to convert from one to the other so that interactions can proceed smoothly.

To formalize the concept of coercive implementations in the context of subtypes
and supertypes with mathematical logic, we consider two types, $\mathcal{A}$ and
$\mathcal{B}$, not necessary subtype of each other. The coercive implementation
implies that values of type $\mathcal{A}$ may need to be converted to type
$\mathcal{B}$ to be treated as values of $\mathcal{B}$.

Let $\mathcal{A}$ and $\mathcal{B}$ be two types within a type system. We define
a coercive conversion process as a function $f$ that maps values from
$\mathcal{A}$ to $\mathcal{B}$:

$$
f: \mathcal{A} \rightarrow \mathcal{B}
$$

This function $f$ takes a value $\mathcal{V}_{\mathcal{A}}$ of type
$\mathcal{A}$ and converts it into a value $\mathcal{V}_{\mathcal{B}}$ of type
$\mathcal{B}$, such that:

$$
\forall \mathcal{V}_{\mathcal{A}} \in \mathcal{A}, \exists \mathcal{V}_{\mathcal{B}} \in \mathcal{B} : \mathcal{V}_{\mathcal{B}} = f(\mathcal{V}_{\mathcal{A}})
$$

````{prf:example} Coercive and Primitive Types
:label: type-theory-01-subtypes-example-coercive-impl-primitive-types

A common example is numeric types, like integers and floating-point numbers. In
many languages, an integer can be used where a floating-point number is
expected. The integer (type `A`) is automatically converted (coerced) into a
floating-point number (type `B`). For instance, in a language like Python, if
you have a function that expects a float but you pass an integer, the integer
will be automatically converted to a float.

```{code-block} md
---
linenos: true
---
# Integer (type A)
int_value = 5

# Floating-point number (type B)
float_value = 2.5

# Adding an integer to a floating-point number
result = int_value + float_value # 7.5
```

Even though `int_value` is an integer and `float_value` is a float, the
programming language automatically converts `int_value` to a float during the
addition operation. This is coercive implementation: the integer is
automatically converted to a float (a related but different type) to make the
operation possible.

In this context:

-   $\mathcal{A}$ corresponds to the type `int`.
-   $\mathcal{B}$ corresponds to the type `float`.
-   The function $f$ represents the implicit conversion process that Python
    performs to convert `int` to `float` before performing the addition.

So, when `int_value` (of type `int`, or $\mathcal{A}$) and `float_value` (of
type `float`, or $\mathcal{B}$) are added, Python implicitly applies the
conversion function $f$ to `int_value` to convert it into a `float` (type
$\mathcal{B}$) before the addition. This can be thought of as:

-   $\mathcal{V}_{\mathcal{A}} = \text{int\_value}$
-   $\mathcal{V}_{\mathcal{B}} = f(\text{int\_value}) = \text{float\_value}$

Here, $f(\text{int\_value})$ effectively "coerces" or converts the integer value
to a floating-point value, ensuring that the addition operation occurs between
two values of the same type ($\mathcal{B}$, or `float`). The result of this
operation is a `float`, demonstrating how the coercive conversion aligns with
the formalism.
````

## References and Further Readings

```{admonition} References
:class: seealso

-   [mypy - Protocols](https://mypy.readthedocs.io/en/stable/protocols.html)
-   [Subtyping schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)
-   [Type Systems: Structural vs. Nominal Typing Explained - Medium](https://medium.com/@thejameskyle/type-systems-structural-vs-nominal-typing-explained-56511dd969f4)
-   [Subtyping - eduNitas](https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html)
-   [PEP 544 – Protocols: Structural subtyping (static duck typing)](https://peps.python.org/pep-0544/)
-   [Nominal Type System - Wikipedia](https://en.wikipedia.org/wiki/Nominal_type_system)
-   [Structural Type System - Wikipedia](https://en.wikipedia.org/wiki/Structural_type_system)
```

[^subtype-wikipedia]:
    [Subtyping - Wikipedia](https://en.wikipedia.org/wiki/Subtyping)

[^subtype-schemes-wikipedia]:
    [Subtyping Schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)

[^subtype-subsumption-wikipedia]:
    [Subtype Subsumption - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subsumption)

[^runtime-checkable]:
    [Using isinstance() with protocols](https://mypy.readthedocs.io/en/stable/protocols.html#using-isinstance-with-protocols)
