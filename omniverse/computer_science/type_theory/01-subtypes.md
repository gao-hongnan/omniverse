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
![Tag](https://img.shields.io/badge/Tag-Structured_Musings-purple)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-Vetted-green)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from collections.abc import Sequence, Sized
from typing import Any
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
check the types at **static-analysis time** (i.e., before the program ever
runs), and rely on the subtyping schemes to determine if one type,
$\mathcal{A}$, is a subtype of another type, $\mathcal{B}$.

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

```{admonition} Declaration, Static-Analysis, and Run Time
:class: note

**Nominal subtype relationships** are established at **declaration time** (i.e.,
when a new subclass is declared), and checked at **static-analysis time**, whereas
**structural subtype relationships** are established at the **point of use**,
and checked at **runtime**. However, when defining via `typing.Protocol`
([PEP 544](https://peps.python.org/pep-0544/), in the standard library since
Python 3.8 — a language feature, not a `mypy` one, so any type checker
understands it), the structural subtyping is actually checked at
**static-analysis time**. We will see the difference later.
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
# Canonical fixture for this series — reused by later chapters
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
print(isinstance(rob, Animal))  # False, Robot is not a nominal subtype of Animal
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

Consider a toy example below, where we construct a generic `Dataset` to hold a
`Sequence` containing elements of type `T` (the `class Dataset[T]:`
type-parameter syntax comes from
[PEP 695](https://peps.python.org/pep-0695/) — we unpack it properly in
{doc}`Generics <04-generics>`). The current implementation does not have any
subtyping schemes to it, and therefore, if we try to check if this `Dataset` is
an instance of
[`Sized`](https://github.com/python/cpython/blob/15309329b65a285cb7b3071f0f08ac964b61411b/Lib/_collections_abc.py#L399),
we would get `False`.

```{code-cell} ipython3
class Dataset[T]:
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
class Dataset[T]:
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

```{code-block} python
---
caption: Reproduced from CPython's Lib/_collections_abc.py (quoted source; annotations added for exposition)
linenos: true
emphasize-lines: 9-13
---
class Sized(metaclass=ABCMeta):

    __slots__ = ()

    @abstractmethod
    def __len__(self):
        return 0

    @classmethod
    def __subclasshook__(cls: type[Sized], C: type) -> bool:
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

The cell runs happily — at runtime the duck check does its job. But watch what
happens when we hand the same code to the static type checkers. The guard
`is_flyable` returns a plain `bool`, which tells a checker *nothing* about
`obj` inside the `if` branch, and the heterogeneous list gives the two
checkers room to disagree about `obj` itself. `mypy --strict` joins the
element type of `objects` up to `object` (the only common ancestor of `Bird`,
`Airplane`, and `Car`) and rejects the call:

```text
duck_check.py:26: error: "object" has no attribute "fly"  [attr-defined]
Found 1 error in 1 file (checked 1 source file)
```

`pyright`, in its default mode, infers the element type as `Unknown` (an
implicit `Any`) and stays silent — `0 errors, 0 warnings, 0 informations` —
though its strict mode flags the unknown-ness instead. Neither checker
*understands* the duck check; they differ only in how loudly they shrug.
(Teaching a checker to trust a boolean predicate is possible, but it must be
declared with `TypeIs` or `TypeGuard` — the subject of a later chapter.)

This gap is precisely what `typing` closes. By defining a
[protocol](https://peps.python.org/pep-0544/) via the `Protocol` class, you can
specify the required methods and properties for a type — making the structural
relationship visible at static-analysis time,

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

can_we_fly(bird)       # OK: Bird is a structural subtype of Flyable
can_we_fly(airplane)   # OK: Airplane is a structural subtype of Flyable
can_we_fly(car)        # runs fine at runtime; rejected at static-analysis time
print("All three calls executed without a runtime error.")
```

Here, both `Bird` and `Airplane` are considered structural subtypes of the
`Flyable` protocol because they implement the required `fly` method, even though
they don't explicitly inherit from `Flyable`. The `Car` class, on the other
hand, does not implement the `fly` method and is not considered a structural
subtype of `Flyable`.

Notice that the cell above executes without a single complaint — annotations
are not enforced while the program runs, so even `can_we_fly(car)` sails
through at runtime. The rejection happens at **static-analysis time**: save the
`Bird`/`Airplane`/`Car` definitions together with the cell above as
`flyable.py` and run a static type checker over it, and the `car` call — and
only the `car` call — is flagged. `pyright` reports

```text
flyable.py:34:12 - error: Argument of type "Car" cannot be assigned to parameter "obj" of type "Flyable" in function "can_we_fly"
    "Car" is incompatible with protocol "Flyable"
      "fly" is not present (reportArgumentType)
1 error, 0 warnings, 0 informations
```

and `mypy --strict` agrees:

```text
flyable.py:34: error: Argument 1 to "can_we_fly" has incompatible type "Car"; expected "Flyable"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

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

### When Structural Subtyping Backfires: LSP

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

```{code-block} python
---
linenos: true
emphasize-lines: 14,17,42,45
---

def _check_methods(C: type, *methods: str) -> bool:
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
    def __subclasshook__(cls: type[Animal], C: type) -> bool:
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
some of these rules.

## Inclusive vs. Coercive Implementations

While nominal and structural subtyping focus on _how_ type relationships are
defined, **inclusive** and **coercive** implementations concern themselves with
_what happens_ to a value when types **interact** in a
program[^subtype-schemes-wikipedia]. In an **inclusive** implementation, the
internal representation of a subtype value is already a valid representation of
the supertype value, so nothing needs to change or be converted — think of it
as direct **"plug-and-play"**. A `Dog` object passed to a function expecting an
`Animal` is not transformed into an `Animal`; it is simply _used_, because its
representation already **includes all necessary aspects** of an `Animal`. This
is the reading that pairs naturally with the subtyping schemes above: every
value of the subtype $\mathcal{A}$ _is_ a value of the supertype $\mathcal{B}$.

In a **coercive** implementation, the internal representations differ, and the
language inserts an "adapter": the value is automatically converted before it
is used. The canonical example is numeric. In `5 + 2.5`, the `int` value `5` is
implicitly converted to the `float` value `5.0` before the addition — `int` and
`float` have different internal representations in CPython — and the result
`7.5` is a `float`. The integer is not used _as_ a float; it is _turned into_
one.

```{prf:remark} Coercion is a conversion function
:label: type-theory-01-subtypes-remark-coercive-conversion

Formally, a coercive implementation between two types $\mathcal{A}$ and
$\mathcal{B}$ (not necessarily subtypes of each other) supplies a conversion
function

$$
f: \mathcal{A} \rightarrow \mathcal{B}
\quad \text{such that} \quad
\forall \mathcal{V}_{\mathcal{A}} \in \mathcal{A}, \exists \mathcal{V}_{\mathcal{B}} \in \mathcal{B} : \mathcal{V}_{\mathcal{B}} = f(\mathcal{V}_{\mathcal{A}}),
$$

which the language applies implicitly wherever a $\mathcal{B}$ is expected but
an $\mathcal{A}$ is supplied — in `5 + 2.5`, $f$ is the `int`-to-`float`
conversion. An inclusive implementation is the degenerate case where $f$ is the
identity. Note that coercion is a statement about _runtime values_, not about
the subtype relation itself: whether Python's `int` should count as a subtype
of `float` at static-analysis time is a different (and subtler) question, which
we take up in {doc}`Type Safety <02-type-safety>`.
```

## Summary

If I had to compress this chapter into a single sentence, it would be the one
we opened with: a subtype is a type whose values can **stand in** for values of
its supertype without the surrounding program noticing. Types are sets of
values, and subtyping is a _substitutability_ promise over those sets — every
program element written to operate on the supertype must keep working when
handed the subtype. Nominal and structural subtyping are not two different
promises; they are two different ways of _establishing_ the same promise. And
as the `Robot` example showed, the structural route can extend the promise to
types that match an interface's letter while violating its spirit, which is
why the Liskov substitution principle remains the semantic yardstick behind
both schemes.

|                           | Nominal subtyping                                     | Structural subtyping                                                     |
| ------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------ |
| Relationship established  | By explicit declaration (`class Dog(Animal):`)        | By shape — the required methods and properties are present               |
| Declared where            | At declaration time, in the class definition          | Nowhere — it holds implicitly at the point of use                        |
| Python mechanism          | Class inheritance                                     | `typing.Protocol` (PEP 544); ABCs via `__subclasshook__`                 |
| Static-analysis check     | Walks the declared class hierarchy                    | Matches members against the protocol                                     |
| Runtime check             | `isinstance(obj, Animal)`                             | `isinstance` only via `@runtime_checkable` or `__subclasshook__`         |
| Characteristic risk       | Rigidity — conformant-but-unrelated types excluded    | Accidental conformance — semantically wrong subtypes slip in (LSP)       |

The next two chapters make the promise precise:
{doc}`Type Safety <02-type-safety>` examines what can go wrong when
substitution is allowed — and why the static type checker exists to stop it —
while {doc}`Subsumption <03-subsumption>` states the formal criterion a
checker applies when it lets a subtype stand in for its supertype.

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
