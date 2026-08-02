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
myst:
    html_meta:
        "description lang=en":
            "What makes one Python type substitutable for another? Nominal vs
            structural subtyping from set-theoretic first principles, with
            mypy and pyright evidence."
        "keywords":
            "python, subtyping, nominal subtyping, structural subtyping,
            protocol, type theory"
---

# Subtyping in Python: Nominal vs Structural Explained

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

When you pass a `Dog` to a function annotated to take `Animal`, `mypy` says
nothing. Pass a `list[int]` to a function that wants `list[float]`, and it
objects. Both feel like "a smaller thing where a bigger thing is expected" —
so why does one substitution type-check while the other fails?

The rule behind both verdicts is **subtyping**: the contract that lets a
value of one type stand in wherever another type is expected, with the
surrounding program none the wiser. This chapter builds that contract from
first principles — types as sets of values, subtypes as substitutable
subsets — and then shows the two ways Python establishes it: **nominally**,
through class inheritance, and **structurally**, through duck typing and
`typing.Protocol`.

By the end you can predict, before running a checker, whether one type may
substitute for another; pick between inheritance and protocols for your own
APIs; and spot the trap where a structural match violates the **Liskov
substitution principle** — the rule that a subtype must _behave_ like its
supertype, not merely look like it. (The `list` refusal is _variance_, which
{doc}`the variance chapter <06-invariance-covariance-contravariance>`
resolves.)

```{admonition} Prerequisites
:class: note

This is the first chapter of the series, and it assumes only comfort with
Python classes and inheritance — no prior type theory. The roadmap and the
series conventions (Python 3.14 syntax, every static-analysis claim verified
against both `mypy` and `pyright`) live in
{doc}`the series introduction <intro>`.
```

## Types Are Sets of Values

The fastest route into subtyping — especially for readers with a
mathematical background — is to identify a **type** with a
[set](<https://en.wikipedia.org/wiki/Set_(mathematics)>) of values: that is
what a type _is_ in type theory[^subtype-subsumption-wikipedia]. `bool`
names a set with exactly two elements, `True` and `False`; `str` names the
infinite set of all strings; and a value "has type `T`" precisely when it is
an element of `T`'s set.

```{prf:example} Integer Type as a Set
:label: type-theory-01-subtypes-example-int-type-as-set

To illustrate, consider the **Integer** (`int`) type in many programming
languages. You can think of this type as a set that includes all whole
numbers from negative infinity to positive infinity. Each such number is an
**element** of the **Integer** type.
```

Once types are sets, subtyping nearly defines itself: a **subtype** is a
type whose set of values sits inside the set of another type — its
**supertype** — in a way that lets every subtype value **stand in** for a
supertype value. We write $S <: T$ for "$S$ is a subtype of $T$"; read it as
"an $S$ can stand in for a $T$"[^notation].

```{prf:definition} Subtype
:label: type-theory-01-subtypes-definition-subtype

A type $S$ is a **subtype** of a type $T$, written $S <: T$, when both of
the following hold[^subtype-wikipedia]:

1. **Values**: every value of $S$ is also a value of $T$ — as sets,
   $S \subseteq T$.
2. **Substitutability**: every program element (typically a function or
   subroutine) written to operate on values of $T$ also works, with
   unchanged meaning, on values of $S$.

$T$ is then called the **supertype** of $S$, and the relation $<:$ is what
a type checker consults whenever a value of one type appears where another
type is expected.
```

Clause 1 is the subset picture. Clause 2 carries the weight: "works" means
_behaves as the supertype's contract promises_, not merely "does not
raise". That behavioral fine print is the Liskov substitution principle
from the opening, stated formally in
{prf:ref}`the LSP theorem <type-theory-liskov-substitution-principle>`, and
{doc}`Subsumption <03-subsumption>` sharpens both clauses into a checkable,
three-part {prf:ref}`subtype criterion <type-theory-subtype-criterion>`.
Subtyping is also called **subtype polymorphism** or **inclusion
polymorphism** — _polymorphism_ ("many forms") because one function written
against $T$ operates, unchanged, on values of every subtype of $T$.

## The Two Subtyping Schemes: Nominal and Structural

Knowing what a subtype _is_ does not yet tell a type checker how to
_decide_ that $S <: T$ holds for two concrete types. Type systems answer
with one of two **subtyping schemes** — and Python, unusually, offers both:

-   [**Nominal subtyping**](https://en.wikipedia.org/wiki/Nominal_type_system)
    bases the relationship on **explicit declarations**: $S <: T$ holds only
    if the code says so — in Python, `class S(T)`. Names and declared
    lineage decide.
-   [**Structural subtyping**](https://en.wikipedia.org/wiki/Structural_type_system)
    bases the relationship on **shape**: $S <: T$ holds if $S$ supplies
    every member (method or property) that $T$ requires, whether or not the
    two classes have ever heard of each other.

```{prf:definition} Nominal and Structural Subtyping
:label: type-theory-01-subtypes-definition-nominal-structural

Let $S$ and $T$ be types.

1. Under **nominal subtyping**, $S <: T$ holds if and only if the program
   explicitly declares $S$ a subtype of $T$ — in Python, `T` appears in the
   inheritance chain (the **method resolution order**) of `S`.
2. Under **structural subtyping**, $S <: T$ holds if and only if $S$
   provides every member that $T$ requires, with compatible signatures.
```

The distinction matters most to **static type checkers** — tools such as
`mypy` and `pyright` that read source and flag type errors at
**static-analysis time**, before the program ever runs. Asked whether
$S <: T$, a nominal check walks the declared class hierarchy looking for
`T` among `S`'s ancestors; a structural check compares members, asking
whether `S` could fulfill `T`'s obligations regardless of ancestry.

```{admonition} When is the relationship established — and when is it checked?
:class: note

A **nominal** subtype relationship is _established_ at declaration time
(the moment `class Dog(Animal)` is written) and _checked_ at
static-analysis time by walking the declared hierarchy — or at runtime by
`isinstance`. A **structural** relationship is never declared: it is
_established_ implicitly at the point of use, and classic duck typing
_checks_ it only at runtime. `typing.Protocol`
([PEP 544](https://peps.python.org/pep-0544/), in the standard library
since Python 3.8 — a language feature, not a `mypy` one, so any type
checker understands it) is precisely the device that lifts the structural
check to static-analysis time. Both timings are demonstrated below.
```

## Nominal Subtyping: Subtypes by Declaration

**Nominal subtyping** is the conservative scheme: a type is a subtype of
another **only if it is explicitly declared as such**. In Python — as in
Java and C# — the declaration is class inheritance (or, in interface-based
languages, an explicit `implements` clause). `class Dog(Animal)` _is_ the
declaration; there is no way for `Dog` to become a nominal subtype of
`Animal` after the fact, and no way to do it by accident.

That explicitness is the scheme's selling point. Subtype relationships are
planned in advance and visible in the source, so the hierarchy is a
controlled environment for polymorphism: readers can trace the lineage, and
the author of a superclass can design for — and document — the behavioral
obligations the Liskov substitution principle places on every subclass. The
cost is rigidity, which the structural scheme exists to relieve.

The cell below is the canonical fixture for this series; later chapters
reuse it. `Dog` and `Cat` inherit from `Animal`, while `Robot` copies
`Animal`'s methods exactly but declares no lineage.

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

`Dog` and `Cat` are nominal subtypes of `Animal` because they explicitly
inherit from it. `Robot` has the exact same method signatures as `Animal` —
under a structural reading it would qualify — but it declares no lineage,
so under the nominal scheme it is simply not an `Animal`, and `isinstance`
agrees. Note, too, that Python happily lets a subclass override methods
unsafely; nothing at runtime stops `Dog.make_sound` from returning an
`int`. It is the static type checker that polices such violations of the
Liskov substitution principle, and
{doc}`Type Safety <02-type-safety>` stages exactly that crash with a
misbehaving `Robot` variant.

## Structural Subtyping: Subtypes by Shape

**Structural subtyping** is the liberal scheme: a type is a subtype of
another based on its **structure** — it possesses every member (property
and method) the supertype requires — regardless of what it inherits from.
It is the typed rendering of
[duck typing](https://en.wikipedia.org/wiki/Duck_typing): if an object
walks like a duck and quacks like a duck, treat it as a duck.

Why would you want this? Loose coupling. Structural subtyping lets classes
that share no ancestor — often classes from libraries that cannot know
about each other — interoperate the moment their shapes line up, enabling
novel and unintended uses of existing code. Forcing every such relationship
through a nominal base class would add ceremony and coupling without adding
safety.

### Duck Typing at Runtime: `Sized` and `__subclasshook__`

Consider a toy example: a generic `Dataset` holding a `Sequence` of
elements of type `T` (the `class Dataset[T]:` type-parameter syntax comes
from [PEP 695](https://peps.python.org/pep-0695/) — we unpack it properly
in {doc}`Generics <04-generics>`). The class declares no lineage at all, so
checking it against
[`Sized`](https://github.com/python/cpython/blob/15309329b65a285cb7b3071f0f08ac964b61411b/Lib/_collections_abc.py#L399)
— the standard library's "has a `__len__`" interface — reports `False`.

```{code-cell} ipython3
class Dataset[T]:
    def __init__(self, elements: Sequence[T]) -> None:
        self.elements = elements

dataset = Dataset([1, 2, 3, 4, 5])
print(isinstance(dataset, Sized))
```

However, once we add `__len__`, the very same check reports `True`. `Sized`
requires just one thing — a `__len__` method returning the size of the
container — and despite `Dataset` inheriting from nothing, the mere
presence of that method satisfies the structural expectation of being
"sized".

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

How did that work with no inheritance in sight? `Sized` is not a
`typing.Protocol` (those arrive below); it is an **abstract base class**
(ABC) that performs older **_structural typing dark magic_**: its
`__subclasshook__` classmethod inspects any candidate class the moment
`isinstance` or `issubclass` asks.

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

The highlighted hook makes `Sized` accept any class for which
`_check_methods(C, "__len__")` succeeds — the helper walks the candidate's
method resolution order looking for a `__len__` definition. To this end,
`Dataset` is a structural subtype of `Sized`, and the check happens at
**runtime**, inside `isinstance` itself. Hold on to `_check_methods`; it
returns in the final section with less charming consequences.

### Why Static Checkers Cannot See a Duck Check

For pedagogical purposes, we can illustrate structural subtyping by
implementing it manually. Our `is_flyable` function checks that an object
has a `fly` attribute and that the attribute is callable — so we know it is
a method or function, not a data attribute.

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

The cell runs happily — at runtime the duck check does its job. But watch
what happens when we hand the same code to the static type checkers. The
guard `is_flyable` returns a plain `bool`, which tells a checker _nothing_
about `obj` inside the `if` branch, and the heterogeneous list gives the
two checkers room to disagree about `obj` itself. `mypy --strict` joins the
element type of `objects` up to `object` (the only common ancestor of
`Bird`, `Airplane`, and `Car`) and rejects the call:

```text
duck_check.py:26: error: "object" has no attribute "fly"  [attr-defined]
Found 1 error in 1 file (checked 1 source file)
```

`pyright`, in its default mode, infers the element type as `Unknown` (an
implicit `Any`) and stays silent — `0 errors, 0 warnings, 0 informations` —
though its strict mode flags the unknown-ness instead. Neither checker
_understands_ the duck check; they differ only in how loudly they shrug.
(Teaching a checker to trust a boolean predicate is possible, but it must
be declared with `TypeIs` or `TypeGuard` — the subject of a later chapter.)

### Protocols: Structural Subtyping at Static-Analysis Time

This gap is precisely what `typing` closes. By defining a
[protocol](https://peps.python.org/pep-0544/) via the `Protocol` class, you
can specify the required methods and properties for a type — making the
structural relationship visible at static-analysis time:

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
`Flyable` protocol because they implement the required `fly` method, even
though they don't explicitly inherit from `Flyable`. The `Car` class, on
the other hand, does not implement the `fly` method and is not considered a
structural subtype of `Flyable`.

Notice that the cell above executes without a single complaint —
annotations are not enforced while the program runs, so even
`can_we_fly(car)` sails through at runtime. The rejection happens at
**static-analysis time**: save the `Bird`/`Airplane`/`Car` definitions
together with the cell above as `flyable.py` and run a static type checker
over it, and the `car` call — and only the `car` call — is flagged.
`pyright` reports

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

### Runtime Protocol Checks with `@runtime_checkable`

If you want to ensure that the check is done at runtime with `isinstance`,
you can use the decorator `runtime_checkable` to enable runtime instance
checks[^runtime-checkable] (you cannot call `isinstance` on `Flyable`
without this decorator):

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

## When Structural Subtyping Backfires: The LSP

Nominal and structural subtyping establish the same promise by different
tests, and the structural test has a blind spot: it evaluates the _presence
and signatures_ of members, not their _meaning_. Any class that happens to
match an interface's shape becomes its subtype — including classes that
satisfy the letter of the contract while violating its spirit. That is
precisely the failure the Liskov substitution principle names: substituting
such a "subtype" changes what the program observably does, breaching
{prf:ref}`the LSP theorem <type-theory-liskov-substitution-principle>`
stated in {doc}`Subsumption <03-subsumption>`.

The fixture already contains the culprit. Under the nominal test, `Robot`
was not an `Animal`. Give `Animal` a structural test instead — the same
`__subclasshook__` device `Sized` uses — and watch the verdict flip. First,
the helper CPython's ABCs rely on, reproduced with type annotations added:

```{code-cell} ipython3
from abc import ABCMeta
from types import NotImplementedType


def _check_methods(C: type, *methods: str) -> bool | NotImplementedType:
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
```

`_check_methods` walks the candidate's method resolution order and reports
whether every requested method is defined somewhere along it — `True` for a
match, `NotImplemented` for "no verdict, fall back to the usual check".
Now the hook goes onto `Animal`. One detail is load-bearing:
`__subclasshook__` is consulted only by `ABCMeta.__subclasscheck__`, so the
class **must** be built with `metaclass=ABCMeta` — on a plain class the
hook is silently ignored, and `issubclass` behaves as if it were never
written. (The cell deliberately redefines the fixture's `Animal`; `Dog`,
`Cat`, and `Robot` are untouched.)

```{code-cell} ipython3
class Animal(metaclass=ABCMeta):
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> str:
        return "Generic Animal Sound!"

    @classmethod
    def __subclasshook__(cls, C: type) -> bool | NotImplementedType:
        if cls is Animal:
            return _check_methods(C, "describe", "make_sound")
        return NotImplemented
```

```{code-cell} ipython3
print(issubclass(Robot, Animal))  # True — structurally admitted
print(isinstance(rob, Animal))    # True — the same rob that failed the nominal test
```

The same `Robot` — the very instance that printed `False` at the top of the
page — is now a subtype of `Animal`, because the structural test asks only
whether `describe` and `make_sound` exist. Semantically the classification
is nonsense: a robot is not an animal, and downstream code that assumes
animal behavior will now cheerfully accept one. The signatures match, the
semantics do not, and the LSP is violated without a line of inheritance in
sight.

````{admonition} How the checkers see these cells
:class: dropdown

Collect the two definition cells, the fixture's `Robot`, and the two
checks into `structural_animal.py`, and the checkers split. `pyright`
accepts the file as written:

```text
0 errors, 0 warnings, 0 informations
```

`mypy --strict`, however, objects to every `return NotImplemented`:

```text
structural_animal.py:11: error: Returning Any from function declared to return "bool | NotImplementedType"  [no-any-return]
structural_animal.py:14: error: Returning Any from function declared to return "bool | NotImplementedType"  [no-any-return]
structural_animal.py:29: error: Returning Any from function declared to return "bool | NotImplementedType"  [no-any-return]
Found 3 errors in 1 file (checked 1 source file)
```

Typeshed declares the type of `NotImplemented`, `types.NotImplementedType`,
as a class deriving from `Any` — which is why `pyright` lets it flow into
any return type, while strict `mypy` (`warn_return_any`) refuses to launder
an `Any` through a declared return. Note that neither checker has anything
to say about the _semantic_ absurdity of `Robot <: Animal`: no tool checks
meaning.
````

In practice the cure is design, not tooling: keep protocols small and
behavior-focused, name them for the capability they demand (`Sized`,
`Flyable`), and never use a shape as a proxy for a semantic category. Go —
a language built almost entirely on structural subtyping — has evolved
exactly these conventions, and
[Effective Go's interface guidance](https://go.dev/doc/effective_go#interfaces)
codifies them.

## Inclusive vs Coercive Implementations

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

If I had to compress this chapter into a single sentence, it would be the
contract we opened with: a subtype is a type whose values can **stand in**
for values of its supertype without the surrounding program noticing. Types
are sets of values, and subtyping is a _substitutability_ promise over those
sets — every program element written to operate on the supertype must keep
working when handed the subtype. Nominal and structural subtyping are not two
different promises; they are two different ways of _establishing_ the same
promise. And as the `Robot` example showed, the structural route can extend
the promise to types that match an interface's letter while violating its
spirit, which is why the Liskov substitution principle remains the semantic
yardstick behind both schemes.

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
checker applies when it lets a subtype stand in for its supertype. And the
opening's other refusal — a `list[int]` where a `list[float]` is wanted — is
deliberately not settled here: that is _variance_, the subject of
{doc}`Invariance, Covariance and Contravariance <06-invariance-covariance-contravariance>`.

## References and Further Readings

```{admonition} References
:class: seealso

-   [mypy - Protocols](https://mypy.readthedocs.io/en/stable/protocols.html)
-   [Python typing specification - Protocols](https://typing.python.org/en/latest/spec/protocol.html)
-   [Subtyping schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)
-   [Type Systems: Structural vs. Nominal Typing Explained - Medium](https://medium.com/@thejameskyle/type-systems-structural-vs-nominal-typing-explained-56511dd969f4)
-   [Subtyping - eduNitas](https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html)
-   [PEP 544 – Protocols: Structural subtyping (static duck typing)](https://peps.python.org/pep-0544/)
-   [Nominal Type System - Wikipedia](https://en.wikipedia.org/wiki/Nominal_type_system)
-   [Structural Type System - Wikipedia](https://en.wikipedia.org/wiki/Structural_type_system)
```

[^subtype-wikipedia]:
    [Subtyping - Wikipedia](https://en.wikipedia.org/wiki/Subtyping)

[^notation]:
    Other texts write the same relation as $S \subseteq T$ or $S \leq T$.
    This series uses $S <: T$ throughout.

[^subtype-schemes-wikipedia]:
    [Subtyping Schemes - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes)

[^subtype-subsumption-wikipedia]:
    [Subtype Subsumption - Wikipedia](https://en.wikipedia.org/wiki/Subtyping#Subsumption)

[^runtime-checkable]:
    [Using isinstance() with protocols](https://mypy.readthedocs.io/en/stable/protocols.html#using-isinstance-with-protocols)
