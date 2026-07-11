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

# Type Safety

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from typing import Protocol, cast
```

A subtype is a foundational concept in type theory and object-oriented
programming that facilitates type safety and polymorphism. The relationship
between a subtype $S$ and a supertype $T$ is denoted as $S <: T$,
$S \subseteq T$, or $S ≤: T$. Before we detail the
**_criterion_**[^what-is-a-criterion] for subtyping in
{doc}`Subsumption <03-subsumption>`, we state an important implication of
subtype — type safety.

```{prf:definition} Subtype and Type Safety
:label: type-theory-subtype-and-type-safety

If $S$ is a subtype of $T$, the subtyping
[relation](<https://en.wikipedia.org/wiki/Relation_(mathematics)>) (written as
$S \leq T$, $S <: T$, $S \subseteq T$, or $S \leq: T$) means that any term of
**type** $S$ can **safely** be used in any **context** where a term of **type**
$T$ is **expected**.
```

In other words, $S$ is a **_subtype_** of $T$ **_if a piece of code written
for variables of type_** $T$ **_can also safely be used on variables of
type_** $S$[^cs2040s-variable-and-type]. That single sentence is **subtype
polymorphism**: one function written against a supertype operates, unchanged,
on every subtype — and the word _safely_ carries all the weight, because
substituting an $S$ where a $T$ is expected must neither crash the program nor
quietly change what it means. The behavioral fine print behind _safely_ is the
Liskov substitution principle, which the next chapter states formally as
{prf:ref}`the LSP theorem <type-theory-liskov-substitution-principle>`, so I
will not re-derive it here. This chapter's job is more operational: to show
what the promise buys us when it holds, and exactly how things break when it
does not.

## A Type Safe Example

Assume for a moment that the classes `Cat` and `Dog` are both valid _subtypes_
of the class `Animal` through class inheritance, which we learned in
{doc}`Subtypes <01-subtypes>` to be called **nominal subtyping** (i.e.
subclasses are subtypes). The cell below reproduces the canonical
`Animal`/`Dog`/`Cat` fixture owned by that chapter — each page of this series
executes in its own kernel, so the classes must be redefined here, but they
are kept identical to chapter 1's definition of record.

```{code-cell} ipython3
# Reproduces the canonical fixture from 01-subtypes for this chapter's kernel.
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
```

Then any instance of `Dog` or `Cat` can safely be used in a context where an
instance of `Animal` is expected. Consider the function `describe_animal`,
which takes in an `animal` of type `Animal` — a signature telling developers
that we can pass in any animal, as long as its type is a subtype of `Animal`.
Note that we annotate each binding below with the type we intend it to have;
those declared types are exactly what the static type checkers will hold us to
in a moment.

```{code-cell} ipython3
def describe_animal(animal: Animal) -> str:
    return animal.describe() + " makes sound " + animal.make_sound()


generic_animal: Animal = Animal()
generic_animal_sound = describe_animal(generic_animal)
print(generic_animal_sound)

generic_dog: Dog = Dog()
generic_dog_sound = describe_animal(generic_dog)
print(generic_dog_sound)

generic_cat: Cat = Cat()
generic_cat_sound = describe_animal(generic_cat)
print(generic_cat_sound)
```

What we have described is inherently related to variable assignment. When you
pass `generic_dog` to `describe_animal`, you are essentially assigning
`generic_dog` to the function's parameter `animal`. By extension, the same
rule governs a plain assignment statement:

```{code-cell} ipython3
generic_animal = generic_dog  # accepted: Dog <: Animal
print(generic_animal.make_sound())
```

Substituting the **expression** `generic_dog` (an instance of `Dog`) for the
**variable** `generic_animal` (declared as `Animal`) is allowed because we
established that `Dog` is a subtype of `Animal` — neither `pyright` nor `mypy`
has any complaint here. The reverse direction, however, is exactly the
substitution that {prf:ref}`type-theory-subtype-and-type-safety` does not
license. Collect the class definitions and two fresh bindings into
`animals.py` and append the reversed assignment:

```python
generic_animal: Animal = Animal()
generic_dog: Dog = Dog()

generic_dog = generic_animal  # Animal is not a subtype of Dog
```

Both checkers reject the final line. `pyright` reports

```text
animals.py:28:15 - error: Type "Animal" is not assignable to declared type "Dog"
    "Animal" is not assignable to "Dog" (reportAssignmentType)
1 error, 0 warnings, 0 informations
```

and `mypy --strict` agrees:

```text
animals.py:28: error: Incompatible types in assignment (expression has type "Animal", variable has type "Dog")  [assignment]
Found 1 error in 1 file (checked 1 source file)
```

The asymmetry is the whole point. Assigning a `Dog` to a variable of type
`Animal` is safe because `Dog` supports everything `Animal` promises
(`describe`, `make_sound`) and possibly more (`fetch`), so no code holding
`generic_animal` can be surprised. Assigning a plain `Animal` to a variable of
type `Dog` is unsafe because the promise now runs backwards: code holding
`generic_dog` is entitled to call `fetch()`, and a plain `Animal` has no such
method.

## Violating Type Safety

In {doc}`Subtypes <01-subtypes>` we met a `Robot` whose methods happened to
mirror `Animal`'s exactly. The `Robot` below is a nastier variant — its
`make_sound` returns an `int` where every `Animal` returns a `str`. It is not
a subtype of `Animal` under any scheme: not nominally (it inherits from
nothing), and not structurally either (its `make_sound` has an incompatible
return type).

```{code-cell} ipython3
class Robot:
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> int:  # returns int, not str
        return 1


robot: Robot = Robot()
```

Passing this `Robot` to `describe_animal` is therefore a type error, and both
checkers say so at static-analysis time — before the program ever runs. With
the fixture, `Robot`, and the offending call collected into `robots.py`:

```python
robot: Robot = Robot()
robot_sound = describe_animal(robot)
```

`pyright` reports

```text
robots.py:22:31 - error: Argument of type "Robot" cannot be assigned to parameter "animal" of type "Animal" in function "describe_animal"
    "Robot" is not assignable to "Animal" (reportArgumentType)
1 error, 0 warnings, 0 informations
```

and `mypy --strict` agrees:

```text
robots.py:22: error: Argument 1 to "describe_animal" has incompatible type "Robot"; expected "Animal"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

At runtime, though, Python enforces none of this — annotations are not
checked while the program runs. To stage the crash in a page whose executed
cells must themselves stay checker-clean, we have to overrule the checker
explicitly, and `typing.cast` is the standard tool for that:
`cast(Animal, robot)` returns `robot` completely unchanged (a no-op at
runtime) while telling the static checker "trust me, treat this as an
`Animal`". It is the honest way to write "imagine this call site had no
checker":

```{code-cell} ipython3
try:
    robot_sound = describe_animal(cast(Animal, robot))
    print(robot_sound)
except TypeError as err:
    print(f"TypeError: {err}")
```

The failure is a `TypeError` raised mid-concatenation: `describe_animal` adds
the string `" makes sound "` to whatever `make_sound()` returned, and for
`Robot` that is the integer `1`. Since `Robot` is not a subtype of `Animal`,
there is no type-safety guarantee — and this loud, immediate crash is the
_good_ outcome. The same violation is far worse if the function is written
with an f-string instead, because then nothing fails at all:

```{code-cell} ipython3
def introduce_animal(animal: Animal) -> str:
    return f"{animal.describe()} makes sound {animal.make_sound()}"


print(introduce_animal(cast(Animal, robot)))
```

The f-string happily formats the integer, the function returns a
plausible-looking sentence, and a hidden bug is now loose: downstream code
believes `1` is what a `Robot` sounds like, and no exception will ever point
here.

And this can happen in real code, sometimes unknowingly. In the loop below we
iterate over entities that are _presumed_ to be `Animal`s — the annotation on
`entities` records precisely that presumption. Sneaking a `Robot` into the
list would once have sailed through review and failed only in production (or
worse, never failed, merely lied). With the element type declared, both
checkers stop the `Robot` at the door, before the loop ever runs. In
`entities.py`:

```python
entities: list[Animal] = [Dog(), Cat(), Robot()]
for entity in entities:
    describe_animal(entity)
```

`pyright` pins the blame on the exact element,

```text
entities.py:31:41 - error: Type "list[Dog | Cat | Robot]" is not assignable to declared type "list[Animal]"
    "Robot" is not assignable to "Animal" (reportAssignmentType)
1 error, 0 warnings, 0 informations
```

and so does `mypy --strict`:

```text
entities.py:31: error: List item 2 has incompatible type "Robot"; expected "Animal"  [list-item]
Found 1 error in 1 file (checked 1 source file)
```

(type-theory-type-safety-int-float-promotion)=

## Why `int` is *not* a subtype of `float`

Sooner or later every discussion of Python's type system meets this
assignment — a name annotated `float`, initialized with an `int` — and
discovers that the checkers wave it through:

```{code-cell} ipython3
x: float = 3
print(x, type(x).__name__)

x = 3.99
print(x, type(x).__name__)
```

Both checkers accept the cell exactly as written. `pyright`:

```text
0 errors, 0 warnings, 0 informations
```

`mypy --strict`:

```text
Success: no issues found in 1 source file
```

It is tempting to conclude that `int` must be a subtype of `float`. An earlier
revision of this very page did exactly that, and then embellished it with a
story about the reverse assignment "truncating to 3 silently". Both claims are
false, and the printed output above already contains the evidence:

1. **No conversion happened.** After `x: float = 3`, the name `x` is bound to
   the very same `int` object `3` — `type(x).__name__` prints `int`, not
   `float`. The annotation constrains what the _checker_ accepts for `x`; it
   does nothing at runtime.
2. **Nothing truncates.** `x = 3.99` simply **rebinds** the name `x` to the
   `float` object `3.99`. The truncation story imports a mental model from
   languages where a variable is a typed memory cell that values are poured
   into. In Python, assignment never converts a value: the name is rebound to
   the new object, whole, every time.

The runtime class hierarchy is just as blunt about the subtype question:

```{code-cell} ipython3
print(issubclass(int, float))  # int does not inherit from float
print(isinstance(3, float))    # a value of int is not a value of float
print(isinstance(3.0, int))    # nor is a value of float a value of int
print(issubclass(bool, int))   # bool, by contrast, IS a genuine subclass of int
```

No value of `int` is a value of `float`, and `int` is not a subclass of
`float` — contrast `bool`, which really is a subclass of `int` and therefore a
true nominal subtype. What the checkers implement instead is a special case
written directly into PEP 484's "numeric tower" section: "when an argument is
annotated as having type `float`, an argument of type `int` is
acceptable"[^pep-484-numeric-tower], and likewise `int` and `float` are both
accepted where `complex` is expected. The maintained typing specification
carries the rule forward under the telling title _Special cases for `float`
and `complex`_[^typing-spec-numeric-tower]. This checker-level rule is called
**promotion**: `int` is _promoted_ to `float` for assignability purposes, by
fiat, without being its subtype.

Promotion is deliberately unsound, and the crack is easy to exhibit. `float`
has methods that `int` lacks — `float.hex()` is one — so code that is
perfectly well-typed against `float` can crash when handed a promoted `int`:

```{code-cell} ipython3
def to_hex(x: float) -> str:
    return x.hex()


print(to_hex(3.14))

try:
    print(to_hex(3))
except AttributeError as err:
    print(f"AttributeError: {err}")
```

Run the checkers over that cell and they bless it — `pyright` again says
`0 errors, 0 warnings, 0 informations` and `mypy --strict` again says
`Success: no issues found in 1 source file` — yet the blessed code raises
`AttributeError` at runtime. Measured against
{prf:ref}`the subtype criterion <type-theory-subtype-criterion>` that the next
chapter builds, `int <: float` fails on its first clause, value inclusion
(`isinstance(3, float)` is `False`), and on its second, function applicability
(`float.hex` is not applicable to `int` values). The typing spec accepts this
sliver of unsoundness in exchange for an enormous ergonomic win: without
promotion, every numeric API would force callers to write `f(float(3))`.

Promotion is also one-way — there is no companion rule demoting `float` to
`int`, so the checkers reject the reverse assignment. In `promotion.py`:

```python
y: int = 3.0
```

`pyright` reports

```text
promotion.py:1:10 - error: Type "float" is not assignable to declared type "int"
    "float" is not assignable to "int" (reportAssignmentType)
1 error, 0 warnings, 0 informations
```

and `mypy --strict` agrees:

```text
promotion.py:1: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
Found 1 error in 1 file (checked 1 source file)
```

The takeaway to carry through the rest of the series: `x: float = 3` is
checker courtesy, not subtyping. When a later chapter needs the fact that
`int` is not a subtype of `float`, this section is the receipt.

## Static, Dynamic, and Gradual Typing

We have come quite a long way in understanding subtyping schemes and the
concept of type safety. I would like to end this discussion with a brief note
on _when_ type checking happens — dynamically, statically, or gradually. In
what follows, we turn to Jeremy Siek — who, together with Walid Taha,
introduced gradual typing[^siek-taha-2006] — and his essay _What is Gradual
Typing_[^what-is-gradual-typing] as a reference.

### Dynamic Type Checking

Dynamic type checking is performed at runtime, by the running program itself.
Consider the following code, where we erroneously pass a `Car` object to the
`can_we_fly` function, which expects a `Flyable` — the same `Protocol` we
built in {doc}`Subtypes <01-subtypes>`. To get the mistaken call written down
at all in a page whose executed cells must stay checker-clean, we launder it
through `cast` once more; the cast changes nothing at runtime, so what
executes below is exactly the buggy call. The mistake manifests as an
`AttributeError` the moment `obj.fly` is looked up:

```{code-cell} ipython3
class Flyable(Protocol):
    def fly(self) -> str: ...


class Bird:
    def fly(self) -> str:
        return "Bird flying"


class Car:
    def drive(self) -> str:
        return "Car driving"


def can_we_fly(obj: Flyable) -> str | None:
    try:
        return obj.fly()
    except AttributeError as err:
        print(err)
        return None


bird = Bird()
car = Car()

_ = can_we_fly(bird)
_ = can_we_fly(cast(Flyable, car))
```

This is dynamic type checking: the object's fitness for the operation is
discovered only when the attribute lookup actually executes, and the error is
caught only on the runs — and code paths — that reach it.

### Static Type Checking

Static type checking, on the other hand, happens at **static-analysis time**:
the checker reads the source before the program ever runs. Python has no such
phase built in, so the checking is done by an external tool such as `pyright`
or `mypy`. Hand them the same code with the cast deleted (as `flyable.py`),

```python
_ = can_we_fly(car)
```

and the bad call is flagged without executing anything. `pyright` reports

```text
flyable.py:30:16 - error: Argument of type "Car" cannot be assigned to parameter "obj" of type "Flyable" in function "can_we_fly"
    "Car" is incompatible with protocol "Flyable"
      "fly" is not present (reportArgumentType)
1 error, 0 warnings, 0 informations
```

and `mypy --strict` agrees:

```text
flyable.py:30: error: Argument 1 to "can_we_fly" has incompatible type "Car"; expected "Flyable"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

It is worth noting that a static type checker makes a _conservative
approximation_[^what-is-gradual-typing] of what can happen to the code at
**runtime**, and raises _potential_ type errors that may happen at runtime. In
fact, the
[**halting problem**](https://en.wikipedia.org/wiki/Halting_problem) implies
that we cannot be 100% sure whether a type error will really occur during
runtime before execution, and thus it is impossible to build a type checker
that can "predict" what type errors will happen at
runtime[^what-is-gradual-typing]. Consequently, static type checkers make
conservative approximations to ensure that the code is type-safe, and these
result in false positives (raising type errors that will not actually occur at
runtime).

We quote verbatim the example given by Jeremy Siek in his essay. Consider the
following Java code:

```java
class A {
    int add1(int x) {
        return x + 1;
    }
    public static void main(String args[]) {
        A a = new A();
        if (false)
            add1(a);
        else
            System.out.println("Hello World!");
    }
}
```

The Java compiler rejects this program even though it would not actually
result in a type error, because the `if (false)` branch is never taken.
However, the Java type checker does not try to figure out which branch of an
`if` statement will be taken at runtime. Instead it conservatively assumes
that either branch could be taken and therefore checks both
branches[^what-is-gradual-typing].

### Comparison between Dynamic and Static Type Checking

Some reasons Jeremy gives include, but are not limited to:

-   Static type checking enhances execution speed by eliminating the need for
    type verification during runtime and allowing for the utilization of more
    efficient data storage formats.
-   Dynamic type checking simplifies the handling of scenarios where the type
    of a value is determined by information available at runtime.

You can find more reasons in his essay[^what-is-gradual-typing].

### Gradual Typing

Python refuses to pick a side, and that refusal has a name. **Gradual
typing** — introduced by Siek and Taha in 2006[^siek-taha-2006] — lets
statically and dynamically checked code coexist in one program, with the
programmer choosing per region: annotated code is checked at static-analysis
time, while unannotated code receives the dynamic treatment above. This is
exactly the design [PEP 483](https://peps.python.org/pep-0483/) adopted for
Python. The seams between the two regimes are the escape hatches we have
already used: an unannotated name is implicitly `Any` — the type on which the
checker gives up — and `cast` overrides the checker one expression at a time.
We even watched the seam split the two checkers apart in chapter 1's
duck-typing exhibit ({doc}`Subtypes <01-subtypes>`), where the same untyped
loop drew a hard error from `mypy --strict` and a silent shrug (`Unknown`, an
implicit `Any`) from `pyright`. Gradual typing is why both reactions are
legal: outside the annotated region, the language only promises to fail at
runtime — exactly as this page's `cast`-staged crashes did.

## Summary

If this chapter had to survive as one sentence, it would be the definition it
opened with: a subtype relation is a **license to substitute**, and type
safety is the guarantee that honoring the license never corrupts the program.
Everything else was that sentence run forwards and backwards. Substituting a
`Dog` where an `Animal` is expected is safe by construction; substituting an
`Animal` where a `Dog` is expected, or a `Robot` anywhere at all, breaks the
guarantee — sometimes loudly (a `TypeError` mid-concatenation), sometimes
silently (an f-string cheerfully reporting that a robot "makes sound 1"). The
static type checkers exist to catch every one of those violations at
static-analysis time, at the price of conservative false positives; the
runtime catches only what actually executes, and only when the failure is
loud enough to notice.

| Demonstration                        | pyright and mypy verdict | Runtime outcome                           |
| ------------------------------------ | ------------------------ | ----------------------------------------- |
| `describe_animal(generic_dog)`       | accepted                 | works — `Dog <: Animal`                   |
| `generic_dog = generic_animal`       | rejected                 | runs, but a later `fetch()` would fail    |
| `describe_animal(robot)`             | rejected                 | loud `TypeError` mid-concatenation        |
| `introduce_animal(robot)` (f-string) | rejected                 | silent wrong output — the worst case      |
| `x: float = 3`                       | accepted (promotion)     | `x` is still an `int` object              |
| `to_hex(3)`                          | accepted (promotion)     | `AttributeError` — promotion's blind spot |

And one myth is now retired for good: `int` is **not** a subtype of `float`.
The checkers' tolerance of `x: float = 3` is the numeric tower's _promotion_
special case — one-way, checker-level only — and nothing about assignment ever
truncates a value.

The obvious next question is the one this chapter kept gesturing at: _by what
rule_ does a checker decide that `Dog <: Animal` holds while `Robot <: Animal`
and `int <: float` do not? {doc}`Subsumption <03-subsumption>` answers with
the formal {prf:ref}`subtype criterion <type-theory-subtype-criterion>` and
the reflexive, transitive structure of the subtyping relation itself.

## References and Further Readings

```{admonition} References
:class: seealso

-   [Unit 2: Variable and Type - CS2030S](https://nus-cs2030s.github.io/2021-s2/02-type.html)
-   [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/)
-   [PEP 484 – Type Hints: The numeric tower](https://peps.python.org/pep-0484/#the-numeric-tower)
-   [Python typing specification: Special cases for `float` and `complex`](https://typing.python.org/en/latest/spec/special-types.html#special-cases-for-float-and-complex)
-   [Siek, J. G., & Taha, W. (2006). Gradual Typing for Functional Languages](https://jsiek.github.io/home/siek06gradual.pdf)
-   [What is Gradual Typing - Jeremy Siek](https://jsiek.github.io/home/WhatIsGradualTyping.html)
-   [Subtyping - eduNitas](https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html)
```

[^what-is-a-criterion]:
    A criterion is a principle or standard by which something may be judged or
    decided. In this context, the criterion for subtype is the principle or
    standard by which we decide if one type is a subtype of another type.

[^cs2040s-variable-and-type]:
    [Unit 2: Variable and Type - CS2030S](https://nus-cs2030s.github.io/2021-s2/02-type.html)

[^pep-484-numeric-tower]:
    [PEP 484 – Type Hints: The numeric tower](https://peps.python.org/pep-0484/#the-numeric-tower)

[^typing-spec-numeric-tower]:
    [Special cases for `float` and `complex` - Python typing specification](https://typing.python.org/en/latest/spec/special-types.html#special-cases-for-float-and-complex)

[^siek-taha-2006]:
    Jeremy G. Siek and Walid Taha.
    [_Gradual Typing for Functional Languages_](https://jsiek.github.io/home/siek06gradual.pdf).
    In _Scheme and Functional Programming Workshop_, 2006, pp. 81–92.

[^what-is-gradual-typing]:
    [What is Gradual Typing - Jeremy Siek](https://jsiek.github.io/home/WhatIsGradualTyping.html),
    the essay previously hosted at `wphomes.soic.indiana.edu`.
