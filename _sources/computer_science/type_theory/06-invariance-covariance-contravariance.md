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
            "Covariance, contravariance, and invariance in Python generics:
            why list[int] is not a list[float], from Guido's classic example
            to PEP 695 inferred variance."
        "keywords":
            "python, covariance, contravariance, invariance, variance,
            generics, type theory"
---

# Covariance, Contravariance, and Invariance in Python Generics

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

from collections.abc import Callable, Iterator, Sequence
```

Pass an `int` where a `float` is expected and the checkers stay silent. Wrap
both sides in a list — pass a `list[int]` where a `list[float]` is expected —
and `mypy` and `pyright` both object. This chapter answers the question that
refusal raises: when does a subtyping relationship between two types carry
over to the generic types built from them?

The answer is **variance**: the rule attached to every generic type —
`list[T]`, `Sequence[T]`, `Callable[[T], R]` — that says whether subtyping
between type arguments lifts to subtyping between the constructed types
(covariance), reverses direction (contravariance), or does not transfer at
all (invariance). It is the corner of the type system a checker confronts
you with earliest, and the one whose error messages are hardest to decode
without the theory.

By the end you can state the three variances formally and predict either
checker's verdict on any generic assignment. You will know why mutability
forces `list` to be invariant while read-only `Sequence` gets to be
covariant, how PEP 695 _infers_ the variance of your own generic classes
where legacy `TypeVar` _declared_ it, and why `Callable` treats parameter
types and return types in opposite directions.

```{admonition} Prerequisites
:class: note

This chapter leans on the subtype relation $S <: T$ ("an $S$ can stand in
for a $T$") from {doc}`Subtyping in Python <01-subtypes>`, the three-part
{prf:ref}`subtype criterion <type-theory-subtype-criterion>` from
{doc}`Subsumption <03-subsumption>`, and generic classes and type variables
from {doc}`Generics and Type Variables <04-generics>`. Examples run on
Python 3.14; every checker claim is verified against `mypy --strict` 2.2.0
and `pyright` 1.1.411.
```

## Why Is `list[int]` Not a `list[float]`?

Start with the classic example Guido van Rossum used to introduce variance
when presenting Python's type hints[^guido-pycon-2015]:

```{code-cell} ipython3
def append_pi(lst: list[float]) -> None:
    lst.append(3.14)


my_list: list[int] = [1, 3, 5]
append_pi(my_list)  # naively, this should be safe...
print(my_list)
```

At runtime nothing objects: Python appends the `float` and moves on. But
`my_list` was declared a `list[int]` and now contains `3.14`, so any
downstream code that trusts the annotation — say, calling `.bit_length()`
on each element — is one iteration away from an `AttributeError`. Save the
same five lines as `append_pi.py` and the static checkers refuse the call.
`mypy --strict`:

```text
append_pi.py:6: error: Argument 1 to "append_pi" has incompatible type "list[int]"; expected "list[float]"  [arg-type]
append_pi.py:6: note: "list" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
append_pi.py:6: note: Consider using "Sequence" instead, which is covariant
Found 1 error in 1 file (checked 1 source file)
```

`pyright` agrees and spells out the reason:

```text
append_pi.py:6:11 - error: Argument of type "list[int]" cannot be assigned to parameter "lst" of type "list[float]" in function "append_pi"
    "list[int]" is not assignable to "list[float]"
      Type parameter "_T@list" is invariant, but "int" is not the same as "float"
      Consider switching from "list" to "Sequence" which is covariant (reportArgumentType)
1 error, 0 warnings, 0 informations
```

Two tempting explanations fail. The first says the checkers must not accept
an `int` where a `float` is expected at all — but they do, by the numeric
tower's promotion rule, and
{ref}`the type-safety chapter holds the receipt <type-theory-type-safety-int-float-promotion>`.
The second says the problem
is that promotion is not _real_ subtyping — but swapping in a genuine
nominal subtype changes nothing. `bool` is a true subclass of `int`, yet a
`list[bool]` argument for a `list[int]` parameter draws the same verdict:

```python
def count_truthy(flags: list[int]) -> int:
    return sum(1 for flag in flags if flag)


bools: list[bool] = [True, False, True]
count_truthy(bools)
```

```text
bools.py:6: error: Argument 1 to "count_truthy" has incompatible type "list[bool]"; expected "list[int]"  [arg-type]
bools.py:6: note: "list" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
bools.py:6: note: Consider using "Sequence" instead, which is covariant
Found 1 error in 1 file (checked 1 source file)
```

(`pyright` mirrors it: `"bool" is not the same as "int"`.) So the refusal is
not about the element types at all. It is a property of `list` itself — the
property both error messages name: **invariance**, glossed for now as "the
type argument must match exactly", and defined formally in the next section.

That property follows from `list`'s own interface. Here are the
load-bearing lines of its typeshed stub, as bundled with `mypy`
2.2.0[^typeshed-builtins] (comments mine):

```python
class list(MutableSequence[_T]):
    def append(self, object: _T, /) -> None: ...           # consumes a _T
    def pop(self, index: SupportsIndex = -1, /) -> _T: ...  # produces a _T
    def __iter__(self) -> Iterator[_T]: ...                # produces _T's
```

Read the type variable `_T` positionally. In `append` it stands in
**parameter position**: the list _consumes_ values of type `_T`. In `pop`
and `__iter__` it stands in **return position**: the list _produces_ them.
A type variable pulled in both directions leaves the checker exactly one
sound option, and the sections below prove it: consuming rules out
covariance, producing rules out contravariance, and what remains is
invariance. The full stub, for the reader who wants every overload, is one
click away.

````{admonition} The full typeshed stub for `list`
:class: dropdown

Verbatim from the typeshed `builtins.pyi` bundled with `mypy`
2.2.0[^typeshed-builtins]:

```python
class list(MutableSequence[_T]):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, iterable: Iterable[_T], /) -> None: ...

    def copy(self) -> list[_T]: ...
    def append(self, object: _T, /) -> None: ...
    def extend(self, iterable: Iterable[_T], /) -> None: ...
    def pop(self, index: SupportsIndex = -1, /) -> _T: ...
    # Signature of `list.index` should be kept in line with `collections.UserList.index()`
    # and multiprocessing.managers.ListProxy.index()
    def index(self, value: _T, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize, /) -> int: ...
    def count(self, value: _T, /) -> int: ...
    def insert(self, index: SupportsIndex, object: _T, /) -> None: ...
    def remove(self, value: _T, /) -> None: ...

    # Signature of `list.sort` should be kept inline with `collections.UserList.sort()`
    # and multiprocessing.managers.ListProxy.sort()
    #
    # Use list[SupportsRichComparisonT] for the first overload rather than [SupportsRichComparison]
    # to work around invariance
    @overload
    def sort(self: list[SupportsRichComparisonT], *, key: None = None, reverse: bool = False) -> None: ...
    @overload
    def sort(self, *, key: Callable[[_T], SupportsRichComparison], reverse: bool = False) -> None: ...

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T]: ...
    __hash__: ClassVar[None]  # type: ignore[assignment]

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> _T: ...
    @overload
    def __getitem__(self, s: slice[SupportsIndex | None], /) -> list[_T]: ...

    @overload
    def __setitem__(self, key: SupportsIndex, value: _T, /) -> None: ...
    @overload
    def __setitem__(self, key: slice[SupportsIndex | None], value: Iterable[_T], /) -> None: ...

    def __delitem__(self, key: SupportsIndex | slice[SupportsIndex | None], /) -> None: ...

    # Overloading looks unnecessary, but is needed to work around complex mypy problems
    @overload
    def __add__(self, value: list[_T], /) -> list[_T]: ...
    @overload
    def __add__(self, value: list[_S], /) -> list[_S | _T]: ...

    def __iadd__(self, value: Iterable[_T], /) -> Self: ...  # type: ignore[misc]
    def __mul__(self, value: SupportsIndex, /) -> list[_T]: ...
    def __rmul__(self, value: SupportsIndex, /) -> list[_T]: ...
    def __imul__(self, value: SupportsIndex, /) -> Self: ...
    def __contains__(self, key: object, /) -> bool: ...
    def __reversed__(self) -> Iterator[_T]: ...
    def __gt__(self, value: list[_T], /) -> bool: ...
    def __ge__(self, value: list[_T], /) -> bool: ...
    def __lt__(self, value: list[_T], /) -> bool: ...
    def __le__(self, value: list[_T], /) -> bool: ...
    def __eq__(self, value: object, /) -> bool: ...
    def __class_getitem__(cls, item: Any, /) -> GenericAlias: ...
```
````

## What Are Covariance, Contravariance, and Invariance?

The definitions need one piece of vocabulary. A **type constructor** is a
type-level function: it takes a type (or several) and returns a type.
`list` is a type constructor — feed it `int` and it returns the concrete
type `list[int]` — and so are `Sequence`, `dict`, and `Callable`;
{doc}`Generics and Type Variables <04-generics>` introduced these as
generic type constructors. Throughout, $C[U]$ denotes the type that a
constructor $C$ returns for the type argument $U$.

**Variance** answers one question: given $S <: T$, how do $C[S]$ and $C[T]$
relate? There are three possible answers, and each is a property of the
constructor $C$ — chosen per constructor, not per program.

```{prf:definition} Covariance
:label: computer-science-type-theory-covariance

A type constructor $C$ is **covariant** in a type parameter if subtyping
between type arguments lifts to the constructed types in the _same_
direction: for all types $S$ and $T$,

$$
S <: T \implies C[S] <: C[T].
$$

Covariance preserves the ordering of types, from more specific to more
generic. It is the sound choice when $C$ only _produces_ values of the
parameter type.
```

```{prf:definition} Contravariance
:label: computer-science-type-theory-contravariance

A type constructor $C$ is **contravariant** in a type parameter if
subtyping between type arguments lifts to the constructed types in the
_reversed_ direction: for all types $S$ and $T$,

$$
S <: T \implies C[T] <: C[S].
$$

Contravariance reverses the ordering of types. It is the sound choice when
$C$ only _consumes_ values of the parameter type.
```

```{prf:definition} Invariance
:label: computer-science-type-theory-invariance

A type constructor $C$ is **invariant** in a type parameter if subtyping
between distinct type arguments does not lift in either direction: for all
types $S$ and $T$,

$$
S \neq T \implies \neg\bigl(C[S] <: C[T]\bigr) \wedge \neg\bigl(C[T] <: C[S]\bigr),
$$

and hence, using reflexivity of $<:$,

$$
C[S] <: C[T] \iff S = T.
$$

An invariant $C[S]$ can be used only where exactly $C[S]$ is expected, even
when $S <: T$ holds. Checkers read the equality as mutual assignability —
`pyright`'s message above says it verbatim: _invariant, but "int" is not
the same as "float"_.
```

```{admonition} Intuition: variance is monotonicity
:class: tip

Subtyping orders types, so picture a type constructor as a function between
ordered sets, $X \mapsto C[X]$. A covariant constructor behaves like the
monotonically increasing $f(x) = 2x$: it preserves order, $x_1 \leq x_2
\implies f(x_1) \leq f(x_2)$. A contravariant constructor behaves like the
monotonically decreasing $g(x) = -x$: it reverses order. An invariant
constructor behaves like $h(x) = x^2$ on all of $\mathbb{R}$: knowing
$x_1 \leq x_2$ tells you nothing about how $h(x_1)$ and $h(x_2)$ compare.
```

## Why Is `list` Invariant in Python?

Invariance is the correct verdict for `list`, and each rejected alternative
exposes one half of the reason. The demonstrations share a pair of genuine
nominal subtypes, $\texttt{Manager} <: \texttt{Employee}$:

```{code-cell} ipython3
class Employee:
    def work(self) -> None: ...


class Manager(Employee):
    def manage(self) -> None: ...
```

### What would go wrong if `list` were covariant?

Covariance would turn $\texttt{Manager} <: \texttt{Employee}$ into
`list[Manager] <: list[Employee]`, so a `list[Manager]` could be passed to
any function expecting `list[Employee]` — including one that _writes_:

```{code-cell} ipython3
def add_employee(employees: list[Employee], employee: Employee) -> None:
    employees.append(employee)


managers: list[Manager] = [Manager(), Manager()]
add_employee(managers, Employee())  # covariance would allow this call

for manager in managers:
    try:
        manager.manage()
    except AttributeError as err:
        print(err)
```

Through the covariant door, a plain `Employee` walked into a
`list[Manager]`. The list's static type still promises `Manager` elements,
the promise is now false, and the crash surfaces later and far from the
call that caused it. This is a violation of the
{prf:ref}`subtype criterion <type-theory-subtype-criterion>`'s second
clause: `list[Manager]` cannot honor `list[Employee]`'s contract, because
its `append` accepts only `Manager`s while `list[Employee].append` must
accept _any_ `Employee`. Collect the cells into `covariant_list.py` and
`mypy --strict` blocks the door (`pyright` concurs: _"Manager" is not the
same as "Employee"_):

```text
covariant_list.py:14: error: Argument 1 to "add_employee" has incompatible type "list[Manager]"; expected "list[Employee]"  [arg-type]
covariant_list.py:14: note: "list" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
covariant_list.py:14: note: Consider using "Sequence" instead, which is covariant
Found 1 error in 1 file (checked 1 source file)
```

### What would go wrong if `list` were contravariant?

The reverse direction fails on reads. Contravariance would turn
$\texttt{Manager} <: \texttt{Employee}$ into
`list[Employee] <: list[Manager]`, letting a `list[Employee]` stand in
where a `list[Manager]` is expected. But code holding a `list[Manager]` is
entitled to _read_ a `Manager` back out — `lst.pop().manage()` — and the
substituted list may hand it a plain `Employee` with no `manage` at all.
Where covariance broke the consuming operations, contravariance breaks the
producing ones.

```{prf:proposition} Consuming and producing together force invariance
:label: type-theory-06-variance-both-positions-invariant

Let $C$ be a generic class whose interface both **consumes** its type
parameter (the parameter appears in parameter position of some method) and
**produces** it (the parameter appears in return position). Then neither
covariance nor contravariance is sound, so $C$ must be **invariant**:

-   If $C$ were covariant, then $C[S] <: C[T]$ for $S <: T$, and a context
    holding a $C[S]$ through the type $C[T]$ may feed a consuming method a
    $T$ that is not an $S$ — the `append` failure above.
-   If $C$ were contravariant, then $C[T] <: C[S]$ for $S <: T$, and a
    context expecting $C[S]$ may receive from a producing method a $T$ that
    is not an $S$ — the `pop` failure above.
```

`list` is exactly such a class — `append` consumes `_T`, `pop` produces it
— so the stub excerpt in the first section already contained the whole
proof. `mypy`'s note that "`list` is invariant" is this proposition applied
to typeshed.

## Why Does Immutability Make Covariance Safe?

Strike every consuming method and the covariant half of the conflict
disappears. A read-only view of a collection cannot be written through, so
the `append` exploit has no door; every remaining operation _produces_
elements, and by the proposition above production is compatible with
covariance. This is why both checkers suggested `Sequence`: typeshed
defines it with no mutating methods and marks its type variable covariant —
the `_co` suffix is the naming convention for one:

```python
class Sequence(Reversible[_T_co], Collection[_T_co]): ...
```

With a covariant constructor, the opening example type-checks. A
`list[int]` is a `Sequence[int]`, and covariance lifts the element-level
acceptability of `int` where `float` is expected up to the container:

```{code-cell} ipython3
def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


whole_numbers: list[int] = [1, 3, 5]
print(mean(whole_numbers))
```

On this file `mypy --strict` reports
`Success: no issues found in 1 source file` and `pyright` reports
`0 errors, 0 warnings, 0 informations`. Same values, same call shape as
`append_pi` — the only change is a parameter type whose constructor cannot
write.

The same trade is available for your own classes. Here is an immutable
list, written with a PEP 695 type parameter (Python 3.12+) — note that
nowhere do we declare a variance:

```{code-cell} ipython3
class ImmutableList[T]:
    def __init__(self, items: list[T]) -> None:
        self._items = list(items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)


def survey(employees: ImmutableList[Employee]) -> None:
    for employee in employees:
        employee.work()


managers_view: ImmutableList[Manager] = ImmutableList([Manager(), Manager()])
survey(managers_view)
```

The runtime is satisfied. The checkers are not — and their refusal is the
most instructive error in this chapter. On the collected
`immutable_list.py`, `mypy --strict`:

```text
immutable_list.py:26: error: Argument 1 to "survey" has incompatible type "ImmutableList[Manager]"; expected "ImmutableList[Employee]"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

`pyright` names the culprit:

```text
immutable_list.py:26:8 - error: Argument of type "ImmutableList[Manager]" cannot be assigned to parameter "employees" of type "ImmutableList[Employee]" in function "survey"
    "ImmutableList[Manager]" is not assignable to "ImmutableList[Employee]"
      Type parameter "T@ImmutableList" is invariant, but "Manager" is not the same as "Employee" (reportArgumentType)
1 error, 0 warnings, 0 informations
```

_Invariant?_ We never declared any variance — with PEP 695 syntax there is
no place to declare it. The variance of a modern type parameter is
**inferred** from the class body, and something in this class made both
checkers infer invariance. That something is the storage.

## How Does PEP 695 Infer Variance — and How Did `TypeVar` Declare It?

PEP 695 removed variance declarations from the syntax; the typing
specification instead has checkers _compute_ each parameter's variance from
the class body[^spec-variance-inference]. The algorithm builds two
specializations of the class — an `upper` copy with the parameter replaced
by `object`, and a `lower` copy with the parameter kept as itself — and
compares them with the normal assignability rules: if `lower` is assignable
to `upper`, the parameter is covariant; if `upper` is assignable to
`lower`, contravariant; if neither, invariant.

Run that on `ImmutableList`. The method side is fine: `__iter__` returns
`Iterator[T]`, `Iterator` is covariant, so the `lower` iterator is
assignable to the `upper` one. The attribute side is not: `self._items` has
type `list[T]`, and `list` — this chapter's own subject — is invariant, so
`list[T]` and `list[object]` are assignable in _neither_ direction. The
comparison fails both ways, and `T` is invariant. Note that every member
counts, including underscore-"private" attributes like `_items`: the
underscore is a convention for human readers, not a boundary the inference
algorithm respects.

The fix is to make the storage as immutable as the interface. `tuple` is
covariant — its stub produces `_T_co` and consumes nothing — so swapping
the internal `list` for a `tuple` flips the inference:

```{code-cell} ipython3
class ImmutableList[T]:
    def __init__(self, items: list[T]) -> None:
        self._items: tuple[T, ...] = tuple(items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)


def survey(employees: ImmutableList[Employee]) -> None:
    for employee in employees:
        employee.work()


managers_view: ImmutableList[Manager] = ImmutableList([Manager(), Manager()])
survey(managers_view)
```

One changed line, and both checkers accept the collected file —
`mypy --strict`: `Success: no issues found in 1 source file`; `pyright`:
`0 errors, 0 warnings, 0 informations`. Under PEP 695 the link between
mutability and variance is not philosophy; it is the literal input to the
variance computation. Store state mutably and your class is invariant;
store it immutably and covariance falls out for free.

### How did legacy `TypeVar` declare variance?

Before PEP 695 — and in any code still using `typing.TypeVar` with
`Generic` — variance was **declared**, not inferred. The flag lives on the
type variable, with a matching naming convention: `_T_co` for
`covariant=True`, `_T_contra` for `contravariant=True`, and a bare `_T`
for the invariant default. This block is legacy syntax, shown as such:

```python
from collections.abc import Iterator
from typing import Generic, TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class FrozenBag(Generic[_T_co]):
    def __init__(self, items: list[_T_co]) -> None:
        self._items = list(items)

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self._items)
```

Both checkers accept a `FrozenBag[Manager]` where a `FrozenBag[Employee]`
is expected — although this is byte for byte the mutable-storage design
that inference just judged invariant. A declaration is trusted, not
re-derived. What the checkers do verify is the declaration against _method
signatures_: declare `_T_co` covariant and then consume it in parameter
position, and both object. For a `Sink(Generic[_T_co])` with
`def push(self, item: _T_co) -> None`, `mypy --strict` reports

```text
sink.py:7: error: Cannot use a covariant type variable as a parameter  [misc]
Found 1 error in 1 file (checked 1 source file)
```

and `pyright` reports

```text
sink.py:7:26 - error: Covariant type variable cannot be used in parameter type (reportGeneralTypeIssues)
1 error, 0 warnings, 0 informations
```

Attributes, however, escape the audit entirely: even a _public_ mutable
attribute typed by a covariant `_T_co` passes both checkers silently. The
legacy system checks what you promised; the modern system computes what is
true. When you migrate a legacy generic to PEP 695 syntax, expect the
effective variance to _change_ wherever the declaration was more generous
than the body — the migrated class does not merely restate your old flags.

````{admonition} Checker divergence: `infer_variance`
:class: warning

For code that must keep the legacy call syntax, PEP 695 added a bridge:
`TypeVar("_T", infer_variance=True)` asks the checker to infer variance
for an old-style type variable[^spec-variance-inference]. On this book's
toolchain the checkers split. `pyright` 1.1.411 supports it — our
`FrozenBag`, redeclared with `infer_variance=True` and tuple storage, is
inferred covariant and passes — while `mypy` 2.2.0 rejects the argument
outright, whether imported from `typing` or `typing_extensions`:

```text
infer.py:6: error: Unexpected argument to "TypeVar()": "infer_variance"  [misc]
```

(followed by cascading errors, since `mypy` then refuses to treat `_T` as
a type variable at all). Until `mypy` supports it, `infer_variance` is a
`pyright`-only convenience; portable code gets inference by moving to
PEP 695 syntax proper.
````

## Why Are `Callable` Parameter Types Contravariant?

Functions are first-class values with types of their own, so the variance
question applies to `Callable[[A], R]` too — and its two slots answer in
opposite directions. That split is where the intuition built on containers
pays off: return position produces, parameter position consumes.

### Return types are covariant

A caller of `get_employee()` consumes what the function produces. Hand it a
function that produces `Manager`s instead, and every produced value is
still an `Employee` — substitution in the same direction as the element
relation:

```{code-cell} ipython3
def dispatch(get_employee: Callable[[], Employee]) -> None:
    get_employee().work()


def get_manager() -> Manager:
    return Manager()


dispatch(get_manager)  # Callable[[], Manager] <: Callable[[], Employee]
```

Both checkers accept the file: `Callable` is **covariant in its return
type**, exactly as `Sequence` is covariant in its element type.

### Parameter types are contravariant

Parameter position points the other way. A function that can process _any_
`Employee` can in particular process a `Manager`; a function that needs
specifically a `Manager` cannot be trusted with an arbitrary `Employee`.
So the _more general_ handler substitutes for the _more specific_ one —
against the direction of the element relation:

```{code-cell} ipython3
def process_employee(employee: Employee) -> None:
    employee.work()


def assign_to_manager(manager: Manager, process: Callable[[Manager], None]) -> None:
    process(manager)


# Callable[[Employee], None] <: Callable[[Manager], None]
assign_to_manager(Manager(), process_employee)
```

Both checkers accept: `process_employee` promises to handle every
`Employee`, and a `Manager` is one. Now the unsafe direction — a
`Manager`-only handler where an any-`Employee` handler is required:

```{code-cell} ipython3
def process_manager(manager: Manager) -> None:
    manager.manage()


def assign_to_any_employee(employee: Employee, process: Callable[[Employee], None]) -> None:
    process(employee)


try:
    assign_to_any_employee(Employee(), process_manager)
except AttributeError as err:
    print(err)
```

The runtime crash is the `append` disaster with the arrow reversed. On the
collected `process.py`, `mypy --strict`:

```text
process.py:21: error: Argument 2 to "assign_to_any_employee" has incompatible type "Callable[[Manager], None]"; expected "Callable[[Employee], None]"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

and `pyright` points at the reversed slot:

```text
process.py:21:40 - error: Argument of type "(manager: Manager) -> None" cannot be assigned to parameter "process" of type "(Employee) -> None" in function "assign_to_any_employee"
    Type "(manager: Manager) -> None" is not assignable to type "(Employee) -> None"
      Parameter 1: type "Employee" is incompatible with type "Manager"
        "Employee" is not assignable to "Manager" (reportArgumentType)
1 error, 0 warnings, 0 informations
```

The formal statement combines both slots into the classic function
subtyping rule:

```{prf:theorem} Function type subtyping
:label: type-theory-06-variance-function-subtyping

For parameter types $A_1, A_2$ and return types $R_1, R_2$, writing
$A \to R$ for `Callable[[A], R]`:

$$
A_2 <: A_1 \ \text{ and } \ R_1 <: R_2
\implies
(A_1 \to R_1) <: (A_2 \to R_2).
$$

The parameter side is compared in the reversed (contravariant) direction
and the return side in the preserved (covariant) direction. PEP 483 states
this rule for Python directly[^pep-483-variance].
```

PEP 483's own illustration is the salary calculator[^pep-483-variance]:

```python
from decimal import Decimal


def calculate_all(employees: list[Manager], salary: Callable[[Manager], Decimal]) -> None: ...
```

A `Callable[[Employee], Decimal]` argument for `salary` is accepted: a
function that can price any `Employee` prices `Manager`s in particular.

## Where Does Variance Show Up in the Wild?

-   Typeshed is a variance atlas once you can read the suffixes:
    `Mapping[_KT, _VT_co]` is invariant in its key type (keys are consumed
    by `__getitem__`) but covariant in its value type (values are only
    produced).
-   [ChromaDB declares a contravariant type variable](https://github.com/chroma-core/chroma/blob/d9a8c28055ca1aa4c602560c0117f7608858d3f0/chromadb/api/types.py#L146)
    for its `Embeddable` payloads, and
    [a covariant one](https://github.com/chroma-core/chroma/blob/d9a8c28055ca1aa4c602560c0117f7608858d3f0/chromadb/api/types.py#L150)
    right next to it.
-   PyTorch's `DataLoader` is generic over a covariant type variable for
    the samples it yields — a producer, exactly as the proposition
    predicts.

## Summary

If I had to compress this chapter into a single sentence: **variance is the
per-constructor rule that says whether subtyping between type arguments
lifts through the constructor preserved (covariant), reversed
(contravariant), or not at all (invariant) — and which rule is sound is
dictated by whether the constructor produces its parameter, consumes it, or
does both.**

| Variance      | Rule                           | Sound when $C$…       | Where you meet it                                   |
| ------------- | ------------------------------ | --------------------- | --------------------------------------------------- |
| Covariant     | $S <: T \implies C[S] <: C[T]$ | only produces         | `Sequence`, `tuple`, `Iterator`, `Callable` returns |
| Contravariant | $S <: T \implies C[T] <: C[S]$ | only consumes         | `Callable` parameters                               |
| Invariant     | $C[S] <: C[T] \iff S = T$      | consumes and produces | `list`, `dict`, mutable containers generally        |

Three receipts to carry forward. `list` is invariant because `append`
consumes `_T` while `pop` produces it, and mutability is what puts a type
variable in both positions. PEP 695 type parameters get their variance
_inferred_ from the whole class body — private attributes included — while
legacy `TypeVar` variance is _declared_ and only partially policed, so a
syntax migration can silently change a class's effective variance. And
`Callable` is the one constructor whose slots split: covariant returns,
contravariant parameters.

Next, {doc}`Function Overloading <07-overload>` stays with function types
and examines `@overload` variants, including the unsafe overlapping
overloads that both checkers flag. Variance also composes with the upper
bounds and constraint sets of
{doc}`Bound and Constraint in Generics and Type Variables <05-typevar-bound-constraints>`,
which restrict what a type parameter may range over before variance decides
how its applications relate.

## References and Further Readings

```{admonition} References
:class: seealso

-   [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/#covariance-and-contravariance)
-   [PEP 484 – Type Hints](https://peps.python.org/pep-0484/#covariance-and-contravariance)
-   [PEP 695 – Type Parameter Syntax](https://peps.python.org/pep-0695/)
-   [Python typing specification – Variance Inference](https://typing.python.org/en/latest/spec/generics.html#variance-inference)
-   [Guido van Rossum – Type Hints (PyCon 2015 keynote)](https://www.youtube.com/watch?v=2wDvzy6Hgxg)
-   [Variance of generics in mypy](https://mypy.readthedocs.io/en/stable/generics.html#variance-of-generics)
-   [python/typeshed – stdlib/builtins.pyi](https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi)
-   [Type Hinting: Covariance & Contra-Variance](https://www.playfulpython.com/type-hinting-covariance-contra-variance/)
-   [CS2030S: Variance](https://nus-cs2030s.github.io/2021-s2/18-variance.html)
-   [Covariance, Contravariance, and Invariance – The Ultimate Python Guide](https://blog.daftcode.pl/covariance-contravariance-and-invariance-the-ultimate-python-guide-8fabc0c24278)
-   [Covariance and contravariance in subtyping](https://eli.thegreenplace.net/2018/covariance-and-contravariance-in-subtyping/)
-   [Python typing: What does TypeVar('A', 'B', covariant=True) mean?](https://stackoverflow.com/questions/61568462/python-typing-what-does-typevara-b-covariant-true-mean)
-   [Why are arrays invariant but lists covariant?](https://stackoverflow.com/questions/6684493/why-are-arrays-invariant-but-lists-covariant)
-   [What is the difference between covariance and contra-variance in programming languages?](https://stackoverflow.com/questions/1163465/what-is-the-difference-between-covariance-and-contra-variance-in-programming-lan)
-   [Covariance and contravariance (computer science) – Wikipedia](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>)
```

[^guido-pycon-2015]:
    Guido van Rossum,
    ["Type Hints"](https://www.youtube.com/watch?v=2wDvzy6Hgxg), PyCon 2015
    keynote. The slide's original body is `lst += [3.14]`; `append` is the
    equivalent single-element form.

[^typeshed-builtins]:
    [python/typeshed, `stdlib/builtins.pyi`](https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi),
    as vendored by `mypy` 2.2.0. Typeshed evolves; line-level details may
    differ in later versions.

[^spec-variance-inference]:
    [Python typing specification, "Variance Inference"](https://typing.python.org/en/latest/spec/generics.html#variance-inference)
    — the algorithm originates in
    [PEP 695](https://peps.python.org/pep-0695/#variance-inference).

[^pep-483-variance]:
    [PEP 483, "Covariance and Contravariance"](https://peps.python.org/pep-0483/#covariance-and-contravariance).
