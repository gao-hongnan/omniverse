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
            "Generics in Python let one class or function serve every type
            without Any: PEP 695, type variables, and parameterized types,
            with mypy and pyright evidence."
        "keywords":
            "python, generics, type variables, PEP 695, TypeVar,
            parameterized types"
---

# Generics in Python: Type Variables and Parameterized Types

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

from dataclasses import dataclass
from typing import Any
```

A `Pair` class that holds two `int`s takes five lines. The moment you need a
pair of `float`s, a pair of `str`s, or a `str`-and-`int` pair, Python offers
a bad menu: copy the class once per element type, or annotate the fields as
`Any` and watch every checker guarantee evaporate. This chapter is about the
third option — **generics**, definitions written against types you have not
chosen yet.

We stage the failure first: an `Any`-typed `Pair` that both checkers wave
through while it silently computes `"164"` where `20` was intended. Then the
repair in Python 3.14's modern syntax from
[PEP 695](https://peps.python.org/pep-0695/) — `class Pair[S, T]:`,
`def first[T](items: list[T]) -> T:` — the vocabulary of type variables,
parameters, and arguments, and the formal view of `list[int]` as a type
constructor applied to a type. A clearly labeled section then covers the
legacy `TypeVar`/`Generic` spelling you will keep meeting in real code.

By the end you can write generic functions and classes that the checkers
enforce, read both checkers' diagnostics about them — including where the two
disagree — and see exactly where a bare type parameter runs out of power,
the problem the next chapter's bounds and constraints exist to solve.

```{admonition} Prerequisites
:class: note

This chapter uses the subtype relation $S <: T$ ("an $S$ can stand in for a
$T$") from {doc}`Subtypes <01-subtypes>` and the substitution criterion from
{doc}`Subsumption <03-subsumption>`. The checker silence we stage below is
gradual typing at work, covered in {doc}`Type Safety <02-type-safety>`. Every
piece of checker output on this page comes from `mypy 2.2.0` (run with
`--strict`) and `pyright 1.1.411` (standard mode, except one output labeled
as strict) on Python 3.14.
```

## Why Not Just Use `Any`?

Start with the smallest useful aggregate: a class that carries two values,
so a function can return both a minimum and a maximum at once.

```{code-cell} ipython3
@dataclass
class IntPair:
    first: int
    second: int


def min_max(values: list[int]) -> IntPair:
    return IntPair(min(values), max(values))


print(min_max([3, 1, 4, 1, 5]))
```

`IntPair` does its one job well — and does exactly one job. The day you need
the extremes of a list of floats, you need a `FloatPair`; lexicographic
extremes of strings, a `StrPair`; a username with its numeric id, yet another
class. Every copy has the same body; only the annotations change. The logic
was never `int`-specific, but the class is.

The tempting exit is `Any`, the special type that is compatible with
everything:

```{code-cell} ipython3
@dataclass
class Pair:
    first: Any
    second: Any


pair = Pair("16", "4")  # strings slipped in where ints were intended
total: int = pair.first + pair.second
print(total, type(total).__name__)
```

One class now holds any two values — and the printed line is the bill for
it. The intent was to add `16 + 4` and get `20`; two strings slipped in,
`+` concatenated them, and a variable annotated `int` now holds the `str`
`"164"`. No exception was raised, at any point. Collect the same code, with
two `reveal_type` probes, into `pair_any.py` and both checkers pass it
wholesale:

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class Pair:
    first: Any
    second: Any


pair = Pair("16", "4")  # strings slipped in where ints were intended
total: int = pair.first + pair.second
reveal_type(pair.first)
reveal_type(total)
```

`mypy --strict`:

```text
pair_any.py:13: note: Revealed type is "Any"
pair_any.py:14: note: Revealed type is "int"
Success: no issues found in 1 source file
```

`pyright`:

```text
pair_any.py:13:13 - information: Type of "pair.first" is "Any"
pair_any.py:14:13 - information: Type of "total" is "int"
0 errors, 0 warnings, 2 informations
```

Name the phenomenon before fixing it. `Any` does not mean "any type, chosen
consistently"; it means _unchecked_. It is
{doc}`gradual typing's <02-type-safety>` escape hatch: a value of static
type `Any` may flow into a slot of any other type — here, straight into
`total: int` — without complaint. Worse, the checker then _believes_ the
annotation: `total` reveals as `int` even though a `str` sits in it at
runtime, so the false belief propagates to everything downstream.

What we actually want is to keep the single class but state a contract: the
first field has _some_ fixed type $S$, the second _some_ fixed type $T$, the
same $S$ and $T$ every time that particular pair is touched. That requires a
variable that ranges over types the way an ordinary variable ranges over
values — a **type variable**.

```{prf:remark} A type variable is not `Any`
:label: type-theory-04-generics-remark-typevar-vs-any

Both promise flexibility, but in opposite currencies. A type variable $T$
stands for a _specific but unspecified_ type: within one use of the
definition, every occurrence of $T$ refers to the same type, so a checker
can enforce consistency without knowing what $T$ is. `Any` abandons the
question: it is compatible with everything in both directions, so every
occurrence is independent and nothing is enforced. Flexibility via `Any`
costs the guarantees; flexibility via $T$ keeps them[^pep483-any].
```

## One Class for Every Type: PEP 695 Type Parameters

Since Python 3.12, [PEP 695](https://peps.python.org/pep-0695/) lets a class
declare **type parameters** — the type variables it is defined over —
directly in its header, in square brackets. This is the modern spelling the
whole series targets:

```{code-cell} ipython3
@dataclass
class Pair[S, T]:
    first: S
    second: T


user = Pair("john_doe", 12345)
print(user)
```

The header `class Pair[S, T]:` introduces two type parameters, `S` and `T`,
scoped to the class body; the field annotations then _use_ them. The
runtime behavior is unchanged — the payoff is entirely static. When you
construct `Pair("john_doe", 12345)`, the checker infers the **type
arguments** from the constructor call — $S \mapsto$ `str`,
$T \mapsto$ `int` — and the value gets the **parameterized type**
`Pair[str, int]`.

That inference is exactly what turns the earlier silent bug into a loud one.
In `pair_generic.py`, a function demands `Pair[str, int]` and the caller
builds the pair with its arguments swapped:

```python
from dataclasses import dataclass


@dataclass
class Pair[S, T]:
    first: S
    second: T


def log_user(user: Pair[str, int]) -> None:
    print(f"User: {user.first}, ID: {user.second}")


record = Pair(12345, "john_doe")  # swapped: Pair[int, str]
reveal_type(record)
log_user(record)
```

`mypy --strict`:

```text
pair_generic.py:15: note: Revealed type is "pair_generic.Pair[int, str]"
pair_generic.py:16: error: Argument 1 to "log_user" has incompatible type "Pair[int, str]"; expected "Pair[str, int]"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

`pyright`:

```text
pair_generic.py:15:13 - information: Type of "record" is "Pair[int, str]"
pair_generic.py:16:10 - error: Argument of type "Pair[int, str]" cannot be assigned to parameter "user" of type "Pair[str, int]" in function "log_user"
    "Pair[int, str]" is not assignable to "Pair[str, int]"
      Type parameter "S@Pair" is invariant, but "int" is not the same as "str"
      Type parameter "T@Pair" is invariant, but "str" is not the same as "int" (reportArgumentType)
1 error, 0 warnings, 1 information
```

The mistake that `Any` let crash — or worse, not crash — at runtime is now
rejected at static-analysis time. Read pyright's explanation closely: it
says each type parameter is **invariant**, meaning `Pair[int, str]` and
`Pair[str, int]` are simply different types, neither substitutable for the
other. When, if ever, one application of a generic class may stand in for
another is _variance_, the subject of
{doc}`Invariance, Covariance and Contravariance <06-invariance-covariance-contravariance>`.

Three terms did the work in this section, and they are worth pinning down —
they recur through the rest of the series:

| Term           | What it names                                          | In the example                       |
| -------------- | ------------------------------------------------------ | ------------------------------------ |
| Type variable  | A variable that ranges over types, not values          | `S`, `T`                             |
| Type parameter | A type variable declared in a definition's header      | the `[S, T]` in `class Pair[S, T]:`  |
| Type argument  | The concrete type supplied (or inferred) at a use site | `str`, `int` in `Pair[str, int]`     |

Under PEP 695 the first two coincide: declaring the parameter _is_ creating
the variable. In the legacy spelling — covered in its own section below —
type variables are standalone `TypeVar` objects created before, and
independently of, the definitions that use them.

## What Is a Generic Type, Formally?

You have been consuming generics since your first annotation. `list[int]`
is the generic class `list` applied to the type argument `int`, and it
polices its elements exactly the way `Pair[str, int]` polices its fields
(`list_append.py`):

```python
nums: list[int] = [1, 2, 3]
nums.append("four")
```

`mypy --strict`:

```text
list_append.py:2: error: Argument 1 to "append" of "list" has incompatible type "str"; expected "int"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

`pyright`:

```text
list_append.py:2:13 - error: Argument of type "Literal['four']" cannot be assigned to parameter "object" of type "int" in function "append"
    "Literal['four']" is not assignable to "int" (reportArgumentType)
1 error, 0 warnings, 0 informations
```

The pattern underneath — _give a type, get a type_ — deserves a precise
statement, because the variance rules of
{doc}`the variance chapter <06-invariance-covariance-contravariance>`
quantify over exactly this structure.

```{prf:definition} Type Constructor and Parameterized Type
:label: type-theory-04-generics-definition-type-constructor

Let $\mathbf{Type}$ denote the universe of types. A **type constructor** of
arity $n \geq 1$ is a mapping

$$
C : \underbrace{\mathbf{Type} \times \cdots \times \mathbf{Type}}_{n}
\;\to\; \mathbf{Type},
\qquad
(T_1, \ldots, T_n) \;\mapsto\; C[T_1, \ldots, T_n],
$$

that takes $n$ type arguments and yields a type. A **generic type** (in
Python, a generic class) is a definition that introduces one or more type
parameters; its name denotes the constructor $C$. Subscripting the name —
`Pair[str, int]`, `list[int]` — is _application_ of the constructor, and
the resulting type $C[T_1, \ldots, T_n]$ is called a **parameterized
type**.

Type constructors are to types what functions are to values: a function
maps values to a value, a type constructor maps types to a
type[^pep483-constructor].
```

Concretely: `list` is a constructor of arity 1, so `list[int]` and
`list[str]` are two of its (infinitely many) applications; `dict` has
arity 2, giving `dict[str, int]`; our `Pair` has arity 2 as well. Note what
this makes of the bare name: `Pair` on its own is the constructor awaiting
arguments, not a finished type. Annotate with it unapplied and strict
`mypy` objects (`Missing type arguments for generic type "Pair"
[type-arg]`) because the parameters silently default to `Any` — the very
hole we just climbed out of. Pyright's standard mode accepts the bare form,
tracking the unknown parameters internally; this asymmetry previews the
divergence admonition below.

```{prf:remark} Applying a constructor does not preserve subtyping
:label: type-theory-04-generics-remark-no-automatic-lifting

Given $S <: T$, nothing follows automatically about the relationship
between $C[S]$ and $C[T]$:

$$
S <: T \quad\not\Longrightarrow\quad C[S] <: C[T].
$$

Whether a constructor preserves the subtype relation, reverses it, or
discards it entirely is a property of the constructor itself, called its
**variance**. You have already seen evidence: pyright justified rejecting
the swapped pair by declaring `Pair`'s parameters invariant — different
arguments, unrelated types. The full classification is the subject of
{doc}`Invariance, Covariance and Contravariance <06-invariance-covariance-contravariance>`.
```

## Generic Functions: One Binding per Call

The same square-bracket syntax works on functions. A generic function
declares its type parameters between the name and the argument list:

```{code-cell} ipython3
def first[T](items: list[T]) -> T:
    return items[0]


print(first(["Alice", "Bob"]), first([3, 1, 4]))
```

Here `T` is scoped to the function alone, and it is bound afresh at every
call: pass a `list[str]` and $T \mapsto$ `str` for that call; pass a
`list[int]` and $T \mapsto$ `int` for the next. Probing the two calls above
with `reveal_type` (`first_infer.py`) shows both checkers agreeing:

`mypy --strict`:

```text
first_infer.py:6: note: Revealed type is "str"
first_infer.py:8: note: Revealed type is "int"
```

`pyright`:

```text
first_infer.py:6:13 - information: Type of "first(names)" is "str"
first_infer.py:8:13 - information: Type of "first(nums)" is "int"
```

The binding is a contract across the whole signature. This function promises
"a list of some type $T$, plus one more $T$, gives back a list of the same
$T$" — so feeding a `str` into a `list[int]` call makes the contract
unsatisfiable (`append_mix.py`):

```python
def append_and_return[T](items: list[T], item: T) -> list[T]:
    items.append(item)
    return items


ints: list[int] = [1, 2, 3]
append_and_return(ints, "four")
```

`mypy --strict`:

```text
append_mix.py:7: error: Cannot infer value of type parameter "T" of "append_and_return"  [misc]
Found 1 error in 1 file (checked 1 source file)
```

`pyright`:

```text
append_mix.py:7:25 - error: Argument of type "Literal['four']" cannot be assigned to parameter "item" of type "T@append_and_return" in function "append_and_return"
    "Literal['four']" is not assignable to "int" (reportArgumentType)
1 error, 0 warnings, 0 informations
```

Same verdict, different diagnosis: `mypy` declines to solve for `T` at all,
while `pyright` solves $T \mapsto$ `int` from the list and then rejects the
string. Keep that split in mind — the admonition at the end of the next
section shows a case where the two checkers' solving strategies produce
_different accepted types_, not just different error messages.

## Generic Classes: One Binding per Instance

Where a generic function binds its type parameter per call, a generic class
binds it per _instance_, and the binding is shared by every attribute and
method that mentions it:

```{code-cell} ipython3
class Stack[T]:
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()


stack = Stack[str]()
stack.push("hello")
stack.push("world")
print(stack.pop())
```

Writing `Stack[str]()` applies the constructor first — producing the
parameterized type `Stack[str]` — and instantiates it second. For this
object's whole lifetime, $T \mapsto$ `str`: `push` only accepts `str`, `pop`
returns `str`, and the private `list[T]` is a `list[str]`. Unlike `Pair`,
the type argument here must be written explicitly, because `__init__` takes
no argument from which the checker could infer it. Collect the class cell
and three lines of use into `stack.py`, and the contract bites exactly as
it did for `list`:

```python
stack = Stack[str]()
stack.push("hello")
stack.push(123)
```

`mypy --strict`:

```text
stack.py:14: error: Argument 1 to "push" of "Stack" has incompatible type "int"; expected "str"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

`pyright`:

```text
stack.py:14:12 - error: Argument of type "Literal[123]" cannot be assigned to parameter "item" of type "str" in function "push"
    "Literal[123]" is not assignable to "str" (reportArgumentType)
1 error, 0 warnings, 0 informations
```

The [Stack page](https://www.gaohongnan.com/dsa/stack/concept.html#the-importance-of-generic-types)
in the data-structures section builds this class out into a full,
production-shaped container if you want a longer worked example.

````{admonition} Where mypy 2.2.0 and pyright 1.1.411 diverge on generics
:class: attention

The typing specification standardizes what generic types _mean_, not how a
checker must infer type arguments — and on two inference questions from
this chapter, the flagship checkers part ways.

**Bare instantiation.** Drop the type argument (`stack_bare.py`) and `mypy`
refuses to guess, demanding an annotation:

```text
stack_bare.py:9: error: Need type annotation for "stack"  [var-annotated]
Found 1 error in 1 file (checked 1 source file)
```

`pyright` (standard mode) reports `0 errors, 0 warnings, 0 informations`,
silently tracking the instance as `Stack[Unknown]`; only its strict mode
surfaces the gap:

```text
stack_bare_strict.py:12:1 - error: Type of "stack" is partially unknown
    Type of "stack" is "Stack[Unknown]" (reportUnknownVariableType)
```

**Mixed-argument solving.** Call `def same[T](x: T, y: T) -> T:` with
`same(1, "a")` (`same_solve.py`) and _both_ checkers accept the call — but
they disagree on what `T` became. `mypy --strict`:

```text
same_solve.py:5: note: Revealed type is "object"
```

`pyright`:

```text
same_solve.py:5:13 - information: Type of "same(1, "a")" is "int | str"
```

`mypy` solves by _join_ — the nearest common supertype of `int` and `str`,
which is `object` — while `pyright` solves by _union_, keeping `int | str`.
Neither is wrong, but the downstream code they permit differs sharply: an
`object` supports almost nothing without narrowing, while a union keeps the
two-case structure. Code that must behave identically under both checkers
should not lean on the inferred `T` of a deliberately mixed call — and if
your intent is that mixing be _rejected_, plain type parameters cannot
express that; the
{doc}`constrained type variables of the next chapter <05-typevar-bound-constraints>` can.
````

## The Legacy Spelling: `TypeVar` and `Generic`

Everything above uses PEP 695 syntax, which landed in Python 3.12. Code
older than that — most production code, most tutorials, and
{doc}`the next chapter <05-typevar-bound-constraints>` of this series —
spells the same ideas with machinery from the `typing` module, and you need
to read both dialects fluently even if you only write the new one. This
section is exactly that legacy sidebar. The `Stack` above, in the old
spelling (`stack_legacy.py`):

```python
from typing import Generic, TypeVar

T = TypeVar("T")


class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)


stack = Stack[str]()
stack.push(123)
```

A `TypeVar` object is created as an ordinary module-level value, and the
class inherits from `Generic[T]` to declare which variables parameterize
it. To the checkers, the two spellings are interchangeable here: this file
draws word-for-word the same diagnostics as the PEP 695 `Stack` —

```text
stack_legacy.py:15: error: Argument 1 to "push" of "Stack" has incompatible type "int"; expected "str"  [arg-type]
Found 1 error in 1 file (checked 1 source file)
```

— and both checkers accept the legacy definitions themselves without
comment. The translation table:

| Modern (PEP 695, Python ≥ 3.12)         | Legacy (PEP 484)                                           |
| --------------------------------------- | ---------------------------------------------------------- |
| `def first[T](items: list[T]) -> T:`    | `T = TypeVar("T")`, then `def first(items: List[T]) -> T:` |
| `class Stack[T]:`                       | `class Stack(Generic[T]):`                                 |
| `list[int]`, `dict[str, int]`           | `typing.List[int]`, `typing.Dict[str, int]`                |
| `int \| None`, `str \| bytes`           | `Optional[int]`, `Union[str, bytes]`                       |

Two differences run deeper than spelling. First, a legacy `TypeVar` is a
reusable value — one `T` can parameterize many definitions, with scoping
rules subtle enough that [PEP 484 devotes a
section](https://peps.python.org/pep-0484/#scoping-rules-for-type-variables)
to them — whereas a PEP 695 parameter exists only inside its definition's
header and body. Second, the `TypeVar(...)` call accepts extra arguments
that _restrict_ what the variable ranges over: an upper bound
(`TypeVar("T", bound=...)`) or a set of constraints
(`TypeVar("T", int, float)`). Those restrictions are the subject of
{doc}`Bound and Constraint <05-typevar-bound-constraints>`, which teaches
them in this legacy spelling — after this section you can read every line
of it. The remaining machinery — PEP 695's scoping rules, the `type` alias
statement, inferred-versus-declared variance, and when a `TypeVar` still
earns its keep in modern code — gets a dedicated chapter on the
{doc}`series roadmap <intro>`.

## Where a Bare Type Parameter Runs Out of Power

Generics look, so far, like a free lunch. Here is the bill. Try the obvious
generic addition function (`add_unconstrained.py`):

```python
def add[T](x: T, y: T) -> T:
    return x + y
```

`mypy --strict`:

```text
add_unconstrained.py:2: error: Returning Any from function declared to return "T"  [no-any-return]
add_unconstrained.py:2: error: Unsupported left operand type for + ("T")  [operator]
Found 2 errors in 1 file (checked 1 source file)
```

`pyright`:

```text
add_unconstrained.py:2:12 - error: Operator "+" not supported for types "T@add" and "T@add" (reportOperatorIssue)
1 error, 0 warnings, 0 informations
```

The rejection is correct, and it is the mirror image of this chapter's
lesson. A bare `T` ranges over _every_ type — `dict[str, int]`, `None`, an
open file handle — and `+` is not defined for every type, so no checker can
defend the body. `T` guarantees consistency ("both arguments and the result
share one type") but promises nothing about what that type _supports_. The
missing expressive power is the ability to shrink `T`'s range: "any type
with `+`", or "one of `int`, `float`, `str` only". That is precisely what
upper bounds and constraints provide, and
{doc}`Bound and Constraint <05-typevar-bound-constraints>` opens with this
very function — including why the fix is _not_ the `Union` you might reach
for first.

## Summary

If this chapter compresses to a single sentence: a generic definition takes
types as parameters the way a function takes values as arguments — `Pair`
is a type constructor, `Pair[str, int]` is its application, and the checker
holds every use to whatever binding it inferred. Along the way:

-   `Any` buys flexibility by turning the checker off: both `mypy --strict`
    and `pyright` blessed code that computed `"164"` where `20` was
    intended. A type variable buys the same flexibility while keeping
    every occurrence consistent.
-   PEP 695 declares type parameters in the header — `class Pair[S, T]:`,
    `def first[T](items: list[T]) -> T:` — and the checker infers type
    arguments at each use, rejecting `Pair[int, str]` where
    `Pair[str, int]` is demanded.
-   Formally, a generic class names a type constructor
    $C : \mathbf{Type}^n \to \mathbf{Type}$; subscripting applies it, and
    the result is a parameterized type. Application does _not_ automatically
    preserve $S <: T$ — that is variance.
-   Functions bind their type parameters per call; classes per instance.
    Where inference is underdetermined, the checkers diverge: join versus
    union, refusal versus `Unknown`.
-   The legacy `TypeVar`/`Generic` spelling is checker-equivalent for
    everything shown here, and it is the dialect of most existing code.

Two threads continue directly from here. A bare `T` cannot promise any
behavior, so
{doc}`Bound and Constraint in Generics <05-typevar-bound-constraints>`
restricts a type variable's range with upper bounds and constraint sets —
picking up the exact `add` function that failed above. And parameterizing a
constructor raises the substitution question this chapter deliberately left
open — when does $S <: T$ lift to $C[S] <: C[T]$? — which
{doc}`Invariance, Covariance and Contravariance <06-invariance-covariance-contravariance>` answers in full.

## References and Further Readings

The motivating arc of this chapter — the `IntPair` duplication problem, the
`Any`-typed `Pair`, and its generic repair — is adapted from
[Unit 20: Generics](https://nus-cs2030s.github.io/2021-s2/20-generics.html)
of the CS2030S course at the National University of Singapore, translated
from Java to Python and re-verified against Python's checkers. The debt is
genuine: their staging of duplication-versus-erasure is what this chapter
modernizes onto PEP 695. The `add` example follows a
[Stack Overflow answer on `Generic` and `TypeVar`](https://stackoverflow.com/questions/68739824/use-of-generic-and-typevar).

```{admonition} References
:class: seealso

-   [PEP 695 – Type Parameter Syntax](https://peps.python.org/pep-0695/)
-   [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/)
-   [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
-   [Python typing specification – Generics](https://typing.python.org/en/latest/spec/generics.html)
-   [Generics - Python Docs](https://docs.python.org/3/library/typing.html#generics)
-   [User-defined generic types - Python Docs](https://docs.python.org/3/library/typing.html#user-defined-generic-types)
-   [Scoping rules for type variables - PEP 484](https://peps.python.org/pep-0484/#scoping-rules-for-type-variables)
-   [Unit 20: Generics - CS2030S](https://nus-cs2030s.github.io/2021-s2/20-generics.html)
-   [Implementing Generics via Type Erasure - CS2030S](https://nus-cs2030s.github.io/2021-s2/21-erasure.html)
-   [Type Hinting: Generics & Inheritance - Playful Python](https://www.playfulpython.com/python-type-hinting-generics-inheritance/)
-   [Use of Generic and TypeVar - Stack Overflow](https://stackoverflow.com/questions/68739824/use-of-generic-and-typevar)
-   [Stack - Omniverse](https://www.gaohongnan.com/dsa/stack/concept.html#the-importance-of-generic-types)
-   [python Generics (intermediate) anthony explains #430](https://www.youtube.com/watch?v=LcfxUU1A-RQ)
```

[^pep483-any]:
    [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/)
    introduces `Any` as consistent with every type, and type variables as
    the mechanism that preserves relationships between annotations.

[^pep483-constructor]:
    The function analogy is PEP 483's own: "a function takes a value and
    returns a value, while generic type constructor takes a type and
    'returns' a type" —
    [PEP 483, Generic types](https://peps.python.org/pep-0483/#generic-types).
