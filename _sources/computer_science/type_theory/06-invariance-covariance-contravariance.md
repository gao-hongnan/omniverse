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

# Invariance, Covariance and Contravariance

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from typing_extensions import reveal_type
from rich.pretty import pprint
```

## The Motivation

This section is not particulary easy. So let's open with an example, a classic
one written by the author of Python, Guido van Rossum.

```{code-cell} ipython3
def append_pi(lst: List[float]) -> None:
    lst.append(3.14) # original is lst += [3.14]

my_list = [1, 3, 5]  # type: List[int]

_ = append_pi(my_list)   # Naively, this should be safe..
pprint(my_list)
```

The function `append_pi` is supposed to append the value of $\pi$ to the list
`lst`. However, the type of `lst` is `List[float]`, and the type of `my_list` is
`List[int]`. The function `append_pi` is supposed to be safe, but it is not. Why
not? Isn't `int` (integers $\mathbb{Z}$) a subtype of `float` (real numbers
$\mathbb{R}$)? Did we not establish this in the section on subsumption?

Yes, we did. However, `int` being a subtype of `float` does not imply that
`List[int]` is a subtype of `List[float]`. Upon reflection, this makes sense,
because the above code would break the second criterion of subtyping (
{prf:ref}`type-theory-subtype-criterion`). The second criterion states that the
if the type $S$ is a subtype of $T$, then if $T$ has $N$ methods, then $S$ must
have at least the same set of $N$ methods. Furthermore, it must preserve the
semantics of the functions. Let's build a mental model to understand why this is
the case. Consider the python implementation of `List`,

```python
_T = TypeVar("_T")

class list(MutableSequence[_T]):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __iterable: Iterable[_T]) -> None: ...
    def copy(self) -> list[_T]: ...
    def append(self, __object: _T) -> None: ...
    def extend(self, __iterable: Iterable[_T]) -> None: ...
    def pop(self, __index: SupportsIndex = -1) -> _T: ...
    # Signature of `list.index` should be kept in line with `collections.UserList.index()`
    # and multiprocessing.managers.ListProxy.index()
    def index(self, __value: _T, __start: SupportsIndex = 0, __stop: SupportsIndex = sys.maxsize) -> int: ...
    def count(self, __value: _T) -> int: ...
    def insert(self, __index: SupportsIndex, __object: _T) -> None: ...
    def remove(self, __value: _T) -> None: ...
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
    def __getitem__(self, __i: SupportsIndex) -> _T: ...
    @overload
    def __getitem__(self, __s: slice) -> list[_T]: ...
    @overload
    def __setitem__(self, __key: SupportsIndex, __value: _T) -> None: ...
    @overload
    def __setitem__(self, __key: slice, __value: Iterable[_T]) -> None: ...
    def __delitem__(self, __key: SupportsIndex | slice) -> None: ...
    # Overloading looks unnecessary, but is needed to work around complex mypy problems
    @overload
    def __add__(self, __value: list[_T]) -> list[_T]: ...
    @overload
    def __add__(self, __value: list[_S]) -> list[_S | _T]: ...
    def __iadd__(self, __value: Iterable[_T]) -> Self: ...  # type: ignore[misc]
    def __mul__(self, __value: SupportsIndex) -> list[_T]: ...
    def __rmul__(self, __value: SupportsIndex) -> list[_T]: ...
    def __imul__(self, __value: SupportsIndex) -> Self: ...
    def __contains__(self, __key: object) -> bool: ...
    def __reversed__(self) -> Iterator[_T]: ...
    def __gt__(self, __value: list[_T]) -> bool: ...
    def __ge__(self, __value: list[_T]) -> bool: ...
    def __lt__(self, __value: list[_T]) -> bool: ...
    def __le__(self, __value: list[_T]) -> bool: ...
    def __eq__(self, __value: object) -> bool: ...
    if sys.version_info >= (3, 9):
        def __class_getitem__(cls, __item: Any) -> GenericAlias: ...
```

and now consider the following code:

```python
def append_pi(lst: List[float]) -> None:
    lst.append(3.14)
```

And since we parametrized the generic `_T` in `List` with `float`, the type
append method `append` is now `append(self, __object: float) -> None` instead of
`append(self, __object: _T) -> None`. And since `my_list` is of type
`List[int]`, the corresponding method `append` should be
`append(self, __object: int) -> None`. If we were to append a `float` to
`my_list`, the method `append` would break the second criterion of subtyping
(and also Liskov's substitution principle). Furthermore, when we pass `my_list`
to `append_pi`, we are essentially **assigning** `my_list: List[int]` to
`lst: List[float]`. This is not a safe operation because their signature in
`append` are different. So even though both have the same method name `append`,
they are not the same method.

And indeed running through this piece of code via a static type checker like
`mypy` will raise an error.

```bash
6: error: Argument 1 to "append_pi" has incompatible type "list[int]"; expected "list[float]"  [arg-type]
    append_pi(my_list)   # Naively, this should be safe..
              ^~~~~~~
6: note: "List" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
6: note: Consider using "Sequence" instead, which is covariant
```

The `mypy` error message even gave you some suggestion that `List` is
**invariant**.

## The Definitions

Let $S$ and $T$ be two types, and $C[U]$ denotes a generic class and an
application of a
[type constructor](https://en.wikipedia.org/wiki/Type_constructor) $C$
parameterized by a type $U$.

It is helpful for one to think of type constructors as classes. For example,
`List` is a type constructor, and `List[int]` is an application of the type.

### Covariance

```{prf:definition} Covariance
:label: computer-science-type-theory-covariance

Within the context of type theory, we say that the type constructor $C$ is
**covariant** if for any types $S$ and $T$, if $S$ is a subtype of $T$, then
$C[S]$ is a subtype of $C[T]$.

$$
S \leq T \implies C[S] \leq C[T]
$$

This behaviour preserves the
[ordering of types](https://en.wikipedia.org/wiki/Subtyping), which orders types
from more specific to more generic.
```

### Contravariance

```{prf:definition} Contravariance
:label: computer-science-type-theory-contravariance

Within the context of type theory, a type constructor $C$ is **contravariant**
if for any types $S$ and $T$, if $S$ is a subtype of $T$, then $C[T]$ is a
subtype of $C[S]$.

$$
S \leq T \implies C[T] \leq C[S]
$$

This behaviour reverses the
[ordering of types](https://en.wikipedia.org/wiki/Subtyping), moving from more
generic to more specific. Contravariance is typically relevant in scenarios
where the type constructor represents a producer or consumer that operates
contrarily to the direction of subtype relationships in covariance.
```

### Invariance

```{prf:definition} Invariance
:label: computer-science-type-theory-invariance

Invariance means that if you have a type constructor $C$ and two types $S$ and
$T$, then $C[S]$ is neither a subtype nor a supertype of $C[T]$ unless $S$ and
$T$ are the same.

Equivalently, for any types $S$ and $T$, $C[S]$ and $C[T]$ are considered
compatible or equivalent only if $S = T$.

Mathematically, this concept does not have a direct inequality representation
like covariance or contravariance because it states a condition of equality
rather than a relational order. However, it implies:

$$
(S \neq T) \implies (C[S] \nleq C[T] \text{ and } C[T] \nleq C[S])
$$

And conversely:

$$
(S = T) \implies (C[S] = C[T])
$$

This means for invariance, the specific type instantiation of $C$ with $S$ can
only be used in contexts expecting exactly $C[S]$, not $C[T]$ where $T$ is
different from $S$, regardless of whether $T$ is a subtype or supertype of $S$.
```

## List is Invariant

Let's connect back to our earlier example on why `List` is invariant. We will
now use backticks instead of latex symbols to denote types and type
constructors.

### Covariance

Covariance describes a relationship between types where a type `T` is considered
a subtype of a type `U` (written as `T ≤ U`) if and only if a container of `T`
(`Container[T]`) is considered a subtype of a container of `U` (`Container[U]`).
This is natural for operations that produce or return values of the generic type
(e.g., getters). In the context of our example, `List[int] ≤ List[float]` would
hold true if `List[T]` were covariant, meaning that a list of integers could be
used wherever a list of floats is expected, because every integer can be
considered a float.

However, this relationship does not hold because the `append` method of a
`List[float]` can accept any `float` as an argument, while `append` for a
`List[int]` can only accept `int` values. Since appending a `float` to a
`List[int]` is not type-safe (a float is not necessarily an int), the `List`
type cannot be considered covariant. More concretely, if we assume that
`List[int]` is indeed a subtype of `List[float]`, then the append method of
`List[int]` should do **_exactly_** the same thing as the append method of
`List[float]`. But this is not the case.

### Contravariance

Contravariance is the opposite: `T ≤ U` implies `Container[U] ≤ Container[T]`.
This makes sense for containers that consume or operate on the generic type
(e.g., setters). Using our example, `List[float] ≤ List[int]` would be true if
`List[T]` were contravariant, meaning you could use a list expecting floats
where a list of integers is required. This is because it's safe to pass an
integer where a float is expected, not the other way around.

The example illustrates that this relationship also doesn't hold because the
`pop` method returns a more specific type (`int`) in `List[int]` than in
`List[float]` (which returns `float`). If `List[T]` were contravariant, we would
expect operations that work on `List[int]` to work on `List[float]`, but the
narrower return type of `pop()` in `List[int]` contradicts this, making it not
contravariant.

### Invariance

Given that the `List` type is neither covariant nor contravariant as
demonstrated, it is termed invariant. Invariance means that you cannot
substitute `List[T]` with `List[U]` or vice versa unless `T` and `U` are exactly
the same type. In practical terms, a `List[int]` is only a `List[int]` and
cannot be treated as a `List[float]`, and vice versa.

## Covariant in the return type vs Contravariant in the argument type

...

## References and Further Readings

-   https://peps.python.org/pep-0483/#covariance-and-contravariance
-   https://peps.python.org/pep-0484/#covariance-and-contravariance
-   https://www.playfulpython.com/type-hinting-covariance-contra-variance/
-   https://nus-cs2030s.github.io/2021-s2/18-variance.html
-   https://mypy.readthedocs.io/en/stable/generics.html#variance-of-generics
-   https://blog.daftcode.pl/covariance-contravariance-and-invariance-the-ultimate-python-guide-8fabc0c24278
