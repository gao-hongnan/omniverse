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

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple, TYPE_CHECKING, Iterable, Iterator
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

## The Connection of Mutability and Variance

Covariance, by definition, allows a type `C[Derived]` to be treated as `C[Base]`
if `Derived` is a subtype of `Base`. This implies that wherever a `Base` is
expected, a `Derived` can be used without issue, preserving the "is-a"
relationship and ensuring that any operation that expects `Base` can work with
`Derived`.

When dealing with immutable structures, the operations you can perform on them
are **read-only**. Since you cannot modify an immutable structure, you can't
perform any operation that might violate the type constraints that covariance
imposes.

1. **No Mutation Means No Type Violations**: Since you cannot add, remove, or
   change the elements of an immutable structure, there's no risk of inserting
   an element of an incorrect type that might violate the expected type of the
   container.

2. **Read-Only Operations Are Safe**: Operations on immutable structures are
   limited to reading or deriving new structures without altering the original.
   This means any operation that works on `C[Base]` will also work on
   `C[Derived]` because it only relies on the presence of `Base` type behaviors
   in `Derived`.

3. **Substitution Principle Is Preserved**: The Liskov Substitution Principle,
   which is central to understanding subtype relationships, states that objects
   of a superclass should be replaceable with objects of a subclass without
   affecting the correctness of the program. Immutability ensures that this
   principle is not violated since the operations allowed on an immutable
   `C[Derived]` would not behave any differently if `C[Derived]` were replaced
   with `C[Base]`.

For mutable structures, the same logic does not apply because the ability to
modify the structure (e.g., adding or removing elements) introduces the
possibility of violating type constraints. If you were allowed to treat a
mutable container of `Derived` as a mutable container of `Base` (covariance),
you could insert a `Base` instance into what is actually a container of
`Derived`, breaking the type safety of the container.

### List vs Immutable List

Consider the below example:

```{code-cell} ipython3
class Employee:
    def work(self) -> None: ...

class Manager(Employee):
    def manage(self) -> None: ...

# Hypothetical scenario where List is treated as covariant
def add_new_employee_using_list(employees: List[Employee], employee: Employee) -> None:
    # Adding an Employee to what was originally a List[Manager]
    employees.append(employee)  # This is where type safety is compromised

# Assume that List[Manager] <: List[Employee]
managers: List[Manager] = [Manager(), Manager()]
add_new_employee_using_list(managers, employee=Employee())  # Assuming managers is treated as List[Employee]

# Now, assuming we want to treat all elements in 'managers' as Managers and call Manager-specific methods:
try:
    for manager in managers:
        manager.manage()
except AttributeError as err:
    print(err)
```

If we naively treat `List` as covariant, we would be able to pass a
`List[Manager]` to a function that expects a `List[Employee]` since
`List[Manager]` is a subtype of `List[Employee]`. In the above example, our
`add_new_employee_using_list` would then be able to accept a list of managers.
However, this would break the type safety of the list, as we would be able to
append an `Employee` into a list of `Manager`s. Then downstream, when we try to
treat the elements in the list as `Manager`s, we would get an `AttributeError`
because the elements are actually `[Manager, Manager, Employee]` and `Employee`
does not have the `manage` method.

But if we were to use an immutable list, the type safety would be preserved
because we would not be able to modify the list in a way that would break the
type constraints. This is because the operations that are allowed on an
immutable list are read-only, and the substitution principle is preserved.

```{code-cell} ipython3
_T_co = TypeVar("_T_co", covariant=True)

class ImmutableList(Generic[_T_co]):
    def __init__(self, items: List[_T_co]) -> None:
        self.items = items

    def __iter__(self) -> Iterator[_T_co]:
        return iter(self.items)


def loop_employees(employees: ImmutableList[Employee], employee: Employee) -> None:
    for employee in employees:
        employee.work()

# Assume that List[Manager] <: List[Employee]
managers: ImmutableList[Manager] = ImmutableList([Manager(), Manager()])
loop_employees(managers, employee=Employee())  # Assuming managers is treated as List[Employee]
```

In a similar fashion, using a list here would yield mypy error because they
really aren't sure what operations you would do on the list within the function.

```{code-cell} ipython3
def loop_employees_using_list(employees: List[Employee], employee: Employee) -> None:
    for employee in employees:
        employee.work()

managers: List[Manager] = [Manager(), Manager()]
loop_employees_using_list(managers, employee=Employee())  # Assuming managers is treated as List[Employee]
```

will result in:

```bash
6: error: Argument 1 to "loop_employees_using_list" has incompatible type "list[Manager]"; expected "list[Employee]"  [arg-type]
    loop_employees_using_list(managers, employee=Employee())  # Assuming managers is treated as List[Employee]
                              ^~~~~~~~
6: note: "List" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
6: note: Consider using "Sequence" instead, which is covariant
```

## Drawing Some Connection to Ordinary Functions

Before we tackle the less intuitive concept of contravariance, it is good to
look at an analogy.

Consider the below 3 functions:

```{code-cell} ipython3
def covariant(x: float) -> float:
    return 2*x

def contravariant(x: float) -> float:
    return -x

def invariant(x: float) -> float:
    return x*x
```

1. **Covariant Function**

    The `covariant` function doubles its input value.

    $$
    f(x) = 2x
    $$

    It is now fairly obvious if we understand covariant via
    [**monotonically increasing function**](https://en.wikipedia.org/wiki/Monotonic_function)
    where if $x_1 <= x_2$, then we _must_ have $f(x_1) <= f(x_2)$. And this
    holds true for $f(x) = 2x$ because if $x_1 <= x_2$, then $2x_1 <= 2x_2$.

2. **Contravariant Function**

    The `contravariant` function negates its input value.

    $$
    g(x) = -x
    $$

    Similarly, if we understand contravariant via
    [**monotonically decreasing function**](https://en.wikipedia.org/wiki/Monotonic_function)
    where if $x_1 <= x_2$, then we _must_ have $g(x_2) <= g(x_1)$. And this
    holds true for $g(x) = -x$ because if $x_1 <= x_2$, then $-x_2 <= -x_1$.

3. **Invariant Function**

    The `invariant` function squares its input value.

    $$
    h(x) = x^2
    $$

    Here it more of a **idempotent** or just non-monotonic function. It does not
    preserve the ordering of the input value.

## Contravariant and Callable Types

Functions, as we know it, is a first class citizens in Python. This means that
functions can be passed around as arguments, returned from other functions, and
assigned to variables. This is why we can use the `Callable` type to represent
functions.

### Callable Return Types (Covariance)

When we talk about the return type of a callable (e.g., a function) being
covariant, we mean that a function's return type can be substituted with its
subtype without affecting the correctness of the program. This is
straightforward and intuitive:

-   If you have a function expected to return an `Employee`, a function that
    returns a `Manager` (assuming `Manager` is a subclass of `Employee`) can be
    used instead because every `Manager` is an `Employee`. This substitution
    follows the intuitive direction of the subtype relationship.
-   `Callable[[], Manager]` is a subtype of `Callable[[], Employee]` because the
    function's return value (a `Manager`) can be used in any context that
    expects an `Employee`, following the principle that a subclass instance can
    be used wherever a superclass instance is expected.

```{code-cell} ipython3
def process_employee(get_employee: Callable[[], Employee]) -> None:
    employee = get_employee()
    print(employee.work())

# A function that returns an Employee instance
def get_employee() -> Employee:
    return Employee()

# A function that returns a Manager instance
def get_manager() -> Manager:
    return Manager()

process_employee(get_employee)
process_employee(get_manager)
```

Consequently, `Callable` is **covariant in its return type**.

### Callable Argument Types (Contravariance)

Contravariance for argument types in callable types is where it gets
counterintuitive for many. If a callable is contravariant in its argument type,
it means that you can pass a callable that accepts a more general type in place
of one that accepts a more specific type. This might seem backward at first, but
it makes sense when you consider the principle of substitutability (LSP) from
the caller's perspective:

-   If a function is designed to handle a `Manager` as its input, you can safely
    replace it with a function designed to handle any `Employee`, because the
    latter can deal with `Manager` instances (being a specific kind of
    `Employee`) and more.
-   `Callable[[Employee], None]` is a subtype of `Callable[[Manager], None]`.
    This is because a function capable of dealing with any `Employee` can, by
    definition, deal with a `Manager` (a specific kind of `Employee`).

```{code-cell} ipython3
class CEO(Employee):
    def manage(self) -> None: ...
    def fire(self) -> None: ...

def process_employee(employee: Employee) -> None:
    """This function has type `Callable[[Employee], None]`."""
    employee.work()

def process_manager(manager: Manager) -> None:
    """This function has type `Callable[[Manager], None]`."""
    manager.work()
    manager.manage()

def process_ceo(ceo: CEO) -> None:
    """This function has type `Callable[[CEO], None]`."""
    ceo.work()
    ceo.manage()
    ceo.fire()

def assign_project(employee: Employee, process: Callable[[Employee], None]) -> None:
    process(employee)
```

Now we ask if we can assign `process_manager` of type
`Callable[[Manager], None]` to `process_employee` of type
`Callable[[Employee], None]` (i.e. `process_employee = process_manager`)? Can we
safely substitute `process_employee` with `process_manager`? If we can do that,
then `Callable[[Manager], None]` is a subtype of `Callable[[Employee], None]`.
We cannot do this:

```{code-cell} ipython3
my_good_employee: Employee = Employee()
try:
    assign_project(employee=my_good_employee, process=process_manager)
except AttributeError as err:
    print(err)
```

This is because `process_manager` expects a `Manager` and not just any
`Employee`. This is where contravariance comes into play. It allows us to safely
substitute a function that expects a more specific type with one that expects a
more general type. This is because the function that expects a more general type
can handle the more specific type and more.

More concretely, can we assign `process_employee` to `process_manager`? This is
counterintuitive because we are more used to a subtype $S$ being able to replace
a supertype $T$ and not the other way around.

```{code-cell} ipython3
def assign_project(manager: Manager, process: Callable[[Manager], None]) -> None:
    process(manager)

my_manager: Manager = Manager()
assign_project(manager=my_manager, process=process_employee)
```

This will be fine because `process_employee` can handle a `Manager` instance
since `Manager` is a subclass of `Employee`. More verbosely, in
`process_employee`, we accept an `Employee` instance that may have $N$
functions/methods, but we know for a fact any subclass of `Employee` will have
at least the same $N$ functions/methods. So we can safely pass a `Manager`
instance to `process_employee`. Consequently, `Callable` is **contravariant in
its argument type**.

One more good example from
[PEP 483](https://peps.python.org/pep-0483/#covariance-and-contravariance):

In the salary calculation example:

```python
from decimal import Decimal

def calculate_all(lst: List[Manager], salary: Callable[[Manager], Decimal]):
    ...
```

A `Callable[[Employee], Decimal]` can indeed replace a
`Callable[[Manager], Decimal]` because the former can handle not just `Manager`
instances but any `Employee`, making it a safe and more general substitution.
This adheres to the principle that functions that operate on broader types can
replace those that operate on more specific types within the context of function
arguments.

## Real World Examples of Covariance and Contravariance

-   Here is how
    [ChromaDB uses contravariant](https://github.com/chroma-core/chroma/blob/d9a8c28055ca1aa4c602560c0117f7608858d3f0/chromadb/api/types.py#L146)
    on `Embeddable`.
-   Here is how
    [ChromaDB uses covariance](https://github.com/chroma-core/chroma/blob/d9a8c28055ca1aa4c602560c0117f7608858d3f0/chromadb/api/types.py#L150).
    -   On a side note, `DataLoader` from PyTorch also uses covariance.

## References and Further Readings

-   [PEP 483 - The Theory of Type Hints](https://peps.python.org/pep-0483/#covariance-and-contravariance)
-   [PEP 484 - Type Hints](https://peps.python.org/pep-0484/#covariance-and-contravariance)
-   [Type Hinting: Covariance & Contra-Variance](https://www.playfulpython.com/type-hinting-covariance-contra-variance/)
-   [CS2030S: Variance](https://nus-cs2030s.github.io/2021-s2/18-variance.html)
-   [Variance of generics in mypy](https://mypy.readthedocs.io/en/stable/generics.html#variance-of-generics)
-   [Covariance, Contravariance, and Invariance - The Ultimate Python Guide](https://blog.daftcode.pl/covariance-contravariance-and-invariance-the-ultimate-python-guide-8fabc0c24278)
-   [Covariance and contravariance in subtyping](https://eli.thegreenplace.net/2018/covariance-and-contravariance-in-subtyping/)
-   [Python typing: What does TypeVar('A', 'B', covariant=True) mean?](https://stackoverflow.com/questions/61568462/python-typing-what-does-typevara-b-covariant-true-mean)
-   [Why are arrays invariant but lists covariant?](https://stackoverflow.com/questions/6684493/why-are-arrays-invariant-but-lists-covariant)
-   [What is the difference between covariance and contra-variance in programming languages?](https://stackoverflow.com/questions/1163465/what-is-the-difference-between-covariance-and-contra-variance-in-programming-lan)
-   [Covariance and contravariance (computer science) - Wikipedia](<https://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)>)
