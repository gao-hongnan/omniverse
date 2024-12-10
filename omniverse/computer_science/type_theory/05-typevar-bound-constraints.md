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

# Bound and Constraint in Generics and Type Variables

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

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Generator, Generic, List, Literal, Tuple, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from rich.pretty import pprint
from typing_extensions import reveal_type
```

## Some Motivation, The Problem With Unconstrained Type Variable

Let's first examine why unconstrained type variables can lead to issues, a
simple `add` function that adds two arguments together:

```python
from typing import TypeVar

T = TypeVar("T")

def add(x: T, y: T) -> T:
    return x + y
```

And note that running `mypy`[^3] above will yield two particular errors:

```text
4: error: Returning Any from function declared to return "T"  [no-any-return]
        return x + y
        ^~~~~~~~~~~~
4: error: Unsupported left operand type for + ("T")  [operator]
        return x + y
               ^~~~~
```

These errors occur because `TypeVar("T")` is _**unconstrained**_, and this
_means_ `T` can represent literally any type, and not all type can do `add`. Why
so? Because for `+` operator, the underlying object must implement the `__add__`
method. For instance, two dictionaries cannot be added together because their
`__add__` is not well defined. This error that `mypy` raised forces you to
rethink your code design - in that what types of type the type variable `T` can
take on. Assume for the sake of simplicity, that your program's `add` would only
operate on a few types:

-   `int`
-   `float`
-   `NDArray[np.float64]`
-   `str`

And if we can somehow tell our type variable `T` to take on any one of the 4
types above, then our job is done. This motivates the need for _constraining_
type variables to a specific set of types that support our desired operations
(i.e. `+` operator).

## Constraining Type Variable

**Constraints** allow you to specify a list of explicit types that a type
variable (`TypeVar`) can take. This means the type variable can represent any
one of these specified types, and no others.

-   **Syntax**: `T = TypeVar("T", Type1, Type2, ...)`
-   **Meaning**: The type variable `T` can be any one of `Type1`, `Type2`, etc.
-   **Concrete**: `T = TypeVar("T", int, float)` means `T` can only be `int` or
    `float`.

To be more formal, a constrained type variable is a type variable $T$ bound to a
finite set of types $\mathcal{S} = \{T_1, T_2, \ldots, T_n\}$ such that $T$ can
only be instantiated with types from $\mathcal{S}$. So if one declares
`T = TypeVar("T", T_1, T_2, ...)`, then $T$ can only be bound to a type
$T_i \in \mathcal{S} = \{T_1, T_2, \ldots, T_n\}$.

In our `add` example, since we only want the type variable `T` to take on 4
specific types, we can thus re-write the type variable as such:

```{code-cell} ipython3
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", int, float, str, NDArray[np.float64])

def add(a: T, b: T) -> T:
    return a + b
```

Then running `mypy` again, will yield no errors, because the static type checker
is intelligent enough to know that `T` can only take on `int`, `float`, `str`,
or `NDArray[np.float64]`, all of which has the add operator well defined.

### Type Binding

When using a constrained `TypeVar`, the type checker ensures that within a
**single usage of the function**, all occurrences of `T` must be the same type.
This is usually defined as
[**Type Binding**](https://courses.cs.vt.edu/cs3304/Fall16/meng/lecture_notes/cs3304-9.pdf).

What does this even mean? Type binding is the association of a type variable
(like `T`) with a **concrete** type (like `int`). We can intuitively think of
this action as a _substitution_ (or _mapping_) of the type variable with a
concrete type - we replace every occurrence of `T` with the concrete (actual)
type.

#### Type Variables And Substitution

Let's use the earlier example of `add` function, and see how type binding works.

In line 6, we declared `T` as
`TypeVar("T", int, float, NDArray[np.float64], str)`. This means:

-   There _exists_ a type variable `T`.
-   `T` can only be _substituted_ from the set of types
    `{int, float, NDArray[np.float64], str}`.
-   **_All_** occurences of `T` in a single **function invocation** must be
    substituted with the same type.

Since type binding happens at **_function call time (runtime)_**, then the
process is like follows (`T` and $T$ are interchangeable):

-   Say we invoke the function `add(a=x, b=y)` where `x` and `y` are of some
    type.
-   Let `type(x)` be denoted as $T_x$ and `type(y)` be denoted as $T_y$.
-   For the checker to pass, we must abide by:
    -   $T_x$ must be one of the types defined in the set
        `{int, float, NDArray[np.float64]}`.
    -   $T_y$ must be one of the types defined in the set
        `{int, float, NDArray[np.float64]}`.
    -   $T_x$ and $T_y$ must be the same type due to both have same type
        variable $T$.
    -   Then $T$ is mapped (or bound) to $T_x$ ($T \mapsto T_x$).

#### Binding Time And Scope

Consider the following two calls:

```python
def add(a: T, b: T) -> T:
    return a + b

z1 = add(2, 5) # T ↦ int
z2 = add(1.0, 2.0) # T ↦ float
```

Each function call creates a new binding scope:

For the call `z1 = add(2, 5)`:

1. Let $x_1 = 2$ and $x_2 = 5$ be the input arguments
2. Given $\text{type}(x_1) = \text{int} \in \mathcal{S}$ where
   $\mathcal{S} = \{\text{int}, \text{float}, \text{NDArray}[\text{np.float64}], \text{str}\}$
3. Create binding $T \mapsto \text{int}$
4. Verify $\text{type}(x_2) = \text{int} = T$
5. Therefore return type $\text{type}(z_1) = T = \text{int}$ is valid

An invalid binding would be:

```python
z3 = add(1, "3")
```

This is because:

-   `type(1) = int`
-   `T` is bound to `int`
-   `type("3") = str != int != T`
-   Type Error ensues. Mypy will also catch this.

### Union Type versus Constrained Type Variable

The definition of constrained type variable may not be immediately clear on why
we cannot just use an `Union` type. After all, `Union` type is a type that can
take on any one of the types defined in the set
`{int, float, str, NDArray[np.float64]}`. Why bother to purposely specify a
constrained type variable? As we shall see shortly, using `Union` type is a
loose constraint, and it does not enforce the type variable to be of the same
type (i.e. within a function scope, `a` and `b` must be of the same type can be
violated).

#### Union Type

Consider the following union type:

```{code-cell}
from typing import Union
from numpy.typing import NDArray
import numpy as np

U = Union[int, float, NDArray[np.float64]]

def add(a: U, b: U) -> U:
    return a + b
```

This type signature allows any combination of types within the Union.

-   For any arguments $a$ and $b$:
    $\text{type}(a), \text{type}(b) \in \{\text{int}, \text{float}, \text{NDArray}[\text{np.float64}]\}$
-   No constraint exists requiring $\text{type}(a) = \text{type}(b)$

This leads to potentially unsafe operations:

```python
result1 = add(3, 4)                     # Works as expected: 7
result2 = add(3, 9.99)                  # Type mixing: 12.99
result3 = add(3, np.array([1, 2, 3]))   # Implicit broadcasting: array([4, 5, 6])
result4 = add(3, "4")                   # Type mixing: error!
```

The Union type permits all these operations because it only enforces that each
argument independently belongs to the set of allowed types.

More verbosely, with `Union`, the operation we use between arguments (i.e. `a`
and `b`) is supported by any _permutation order_[^2]. As we can see, we added
two `int`, added `int` and `float`, added `int` and an `NDArray`, and lastly,
added `int` and a `str` (which is an error)! Is this really what we want? Do we
really want to allow `int` and `NDArray` to be added together freely (they can,
but some may regard it as not safe as it might lead to undesirable consequences
via broadcasting). Consequently, there will be no error raised for certain
operations that are unsafe. We fear not those that raises an error, but those
that does not (silent errors).

Furthermore, let's consider a scenario where you want to restrict the `add`
function to only work with `int` or `str` types. Using a `Union` type would be
problematic because it allows mixing these types in a single operation (like
adding an `int` with a `str`), which would raise a runtime error.

```python
U = Union[int, str]

def add(a: U, b: U) -> U:
    return a + b
```

Our static type checker is fast to spot this potential error and raised aptly
the following:

```bash
4: error: Unsupported operand types for + ("int" and "str")  [operator]
        return a + b
               ^
4: error: Unsupported operand types for + ("str" and "int")  [operator]
        return a + b
               ^
4: note: Both left and right operands are unions
```

This suggests there's a better way to type hint rather than using `Union` type.

#### Constrained TypeVar Approach

Now consider using a constrained TypeVar:

```python
from typing import TypeVar

T = TypeVar("T", int, float, str, NDArray[np.float64])

def add(a: T, b: T) -> T:
    return a + b
```

This enforces a stronger type constraint. Formally:

-   $\exists T \in \{\text{int}, \text{float}, \text{NDArray}[\text{np.float64}]\}: \text{type}(a) = T \land \text{type}(b) = T$
-   The return type must be the same $T$ that was bound to the arguments

This prevents type mixing:

```python
result1 = add(3, 4)                     # Valid: T ↦ int
result3 = add(3, np.array([1, 2, 3]))   # Type Error: Cannot bind T to both int and NDArray
result4 = add(3, "4")                   # Type Error: Cannot bind T to both int and str
```

TypeVar provides stronger compile-time guarantees and ensures type consistency
within a function scope. Naively, I _guess_ the type checker applies some rules
for each approach above, my guess is:

1. Union types:

    1. For each argument `x`, the type checker will check if `x` is of any of
       the types in the union `Union[T_1, T_2, ...]`.
    2. If `x` is of any of the types in the union, then the type checker will
       proceed to check the next argument.
    3. If `x` is not of any of the types in the union, then the type checker
       will raise an error.

2. Constrained TypeVar:
    1. First occurrence of `T` is checked against the set of types defined in
       the `TypeVar` declaration.
    2. If the type of the first occurrence of `T` is not in the set, then the
       type checker will raise an error.
    3. If the type of the first occurrence of `T` is in the set, then `T` is
       bound to the type of the first occurrence of `T` and **_all subsequent
       occurrences of `T` must be of the same type_**.

## Upper Bounding Type Variables

**Bounds** specify an upper bound for the type variable. This means that the
type variable can represent any type that is a subtype of the specified bound.

-   **Syntax**: `T = TypeVar("T", bound=SuperType)`
-   **Meaning**: The type variable `T` can be any type that is a subtype of
    `SuperType` (including `SuperType` itself).
-   **Concrete**: `T = TypeVar("T", bound=BaseModel)` means `T` can be any type
    that is a subclass of `BaseModel` (from `pydantic`).

### Defining Type Variables with Upper Bounds

In Python's type hinting system, you can define a type variable that restricts
which types can be used in place of it by specifying an upper bound. This is
done using the `bound=<type>` argument in the `TypeVar` function. The key point
is that any type that replaces this type variable must be a subtype of the
specified boundary type.

It's important to note that the boundary type itself cannot be another type
variable or a parameterized type. For example:

```python
# NOT ALLOWED!
T1 = TypeVar("T1")
T2 = TypeVar("T2", bound=T1)        # Cannot use T1 as a bound
```

You also cannot use a parameterized generic type as a bound. For example:

```python
# This is NOT allowed
from typing import List
T = TypeVar('T', bound=List[int])   # Error! Can't use List[int] as a bound

# This is NOT allowed either
S = TypeVar('S')
T = TypeVar('T', bound=List[S])     # Error! Can't use parameterized List as bound
```

### Some Motivation, Simple Length Comparison

Let's say we want to write a function that compares the lengths of two
sequences.

```python
from typing import TypeVar

T = TypeVar("T")

def longer(x: T, y: T) -> T:
    return x if len(x) > len(y) else y
```

This fails the type checker because `T` could be any type - there's no guarantee
it supports `len()`. For example, the below code works because `list` and `str`
both support `len()`:

```python
compare_lengths([1, 2], [3])
compare_lengths("hello", "hi")
```

However, the below code will fail at runtime because `int` does not support
`len()`:

```python
compare_lengths(42, 100)
```

### A Case Study On `Sized`

We can fix this using `bound` to ensure our type variable only accepts types
that support `len()`. Consider the `Sized` protocol from Python's `typing`
module, which represents any type that supports the `len()` function. We define
a type variable `ST` with `Sized` as its upper bound:

```python
from typing import TypeVar, Sized

ST = TypeVar("ST", bound=Sized)
```

This definition means that `ST` can be replaced by any type that has a `len()`
method, ensuring that objects of type `ST` can be measured for their size.

The function `longer` takes two parameters, `x` and `y`, both of type `ST`. It
returns the object with the greater length:

```python
def longer(x: ST, y: ST) -> ST:
    return x if len(x) > len(y) else y
```

Because `ST` is bound to `Sized`, we can safely use `len()` on `x` and `y`. This
allows the function to work with any sized collection, such as lists or sets.

-   `longer([1], [1, 2])` correctly returns the longer list, with the return
    type being `List[int]`.
-   `longer({1}, {1, 2})` operates on sets, returning the larger set as
    `Set[int]`.
-   What's interesting is that `longer([1], {1, 2})` being okay and returning a
    type `Collection[int]` is correct as well. This is because **unlike**
    constraints, we do not need both `x` and `y` to be of the same exact type,
    they just need to be subclass of the bound super type.

### Why Not Use Constraints?

You might wonder why not use constraints, we can try, say, let's say we want to
let `T` be a type that is either a `list`, `str`, or `tuple`.

```python
T = TypeVar('T', list, str, tuple)
```

This approach has limitations because we need to add and list every possible
type that we want to allow, or that has `__len__` method. This is not scalable
and so bound is more applicable.

More formally, we can say something like, let $S$ be the set of all types that
implement the `Sized` protocol. For type variable $T$ with bound `Sized`:

-   $\forall t \in T \implies t \in S$
-   That is, any type $t$ that can be bound to $T$ must be in the set of `Sized`
    types.

### Bounding and Semantic Clarity

Bounding also offers more clarity and semantic meaning, than say, an `Union`
type.

```python
class Animal:
    ...


class Dog(Animal):
    ...


class Cat(Animal):
    ...


class Car:
    ...


AnimalType = TypeVar("AnimalType", bound=Animal)


def function_with_bound(arg: AnimalType) -> AnimalType:
    return arg


def function_with_union(arg: Union[Dog, Cat, Car]) -> Union[Dog, Cat, Car]:
    return arg
```

In `function_with_bound`, the argument arg must be an instance of `Animal` or a
subclass of `Animal`. This means you could pass in an instance of `Dog` or
`Cat`, but not `Car`, because `Car` is not a subclass of `Animal`.

In `function_with_union`, the argument arg can be an instance of `Dog`, `Cat`,
or `Car`. There's no requirement that these types are related in any way.

### A Case Study On `Addable`

Let's say we want to write a function that doubles any number. Without `bound`:

```python
from typing import TypeVar

T = TypeVar('T')

def double(x: T) -> T:
    return x + x
```

If we run through `mypy`, we get the below errors:

```text
tmp.py:237: error: Returning Any from function declared to return "T"  [no-any-return]
        return x + x
        ^~~~~~~~~~~~
tmp.py:237: error: Unsupported left operand type for + ("T")  [operator]
        return x + x
               ^~~~~
Found 2 errors in 1 file (checked 1 source file)
```

Rightfully so, same logic, not all types support `+` operation, like
`double(x={"key": "value"})` will cause errors. To resolve this, we seek to find
a class like `Sized` that has the `__add__` method. But let's say it's hard to
find, we can define our own protocol.

```python
from __future__ import annotations

from typing import TypeVar, Protocol


class Addable(Protocol):
    def __add__(self, other: T) -> T: ...


T = TypeVar("T", bound=Addable)


def double(x: T) -> T:
    return x + x
```

## Bound versus Constraints

-   **Bounds** are used to specify that a type variable must be a subtype of a
    particular type. This is akin to setting an upper limit (or in some
    contexts, a lower limit) on what the type variable can be. The purpose of
    bounds is to ensure that the type variable adheres to a hierarchical type
    constraint, typically ensuring that it inherits certain methods or
    properties.
-   **Constraints**, on the other hand, specify a list of explicit types that a
    type variable can represent, without implying any hierarchical relationship
    between them. The purpose of constraints is to allow a type variable to be
    more flexible by being one of several types, rather than restricting it to a
    subtype of a specific class or interface.

The comparison is pretty superficial but something to remember is that you can
mix types within arguments if you use bound, which behaves a little like
`Union`, whereas in constraint, all arguments must be of the exact same type.

Let's see an example:

```python
AnimalType = TypeVar("AnimalType", bound=Animal)

def function_with_bound(arg1: AnimalType, arg2: AnimalType) -> Tuple[AnimalType, AnimalType]:
    return arg1, arg2

cat = Cat()
tabby = Cat()
dog = Dog()

_, _ = function_with_bound(cat, dog)
```

This above code will not raise any issue when compared to the below code:

```python
AnimalType = TypeVar("AnimalType", Cat, Dog)

def function_with_bound(arg1: AnimalType, arg2: AnimalType) -> Tuple[AnimalType, AnimalType]:
    return arg1, arg2

cat = Cat()
tabby = Cat()
dog = Dog()

_, _ = function_with_bound(cat, dog)
```

This is because if we use constraint, our contract is that within the scope, all
arguments must be of type `AnimalType`, whereas when in bound, `arg1` and `arg2`
can be different, as long as both are upper bounded by `Animal`.

## References and Further Readings

-   [What's the difference between a constrained TypeVar and a Union?](https://stackoverflow.com/questions/58903906/whats-the-difference-between-a-constrained-typevar-and-a-union)
-   [Type variables with an upper bound - PEP484](https://peps.python.org/pep-0484/#type-variables-with-an-upper-bound)
-   [Difference between TypeVar("T", A, B) and TypeVar("T", bound=Union[A, B])](https://stackoverflow.com/questions/59933946/difference-between-typevart-a-b-and-typevart-bound-uniona-b)

[^1]: `mypy run <file_name>.py`
[^2]:
    [What's the difference between a constrained TypeVar and a Union?](https://stackoverflow.com/questions/58903906/whats-the-difference-between-a-constrained-typevar-and-a-union)

[^3]:
    [Difference between TypeVar("T", A, B) and TypeVar("T", bound=Union[A, B])](https://stackoverflow.com/questions/59933946/difference-between-typevart-a-b-and-typevart-bound-uniona-b)
