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
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from typing_extensions import reveal_type
from rich.pretty import pprint
```

If you recall from
{ref}`computer-science-type-theory-04-generics-generic-functions` section, we
made up a simple example on `add`:

```python
T = TypeVar('T')

def add(x: T, y: T) -> T:
    return x + y
```

And note that running mypy above will yield two particular errors:

```bash
4: error: Returning Any from function declared to return "T"  [no-any-return]
        return x + y
        ^~~~~~~~~~~~
4: error: Unsupported left operand type for + ("T")  [operator]
        return x + y
               ^~~~~
```

As a recap, this is because `T` can represent literally any type, and not all
type can do `add`. For instance, two dictionaries cannot be added together
because their `__add__` is not well defined. This error that `mypy` raised
forces you to rethink your code design - in that what types of type the type
variable `T` can take on. Assume for the sake of simplicity, that your program's
`add` would only operate on a few types:

-   `int`
-   `float`
-   `NDArray[np.float64]`

And if we can somehow tell our type variable `T` to take on any one of the 3
types above, then our job is done. This is where we would use the
**constraints**.

## Constraining Type Variable

**Constraints** allow you to specify a list of explicit types that a type
variable (`TypeVar`) can take. This is akin to saying that the type variable can
represent any one of these specified types, and no others.

-   **Syntax**: `T = TypeVar('T', Type1, Type2, ...)`
-   **Meaning**: The type variable `T` can be any one of `Type1`, `Type2`, etc.

### Revisiting Add Example

In our `add` example, since we only want the type variable `T` to take on 3, and
only 3 types, we can thus re-write the type variable as such:

```{code-cell} ipython3
T = TypeVar('T', int, float, NDArray[np.float64])

def add(a: T, b: T) -> T:
    return a + b
```

Then running `mypy` again, will yield no errors, because the static type checker
is intelligent enough to know that `T` can only take on `int`, `float` or
`NDArray[np.float64]`, all of which has the add operator well defined.

### Union Type versus Constrained Type Variable

The definition of constrained type variable may not be immediately clear on why
we cannot just use an `Union` type.

Consider the following union type:

```{code-cell}
U = Union[int, float, NDArray[np.float64]]

def add(a: U, b: U) -> U:
    return a + b


add_two_int = add(a=3, b=4)
pprint(add_two_int)

add_int_and_float = add(a=3, b=9.99)
pprint(add_int_and_float)

add_int_and_ndarray = add(a=3, b=np.array([1, 2, 3]))
pprint(add_int_and_ndarray)
```

With `Union`, the operation we use between arguments (i.e. `a` and `b`) is
supported by any _permutation order_[^1]. As we can see, we added two `int`,
added `int` and `float`, and lastly, added `int` and an `NDArray`! Is this
really what we want? Do we really want to allow `int` and `NDArray` to be added
together freely (they can, but some may regard it as not safe as it might lead
to undesirable consequences via broadcasting). Consequently, there will be no
error raised.

Furthermore, what if your programming logic changes and now you only want to add
`int` and `str`. This will be problematic because if we use union type, then it
has potential of adding an `int` and a `str`, which is likely lead a type error
saying unsupported operand type for the add operator between `int` and `str`.

```{code-cell}
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

This is when type variable prove to be more type safe here.

```{code-cell} ipython3
T = TypeVar("T", int, float, NDArray[np.float64])


def add(a: T, b: T) -> T:
    return a + b


add_int_and_ndarray = add(a=3, b=np.array([1, 2, 3]))
```

And now, `mypy` will raise an error here, telling you that you need to abide to
the contract, that within the scope of the function `add`, all type variable
must be the same type! So `add_int_and_ndarray` will raise an error.

Furthermore, now the below example will be okay because the static type checker
has piece of mind that both `a` and `b` must of same type and no mixing is
involved.

```{code-cell} ipython3
T = TypeVar("T", int, str)


def add(a: T, b: T) -> T:
    return a + b
```

## Upper Bounding Type Variables

**Bounds** specify an upper bound for the type variable. This means that the
type variable can represent any type that is a subtype of the specified bound.

-   **Syntax**: `T = TypeVar('T', bound=SuperType)`
-   **Meaning**: The type variable `T` can be any type that is a subtype of
    `SuperType` (including `SuperType` itself).

The excerpt discusses how to enforce type constraints in Python using the
`TypeVar` function from the `typing` module, focusing on the concept of an upper
bound. Here's a more structured explanation for clarity:

### Defining Type Variables with Upper Bounds

In Python's type hinting system, you can define a type variable that restricts
which types can be used in place of it by specifying an upper bound. This is
done using the `bound=<type>` argument in the `TypeVar` function. The key point
is that any type that replaces this type variable must be a subtype of the
specified boundary type. It's important to note that the boundary type itself
cannot be another type variable or a parameterized type.

### Example: Ensuring Type Safety with `Sized`

Consider the `Sized` protocol from Python's `typing` module, which represents
any type that supports the `len()` function. We define a type variable `ST` with
`Sized` as its upper bound:

```python
from typing import TypeVar, Sized

ST = TypeVar('ST', bound=Sized)
```

This definition means that `ST` can be replaced by any type that has a `len()`
method, ensuring that objects of type `ST` can be measured for their size.

The function `longer` takes two parameters, `x` and `y`, both of type `ST`. It
returns the object with the greater length:

```python
def longer(x: ST, y: ST) -> ST:
    if len(x) > len(y):
        return x
    else:
        return y
```

Because `ST` is bound to `Sized`, we can safely use `len()` on `x` and `y`. This
allows the function to work with any sized collection, such as lists or sets.

-   `longer([1], [1, 2])` correctly returns the longer list, with the return
    type being `List[int]`.
-   `longer({1}, {1, 2})` operates on sets, returning the larger set as
    `Set[int]`.
-   The statement about `longer([1], {1, 2})` being okay and returning a type
    `Collection[int]` is correct as well. This is because unlike constraints, we
    do not need both `x` and `y` to be of the same exact type, they just need to
    be subclass of the bound super type.

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
-   [Difference between TypeVar('T', A, B) and TypeVar('T', bound=Union[A, B])](https://stackoverflow.com/questions/59933946/difference-between-typevart-a-b-and-typevart-bound-uniona-b)

[^1]:
    [What's the difference between a constrained TypeVar and a Union?](https://stackoverflow.com/questions/58903906/whats-the-difference-between-a-constrained-typevar-and-a-union)

[^2]:
    [Difference between TypeVar('T', A, B) and TypeVar('T', bound=Union[A, B])](https://stackoverflow.com/questions/59933946/difference-between-typevart-a-b-and-typevart-bound-uniona-b)
