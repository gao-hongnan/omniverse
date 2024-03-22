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

# Function Overloading

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

from abc import ABC, abstractmethod
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple, TYPE_CHECKING, Iterable, Iterator, Sequence, overload, Optional

from rich.pretty import pprint
```

## Motivation

The most common use case for overloading is to define multiple type signatures
for a single function. This is useful when a function can accept different types
of arguments and return different types of results. But why? Most of the times
it is because of a type like `Union`, which tries to capture the different types
of the arguments and return types.

Consider the following example, where we double an integer or a sequence of
integers:

```{code-cell} ipython3
def double_int_or_ints(data: int | Sequence[int]) -> int | List[int]:
    if isinstance(data, Sequence):
        return [i * 2 for i in data]
    return data * 2
```

And if we encounter `data` as a sequence of integers, we will get double each
integer in the sequence and return as `List[int]`. If `data` is an integer, we
simply double it and return as `int`.

What could go wrong here? The problem is because of the `Union` type, an
immediate problem is that it returns either `int` or `List[int]`. Therefore, as
we see below, calling `mypy` on this script with `reveal_type` will show that
the type of `x = double_int_or_ints(data=10)` is `Union[int, List[int]]`.

```{code-cell} ipython3
x = double_int_or_ints(data=10)
```

```python
reveal_type(x)  # note: Revealed type is "Union[builtins.int, builtins.list[builtins.int]]"
```

This is not ideal, because we would like to know the exact type of `x` after
calling `double_int_or_ints`. Although we know for a fact that the `x` in this
context returns an `int`, and is _safe_ to pass into a division function that
divides two integers. But `mypy` will not be able to infer the type of `x` as
`int` and will complain about the division function.

```{code-cell} ipython3
def do_division_on_ints(x: int, y: int) -> float:
    return x / y

y = 100
z = do_division_on_ints(x=x, y=y)
```

Indeed, `mypy` complains the following on incompatible types:

```bash
40: note: Revealed type is "Union[builtins.int, builtins.list[builtins.int]]"
48: error: Argument "x" to "do_division_on_ints" has incompatible type "Union[int, list[int]]"; expected "int"  [arg-type]
    z = do_division_on_ints(x=x, y=y)
                              ^
```

Rightfully, `mypy` is worried that `x` could _potentially_ be a `List[int]` and
not an `int`, and it will definitely throw an error if we try to divide a list
by an integer.

We can signal to `mypy` that `x` is an `int` by using the `isinstance` function
to check if `x` is an `int` and then call the division function.

```{code-cell} ipython3
assert isinstance(x, int)
z = do_division_on_ints(x=x, y=y)
```

This is a form of
[type narrowing](https://mypy.readthedocs.io/en/stable/type_narrowing.html) in
Python, and it is a common pattern to use `isinstance` to narrow the type of a
variable - but it is verbose and can scatter everywhere in the code.

This is where function overloading comes in.

```{code-cell} ipython3
@overload
def double_int_or_ints(data: int) -> int:
    ...

@overload
def double_int_or_ints(data: Sequence[int]) -> List[int]:
    ...

def double_int_or_ints(data: int | Sequence[int]) -> int | List[int]:
    if isinstance(data, Sequence):
        return [i * 2 for i in data]
    return data * 2
```

What did we do here?

-   The `overload` decorator tells us that the function is an allowed
    combination of types and return types.
-   The ellipsis `...` is a special syntax that tells `mypy` that the function
    is an overload variant and not a real function and won't be called at
    runtime.
-   The last function definition is the actual implementation of the function.
-   Now when we pass in `10` to `double_int_or_ints`, `mypy` will know that the
    return type is `int` and not `Union[int, List[int]]` because it will "look
    for" the overload variant that matches the type of the argument.

```python
x = double_int_or_ints(data=10)
reveal_type(x)  # Revealed type is 'builtins.int*'

x_list = double_int_or_ints(data=[1, 2, 3])
reveal_type(x_list)  # Revealed type is 'builtins.list[builtins.int]'

def do_division_on_ints(x: int, y: int) -> float:
    return x / y


y = 100
z = do_division_on_ints(x=x, y=y)
```

Now this code yields no errors, and `mypy` is happy with the type of `x` as it
is inferred to be `int` and can be _safe_ to pass into the division function.

## Function Overloading and Single/Dynamic Dispatch

Both concepts are quite similar, but they are different in the sense that
function overloading is a compile-time polymorphism feature available in
statically typed languages like Java or C++, while single dispatch is a runtime
polymorphism mechanism typically found in dynamically typed languages like
Python.

### Function Overloading

Function or method overloading is a compile-time polymorphism feature available
in statically typed languages like Java or C++. It allows multiple functions or
methods to be defined with the same name but different signatures (i.e.,
different parameter types, numbers, or both). The correct function or method
version is determined at compile time based on the argument types provided in
the call. This decision is made based on the static types of the arguments, and
it enables a form of polymorphism where a single function or method name can
encompass multiple behaviors, depending on the compile-time types of the
arguments it is invoked with.

### Single Dispatch

Single dispatch is a runtime polymorphism mechanism typically found in
dynamically typed languages like Python. It focuses on the type of the object
that a method is invoked upon (i.e., the method's first argument, usually
referred to as `self` in object-oriented languages). In single dispatch, the
method to be executed is determined at runtime based on the type of the first
argument. This allows different methods to be executed for objects of different
types, even if those methods share the same name. Single dispatch enables
polymorphism by allowing a single method call to exhibit various behaviors
depending on the runtime type of the object it is called on.

## Runtime Behavior

Overloaded functions in Python should be structured with multiple overload
variants leading up to a single implementing function. These elements need to be
placed consecutively, essentially forming a single cohesive unit in your code.

For the overload variants, only empty bodies are permitted, marked
conventionally by the ellipsis (`...`) rather than actual code. This is because,
during runtime, Python ignores these variants and only the final, implementing
function is executed. Essentially, despite the presence of overload
declarations, an overloaded function operates just like any standard Python
function. This means there's no built-in mechanism to automatically choose
between variants based on input types; such logic (e.g., using `if` statements
and `isinstance` checks) needs to be manually coded into the implementing
function[^2].

## Overloading with Container

If you take a look into Pythons `builtins.pyi` file, you will see that the
majority of the built-in types have overloads. Here we pick a simple use case in
the context of a list (full type definition can be found in the file
[`builtins.pyi`](https://github.com/python/typeshed/blob/3c08a976564daf4d6f54fbee2fba20ec1d99dbef/stdlib/builtins.pyi#L972)):

```{code-cell} ipython3
T = TypeVar('T')

class SimpleList(Sequence[T]):

    def __init__(self, data: Sequence[T]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        if isinstance(index, int):
            return self.data[index]
        elif isinstance(index, slice):
            return self.data[index]
        else:
            raise TypeError(...)

list_ = SimpleList[int]([1, 2, 3])
print(list_[0])
print(list_[1:3])
```

Here the custom `SimpleList` class has two overload variants for the
`__getitem__` that gives us more precise information. Why? Now it is clear that
if the `__getitem__` receives an `int`, it will return a `T`, and if it receives
a `slice`, it will return a `Sequence[T]`. This is a good example of how to use
overloading to narrow the type of the return value.

## Unsafe Overloading Variants

Mypy will also type check the different variants and flag any overloads that
have inherently unsafely overlapping variants. For example, consider the
following unsafe overload definition:

```{code-cell} ipython3
@overload
def unsafe_func(x: int) -> int: ...

@overload
def unsafe_func(x: object) -> str: ...

def unsafe_func(x: object) -> int | str:
    if isinstance(x, int):
        return 42
    else:
        return "some string"
```

The overloading looks fine, but during runtime it will crash if we do:

```{code-cell} ipython3
some_obj: object = 42
try:
    unsafe_func(some_obj) + " danger danger"  # Type checks, yet crashes at runtime!
except TypeError as err:
    print(err)
```

If you declare `some_obj` as an `object`, and assign an `int` (you can because
`int` is a subclass/subtype of `object`), you would think that the second
overload variant would be called, and return a `str`, which you can do an
addition with another `str`. But it will crash at runtime because it actually
returns an `int`.

The first overload `def unsafe_func(x: int) -> int: ...` is overlapped by the
second overload `def unsafe_func(x: object) -> str: ...` because `int` is a
subtype of `object`. This means that an `int` can be passed to either of the two
overloaded functions, but they have different return types (`int` and `str`),
which is not allowed. So `mypy` will flag this as an error because they define
that two variants are considered unsafely overlapping when both of the following
are true[^2]:

1. All of the arguments of the first variant are potentially compatible with the
   second.
2. The return type of the first variant is not compatible with (e.g. is not a
   subtype of) the second.

Indeed, the first variant `def unsafe_func(x: int) -> int: ...` is compatible
with the second variant `def unsafe_func(x: object) -> str: ...` because `int`
is a subtype of `object`. The return type of the first variant `int` is not
compatible with the second `str` because `int` is not a subtype of `str`.
Consequently, `mypy` will flag this as `overload-overlap`.

## A Not So Good Example on Implementing Base Estimator

Here's an example of how to use overloading to implement a base estimator that
uses `overload` to define two different `fit` methods, one for supervised
learning and one for unsupervised learning.

```python
T = TypeVar("T")

class Unsupervised:
    def __repr__(self) -> Literal["Unsupervised()"]:
        return "Unsupervised()"

UNSUPERVISED = Unsupervised()

class BaseEstimator(ABC):
    @overload
    def fit(self, X: T, y: T) -> BaseEstimator:
        """Overload for supervised learning."""

    @overload
    def fit(self, X: T, y: Unsupervised = ...) -> BaseEstimator:
        """Overload for unsupervised learning."""

    @abstractmethod
    def fit(self, X: T, y: T | Unsupervised= UNSUPERVISED) -> BaseEstimator:
        """
        Fit the model according to the given training data.

        For supervised learning, y should be the target data.
        For unsupervised learning, y should be Unsupervised.
        """


class MyEstimator(BaseEstimator):
    def fit(self, X: T, y: T | Unsupervised = UNSUPERVISED) -> MyEstimator:
        ...
```

## References and Further Readings

MyPy's
[Function Overloading](https://mypy.readthedocs.io/en/stable/more_types.html#function-overloading)
has an extensive collection of the do's and don'ts of function overloading.

Here are the provided links formatted as proper Markdown links:

-   [Python Type Hints: How to Use Overload](https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/)
    by Adam Johnson
-   [Function Overloading in mypy documentation](https://mypy.readthedocs.io/en/stable/more_types.html#function-overloading)
-   [Python Type Hints: How to Narrow Types with isinstance, assert, Literal](https://adamj.eu/tech/2021/05/17/python-type-hints-how-to-narrow-types-with-isinstance-assert-literal/)
    by Adam Johnson
-   [Type Narrowing in mypy documentation](https://mypy.readthedocs.io/en/stable/type_narrowing.html)
-   [Type Coercion and Overloading](https://vaibhavkarve.github.io/pytype.html#orgda87b0a)
    by Vaibhav Karve
-   [PEP 3124 - Overloading, Generic Functions, Interfaces, and Adaptation](https://peps.python.org/pep-3124/)

[^1]:
    [Python Type Hints - How to Use @overload](https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/)

[^2]:
    [Function Overloading - MyPy](https://mypy.readthedocs.io/en/stable/more_types.html#function-overloading)
