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

# Generics and Type Variables

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple, TYPE_CHECKING
from typing_extensions import reveal_type
from rich.pretty import pprint
```

## The Motivation

Sometimes it is useful to have a lightweight class to bundle a pair of variables
together. One could, for instance, write a method that returns two values. The
example defines a class `IntPair` that bundles two int variables together. This
is a utility class with no semantics nor methods associated with it and so, we
did not attempt to hide the implementation details.

```{code-cell} ipython3
from dataclasses import dataclass

@dataclass
class IntPair:
    first: int
    second: int

    def get_first(self) -> int:
        return self.first

    def get_second(self) -> int:
        return self.second
```

This class can be used, for instance, in a function that returns two int values.
What the author mentioned here is trying to say is this `IntPair` works well
with a function such as `find_min_max` below, but what if `find_min_max` wants
to find the min and max of `float` too?

```{code-cell} ipython3
def find_min_max(array: List[int]) -> IntPair:
    min_val = sys.maxsize  # Largest possible int
    max_val = -sys.maxsize - 1  # Smallest possible int

    for num in array:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num

    return IntPair(min_val, max_val)
```

We could similarly define a pair class for two (DoublePair), two booleans
(BooleanPair), etc. In other situations, it is useful to define a pair class
that bundles two variables of two different types, say, a Customer and a
ServiceCounter; a String and an int; etc.

We should not, however, create one class for each possible combination of types.
A better idea is to define a class that stores two Object references:

```{code-cell} ipython3
@dataclass
class Pair:
    first: Any
    second: Any

    def get_first(self) -> Any:
        return self.first

    def get_second(self) -> Any:
        return self.second
```

This `Pair` class can now hold any two types specified at instantiation.

At the cost of using a wrapper class in place of primitive types, we get a
single class that can be used to store any type of values.

You might recall that we used a similar approach for our contains method to
implement a general method that works for any type of object. Here, we are using
this approach for a general class that encapsulates any type of object.

Unfortunately, the issues we faced with narrowing type conversion and potential
run-time errors apply to the Pair class as well. Suppose that a function returns
a Pair containing a String and an Integer, and we accidentally treat this as an
Integer and a String instead, the compiler will not be able to detect the type
mismatch and stop the program from crashing during run-time.

```{code-cell} ipython3
def create_misleading_pair() -> Pair:
    """This function creates a Pair with a string and an integer."""
    return Pair("hello", 4)

def process_misleading_pair(pair: Pair) -> None:
    """
    Process elements of the pair, with incorrect assumptions about their types.
    """

    # The programmer incorrectly assumes the first element is an integer
    # and the second element is a string.
    try:
        first_element = int(pair.get_first())  # Mistakenly assuming it's an int
        second_element = str(pair.get_second())  # Mistakenly assuming it's a str
        if TYPE_CHECKING:
            reveal_type(first_element)
            reveal_type(second_element)
            reveal_locals()

        print(f"First element (assumed int): {first_element * 2}")
        print(f"Second element (assumed str): {second_element.upper()}")
    except ValueError as err:
        print(f"Error: {err}")

# Example Usage
pair = create_misleading_pair()
process_misleading_pair(pair)
```

Why does the author say the compiler will not be able to detect the type
mismatch and stop the program from crashing during run-time? Yes, in fact, if
you run the type checker `mypy` on the above code, it will not raise any error.
You can also add `reveal_type` and `reveal_locals` to the code to see the type
of `first_element` and `second_element` and the local variables.

```python
16: note: Revealed type is "builtins.int"
17: note: Revealed type is "builtins.str"
18: note: Revealed local types are:
18: note:     first_element: builtins.int
18: note:     pair: Pair
18: note:     second_element: builtins.str
```

Only when you run the code, then the error occur. What's worse, is that the
error is silent in both compile and run time.

```{code-cell} ipython3
def create_silent_error_pair() -> Pair:
    """This function creates a Pair with a string and an integer."""
    return Pair('16', '4')

def process_silent_error_pair(pair: Pair) -> str:
    """
    Process elements of the pair, with incorrect assumptions about their types.
    """

    try:
        first_element = str(pair.get_first())
        second_element = str(pair.get_second())
        if TYPE_CHECKING:
            reveal_type(first_element)
            reveal_type(second_element)
            reveal_locals()

    except ValueError as err:
        print(f"Error: {err}")

    return first_element + second_element
```

And what is wrong in this code? The programmer actually wanted to create a pair
of `int` but mistakenly created a pair of `str` and want to add them together.
Instead of getting `20`, the result is `164`.

Running `mypy` on the above code will not raise any error as well, as they
inferred "correctly" here.

The problem here is that `Any` is too "generic" literally. When the programmer
write the return type of `create_misleading_pair` and `create_silent_error_pair`
as `Pair`, the there is no type binding to the `Pair` class for the type of
`first` and `second` attribute.

If there is a way to parameterize the `Pair` class, then the type checker can
detect the type mismatch and stop the program from crashing during run-time. If
we can create two _generic_ type variables, `S` and `T`, both not necessarily
the same, then we can have a contract that the `first` attribute of `Pair` is of
type `S` and the `second` attribute of `Pair` is of type `T`, then necessarily,
the return type of `get_first` and `get_second` will be `S` and `T`. If we can
do that, our problems may be solved.

## Containers are Generics

Let's defer the solution to the motivation above and first understand some
practical examples of generics in Python. This is adapted from
[Type Hinting: Generics & Inheritance](https://www.playfulpython.com/python-type-hinting-generics-inheritance/)

```{code-cell} ipython3
@dataclass
class Employee:
    name: str
    id: int

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"

    def unique_id(self) -> str:
        return self.__str__()
```

Python's list is a dynamic array that can hold any type of object. This has its
advantages, but it also means that the type of the elements in the list is not
enforced. This can lead to bugs and errors that are difficult to track down.

For example, consider the following list of dynamic types:

```{code-cell} ipython3
list_of_dynamic_types: List = [Employee("Alice", 1), 1, "hello", 3.14]

try:
    for item in list_of_dynamic_types:
        print(item.__str__())
        print(item.unique_id())
except AttributeError as err:
    print(f"Error: {err}")
```

In the above code, the `list_of_dynamic_types` contains a mix of `Employee`,
`int`, `str`, and `float` objects. When we try to call the `__str__` and
`unique_id` methods on each item in the list, we get an `AttributeError` because
not all the items in the list have these methods.

To address this issue, we can use Python's type hinting system to specify the
type of the elements in the list. This will allow the type checker to catch
errors at compile time, rather than at run time.

```{code-cell} ipython3
list_of_employees: List[Employee] = [Employee("Alice", 1), Employee("Bob", 2)]
list_of_ints: List[int] = [1, 2, 3, 4]
list_of_strings: List[str] = ["hello", "world"]

if TYPE_CHECKING:
    reveal_type(list_of_employees)  # Revealed type is 'List[Employee]'
    reveal_type(list_of_ints)  # Revealed type is 'List[int]'
    reveal_type(list_of_strings)  # Revealed type is 'List[str]'

for employee in list_of_employees:
    print(employee.unique_id()) # this will not raise any error because mypy knows each employee is of type Employee and has unique_id method
```

And if you append `Employee` to `list_of_ints`:

```{code-cell} ipython3
list_of_ints.append(Employee("Charlie", 3))  # Error: Argument 1 to "append" of "list" has incompatible type "Employee"; expected "int"
```

Running `mypy` will yield:

```bash
150: error: Argument 1 to "append" of "list" has incompatible type "Employee"; expected "int"  [arg-type]
    list_of_ints.append(Employee("Charlie", 3))  # Error: Argument 1 to "append" of "list" has incompatible type "Employee"; expected "int"
```

So why do we say containers are generics? If you notice, when we type hint
`list_of_employees`, `list_of_ints`, and `list_of_strings`, we are actually
using a generic type `List` with a type parameter `Employee`, `int`, and `str`
respectively via `List[Employee]`, `List[int]`, and `List[str]`. This is the
same as the `Pair` class we defined in the motivation above. The `List` class is
a generic class that can hold any type of object, and the type parameter
specifies the type of the elements in the list.

How do we use generics and typevar in this context?

Consider a very simple function that just want to append an `int` element to a
list and return the list:

```{code-cell} ipython3
def append_int_and_return_list(list_: List[int], element: int) -> List[int]:
    list_.append(element)
    return list_

list_of_ints = [1, 2, 3]
new_list_of_ints = append_int_and_return_list(list_of_ints, 4)
print(new_list_of_ints)
```

and another function that just want to append a `str` element to a list and
return the list:

```{code-cell} ipython3
def append_str_and_return_list(list_: List[str], element: str) -> List[str]:
    list_.append(element)
    return list_

list_of_strings = ["hello", "world"]
new_list_of_strings = append_str_and_return_list(list_of_strings, "!")
print(new_list_of_strings)
```

The above two functions are very similar, and the only difference is the type of
the list and the type of the element. We really do not want two functions doing
the same thing as the `append` method is well defined for the `List` class. We
can use `Any` as the type of the list and the element, but this will not catch
type errors at compile time if we silently slip in a `str` to a list of `int` or
vice versa.

```{code-cell} ipython3
def append_and_return_list(list_: List[Any], element: Any) -> List[Any]:
    list_.append(element)
    return list_

list_of_ints: List[int] = [1, 2, 3]
new_list_of_ints: List[Any] = append_and_return_list(list_of_ints, "hello")
print(new_list_of_ints)
```

In comes the type variable.

```{code-cell} ipython3
T = TypeVar('T')

def append_and_return_list(list_: List[T], element: T) -> List[T]:
    list_.append(element)
    return list_

list_of_ints: List[int] = [1, 2, 3, 4, 5]

# This will cause a mypy error
new_list_of_ints: List[int] = append_and_return_list(list_=list_of_ints, element="abcedf")
print(new_list_of_ints)
```

Running `mypy` on the above code will yield:

```bash
10: error: Argument "element" to "append_and_return_list" has incompatible type "str"; expected "int"  [arg-type]
    new_list_of_ints: List[int] = append_and_return_list(list_=list_of_ints, element="abcedf")
```

By now, we can spot the pattern from the examples above. Our list is a
container, it is called a generic container because it can hold any type of
object parameterized by the type parameter/variable `T`. The `T` is a type
variable, and it is a placeholder for the actual type that will be used when the
function is called.

`T` _generalizes_ the type in the list: `List[int]`, `List[str]`,
`List[Employee]`, etc and `T` is called a variable because it is literally a
_variable of types_. `List[str]` is a _specific_ type, and `List[T]` is a
_generic_ type that can represent any specific type. Consequently, `List[T]` is
a _generic type_ and `T` is a _type variable_[^1].

```{prf:remark} Why not just use Any?
:label: computer-science-type-theory-generics-why-use-t

You might wonder why we use a type variable `T` to denote generality in types,
rather than simply using `Any`, given that both seem to imply a kind of
universality. However, there's a crucial distinction between them in terms of
type safety.

The type variable `T` is generic, meaning it represents a specific but
unspecified type across the context in which it's used. This specificity allows
for type consistency within a given scope. For example, if a function is defined
to accept a list of type `List[T]` and an element of type `T`, `T` will be the
same type for both the list and the element. This generic but consistent typing
ensures that the function can operate on a list and an element of the same,
though unspecified, type, thereby maintaining type safety by preventing type
mismatches.

On the other hand, `Any` is indeed "more generic" but in a way that bypasses
type safety checks. Declaring a list as `List[Any]` tells the type checker that
the list can contain elements of any type, effectively disabling type safety for
elements of that list. This means that both integers and strings (and any other
type) can coexist in such a list without raising type errors at compile-time or
during static analysis. While this provides flexibility, it sacrifices the
guarantees that come with stricter type checking.

Using `Any` indiscriminately can lead to silent type errors where, for instance,
a string is accidentally added to a list of integers. The type system won't
catch this mix-up at compile-time or during static analysis with `mypy`, leading
to potential runtime errors or bugs that are harder to trace and fix.
```

You can find another custom example on implementing a stack using generics in
the
[Stack - Omniverse](https://www.gaohongnan.com/dsa/stack/concept.html#the-importance-of-generic-types)
page.

## Generics and Type Variables

### Generics

Generics enable the construction of new, more specific types from abstract type
definitions by using what are known as **generic type constructors**. These
constructors operate at the type level, similar to how functions operate at the
value level.

-   **Generic Type**: A generic type is a type definition that includes one or
    more type variables. It's a template from which concrete types can be
    constructed. For example, `Tuple[T]` or more broadly `Tuple[T, ...]` (using
    Python's typing syntax) is the generic type. It's not `Tuple` itself that's
    the generic type, but rather `Tuple` parameterized with type variables like
    `T` or with specific types (we usually call the realization of the generic
    type as a _parameterized type_[^2] (concrete type)).

-   **Generic Type Constructor**: This term refers to the mechanism by which
    generic types like `Tuple[T, ...]` are used to create new, parameterized
    types. The constructor takes a type (or types) as an argument and "returns"
    a new parameterized type based on the generic definition. For instance, when
    you use `Tuple[float, ...]`, you're invoking the generic type constructor of
    `Tuple` with `float` as the argument, resulting in a new parameterized type
    that can be thought of as a vector of floats.

Here are two examples illustrating this concept:

1. **Vector Example**: When `Tuple` is parameterized with `float` and an
   ellipsis (`...`) indicating a variable length, `Tuple[float, ...]` acts as a
   parameterized type that could represent a mathematical vector of floats. In
   this scenario, `Tuple[float, ...]` is a parameterized type derived from the
   generic type `Tuple[T, ...]` by specifying `float` as the type argument.

2. **Registry Example**: Similarly, by parameterizing `Tuple` with another type,
   such as `UserID` (assuming `UserID` is a type you've defined elsewhere),
   `Tuple[UserID, ...]` becomes a parameterized type that could represent a
   registry of user IDs. Again, `Tuple[UserID, ...]` is a parameterized type
   constructed from the generic type `Tuple[T, ...]` using `UserID` as the type
   argument.

Such semantics is known as _generic type constructor_, which is similar to the
semantics of functions, where a function takes a value and returns a value. In
this case, a generic type constructor takes a type and returns a type[^3].

### Type Variable and Type Parameter

A **type variable** is a placeholder used in generic programming to denote a
type that is not specified at the point of declaration but will be determined
later, at the time of use. Type variables allow the definition of generic types
or functions that can operate on any data type. They are essentially the
variables of type expressions.

A **type parameter** is similar to a type variable, but the term is often used
in the context of defining generic classes, interfaces, or methods. The type
parameter is declared as a part of a generic type or method definition and
specifies a placeholder that will be replaced with an actual type when the
generic type is instantiated or the generic method is invoked. It allows for the
creation of parameterized types or methods where the specific type(s) to be used
are specified when a class is instantiated or a method is called.

The `T` defined via `TypeVar` is a type variable, and it is a placeholder for
the actual type that will be used when the function is called. For instance,
looking at the function below:

```python
T = TypeVar('T')

def append_and_return_list(list_: List[T], element: T) -> List[T]:
    list_.append(element)
    return list_
```

We do not know what `T` is at the time of defining the function, but we know
that `T` will be a type of the list and the element when the function is called.

### Type Argument

A **type argument** refers to the concrete type that is supplied in place of a
type variable or parameter when a generic class, interface, or method is
actually used. It effectively "fills in" the generic placeholder with a specific
type, thereby instantiating or invoking the generic entity with that specific
type. The type argument provides the actual type information that allows the
generic mechanism to operate on specific data types, ensuring type safety and
consistency.

For example, in a generic class `List[T]`, if you create a new instance with
`new List[int]()`, the `int` is the type argument replacing the type parameter
`T` for that specific instance.

Consider the earlier example of the `append_and_return_list` function:

```python
T = TypeVar('T')

def append_and_return_list(list_: List[T], element: T) -> List[T]:
    list_.append(element)
    return list_
```

When we invoke this function, we supply type arguments through the types of the
arguments passed to the function. If we call `append_and_return_list` with a
list of integers and an integer element:

```python
list_of_ints: List[int] = [1, 2, 3, 4, 5]
new_list_of_ints: List[int] = append_and_return_list(list_=list_of_ints, element=6)
```

In this case, `int` serves as the type argument for `T`. The generic `T` in the
function's definition is replaced by `int`, making the function operate
specifically on a list of integers. The type argument ensures that the generic
function can be applied to a specific data type, in this context, integers,
thereby tailoring the generic function to a particular use case while preserving
type safety.

## Scopes of Type Variables

## I do not know how to connect the below

We should note the example given by the author is about pairs, but in fact even
single type annotation of `Any` can lead to the same kind of errors!

Next, let's define the `find_min_max` function, which will return a
`Pair[int, int]`:

```python
def find_min_max(array: Tuple[int, ...]) -> Pair[int, int]:
    if not array:
        raise ValueError("Array must not be empty")

    min_val, max_val = float('inf'), float('-inf')
    for i in array:
        if i < min_val:
            min_val = i
        if i > max_val:
            max_val = i
    return Pair(min_val, max_val)
```

In this function, `Tuple[int, ...]` is used to specify that the input should be
a tuple of integers. The function then calculates the minimum and maximum values
and returns them as a `Pair[int, int]`.

Finally, let's demonstrate the use of this `Pair` class with a function that
returns a `Pair` of different types, like `str` and `int`:

```python
def example_function() -> Pair[str, int]:
    return Pair("hello", 4)

# Example Usage
p = example_function()
first_element = p.get_first()  # Will be inferred as str
second_element = p.get_second()  # Will be inferred as int
```

This implementation in Python, with strict type hinting, provides the benefits
of type safety and clarity while maintaining Python's dynamic nature. It
addresses the issues of type safety and human error in the original example by
leveraging Python's type hinting system.

## References and Further Readings

[Type Hinting: Generics & Inheritance](https://www.playfulpython.com/python-type-hinting-generics-inheritance/)

-   [Unit 20: Generics - CS2030S](https://nus-cs2030s.github.io/2021-s2/20-generics.html)
-   [Generics - Python Docs](https://docs.python.org/3/library/typing.html#generics)
-   [User-defined generic types - Python Docs](https://docs.python.org/3/library/typing.html#user-defined-generic-types)
-   [Use of Generic and TypeVar](https://stackoverflow.com/questions/68739824/use-of-generic-and-typevar)
-   [Stack - Omniverse](https://www.gaohongnan.com/dsa/stack/concept.html#the-importance-of-generic-types)
-   [Implementing Generics via Type Erasure - CS2030S](https://nus-cs2030s.github.io/2021-s2/21-erasure.html)

[^1]:
    [Type Hinting: Generics & Inheritance](https://www.playfulpython.com/python-type-hinting-generics-inheritance/)

[^2]:
    [Unit 20: Generics - CS2030S](https://nus-cs2030s.github.io/2021-s2/20-generics.html)

[^3]:
    [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/#generic-types)
