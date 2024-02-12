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

This article is closely aligned with and draws inspiration from the materials
provided in
[Unit 20: Generics](https://nus-cs2030s.github.io/2021-s2/20-generics.html) of
the CS2030S course at the National University of Singapore. Some sections have
been adapted and rephrased for clarity and context, in particular translating
the Java code to Python.

## The Motivation

In some scenarios, crafting a simple class to aggregate a duo of variables
proves beneficial, especially when a method needs to output two distinct values.
Consider, for instance, the `IntPair` class, which groups two integer variables.
This class serves merely as a utility, lacking any complex semantics or methods,
and thus, its internal workings are left exposed for simplicity.

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

Such a class is ideal for use cases like a function that needs to return a pair
of integers, exemplified by a hypothetical `find_min_max` function that
determines the minimum and maximum values in an array of integers. However, this
raises a question: what if the need arises to identify the minimum and maximum
in an array of floats?

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

To address a broader range of types, one could theorize the creation of
additional pair classes, such as `DoublePair` for floats or `BooleanPair` for
booleans. Alternatively, designing a pair class that accommodates a combination
of different types, like pairing a `Customer` object with a `ServiceCounter`, or
a string with an integer, could offer more versatility.

However, it's impractical and inefficient to design a separate class for each
potential type pairing. A more elegant solution is to design a generic pair
class capable of encapsulating any two types.

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

This generic `Pair` class can encompass any two types designated upon its
instantiation, providing a versatile structure that can hold any value types,
mirroring the flexibility previously applied in methods to accommodate various
object types.

Nevertheless, this approach is not without its drawbacks, notably the issue of
type conversion narrowing and the potential for runtime errors. For example, if
a function mistakenly returns a `Pair` with a string and an integer but treats
it inversely, static type checking at compile-time won't catch this discrepancy,
possibly leading to errors or crashes during execution. This highlights the
importance of careful type management and the limitations of relying solely on
runtime type identification.

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
            reveal_type(pair.get_first())
            reveal_type(pair.get_second())
            reveal_type(first_element)
            reveal_type(second_element)
            reveal_locals()

    except ValueError as err:
        print(f"Error: {err}")

# Example Usage
pair = create_misleading_pair()
process_misleading_pair(pair)
```

Why does the compiler not be able to detect the type mismatch and stop the
program from crashing during run-time? In fact, if you run the type checker
`mypy` on the above code, it will not raise any error. You can also add
`reveal_type` and `reveal_locals` to the code to see the type of `first_element`
and `second_element` and the local variables.

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

### Pair Problem Revisited

To resolve the issues with the `Pair` class in the motivation, we can use type
variables to parameterize the `Pair` class. This will allow us to specify the
types of the `first` and `second` attributes, as well as the return types of the
`get_first` and `get_second` methods.

```{code-cell} ipython3
S = TypeVar("S")
T = TypeVar("T")

@dataclass
class Pair(Generic[S, T]):
    first: S
    second: T

    def get_first(self) -> S:
        return self.first

    def get_second(self) -> T:
        return self.second

def create_misleading_pair() -> Pair[str, int]: # 101: error: Missing type parameters for generic type "Pair"  [type-arg]
    """This function creates a Pair with a string and an integer."""
    return Pair("hello", 4)

def process_misleading_pair(pair: Pair[str, int]) -> None:
    """
    Process elements of the pair, with incorrect assumptions about their types.
    """

    # The programmer incorrectly assumes the first element is an integer
    # and the second element is a string.
    try:
        first_element = int(pair.get_first())  # Mistakenly assuming it's an int
        second_element = str(pair.get_second())  # Mistakenly assuming it's a str

        if TYPE_CHECKING:
            reveal_type(pair.get_first())
            reveal_type(pair.get_second())
            reveal_type(first_element)
            reveal_type(second_element)
            reveal_locals()

    except ValueError as err:
        print(f"Error: {err}")

# Example Usage
pair = create_misleading_pair()
process_misleading_pair(pair)
```

To recap:

-   `S` and `T` are type variables/parameters that represent the types of the
    `first` and `second` attributes, respectively.
-   `Pair[S, T]` is a generic class that is parameterized with `S` and `T`,
    allowing the `first` and `second` attributes to be of different types.
-   `Pair[str, int]` is a parameterized type that represents a pair of a string
    and an integer. This just means "subsitution" of `S` and `T` with `str` and
    `int` respectively, just like how we assign normal variables.

However, `mypy` actually does not raise any issue on line `27` because Python
allows a string type to be _dynamically converted_ to an integer type if and
only if the string is a valid "integer" string.

But the example, if set in Java, will allow the type checker to catch the type
mismatch and stop the program from crashing during run-time. Why? Because Java
does not allow a string type to be coerced to an integer type. Below I show a
small snippet of Java code that will raise a `ClassCastException` at run-time.

```java
public class Main {

    public static void main(String[] args) {
        // Create a String object
        String stringValue = "hello";

        // Attempt to cast the String to an Integer
        Integer intValue = (Integer) stringValue;
        System.out.println(intValue);
    }
}
```

````{prf:example} I came all the way just for a Moot Example
:label: computer-science-type-theory-generics-moot-example

I converted the notes of CS2030S from Java to Python, just to reach a moot point
where the type checker does not raise any issue on line 27. But the point should
be clear. If we do not parameterize the `Pair` class and use `Any` instead, then
even in a strongly typed language like Java, the type mismatch may not be caught
at compile time.

Let's just construct another simple example where `mypy` will actually raise an
error.

```python
K = TypeVar("K")
V = TypeVar("V")

@dataclass
class Pair(Generic[K, V]):
    key: K
    value: V

    def get_key(self) -> K:
        return self.key

    def get_value(self) -> V:
        return self.value

def log_user_info(user_info: Pair[str, int]) -> None:
    """
    Logs user information, expecting a username (str) and user ID (int).
    """
    username: str = user_info.get_key()
    user_id: int = user_info.get_value()
    print(f"User: {username}, ID: {user_id}")

def create_incorrect_pair() -> Pair[int, str]:
    """
    Incorrectly creates a pair intended to represent user info,
    but swaps the types of the key and value.
    """
    # Mistakenly swapped the order of types, creating a Pair[int, str] instead of Pair[str, int]
    return Pair(12345, "john_doe")

# Example Usage
incorrect_pair = create_incorrect_pair()
log_user_info(user_info=incorrect_pair)
```

Indeed, running `mypy` on the above code will yield:

```bash
33: error: Argument 1 to "log_user_info" has incompatible type "Pair[int, str]"; expected "Pair[str, int]"  [arg-type]
    log_user_info(incorrect_pair)
```
````

## Scope of Generic Methods and Functions

In the course
[Unit 20: Generics - CS2030S](https://nus-cs2030s.github.io/2021-s2/20-generics.html),
it is written in Java, so there is only mention of generic methods. But in
Python, we can also define generic functions where the function signature and
return type (need not be both) are parameterized by type variables.

(computer-science-type-theory-04-generics-generic-functions)=

### Generic Functions

It is possible to use `TypeVar` (type variables) to parameterize the types of
the arguments and return type of a **function** _without_ the need to define a
`Generic` (generic) class. This is because functions are **_naturally and
automatically_** scoped to the type variables defined in the function. What does
that mean?

Let's walk through an example[^4] to find out.

Consider a function that adds two _things_ (not necessarily numbers) together.
In this particular case, if we want to find out the answer of `3 + 4`, we can
use the `add` function to add `3` and `4` together.

```python
def add():
    return 3 + 4
```

Adding two _literal_ things together is **valid** but **restricted** and
obviously is a bad pattern. Polymorphism taught us to _parameterize_ the the two
_things_ to add:

```python
def add(x, y):
    return x + y
```

And injecting `x` and `y` is the act of parameterizing the function with what we
call the **_regular variables_**.

Now, we want to type hint the `add` function. You may likely do the following:

```python
def add(x: int, y: int) -> int:
    return x + y
```

This is again **valid** but **restricted**. The `add` function is now restricted
to adding two integers together. What if we want to add two strings together?
What if we want to add two floats together? What if we want to add two `Item`
objects with `__add__` method defined together?

This is where you would consider using **_type variables_** to parameterize the
types of the arguments and return type of the function.

```python
T = TypeVar('T')

def add(x: T, y: T) -> T:
    return x + y
```

This means that `x` and `y` can be of any type, and the return type of the
function will be the same as the type of `x` and `y`. A contract is born because
now if `x` is defined as an `int` type, then `y` must also be an `int` type, and
the return type of the function will be an `int` type.

Note however, running mypy above will yield two particular errors:

```bash
4: error: Returning Any from function declared to return "T"  [no-any-return]
        return x + y
        ^~~~~~~~~~~~
4: error: Unsupported left operand type for + ("T")  [operator]
        return x + y
               ^~~~~
```

This is because there is no guarantee that the `+` operator is defined for the
type `T`. What if `T` is a `Dict[int, int]` type, then adding two dictionaries
together is not defined (likely no `__add__` method defined for the `dict`
type). But let's ignore this and appreciate the bigger picture.

We will talk about bound and constraints later, which will easily solve this
problem:

```python
T = TypeVar('T', int, float, complex)

def add(x: T, y: T) -> T:
    return x + y # yields no error from mypy
```

In conclusion, the `add` function is now a **_generic function_** because it can
operate on any type of object, and the type of the arguments and return type of
the function are parameterized by type variables. But note that the type
variables are **only** scoped to the function, and they are not available
outside the function.

### Generic Methods

In the context of classes, it is also possible to define **_generic methods_**
where the method signature and return type (need not both) are parameterized by
type variables.

We first look at another violation of type safety via the `Stack` class.

```{code-cell} ipython3
from typing import Any, Generic, List, TypeVar

from rich.pretty import pprint


class Stack:
    def __init__(self) -> None:
        self._container: List[Any] = []

    def push(self, item: Any) -> None:
        self._container.append(item)

    def contains(self, item: Any) -> bool:  # __contains__ method
        for curr in self._container:
            if curr == item:
                return True
        return False


stack = Stack()
stack.push("hello")
stack.push("world")
pprint(stack._container)
```

As you can see, the `contains` method is not type safe. The `item` parameter is
of type `Any`, and the method will return `True` if the `item` is found in the
stack. The problem with `Any`, as repeated many times, is that it is too generic
and does not provide a contract to abide by. For example, our stack above as we
see it, contains strings, and only strings. Now if we want to search for an
integer `to_find = 123` in the stack, the `contains` method will return `False`
because the `item` is not found in the stack. The problem is that the integer
type will never be found in the stack because the stack only contains strings.
However, because both the `self._container` and `item` are of type `Any`, the
type checker will not raise any error.

```{code-cell} ipython3
to_find: int = 123
result: bool = stack.contains(to_find)
pprint(result)
```

We want to bind a contract such that if the `self._container` is of type
`List[str]`, then the `item` must also be of type `str`. If the
`self._container` is of type `List[int]`, then the `item` must also be of type
`int`. This is where we can use type variables to parameterize the types of the
arguments and return type of the method.

```{code-cell} ipython3
from typing import Any, Generic, List, TypeVar

from rich.pretty import pprint

T = TypeVar("T")


class Stack(Generic[T]):
    def __init__(self) -> None:
        self._container: List[T] = []

    def push(self, item: T) -> None:
        self._container.append(item)

    def contains(self, item: T) -> bool:  # __contains__ method
        for curr in self._container:
            if curr == item:
                return True
        return False


stack_of_strings = Stack[str]()
stack_of_strings.push("hello")
stack_of_strings.push("world")
pprint(stack_of_strings._container)

to_find: int = 123
result: bool = stack_of_strings.contains(to_find) # error from mypy
pprint(result)

to_find: str = "hello"
result: bool = stack_of_strings.contains(to_find) # no error from mypy
```

Notably, if we define the `Stack` class as `Stack[str]`, then the `T` is now
bound to `str`. This means that any type hints in the `Stack` class that uses
`T` will now be bound to `str`. This would include the `item` parameter in the
`contains` method. Consequently, the above code will actually yield an error
from `mypy`:

```bash
28: error: Argument 1 to "contains" of "Stack" has incompatible type "int"; expected "str"  [arg-type]
    result: bool = stack.contains(to_find)
```

Similarly, you can define the `Stack` class as `Stack[int]`, and the `T` is now
bound to `int`.

The above example **scopes** the type variable `T` to the `Stack` class in its
entirety. Unlike functions, creating a generic class does not automatically
scope the type variables across attributes and methods. You have to explicitly
define the type hints yourself. Consequently, it is not uncommon to only scope
the type variable to some attributes and methods to the generic class. How you
scope them depends on the use cases.

## References and Further Readings

-   [Type Hinting: Generics & Inheritance](https://www.playfulpython.com/python-type-hinting-generics-inheritance/)
-   [Unit 20: Generics - CS2030S](https://nus-cs2030s.github.io/2021-s2/20-generics.html)
-   [Generics - Python Docs](https://docs.python.org/3/library/typing.html#generics)
-   [User-defined generic types - Python Docs](https://docs.python.org/3/library/typing.html#user-defined-generic-types)
-   [Use of Generic and TypeVar](https://stackoverflow.com/questions/68739824/use-of-generic-and-typevar)
-   [Stack - Omniverse](https://www.gaohongnan.com/dsa/stack/concept.html#the-importance-of-generic-types)
-   [Implementing Generics via Type Erasure - CS2030S](https://nus-cs2030s.github.io/2021-s2/21-erasure.html)
-   [python Generics (intermediate) anthony explains #430](https://www.youtube.com/watch?v=LcfxUU1A-RQ)
-   [Scoping rules for type variables](https://peps.python.org/pep-0484/#scoping-rules-for-type-variables)
-   [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/)
-   [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)

[^1]:
    [Type Hinting: Generics & Inheritance](https://www.playfulpython.com/python-type-hinting-generics-inheritance/)

[^2]:
    [Unit 20: Generics - CS2030S](https://nus-cs2030s.github.io/2021-s2/20-generics.html)

[^3]:
    [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/#generic-types)

[^4]:
    [Use of Generic and TypeVar](https://stackoverflow.com/questions/68739824/use-of-generic-and-typevar)
