---
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
```

## Type Safety

A subtype is a foundational concept in type theory and object-oriented
programming that facilitates type safety and polymorphism. The relationship
between a subtype $S$ and a supertype $T$ is denoted as $S <: T$,
$S \subseteq T$, or $S ≤: T$. Before we detail the
**_criterion_**[^what-is-a-criterion] for subtype in the next section, we state
an important implication of subtype - type safety.

```{prf:definition} Subtype and Type Safety
:label: type-theory-subtype-and-type-safety

If $S$ is a subtype of $T$, the subtyping
[relation](<https://en.wikipedia.org/wiki/Relation_(mathematics)>) (written as
$S \leq T$, $S <: T$, $S \subseteq T$, or $S \leq: T$) means that any term of
**type** $S$ can **safely** be used in any **context** where a term of **type**
$T$ is **expected**.
```

In other words, we say that $S$ is a **_subtype_** of $T$ **_if a piece of code
written for variables of type_** $T$ **_can also safely be used on variables of
type_** $S$ [^cs2040s-variable-and-type].

What does this mean?

This means that if $S$ is a subtype of $T$, you can use an instance of $S$ in
any place where an instance of $T$ is required, without any issues related to
type compatibility. That is what we call **subtype polymorphism**.

This concept of subtyping forms the basis of subtype polymorphism in
object-oriented programming. Subtype polymorphism allows objects of a subtype to
be treated as objects of a supertype, enabling methods to operate on objects of
different types as long as they share a common supertype. This mechanism is
critical for implementing interfaces and abstract classes in a type-safe manner.

More formally, in subtype polymorphism, if $S$ is a subtype of $T$ (denoted as
$S <: T$), then objects of type $S$ can be used in contexts expecting objects of
type $T$. This interoperability is guaranteed without loss of integrity or
behavior of the original type $S$, ensuring that operations performed on $T$ are
valid on $S$. This allows for greater flexibility and code reuse while
maintaining strict type safety, as it ensures that the substitutability of
subtypes for their supertypes does not lead to runtime type errors or unexpected
behaviors.

## A Type Safe Example

Assume for a moment that the class `Cat` and `Dog` are both valid _subtype_ of
the class `Animal` through class inheritance, which we have learned earlier to
be called [**nominal subtyping**](../type_theory/01-subtypes.md) (i.e.
subclasses are subtypes).

```{code-cell} ipython3
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

Then this means that _any instance of `Dog` or `Cat` can safely be used in a
context where an instance of `Animal` is expected_.

For example, consider the following function `describe_animal` that takes in an
`animal` of type `Animal`. This is telling developers that we can pass in any
animal as long as it is a subtype of the `Animal` class.

```{code-cell} ipython3
def describe_animal(animal: Animal) -> str:
    return animal.describe() + " makes sound " + animal.make_sound()

generic_animal = Animal()
generic_animal_sound = describe_animal(generic_animal)
print(generic_animal_sound)

generic_dog = Dog()
generic_dog_sound = describe_animal(generic_dog)
print(generic_dog_sound)

generic_cat = Cat()
generic_cat_sound = describe_animal(generic_cat)
print(generic_cat_sound)
```

In fact, what we have described is also inherently related with variable
assignment. You can also think of variable assigning from the function
`describe_animal` above. How so? When you pass in an instance of `Dog`, say
`generic_dog`, to the function `describe_animal`, you are essentially assigning
`generic_dog` to the parameter/variable `animal` in the function.

By _extension_, the following assignment:

```{code-cell} ipython3
generic_animal = generic_dog  # Safe because Dog <: Animal
```

is allowed and considered safe because we are substituting (assigning) an
**expression** `generic_dog` of type instance `Dog` to the **variable**
`generic_animal` is allowed because we established that `Dog` is a subtype of
`Animal` - so it is safe. A static type checker such as `mypy` will not raise an
error here. However, if you were to do the reverse:

```{code-cell} ipython3
generic_dog = generic_animal  # Unsafe because Animal is not a subtype of Dog
```

Then the static type checker will raise an error:

```python
error: Incompatible types in assignment (expression has type "Animal", variable has type "Dog")  [assignment]
    generic_dog = generic_animal
```

indicating that you are trying to assign an expression `generic_animal` of type
`Animal` to the variable `generic_dog` of type `Dog`. This is unsafe because
there is no guarantee that the `Animal` class has say, all methods that a `Dog`
instance might have!

In short, we have:

> Therefore, it's safe to assign an instance of `Dog` to a variable of type
> `Animal` since `Dog` contains all functionalities (`make_sound`) of `Animal`
> and possibly more (`fetch`) so there won't be any surprise here. But it is
> deemed unsafe to assign `generic_animal` to `generic_dog` because not every
> `Animal` is a `Dog`. While every `Dog` instance is an `Animal` (fulfilling the
> subtype criteria), the reverse isn't true. An `Animal` instance might not have
> all functionalities of a `Dog` (like `fetch()`), leading to potential errors
> or undefined behaviors if treated as a `Dog`. This violates the principle that
> the subtype should be able to handle everything the supertype can, plus
> potentially more.

## Violating Type Safety

Consider one example that violates type safety:

```{code-cell} ipython3
class Robot:
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> int:
        return 1

robot = Robot()

try:
    robot_sound = describe_animal(robot)
    print(robot_sound)
except Exception as err:
    print(f"Error: {err}")
```

In python there is no notion of type checking during **compile** time unless you
have a static type checker. Consequently, the above code will only throw an
error during **runtime** because we are adding an integer `1` to the string
`animal.describe() + " makes sound "`. This is because we are errorneously
passing in an instance of `Robot` to a function that accepts `Animal` only.
Since `Robot` is not a subtype of `Animal`, there is no type safety guarantee.
This example can be way worse if we were to just change the `describe_animal` to
return a `f-string` instead - which will not throw any error at all, leading to
hidden bugs!

```{code-cell} ipython3
def describe_animal(animal: Animal) -> str:
    return f"{animal.describe()} makes sound {animal.make_sound()}"

robot = Robot()
robot_sound = describe_animal(robot)
print(robot_sound)
```

and this can often happen in code, sometimes unknowingly.

In the for loop below, we iterate over entities that are presumed to be `Animal`
types. However, including a `Robot` in the list leads to a violation of type
safety that might only be caught at runtime, or worse, not caught at all,
potentially allowing a bug to go unnoticed until it causes a failure in a
production environment.

```python
entities = [Dog, Cat, Robot]
for entity in entities:
    describe_animal(entity)
```

## Further Violation of Type Safety

```{code-cell} ipython3
old: float = 3.01
new: int = 5
old = new  # Safe because int <: float
```

```{code-cell} ipython3
old: int = 3
new: float = 3.03
old = new  # Unsafe because int <: float
# assume the static language doesnt compile error then old will truncate to 3 silently because it is defined as an `int`!
```

## On Dynamic vs Static Type Checking

We have come quite a long way in understanding subtyping schemes and the concept
of type safety. We would like to end this discussion with a brief note on the
difference between dynamic and static type checking. In what follows, we would
turn to the seminal work by Jeremy Siek, titled
[_What is Gradual Typing_](https://wphomes.soic.indiana.edu/jsiek/what-is-gradual-typing/),
as a reference.

### Dynamic Type Checking

Dynamic type checking is a form of type checking that is performed at runtime.
Consider the following Python code, where we erroneously pass a `Car` object to
the `can_we_fly` function, which expects a `Flyable` object. The error manifests
as an `AttributeError` at runtime, indicating that the `Car` object does not
have the `fly` method.

```{code-cell} ipython3
from __future__ import annotations

from typing import Protocol, runtime_checkable

@runtime_checkable
class Flyable(Protocol):
    def fly(self) -> str:
        ...

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
_ = can_we_fly(car)
```

This is deemed as dynamic type checking because the type of the object is only
checked at runtime and the error is only caught when the code is executed.

### Static Type Checking

Static type checking, on the other hand, is a form of type checking that is
performed at compile time. Consider the same piece of Python code, but now we
use the `mypy` static type checker to catch the error at compile time (a better
example would be to use a language like Java or C# that has a more robust static
type checking system).

```bash
sandbox.py:30: error: Argument 1 to "can_we_fly" has incompatible type "Car"; expected "Flyable"  [arg-type]
    _ = can_we_fly(car)
                   ^~~
Found 1 error in 1 file (checked 1 source file)
```

It is worth noting that static type checking make a _conservative
approximation_[^what-is-gradual-typing] of what can happen to the code at
**runtime**, and raise _potential_ type errors that may happen at runtime. In
fact, the
[**the halting problem**](https://en.wikipedia.org/wiki/Halting_problem) implies
that we cannot be 100% sure whether a type error will really occur during
runtime before execution, and thus impossible to build a type checker that can
"predict" what type errors will happen in runtime[^what-is-gradual-typing].
Consequently, static type checkers often make conservative approximations to
ensure that the code is type-safe and result in false positives (i.e. raising
type errors that will not actually occur at runtime).

We quote verbatim the example given by Jeremy Siek in his article. Consider the
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

The Java compiler rejects the following program even though it would not
actually result in a type error because the `if (false)` branch is never taken.
However,the Java type checker does not try to figure out which branch of an if
statement will be taken at runtime. Instead it conservatively assumes that
either branch could be taken and therefore checks both
branches[^what-is-gradual-typing].

### Comparison between Dynamic and Static Type Checking

Some reasons Jeremy gave includes, but not limited to:

-   Static type checking enhances execution speed by eliminating the need for
    type verification during runtime and allowing for the utilization of more
    efficient data storage formats.
-   Dynamic type checking simplifies the handling of scenarios where the type of
    a value is determined by information available at runtime.

You can find more
[reasons in his article](https://wphomes.soic.indiana.edu/jsiek/what-is-gradual-typing/).

## References and Further Readings

-   [Unit 2: Variable and Type - CS2030S](https://nus-cs2030s.github.io/2021-s2/02-type.html)
-   [PEP 483 – The Theory of Type Hints](https://peps.python.org/pep-0483/)
-   [Subtyping - eduNitas](https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html)
-   [What is Gradual Typing](https://wphomes.soic.indiana.edu/jsiek/what-is-gradual-typing/)

[^what-is-a-criterion]:
    A criterion is a principle or standard by which something may be judged or
    decided. In this context, the criterion for subtype is the principle or
    standard by which we decide if one type is a subtype of another type.

[^cs2040s-variable-and-type]:
    [Unit 2: Variable and Type - CS2030S](https://nus-cs2030s.github.io/2021-s2/02-type.html)

[^what-is-gradual-typing]:
    [What is Gradual Typing](https://wphomes.soic.indiana.edu/jsiek/what-is-gradual-typing/)
