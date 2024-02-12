-   https://stackoverflow.com/questions/68739824/use-of-generic-and-typevar
    -   explains main diff using contraints versus union,
    -   Why ask this question? Because see below it says oh if you bound a type
        variable `T` to say type `str`, `int` and `float`, then the type
        variable can be any of those types - behaving like a union type.
    -   With Union, the operation you use between arguments has to be supported
        by all arguments in any permutation order:

See second comment in
https://stackoverflow.com/questions/58903906/whats-the-difference-between-a-constrained-typevar-and-a-union

```
from typing import Union

U = Union[int, str]

def add(a: U, b: U):
    return a + b
```

### TypeVar Constraints

**Constraints** allow you to specify a list of explicit types that a type
variable (`TypeVar`) can take. This is akin to saying that the type variable can
represent any one of these specified types, and no others.

-   **Syntax**: `T = TypeVar('T', Type1, Type2, ...)`
-   **Meaning**: The type variable `T` can be any one of `Type1`, `Type2`, etc.

**Example**:

```python
from typing import TypeVar

Animal = TypeVar('Animal', Dog, Cat)

def play_with_animal(animal: Animal) -> None:
    animal.play()
```

In this example, `Animal` can be either a `Dog` or a `Cat` type. The function
`play_with_animal` accepts either a `Dog` or a `Cat` instance as its argument,
leveraging polymorphism while ensuring type safety by restricting the types to
those specified.

### TypeVar Bounds

**Bounds** specify an upper bound for the type variable. This means that the
type variable can represent any type that is a subtype of the specified bound.

-   **Syntax**: `T = TypeVar('T', bound=SuperType)`
-   **Meaning**: The type variable `T` can be any type that is a subtype of
    `SuperType` (including `SuperType` itself).

**Example**:

```python
from typing import TypeVar, Generic

Shape = TypeVar('Shape', bound=GeometricShape)

class Box(Generic[Shape]):
    def __init__(self, shape: Shape) -> None:
        self.shape = shape

    def describe(self) -> None:
        print(f"This box contains a {self.shape.description()}.")
```

In this example, `Shape` can be any type that is a subtype of `GeometricShape`.
This allows the `Box` class to be instantiated with any geometric shape that
derives from `GeometricShape`, ensuring that all types used with `Box` have a
common interface or set of features defined by the `GeometricShape` superclass.

### Comparison and Use Cases

-   **Constraints** are used when you want to allow a type variable to represent
    one of several specific types. This is particularly useful when those types
    do not share a common superclass other than `object`, or when you want to
    include unrelated types.

-   **Bounds** are used when the types you want to allow have a common
    hierarchy, and you want to include all subtypes of a particular superclass.
    This is useful for enforcing that all types used with the type variable
    share certain attributes or methods.
