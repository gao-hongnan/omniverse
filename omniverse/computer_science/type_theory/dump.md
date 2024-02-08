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

# Type Theory (A Very Rudimentary Guide)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

import math
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple
from rich.pretty import pprint
```

```
from typing import List, Literal, Tuple, Dict, Any

import sys
from dataclasses import dataclass


@dataclass
class IntPair:
    first: int
    second: int

    def get_first(self) -> int:
        return self.first

    def get_second(self) -> int:
        return self.second


@dataclass
class FloatPair:
    first: float
    second: float

    def get_first(self) -> float:
        return self.first

    def get_second(self) -> float:
        return self.second


@dataclass
class Pair:
    first: Any
    second: Any

    def get_first(self) -> Any:
        return self.first

    def get_second(self) -> Any:
        return self.second


def find_min_max(array: List[int]) -> IntPair:
    min_val = sys.maxsize  # Largest possible int
    max_val = -sys.maxsize - 1  # Smallest possible int

    for num in array:
        if num < min_val:
            min_val = num
        if num > max_val:
            max_val = num

    return IntPair(min_val, max_val)


def create_misleading_pair() -> Pair:
    """This function creates a Pair with a string and an integer."""
    return Pair("hello", 4)


def process_pair(pair: Pair) -> None:
    """
    Process elements of the pair, with incorrect assumptions about their types.
    """

    # The programmer incorrectly assumes the first element is an integer
    # and the second element is a string.
    try:
        first_element = int(pair.get_first())  # Mistakenly assuming it's an int
        second_element = str(pair.get_second())  # Mistakenly assuming it's a str

        print(f"First element (assumed int): {first_element * 2}")
        print(f"Second element (assumed str): {second_element.upper()}")
    except ValueError as err:
        print(err)


# Example Usage
pair = create_misleading_pair()
process_pair(pair)
```

```
import torch
from typing import TypeVar, Generic, List
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification  # type: ignore[import-untyped]
from typing import List, Tuple


# Generic types for the model and the dataset
TModel = TypeVar("TModel", bound=torch.nn.Module, covariant=False, contravariant=False)
TData = TypeVar("TData", bound=Dataset, covariant=False, contravariant=False)


class ModelTrainer(Generic[TModel, TData]):
    def __init__(self, model: TModel, dataset: TData, batch_size: int = 32):
        self.model: TModel = model
        self.dataset: TData = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size)
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, target in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss / len(self.dataloader)}")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.dataloader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}%")


class ImageModelTrainer(ModelTrainer[ResNet, ImageFolder]):
    def __init__(self, dataset_path: str, batch_size: int = 32):
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
        dataset = ImageFolder(root=dataset_path, transform=transform)
        # model = resnet18(pretrained=True)
        model: str = resnet18(pretrained=True)
        super().__init__(model, dataset, batch_size)


# Define a simple text dataset
class TextDataset(Dataset):
    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class NLPModelTrainer(ModelTrainer[BertForSequenceClassification, TextDataset]):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 16,
        max_length: int = 128,
    ):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = TextDataset(texts, labels, tokenizer, max_length)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        super().__init__(model, dataset, batch_size)

    def train(self, epochs: int = 3):
        # Overriding the train method to handle specifics of NLP model training
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}, Loss: {total_loss / len(self.dataloader)}")
```


### Type Safety

A subtype is a foundational concept in type theory and object-oriented
programming that facilitates type safety and polymorphism. The relationship
between a subtype $S$ and a supertype $T$ is denoted as $S <: T$,
$S \subseteq T$, or $S ≤: T$. Before we detail the **_criterion_** for subtype
in the next section, we state an important implication of subtype - type safety.

```{prf:definition} Subtype and Type Safety
:label: type-theory-subtype-and-type-safety

If $S$ is a subtype of $T$, the subtyping relation (written as $S <: T$,
$S \subseteq T$, or $S ≤: T$ ) means that any term of **type** $S$ can
**safely** be used in any **context** where a term of **type** $T$ is
**expected**.

In other words, we say that $S$ is a **_subtype_** of $T$ **_if a piece of code
written for variables of type_** $T$ **_can also safely be used on variables of
type_** $S$ [^cs2040s-variable-and-type].
```

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

#### An Example on Type Safety

**DEPRECATED BECAUSE WE DEFINED NOMINAL SUBTYPING IN AN EARLIER SECTION
ALREADY.**

Assume for a moment that the class `Cat` and `Dog` are both _subtype_ of the
class `Animal`. We will see later that this type of subtype through class
inheritance is called **nominal subtyping** (i.e. subclasses are subtypes).

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

You can also think of variable assigning from the function `describe_animal`
above.

For example:

```{code-cell} ipython3
generic_animal = generic_dog  # Safe because Dog <: Animal
```

is allowed because we are substituting (assigning) an **expression**
`generic_dog` of type instance `Dog` to the **variable** `generic_animal` is
allowed because we established that `Dog` is a subtype of `Animal` - so it is
safe. A static type checker such as `mypy` will not raise an error here.
However, if you were to do the reverse:

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

Consider one more example that violates type safety:

```{code-cell} ipython3
class Robot:
    def describe(self) -> str:
        return str(self.__class__.__name__)

    def make_sound(self) -> int:
        return 1

# robot = Robot()
# robot_sound = describe_animal(robot)
# print(robot_sound)
```

In python there is no notion of type checking during compile time unless you
have a static type checker. Consequently, the above code will only throw an
error during runtime because we are adding an integer `1` to the string
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

```python
entities = [Dog, Cat, Robot]
for entity in entities:
    describe_animal(entity)
```

#### Some More Examples

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

### Liskov Substitution Principle

The Liskov Substitution Principle (LSP) is a key concept in understanding
subtype relationships and type safety. Formulated by Barbara Liskov, the
principle states that objects of a superclass shall be replaceable with objects
of its subclasses without affecting the correctness of the program. This
principle is crucial for ensuring that subtype polymorphism is used correctly in
object-oriented programming.

```{prf:theorem} Liskov Substitution Principle
:label: type-theory-liskov-substitution-principle

If `S` is a subtype of `T`, then objects of type `T` in a program may be replaced with objects of type `S` (i.e., objects of the subtype) without altering any of the desirable properties of that program (correctness, task performed, etc.).
```

The LSP emphasizes that a subtype should not only fulfill the structural
requirements of its supertype but also adhere to its behavioral expectations.
This means that the subtype should meet the following conditions:

1. **Signature Matching**: The subtype should offer all methods and properties
   that the supertype offers, maintaining the same signatures.

2. **Behavioral Compatibility**: The behavior of these methods and properties in
   the subtype should conform to the behavior expected by the supertype. This
   includes respecting invariants, postconditions, and preconditions defined by
   the supertype.

The LSP is a foundational guideline in object-oriented design, ensuring that
subclasses extend superclasses in a manner that does not compromise the
functionality and integrity of the superclass.

Following the Liskov Substitution Principle, let's explore the criteria that
define a valid subtype relationship.

### Subsumption, Criterion for Subtype Relationships

See [subsumption](https://en.wikipedia.org/wiki/Subtyping#Subsumption) for a
more rigorous treatment.

In type theory the concept of subsumption is used to define or evaluate whether
a type $S$ is a subtype of type $T$.

Now we will formalize the subtype relationship via
[**mathematical relation**](<https://en.wikipedia.org/wiki/Relation_(mathematics)>).

```{prf:criterion} Subtype Criterion
:label: type-theory-subtype-criterion

Let $\mathcal{T}_1$ and $\mathcal{T}_2$ represent two types in a type system. We
say that $\mathcal{T}_2$ is a subtype of $\mathcal{T}_1$ (denoted as
$\mathcal{T}_2 <: \mathcal{T}_1$) if and only if the following criteria are met:

1. **Value Inclusion (Set Membership)**:
   Every value of the subtype is also a value of the supertype, formally
   expressed as:

    $$
    \forall v \in \mathcal{T}_2, v \in \mathcal{T}_1
    $$

    This criterion ensures that instances of the subtype can be used in any
    context where instances of the supertype are expected.

2. **Function Applicability (Behavioral Compatibility and Preservation of
   Semantics)**:
   This criterion ensures that not only are the operations or functions
   applicable to both the supertype and the subtype, but they also preserve
   their semantics and invariants across these types. This can be formally
   expressed in two parts:

    - **Applicability**: Every function or operation applicable to instances of
      the supertype must also be applicable to instances of the subtype.
      Formally:

        $$
        \forall f, \left( f: \mathcal{T}_1 \rightarrow \mathcal{Y} \right) \Rightarrow \left( f: \mathcal{T}_2 \rightarrow \mathcal{Y} \right)
        $$

        where $f$ is a function from type $\mathcal{T}_1$ (or $\mathcal{T}_2$)
        to some type $\mathcal{Y}$.

    - **Semantic Preservation**: The behavior and semantics of the functions or
      operations must be consistent when applied to either the subtype or the
      supertype. This includes maintaining the expected results, side effects,
      and invariants. Formally, for every function $f$ and for all values
      $v_1 \in \mathcal{T}_1$ and $v_2 \in \mathcal{T}_2$, if $v_1$ and $v_2$
      are considered equivalent in the context of $f$, then $f(v_1)$ and
      $f(v_2)$ must yield equivalent results. This can be represented as:

        $$
        \forall f, \forall v_1 \in \mathcal{T}_1, \forall v_2 \in \mathcal{T}_2, \left( v_1 \sim v_2 \right) \Rightarrow \left( f(v_1) \sim f(v_2) \right)
        $$

        where $\sim$ denotes an equivalence relation appropriate to the context
        of $f$.

3. **Property Preservation (Invariant Maintenance)**:
   All invariants or properties that are true for the supertype must also be
   true for the subtype. This can be formally represented using predicate logic:

    $$
    \forall P, \left( P(\mathcal{T}_1) \right) \Rightarrow \left( P(\mathcal{T}_2) \right)
    $$

    where $P$ is a predicate representing a property or invariant of the type.

    For instance, if $\mathcal{T}_1$ has a property P, then $\mathcal{T}_2$ should
    also exhibit property P.
```

1. **Function Applicability**:

    Given a function $f$, if $f$ is applicable to $\mathcal{T}_1$, then it
    should also be applicable to $\mathcal{T}_2$.

    $$
    \forall f, \left(f: \mathcal{T}_1 \rightarrow \dots \right) \Rightarrow \left(f: \mathcal{T}_2 \rightarrow \dots \right)
    $$

    In other words, consider without loss of generality that $\mathcal{T}_1$ has
    $N$ functionalities, $f_1, f_2, \ldots, f_N$, then $\mathcal{T}_2$ also have
    these $N$ functionalities, $f_1, f_2, \ldots, f_N$ or more.

#### Reflexivity, Transivity and Antisymmetry

A subtype relation is transitive, reflexive and obeys antisymmetry. These are
fundamental properties of the subtype relationship in type theory:

1. **Reflexive**: This means every type is a subtype of itself. Formally, for
   any type $T$, $T <: T$. This reflexivity indicates that you can always use a
   type in the same context as itself, which is a basic and necessary property
   for any type system.

2. **Transitive**: This property means that if type $S$ is a subtype of type
   $T$, and type $T$ is a subtype of type $U$, then type $S$ is also a subtype
   of type $U$. In formal notation, if $S <: T$ and $T <: U$, then $S <: U$. The
   transitive property is crucial for maintaining consistent type relationships
   across a hierarchy or chain of types.

3. **Antisymmetry**: This property states that if type $S$ is a subtype of type
   $T$, and type $T$ is a subtype of type $S$, then $S$ and $T$ are the same
   type. In formal notation, if $S <: T$ and $T <: S$, then $S = T$.
   Antisymmetry ensures that the subtype relationship helps define a clear
   hierarchy or ordering of types, preventing circular relationships where two
   distinct types are each other's subtypes.

#### Narrowing Values, Widening Functions

In the subtype process, the set of values of $\mathcal{T}_2$ is a subset (or
equal to) of $\mathcal{T}_1$, and the set of functions applicable to
$\mathcal{T}_2$ is a superset (or equal to) of those applicable to
$\mathcal{T}_1$. Stated in points for clarity:

-   Point 1: The set of values for which $\mathcal{T}_2$ can take becomes
    smaller in the process of subtyping;
-   Point 2: The set of functions which is applicable to $\mathcal{T}_2$ becomes
    larger in the process of subtyping.

#### An Example on Real Numbers

Let's look at a few examples.

Consider the real number system $\mathbb{R}$, then we say that the integers
(whole numbers) $\mathbb{Z}$ is a **_subtype_** of $\mathbb{R}$ because it
fulfills:

1.  For all values $v \in \mathbb{Z}$, it follows that $v \in \mathbb{R}$ as
    well.
2.  All operations well defined on $\mathbb{R}$ is well defined on the integer
    system $\mathbb{Z}$. For example, all plus, subtraction, multiplication and
    division are applicable on $\mathbb{R}$ and also on $\mathbb{Z}$. But
    $\mathbb{Z}$ can have more, for instance the bitshift operator $<<$, but is
    not defined on $\mathbb{R}$.

Indeed, if we consider the integer number system as a subtype of the real number
system, then the set of integers values is indeed a subset of real numbers
(narrowing values), and the set of operations/functions applicable by the set of
integers widens (widening functions) since integers can do bitshift but real
numbers cannot.

In python, the $\mathbb{R}$ can be denoted as type `float` and $\mathbb{Z}$ as
`int` (ignoring the fact that not all real numbers can be represented precisely
in computer systems).

#### An Example on 2D Shapes

##### Criteria 1: Value Inclusion (Set Membership)

Consider the class `Shape` to represent all 2 dimensional Euclidean Geometry. We
will show how to use the criterion to deduce that the `Circle` class is not only
a subclass of `Shape`, but is also a subtype. We denote $\mathcal{T}_1$ as
`Circle` and $\mathcal{T}_2$ as `Shape`.

```{code-cell} ipython3
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    """Assume `Shape` is 2D and obeys Euclidean Geometry."""

    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError("All 2D Shapes must implement area method")

    @abstractmethod
    def perimeter(self) -> float:
        raise NotImplementedError("All 2D Shapes must implement area method")


class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return math.pi * self.radius**2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius
```

In the given example, `Circle` is a subclass of `Shape`. According to the Value
Inclusion criterion, every instance of `Circle` should also be considered an
instance of `Shape`. This relationship is demonstrated through inheritance in
Python, where `Circle` inherits from the abstract class `Shape`.

This is semantically true in the Euclidean space since circle is definitely a
subset of the 2 dimensional shapes. In python, to illustrate this, we create
instances of `Circle` and verify that they are indeed instances of `Shape`.

```{code-cell} ipython3
circle = Circle(radius=10)
isinstance(circle, Shape)
```

##### Criteria 2: Function Applicability

This criterion ensures that all methods/functions defined in the supertype
(`Shape`) are also implemented in the subtype (`Circle`), with the same
signature and expected behavior.

Consider $f$ be all the methods that `Shape` has (i.e. $f_1$ and $f_2$
corresponding to `area` and `perimeter` respectively), then, according to the
criterion:

$$
\forall f, \left( f: \mathcal{T}_1 \rightarrow \mathcal{Y} \right) \Rightarrow \left( f: \mathcal{T}_2 \rightarrow \mathcal{Y} \right)
$$

means that for all functions defined in the supertype `Shape` ($\mathcal{T}_1$)
with return type $\mathcal{Y}$, we **_must_** have the subtype `Circle`
($\mathcal{T}_2$) to have the same method/signature.

-   **Applicability**: `Circle` implements the `area` and `perimeter` method
    with the same input (none in this case) and output signature (returns a
    float), satisfying the applicability criterion.
-   **Preserve Behavior and Semantics**: The `area` and `perimeter` method in
    `Circle` correctly calculates the area and perimeter of a circle, which is
    consistent with the expected behavior of an `area` and `perimeter` method in
    a 2D shape. This can be checked by verifying that the method returns correct
    values for known inputs.

```{code-cell} ipython3
circle = Circle(radius=5)
assert math.isclose(circle.area(), math.pi * 5**2), "Area method in Circle does not behave as expected."
```

##### Criteria 3: Property Preservation (Invariant Maintenance)

This part is usually associated with the inherent invariance of the parent
class.

An invariant in the context of the `Shape` class could be a property that is
universally true for all shapes. Since `Shape` is an abstract class representing
2D Euclidean shapes, a possible invariant might be:

-   **Non-Negative Area**: All shapes must have a non-negative area.

This property is inherent to the concept of a 2D shape in Euclidean geometry.
The area, being a measure of the extent of a shape in a plane, cannot be
negative.

##### Circle is a Subtype of Shape

Thus, we have shown `Circle` to obey all 3 criterion, and is therefore a subtype
of `Shape`.

##### Reflexivity, Transivity and Antisymmetry

We will use the `Dog` and `Animal` to illustrate these 3 points, but this is a
loose illustration.

```{code-cell} ipython3
class PoliceDog(Dog):
    def search(self) -> str:
        print("Found something!")
```

**Reflexivity** in type theory can be roughly illustrated in Python using the
`isinstance` function. Reflexivity states that a type is a subtype of itself.

In Python, this can be demonstrated by showing that an instance of a class
(e.g., `generic_dog`) is indeed an instance of that same class (e.g., `Dog`).
This is a practical demonstration of the concept that a type (or class) is
compatible with itself.

```{code-cell} ipython3
isinstance(generic_dog, Dog)
```

This code checks whether `generic_dog` is an instance of `Dog`, which will
return `True`. This result aligns with the principle of reflexivity, showing
that an object of a type is always an instance of its own type.

**Transitivity** in type theory states that if Type A is a subtype of Type B,
and Type B is a subtype of Type C, then Type A is also a subtype of Type C. In
the context of our example:

1. `PoliceDog` is a subtype of `Dog`.
2. `Dog` is a subtype of `Animal`.
3. Therefore, `PoliceDog` should be a subtype of `Animal`.

```{code-cell} ipython3
police_dog = PoliceDog()

is_police_dog_instance_of_dog = isinstance(police_dog, Dog)  # Expected to be True
is_police_dog_instance_of_animal = isinstance(police_dog, Animal)  # Expected to be True

print(is_police_dog_instance_of_dog, is_police_dog_instance_of_animal)
```

Anti-symmetry is hard to show in practice as it is more of a theoretical
property.

## Generics and Type Variable (TypeVar)

Reference:

-   https://www.playfulpython.com/python-type-hinting-generics-inheritance/
-   https://docs.python.org/3/library/typing.html#user-defined-generic-types
-   https://docs.python.org/3/library/typing.html#generics
-   https://peps.python.org/pep-0483/#generic-types
-   https://nus-cs2030s.github.io/2021-s2/20-generics.html#generic-types

### The Definitions of Generic and Type Variables

Sometimes it is useful to have a lightweight class to bundle a pair of variables
together. One could, for instance, write a method that returns two values. The
example defines a class IntPair that bundles two int variables together. This is
a utility class with no semantics nor methods associated with it and so, we did
not attempt to hide the implementation details.

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

def process_pair(pair: Pair) -> None:
    """
    Process elements of the pair, with incorrect assumptions about their types.
    """

    # The programmer incorrectly assumes the first element is an integer
    # and the second element is a string.
    try:
        first_element = int(pair.get_first())  # Mistakenly assuming it's an int
        second_element = str(pair.get_second())  # Mistakenly assuming it's a str

        print(f"First element (assumed int): {first_element * 2}")
        print(f"Second element (assumed str): {second_element.upper()}")
    except ValueError as err:
        print(err)


# Example Usage
pair = create_misleading_pair()
process_pair(pair)
```

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

## Covariance, Contravariance and Invariance

Variance refers to places where one data type can be substituted for another.

-   Covariance means that a data type can be substituted with a more specific
    type
-   Contravariance means that a data type can be substituted with a more general
    type
-   Invariance means that a data type cannot be substituted either way

### Covariant?

In the context of type generics and `TypeVar` in Python, covariance (covariant)
refers to a way of defining how types can change through inheritance in a
type-safe manner.

To understand covariance, it's important to first grasp the concepts of
subtyping and inheritance. In object-oriented programming, a class can inherit
from another class, becoming a subtype of it. For instance, if you have a class
`Animal` and a class `Dog` that inherits from `Animal`, `Dog` is a subtype of
`Animal`.

Now, let's consider generics and `TypeVar`:

-   `TypeVar` is a utility for defining generic types in Python. You can create
    a `TypeVar` with or without constraints, and it can be invariant, covariant,
    or contravariant.
-   A type variable is covariant if it allows a subtype relationship to be
    transferred from its base types to the constructed types. In simpler terms,
    if `Dog` is a subtype of `Animal`, and you have a generic type
    `Container[T]`, then `Container[Dog]` can be considered a subtype of
    `Container[Animal]` if `T` is defined as covariant.

Here's a basic example:

```python
from typing import TypeVar, Generic

T = TypeVar('T', covariant=True)

class Container(Generic[T]):
    ...
```

In this example, `T` is covariant. This means if you have a function that
expects `Container[Animal]`, you can safely pass `Container[Dog]` to it, because
`Dog` is a subtype of `Animal`, and due to covariance, `Container[Dog]` is
considered a subtype of `Container[Animal]`.

Covariance is particularly useful for return types. If a method in a base class
returns a type `T`, and in a derived class this method returns a subtype of `T`,
this is a safe and natural use of covariance.

However, covariance has its limitations and isn't suitable for all situations,
especially when dealing with method arguments where contravariance might be more
appropriate. Proper use of covariance in generics ensures type safety and
consistency, adhering to the Liskov Substitution Principle in object-oriented
design.

## Overload and Coercion

```
a: int   = 2
b: float = 3.0
c = a + b
```

implement `__add__` as overload variant.

## WHAT NEXT

-   LSP and justify why we need it esp oop: The concept of subtyping has gained
    visibility (and synonymy with polymorphism in some circles) with the
    mainstream adoption of object-oriented programming. In this context, the
    principle of safe substitution is often called the Liskov substitution
    principle, :
    -   https://wiki.edunitas.com/IT/en/114-10/Subtyping_4238_eduNitas.html
    -   https://wiki.edunitas.com/IT/en/114-10/substitutability_10964_eduNitas.html
-   inclusive vs coercive: Implementations of programming languages with
    subtyping fall into two general classes: inclusive implementations, in which
    the representation of any value of type A also represents the same value at
    type B if A<:B, and coercive implementations, in which a value of type A can
    be automatically converted into one of type B. The subtyping induced by
    subclassing in an object-oriented language is usually inclusive; subtyping
    relations that relate integers and floating-point numbers, which are
    represented differently, are usually coercive.
-   In type theory, which is a theoretical framework for describing types in
    programming languages, there are several different kinds of types that are
    used to describe the structure and behavior of data. Three important types
    are record types, function types, and object types:

1. **Record Types:**

    - **Description**: Record types (also known as composite types, struct
      types, or tuple types) are compound data types that group together several
      fields, each of which can have a type of its own. Each field in a record
      type is identified by a name and has a specific type.
    - **Usage**: Record types are used to model data that has a fixed number of
      components, each with potentially different types. For example, a `Person`
      record type might consist of a `string` for a name, an `int` for age, and
      a `string` for address.
    - **Example**: In many languages, record types are represented as `structs`
      or `classes` without methods.
    - **Syntax Example**:
        ```python
        class Person:
            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age
        ```

2. **Function Types:**

    - **Description**: Function types describe the types of functions,
      specifying the types of their arguments and return values. A function type
      generally has the form `A → B`, indicating a function that takes an
      argument of type `A` and returns a value of type `B`.
    - **Usage**: Function types are central in functional programming and are
      also important in other paradigms for describing the interfaces of
      functions.
    - **Example**: The type of a function that takes an integer and returns a
      string could be written as `int → string`.
    - **Syntax Example**:
        ```python
        def int_to_string(num: int) -> str:
            return str(num)
        ```

3. **Object Types:**

    - **Description**: Object types, commonly found in object-oriented
      programming, define a structure consisting of both data fields
      (attributes) and functions (methods) that operate on the data. The object
      type defines the interface to objects of that type.
    - **Usage**: Object types are used to encapsulate data and behavior
      together, allowing for concepts like inheritance, polymorphism, and
      encapsulation.
    - **Example**: An object type might describe a `Circle` with properties like
      `radius` and methods like `area()`.
    - **Syntax Example**:

        ```python
        class Circle:
            def __init__(self, radius: float):
                self.radius = radius

            def area(self) -> float:
                return 3.14 * self.radius * self.radius
        ```

## References and Further Readings

-   https://wphomes.soic.indiana.edu/jsiek/what-is-gradual-typing/
-   https://nus-cs2030s.github.io/2021-s2/02-type.html
-   https://mypy.readthedocs.io/en/stable/protocols.html
-   https://en.wikipedia.org/wiki/Subtyping
-   https://en.wikipedia.org/wiki/Subtyping#Subsumption
-   https://en.wikipedia.org/wiki/Subtyping#Subtyping_schemes
-   https://www.playfulpython.com/python-type-hinting-generics-inheritance/
-   https://docs.python.org/3/library/typing.html#generics
-   https://docs.python.org/3/library/typing.html#user-defined-generic-types
-   https://peps.python.org/pep-0483/#generic-types
-   https://nus-cs2030s.github.io/2021-s2/20-generics.html#generic-types

[^subtype-wikipedia]: https://en.wikipedia.org/wiki/Subtyping
[^subtype-subsumption-wikipedia]:
    https://en.wikipedia.org/wiki/Subtyping#Subsumption

[^cs2040s-variable-and-type]: https://nus-cs2030s.github.io/2021-s2/02-type.html
