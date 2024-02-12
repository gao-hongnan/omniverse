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

## Tricky Subtype example

The tricky example you mentioned highlights a subtle issue in type theory
related to subtype relationships and the concept of variance in generic types.

1. **Subtyping Condition**: While every `int` is a `float`, making one inclined
   to think `List[int]` is a subtype of `List[float]`, this is not the case. The
   reason is due to the second condition of subtyping related to function
   applicability.

2. **Function Applicability Issue**: `List[int]` and `List[float]` differ in how
   they react to certain operations. You can append a `float` to a
   `List[float]`, but not to a `List[int]`. Therefore, passing `List[int]` where
   `List[float]` is expected can lead to operations that are invalid for
   `List[int]`.

3. **Example Breakdown**: In the given code, appending `3.14` (a `float`) to
   `my_list` (typed as `List[int]`) would be invalid. This invalid operation
   would either fail or incorrectly alter the list's type integrity. The
   subsequent operation `my_list[-1] << 5` expects an integer (due to the
   bitwise shift operation), which would fail or behave unexpectedly if
   `my_list[-1]` is a float.

This example underscores the importance of understanding variance (covariance
and contravariance) in generic types and why certain intuitive subtype
relationships might not hold in practice, especially for mutable types like
lists.

It fails the second criterion of subtyping (function applicability) because the
set of valid operations for `List[int]` is not the same as for `List[float]`.
Specifically, appending a `float` to a `List[int]` is not valid. This difference
means that while `List[int]` can be used wherever `List[float]` is expected in
terms of containing only numerical values, it cannot be used in contexts where
operations specific to `List[float]` (like appending a `float` value) are
required. This violation of function applicability criteria makes `List[int]`
not a subtype of `List[float]`.

```python
def append_pi(lst: List[float]) -> None:
    lst += [3.14]

my_list = [1, 3, 5]  # type: List[int]

append_pi(my_list)   # Naively, this should be safe...

my_list[-1] << 5     # ... but this fails
```

## Covariant?

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

## Overload

### Example BaseEstimator

Use

```
class Unsupervised:
    def __repr__(self):
        return "Unsupervised()"

UNSUPERVISED = Unsupervised()

class BaseEstimator(ABC):
    @overload
    def fit(self, X: T, y: T) -> BaseEstimator:
        """Overload for supervised learning."""

    @overload
    def fit(self, X: T, y: Unsupervised = UNSUPERVISED) -> BaseEstimator:
        """Overload for unsupervised learning."""

    @abstractmethod
    def fit(self, X: T, y: Union[T, Unsupervised] = UNSUPERVISED) -> BaseEstimator:
        """
        Fit the model according to the given training data.

        For supervised learning, y should be the target data.
        For unsupervised learning, y should be Unsupervised.
        """
        pass

# Example subclass
class MyEstimator(BaseEstimator):
    def fit(self, X: T, y: Union[T, Unsupervised] = UNSUPERVISED) -> BaseEstimator:
        if y is UNSUPERVISED:
            # Unsupervised learning logic
            ...
        else:
            # Supervised learning logic
            ...
        return self
```

### Overload and Coercion Relationship

```
a: int   = 2
b: float = 3.0
c = a + b
```

implement `__add__` as overload variant.

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
