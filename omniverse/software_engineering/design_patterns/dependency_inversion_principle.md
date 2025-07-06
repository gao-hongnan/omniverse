---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Dependency Inversion Principle

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/tree/e96d19fc2cc5d4a1f9311fe91aced78ab5f4a910/omnixamples/software_engineering/design_patterns/dii)

```{contents}
```

## Definition

In object-oriented design, the
[dependency inversion principle](https://en.wikipedia.org/wiki/Dependency_inversion_principle)
is a specific methodology for loosely coupling software modules. When following
this principle, the conventional dependency relationships established from
high-level, policy-setting modules to low-level, dependency modules are
reversed, thus rendering high-level modules independent of the low-level module
implementation details. The principle states[^1]:

-   High-level modules should not import anything from low-level modules. Both
    should depend on abstractions (e.g., interfaces).
-   Abstractions should not depend on details. Details (concrete
    implementations) should depend on abstractions.

By dictating that both high-level and low-level objects must depend on the same
abstraction, this design principle inverts the way some people may think about
object-oriented programming[^2].

## Low Level and High Level Modules

Low level modules are "low level" because they have no dependencies, or no
relevant dependencies. Very often, they can be easily reused in different
contexts without introducing any separate, formal interfaces - which means,
reusing them is straightforward, simple and does not require any Dependency
Inversion.

High level modules, however, are "high level", because they require other, lower
level modules to work. But if they are tied to a specific low-level
implementation, this often prevents to reuse them in a different context.

High level modules depend on low level modules, but shouldn't depend on their
implementation. This can be achieved by using interfaces, thus decoupling the
definition of the service from the implementation[^3].

## A Violation of Dependency Inversion Principle

Let's look at a code example that violates the **Dependency Inversion
Principle**.

```{code-cell} ipython3
"""Violation of DII."""
from typing import Any, Callable, List, Literal, Union

from rich.pretty import pprint

TransformFunc = Callable[[Any], str]


class ImageClassificationTransforms:
    """Dummy class for image classification transforms."""

    def get_train_transforms(self) -> TransformFunc:
        """Get train transforms."""
        return lambda x: f"Using {self.__class__.__name__} for training: {x}"

    def get_test_transforms(self) -> TransformFunc:
        """Get test transforms."""
        return lambda x: f"Using {self.__class__.__name__} for testing: {x}"


class ImageSegmentationTransforms:
    """Dummy class for image segmentation transforms."""

    def get_train_transforms(self) -> TransformFunc:
        """Get train transforms."""
        return lambda x: f"Using {self.__class__.__name__} for training: {x}"

    def get_test_transforms(self) -> TransformFunc:
        """Get test transforms."""
        return lambda x: f"Using {self.__class__.__name__} for testing: {x}"


# violates DIP
class CustomDataset:
    """Enhanced class for a custom dataset, with a real __getitem__ method."""

    def __init__(self, data: List[Any], stage: Literal["train", "test"] = "train") -> None:
        self.data: List[Any] = data
        self.stage: str = stage

        # Directly using ImageClassificationTransforms without interface/abstraction
        self.transforms: ImageClassificationTransforms = ImageClassificationTransforms()

    def apply_transforms(self, item: Any) -> str:
        """Apply transforms to a single data item based on stage."""
        if self.stage == "train":
            transformed = self.transforms.get_train_transforms()(item)
        else:
            transformed = self.transforms.get_test_transforms()(item)
        return transformed

    def __getitem__(self, index: Union[int, slice]) -> Union[str, List[str]]:
        """Fetch and transform item(s) from dataset by index."""
        if isinstance(index, int):  # Single item requested
            item = self.data[index]
            return self.apply_transforms(item)
        elif isinstance(index, slice):  # Slice of items requested
            items = self.data[index]
            return [self.apply_transforms(item) for item in items]
        else:
            raise TypeError("Invalid index type. Must be int or slice.")


if __name__ == "__main__":
    dummy_data = ["data1", "data2", "data3", "data4", "data5"]
    dataset_train = CustomDataset(data=dummy_data, stage="train")
    dataset_test = CustomDataset(data=dummy_data, stage="test")

    # Access single item
    single_item_train = dataset_train[2]  # Should apply training transform to 'data3'
    pprint(single_item_train)
    single_item_test = dataset_test[2]  # Should apply testing transform to 'data3'
    pprint(single_item_test)

    # Access a slice of items
    slice_items_train = dataset_train[1:4]  # Should apply training transform to 'data2', 'data3', 'data4'
    slice_items_test = dataset_test[1:4]  # Should apply testing transform to 'data2', 'data3', 'data4'
    pprint(slice_items_train)
    pprint(slice_items_test)

    # you cannot change transforms from ImageClassification to ImageSegmentation
    # without changing the code in CustomDataset class.
```

In the provided code example, we encounter a clear violation of the Dependency
Inversion Principle (DIP), which affects the design's flexibility and
maintainability. Let's dissect the relationship and dependency structure between
the modules involved:

-   **`CustomDataset`**: This is a **high-level** module designed to manage
    dataset items and apply specific transformations to these items based on the
    dataset's stage (training or testing). Its responsibilities include fetching
    data items by index and applying the appropriate transformation functions to
    prepare data for either training or testing scenarios.

-   **`ImageClassificationTransforms` and `ImageSegmentationTransforms`**: These
    **low-level** modules are responsible for providing the actual
    transformation functions applicable to the dataset items. Each module
    defines two methods, `get_train_transforms` and `get_test_transforms`, which
    return transformation functions tailored to either training or testing.

-   In our code, the high level module depends on the low level module such that
    the **creation** of `ImageClassificationTransforms` is done inside the
    `CustomDataset` constructor. This leads to **high coupling**.

-   **Direct Dependency**: Within the `CustomDataset` constructor, an instance
    of `ImageClassificationTransforms` is directly instantiated. This direct
    instantiation tightly couples the `CustomDataset` class to the
    `ImageClassificationTransforms` class. As a result, `CustomDataset` is not
    just dependent on the abstraction of transformation functions but on a
    specific implementation of these functions provided by
    `ImageClassificationTransforms`.

-   **Coupling and Flexibility Issues**: This coupling between the high-level
    `CustomDataset` module and the low-level `ImageClassificationTransforms`
    module restricts the flexibility of the dataset management system. Should
    there be a need to apply a different set of transformations (e.g., using
    `ImageSegmentationTransforms` for a segmentation task), the current design
    requires modifying the `CustomDataset` class itself to change the
    dependency. This design goes against the DIP's guidance, which suggests that
    both high-level and low-level modules should depend on abstractions, not on
    concrete implementations.

In other words:

-   The code looks fine but if we want to change the
    `ImageClassificationTransforms` to `ImageSegmentationTransforms`, then we
    have to change the `CustomDataset` code in two places:
    -   Type hint of `ImageClassificationTransforms` to
        `ImageSegmentationTransforms`;
    -   Change manually the `ImageClassificationTransforms` to
        `ImageSegmentationTransforms` in the constructor.
-   Things soon get out of hand if we have a lot more of such dependencies, such
    as `ObjectDetectionTransforms`, `ImageCaptioningTransforms`, etc.

## Correcting the Violation

To adhere to the Dependency Inversion Principle, the design should be refactored
such that:

1. **Depend on Abstractions**: Both `CustomDataset` and the transformation
   providing modules (`ImageClassificationTransforms`,
   `ImageSegmentationTransforms`) should depend on a common abstraction (e.g.,
   an interface or a base class) that defines the contract for transformation
   functions.

2. **Injection of Dependencies**: Instead of directly instantiating a specific
   transformations class within the `CustomDataset` constructor, the
   transformation provider should be passed in as a parameter (dependency
   injection). This approach allows the `CustomDataset` to remain agnostic of
   the concrete implementation of transformation functions, enhancing
   modularity, flexibility, and the ease of testing.

```{figure} ./assets/uml.drawio.svg
---
name: UML Diagram.
---

A very ugly UML diagram.
```

More concretely, we can create an interface `Transforms` that will be
implemented by `ImageClassificationTransforms`, `ImageSegmentationTransforms`,
etc. Then, we can pass the `Transforms` object to the `CustomDataset`
constructor `__init__` method. This way, the `CustomDataset` will depend on the
`Transforms` interface and not on the `ImageClassificationTransforms` class.
This way, we can change the `ImageClassificationTransforms` to
`ImageSegmentationTransforms` without changing the `CustomDataset` code. This is
called **Dependency Inversion**.

The abstraction does not depend on details simply mean the abstract class should
not hold any implementation. The implementation should be done in the concrete
class.

For example, in my `Transforms(ABC)` abstract class/interface below, I have two
abstract methods `get_train_transforms` and `get_test_transforms`. These methods
are not implemented in the abstract class. They are implemented in the concrete
class `ImageClassificationTransforms`. This is the second rule in **Dependency
Inversion Principle**.

In the high level module `CustomDataset`, I have a constructor `__init__` that
takes in a `Transforms` abstract class/interface. Now my `CustomDataset` depends
on the `Transforms` abstraction and not on the `ImageClassificationTransforms`
class. This is the first rule in **Dependency Inversion Principle**.
Furthermore, if you were to switch your task from image classification to image
segmentation, you can simply change the `ImageClassificationTransforms` to
`ImageSegmentationTransforms` without changing the `CustomDataset` code as you
are not **creating/coupled** to the `ImageClassificationTransforms` class.

```{code-cell} ipython3
"""Less violation but everything is contained in one script, the inversion is not obvious."""
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Union

from rich.pretty import pprint

TransformFunc = Callable[[Any], str]


class Transforms(ABC):
    """Abstract class for transforms."""

    @abstractmethod
    def get_train_transforms(self) -> TransformFunc:
        """Get train transforms."""

    @abstractmethod
    def get_test_transforms(self) -> TransformFunc:
        """Get test transforms."""


class ImageClassificationTransforms(Transforms):
    """Dummy class for image classification transforms."""

    def get_train_transforms(self) -> TransformFunc:
        """Get train transforms."""
        return lambda x: f"Using {self.__class__.__name__} for training: {x}"

    def get_test_transforms(self) -> TransformFunc:
        """Get test transforms."""
        return lambda x: f"Using {self.__class__.__name__} for testing: {x}"


class ImageSegmentationTransforms(Transforms):
    """Dummy class for image segmentation transforms."""

    def get_train_transforms(self) -> TransformFunc:
        """Get train transforms."""
        return lambda x: f"Using {self.__class__.__name__} for training: {x}"

    def get_test_transforms(self) -> TransformFunc:
        """Get test transforms."""
        return lambda x: f"Using {self.__class__.__name__} for testing: {x}"


class CustomDataset:
    """Enhanced class for a custom dataset, with a real __getitem__ method."""

    def __init__(self, transforms: Transforms, data: List[Any], stage: Literal["train", "test"] = "train") -> None:
        self.data: List[Any] = data
        self.stage: str = stage

        # Directly using ImageClassificationTransforms without interface/abstraction
        self.transforms = transforms

    def apply_transforms(self, item: Any) -> str:
        """Apply transforms to a single data item based on stage."""
        if self.stage == "train":
            transformed = self.transforms.get_train_transforms()(item)
        else:
            transformed = self.transforms.get_test_transforms()(item)
        return transformed

    def __getitem__(self, index: Union[int, slice]) -> Union[str, List[str]]:
        """Fetch and transform item(s) from dataset by index."""
        if isinstance(index, int):  # Single item requested
            item = self.data[index]
            return self.apply_transforms(item)
        elif isinstance(index, slice):  # Slice of items requested
            items = self.data[index]
            return [self.apply_transforms(item) for item in items]
        else:
            raise TypeError("Invalid index type. Must be int or slice.")


if __name__ == "__main__":
    dummy_data = ["data1", "data2", "data3", "data4", "data5"]

    image_classification_transforms = ImageClassificationTransforms()
    dataset_train = CustomDataset(transforms=image_classification_transforms, data=dummy_data, stage="train")
    pprint(dataset_train[0])

    # you can change transforms from ImageClassification to ImageSegmentation
    image_segmentation_transforms = ImageSegmentationTransforms()
    dataset_test = CustomDataset(transforms=image_segmentation_transforms, data=dummy_data, stage="test")
    pprint(dataset_test[0])
```

Originally, `CustomDataset` **creates** its own dependency and it is the one
controlling the dependency. Now after applying **Dependency Inversion
Principle**, `CustomDataset` is no longer creating its own dependency. It is now
**injected** with the dependency. This **inverts the control** of the dependency
from `CustomDataset` to the caller of `CustomDataset`. This is the **Dependency
Inversion Principle**.

In traditional sense, since class A depends on class B, then class A is the one
creating the dependency. But after applying **Dependency Inversion Principle**,
class A is no longer creating the dependency. Instead, the dependency is
instantiated outside of class A at runtime and is injected into class A. This is
the **Dependency Inversion Principle**, a form of **Inversion of Control**.

## Dependency Inversion Principle and Dependency Injection

The Dependency Inversion Principle (DIP) is just one part of the larger concept
of Dependency Injection (DI). While DIP is about the static structure of your
code (i.e., how classes and modules are related to each other), DI is about how
the dependencies are provided to an object at runtime.

DIP is primarily concerned with the design and static structure of code, guiding
developers on how to write modules that remain as independent as possible from
the implementations of the modules they rely on.

DI, on the other hand, is a design pattern that **implements** the Dependency
Inversion Principle. It's about the actual mechanics of providing an object with
the outside resources (dependencies) it needs to perform its functions.

In summary, DIP advocates for a particular structure of code relationships to
reduce the coupling between high-level and low-level modules, while DI is a
practical way to achieve this by handling the creation and binding of
dependencies externally. Adhering to DIP will allow your software to become more
modular, easier to test, and more maintainable.

## References and Further Readings

-   [DIP in the Wild - Martin Fowler](https://martinfowler.com/articles/dipInTheWild.html)
-   [Dependency Inversion - ArjanCodes](https://www.youtube.com/watch?v=Kv5jhbSkqLE)

[^1]:
    [Abstractions should not depend on implementations.](https://stackoverflow.com/questions/52857145/what-is-mean-by-abstractions-should-not-depend-on-details-details-should-depen)

[^2]:
    [Wikipedia: Dependency Inversion Principle](https://en.wikipedia.org/wiki/Dependency_inversion_principle)

[^3]:
    [What are "High-level modules" and "low-level modules" (in the context of Dependency inversion principle)?](https://stackoverflow.com/questions/3780388/what-are-high-level-modules-and-low-level-modules-in-the-context-of-depende)
