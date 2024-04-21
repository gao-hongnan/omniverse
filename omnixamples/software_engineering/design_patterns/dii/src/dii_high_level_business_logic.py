"""
This module is the high-level business logic of the project. This module will
only depend on the abstract interface module. This module will not depend on any
low-level concrete implementations. Violating will cause your code to be highly
coupled.

Contrary to the violation example earlier, we have successfully decoupled the
**injected** our dependency (the original ImageClassificationTransforms) as an
abstract interface (Transforms).

Note we are using the Constructor Dependency Injection. There are other types
such as Setter Dependency Injection and Method Dependency Injection.

Originally, `CustomDataset` **creates** its own dependency and it is the one
controlling the dependency. Now after applying **Dependency Inversion
Principle**, `CustomDataset` is no longer creating its own dependency. It is now
**injected** with the dependency. This **inverts the control** of the dependency
from `CustomDataset` to the caller of `CustomDataset`. This is the **Dependency
Inversion Principle**.

More concretely, in traditional sense, since class A depends on class B, then
class A is the one creating the dependency. But after applying **Dependency
Inversion Principle**, class A is no longer creating the dependency. Instead,
the dependency is instantiated outside of class A at runtime and is injected
into class A. This is the **Dependency Inversion Principle**, a form of
**Inversion of Control**.

Note that it is similar to Strategy Pattern. But we will not be changing the
behavior of the algorithm at runtime. We will be changing the algorithm at
compile time.
"""
from typing import Any, List, Literal, Union

from omnixamples.software_engineering.design_patterns.dii.src.dii_base import (  # from abstract interface import Transforms
    Transforms,
)


class CustomDataset:
    """Enhanced class for a custom dataset, with a real __getitem__ method."""

    def __init__(self, transforms: Transforms, data: List[Any], stage: Literal["train", "test"] = "train") -> None:
        self.data: List[Any] = data
        self.stage: str = stage

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
