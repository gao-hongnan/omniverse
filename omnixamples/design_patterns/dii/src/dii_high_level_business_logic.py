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
from typing import Any

from src.dii_base import Transforms  # from abstract interface import Transforms


class CustomDataset:
    """Dummy class for custom dataset."""

    def __init__(self, transforms: Transforms, stage: str = "train") -> None:
        self.stage = stage
        self.transforms = transforms

    def apply_transforms(self, dummy_data: Any = None) -> Any:
        """Apply transforms to dataset based on stage."""
        if self.stage == "train":
            transformed = self.transforms.get_train_transforms()(dummy_data)
        else:
            transformed = self.transforms.get_test_transforms()(dummy_data)
        return transformed

    def getitem(self, dummy_data: Any = None) -> Any:
        """Replace __getitem__ method as normal method for simplicity."""
        return self.apply_transforms(dummy_data=dummy_data)
