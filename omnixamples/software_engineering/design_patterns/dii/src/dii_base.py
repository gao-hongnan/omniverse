"""Interface class for Transforms."""


from abc import ABC, abstractmethod
from typing import Any, Callable

TransformFunc = Callable[[Any], str]


class Transforms(ABC):
    """Abstract class for transforms."""

    @abstractmethod
    def get_train_transforms(self) -> TransformFunc:
        """Get train transforms."""

    @abstractmethod
    def get_test_transforms(self) -> TransformFunc:
        """Get test transforms."""
