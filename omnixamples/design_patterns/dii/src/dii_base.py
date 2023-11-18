"""Interface class for Transforms."""
from abc import ABC, abstractmethod
from typing import Callable


class Transforms(ABC):
    """Abstract class for transforms."""

    @abstractmethod
    def get_train_transforms(self) -> Callable:
        """Get train transforms."""

    @abstractmethod
    def get_test_transforms(self) -> Callable:
        """Get test transforms."""
