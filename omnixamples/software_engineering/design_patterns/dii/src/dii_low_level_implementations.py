"""
This module is the low-level concrete implementations of the project. This
module depend only on the abstract interface module.
"""
from src.dii_base import TransformFunc, Transforms  # from abstract interface import Transforms


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
