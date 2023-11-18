"""Less violation but everything is contained in one script, the inversion is not obvious."""
from abc import ABC, abstractmethod
from typing import Any, Callable


class Transforms(ABC):
    """Abstract class for transforms."""

    @abstractmethod
    def get_train_transforms(self) -> Callable:
        """Get train transforms."""

    @abstractmethod
    def get_test_transforms(self) -> Callable:
        """Get test transforms."""


class ImageClassificationTransforms(Transforms):
    """Dummy class for image classification transforms."""

    def get_train_transforms(self) -> Callable:
        """Get train transforms."""
        print("Getting image classification train transforms.")
        return lambda x: None

    def get_test_transforms(self) -> Callable:
        """Get test transforms."""
        print("Getting image classification test transforms.")
        return lambda x: None


class ImageSegmentationTransforms(Transforms):
    """Dummy class for image segmentation transforms."""

    def get_train_transforms(self) -> Callable:
        """Get train transforms."""
        print("Getting image segmentation train transforms.")
        return lambda x: None

    def get_test_transforms(self) -> Callable:
        """Get test transforms."""
        print("Getting image segmentation test transforms.")
        return lambda x: None


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


if __name__ == "__main__":
    dataset = CustomDataset(transforms=ImageClassificationTransforms(), stage="train")
    dataset.getitem(dummy_data=None)

    # you can change transforms from ImageClassification to ImageSegmentation
    dataset = CustomDataset(transforms=ImageSegmentationTransforms(), stage="train")
    dataset.getitem(dummy_data=None)
