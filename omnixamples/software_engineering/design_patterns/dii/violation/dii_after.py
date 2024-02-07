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
