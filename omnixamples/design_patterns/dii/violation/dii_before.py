"""Violation of DII."""
from typing import Any, Callable


class ImageClassificationTransforms:
    """Dummy class for image classification transforms."""

    def get_train_transforms(self) -> Callable:
        """Get train transforms."""
        print("Getting image classification train transforms.")
        return lambda x: None

    def get_test_transforms(self) -> Callable:
        """Get test transforms."""
        print("Getting image classification test transforms.")
        return lambda x: None


class ImageSegmentationTransforms:
    """Dummy class for image segmentation transforms."""

    def get_train_transforms(self) -> Callable:
        """Get train transforms."""
        print("Getting image segmentation train transforms.")
        return lambda x: None

    def get_test_transforms(self) -> Callable:
        """Get test transforms."""
        print("Getting image segmentation test transforms.")
        return lambda x: None


# violates DIP
class CustomDataset:
    """Dummy class for custom dataset."""

    def __init__(self, stage: str = "train") -> None:
        self.stage = stage
        self.transforms: ImageClassificationTransforms = ImageClassificationTransforms()

    def apply_transforms(
        self,
        dummy_data: Any = None,
    ) -> Any:
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
    dataset = CustomDataset(stage="train")
    dataset.getitem(dummy_data=None)

    # you cannot change transforms from ImageClassification to ImageSegmentation
    # without changing the code in CustomDataset class.
