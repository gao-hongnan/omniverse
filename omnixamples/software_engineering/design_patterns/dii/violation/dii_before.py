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
