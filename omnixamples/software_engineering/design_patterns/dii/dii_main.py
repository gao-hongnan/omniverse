"""
This module is the composition of all the modules in the project. This module
will depend on both the low-level and high-level modules. This is the final step
of the Dependency Inversion Principle (DIP) implementation + Dependency
Injection.
"""
from typing import List, Union

from rich.pretty import pprint
from src.dii_high_level_business_logic import CustomDataset
from src.dii_low_level_implementations import ImageClassificationTransforms, ImageSegmentationTransforms


# This is the client "code". It is the composition of all the modules in the project.
def run_dataset(dataset: CustomDataset, index: Union[int, slice]) -> Union[str, List[str]]:
    """Run ML pipeline."""
    return dataset[index]


if __name__ == "__main__":
    dummy_data = ["data1", "data2", "data3", "data4", "data5"]

    image_classification_transforms = ImageClassificationTransforms()
    image_classification_dataset = CustomDataset(
        transforms=image_classification_transforms, data=dummy_data, stage="train"
    )
    pprint(run_dataset(image_classification_dataset, index=0))

    # # you can change transforms from ImageClassification to ImageSegmentation
    image_segmentation_transforms = ImageSegmentationTransforms()
    image_segmentation_dataset = CustomDataset(image_segmentation_transforms, data=dummy_data, stage="train")
    pprint(run_dataset(image_segmentation_dataset, index=0))
