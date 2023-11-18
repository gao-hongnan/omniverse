"""
This module is the composition of all the modules in the project. This module
will depend on both the low-level and high-level modules. This is the final step
of the Dependency Inversion Principle (DIP) implementation + Dependency
Injection.
"""
from src.dii_high_level_business_logic import CustomDataset
from src.dii_low_level_implementations import (
    ImageClassificationTransforms,
    ImageSegmentationTransforms,
)


# This is the client "code". It is the composition of all the modules in the project.
def run_dataset(dataset: CustomDataset) -> None:
    """Run ML pipeline."""
    dataset.getitem(dummy_data=None)


if __name__ == "__main__":
    image_classification_transforms = ImageClassificationTransforms()
    image_classification_dataset = CustomDataset(
        image_classification_transforms, stage="train"
    )
    run_dataset(image_classification_dataset)

    # # you can change transforms from ImageClassification to ImageSegmentation
    image_segmentation_transforms = ImageSegmentationTransforms()
    image_segmentation_dataset = CustomDataset(
        image_segmentation_transforms, stage="train"
    )
    run_dataset(image_segmentation_dataset)
