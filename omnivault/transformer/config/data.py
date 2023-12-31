from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """The data config."""

    context_length: int = Field(default=128, description="The context length.")
    dataset_size: int = Field(default=2, description="The size of the dataset.")
    dataset_path: Union[str, None] = Field(default=None, description="The path to the dataset.")

    split: Union[List[float], None] = Field(default=[0.7, 0.1, 0.2], description="The split ratio of the dataset.")

    collate_fn: Dict[str, Any] = Field(
        default={
            "batch_first": True,
            "pad_token_id": 16,
        },  # TODO: `pad_token_id` should be interpolated from `MaybeConstant`.
        description="The collate function config.",
    )

    train_loader: Dict[str, Any] = Field(
        default={"batch_size": 32, "shuffle": True, "num_workers": 0, "pin_memory": False, "drop_last": False},
        description="The train loader config.",
    )
    val_loader: Dict[str, Any] = Field(
        default={"batch_size": 32, "shuffle": False, "num_workers": 0, "pin_memory": False, "drop_last": False},
        description="The validation loader config.",
    )
    test_loader: Dict[str, Any] = Field(
        default={"batch_size": 32, "shuffle": False, "num_workers": 0, "pin_memory": False, "drop_last": False},
        description="The test loader config.",
    )
