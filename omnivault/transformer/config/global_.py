"""Dump all global configs here or whatever is not decided here."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class MaybeGlobal(BaseModel):
    seed: int = Field(default=42, description="The seed for reproducibility.")
    dataset_size: int = Field(..., description="The size of the dataset.")

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
