from __future__ import annotations

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

from omnivault._types._alias import Missing
from omnivault._types._sentinel import MISSING
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.optim import OptimizerConfig


class DataConfig(BaseModel):
    """The data config."""

    dataset_size: int = Field(default=2, description="The size of the dataset.")

    split: List[float] = Field(default=[0.7, 0.1, 0.2], description="The split ratio of the dataset.")

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


class Composer(BaseModel):  # TODO: add generic subclassing - see if got time lols
    constants: MaybeConstant = Field(default_factory=MaybeConstant)
    global_: MaybeGlobal = Field(default_factory=MaybeGlobal)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: Union[OptimizerConfig, Missing] = Field(default=MISSING, description="The optimizer config.")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
