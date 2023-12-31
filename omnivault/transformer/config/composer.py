from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field

from omnivault._types._alias import Missing
from omnivault._types._sentinel import MISSING
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.data import DataConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.optim import OptimizerConfig


class Composer(BaseModel):  # TODO: add generic subclassing - see if got time lols
    constants: MaybeConstant = Field(default_factory=MaybeConstant)
    global_: MaybeGlobal = Field(default_factory=MaybeGlobal)
    data: DataConfig = Field(default_factory=DataConfig)
    optimizer: Union[OptimizerConfig, Missing] = Field(default=MISSING, description="The optimizer config.")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
