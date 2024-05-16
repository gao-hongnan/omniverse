from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field
from rich.pretty import pprint

from omnivault._types._alias import Missing
from omnivault._types._sentinel import MISSING
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.criterion import CriterionConfig
from omnivault.transformer.config.data import DataConfig
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.config.distributed import DistributedConfig
from omnivault.transformer.config.generator import GeneratorConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.logger import LoggerConfig
from omnivault.transformer.config.optim import OptimizerConfig
from omnivault.transformer.config.scheduler import SchedulerConfig
from omnivault.transformer.config.trainer import TrainerConfig


class Composer(BaseModel):  # TODO: add generic subclassing - see if got time lols
    constants: MaybeConstant = Field(default_factory=MaybeConstant)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    global_: MaybeGlobal = Field(default_factory=MaybeGlobal)
    data: DataConfig = Field(default_factory=DataConfig)
    model: Union[DecoderConfig, Missing] = Field(default=MISSING, description="The model config.")
    optimizer: Union[OptimizerConfig, Missing] = Field(default=MISSING, description="The optimizer config.")
    criterion: Union[CriterionConfig, Missing] = Field(default=MISSING, description="The criterion config.")
    scheduler: Union[SchedulerConfig, Missing] = Field(default=MISSING, description="The scheduler config.")
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def pretty_print(self) -> None:
        """Pretty print the config."""
        pprint(self)
