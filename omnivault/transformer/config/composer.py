from pydantic import BaseModel, Field

from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.global_ import MaybeGlobal


class Composer(BaseModel):  # TODO: add generic subclassing - see if got time lols
    constants: MaybeConstant = Field(default_factory=MaybeConstant)
    global_: MaybeGlobal = Field(default_factory=MaybeGlobal)

    # say
    # optimizer: OptimizerConfig = Field(default_factory=AdamConfig)...
