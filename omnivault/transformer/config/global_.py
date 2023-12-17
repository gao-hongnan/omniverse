"""Dump all global configs here or whatever is not decided here."""

from pydantic import BaseModel, Field

# from omnivault.transformer.utils.device import get_device
# import torch


class MaybeGlobal(BaseModel):
    seed: int = Field(default=42, description="The seed for reproducibility.")
    debug: bool = Field(default=False, description="Debug mode.")


# MAX_SEED = 2**32 - 1  # seeds must be 32-bit unsigned integers
