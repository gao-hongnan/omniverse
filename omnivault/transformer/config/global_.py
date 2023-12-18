"""Dump all global configs here or whatever is not decided here."""
from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field, field_validator

# from omnivault.transformer.utils.device import get_device
# import torch


class MaybeGlobal(BaseModel):
    seed: int = Field(default=42, description="The seed for reproducibility.")
    debug: bool = Field(default=False, description="Debug mode.")

    @field_validator("seed")
    @classmethod
    def seed_non_negative_and_within_32_bit_unsigned_integer(cls: Type[MaybeGlobal], v: int) -> int:
        if not (0 <= v <= 2**32 - 1):
            raise ValueError(f"Seed must be within 0 and {2 ** 32 - 1} inclusive.")
        return v
