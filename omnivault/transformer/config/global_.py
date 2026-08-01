"""Dump all global configs here or whatever is not decided here."""

from typing import Annotated, Self

from pydantic import BaseModel, Field, field_validator

__all__ = ["MaybeGlobal"]


class MaybeGlobal(BaseModel):
    seed: Annotated[int, Field(description="The seed for reproducibility.")] = 42
    debug: Annotated[bool, Field(description="Debug mode.")] = False
    debug_samples: Annotated[int | None, Field(description="Number of samples to debug.")] = 256

    @field_validator("seed")
    @classmethod
    def seed_non_negative_and_within_32_bit_unsigned_integer(cls: type[Self], v: int) -> int:
        if not (0 <= v <= 2**32 - 1):
            raise ValueError(f"Seed must be within 0 and {2**32 - 1} inclusive.")
        return v
