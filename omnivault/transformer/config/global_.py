"""Dump all global configs here or whatever is not decided here."""

from pydantic import BaseModel, Field


class MaybeGlobal(BaseModel):
    seed: int = Field(default=42, description="The seed for reproducibility.")
