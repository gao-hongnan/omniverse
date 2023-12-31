from __future__ import annotations

from typing import Type

import torch
from pydantic import BaseModel, Field, field_validator

from omnivault.transformer.utils.device import get_device

__all__ = ["TrainerConfig"]


class TrainerConfig(BaseModel):
    device: str = Field(default="auto", description="Device to use for training.")

    @field_validator("device")
    @classmethod
    def set_device(cls: Type[TrainerConfig], v: str) -> torch.device:
        if v == "auto":
            return get_device()
        return torch.device(v)
