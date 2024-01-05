from __future__ import annotations

from typing import Type

import torch
from pydantic import BaseModel, Field, field_validator

from omnivault.transformer.utils.device import get_device

__all__ = ["TrainerConfig"]


class TrainerConfig(BaseModel):
    device: str = Field(default="auto", description="Device to use for training.")
    apply_weight_decay_to_different_param_groups: bool = Field(
        default=False, description="Whether to apply weight decay to different parameter groups."
    )
    num_epochs: int = Field(default=10, description="Number of epochs to train for.")
    eval_interval: int = Field(default=1, description="Number of epochs between evaluations.")

    # saving stuff
    save_dir: str = Field(default="checkpoints", description="Directory to save checkpoints to.")
    save_every_epoch: bool = Field(default=False, description="Always save the model after each epoch.")

    @field_validator("device")
    @classmethod
    def set_device(cls: Type[TrainerConfig], v: str) -> torch.device:
        if v == "auto":
            return get_device()
        return torch.device(v)
