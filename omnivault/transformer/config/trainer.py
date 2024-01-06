from __future__ import annotations

from typing import Any, Dict, Type, Union

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

    # training stability
    # 1. gradient clipping
    clip_grad_norm: Union[Dict[str, Any], None] = Field(
        default={"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False, "foreach": None},
        description="Gradient clipping, for details of the params, see `torch.nn.utils.clip_grad_norm_`.",
    )

    # saving stuff
    save_dir: str = Field(default="checkpoints", description="Directory to save checkpoints to.")
    save_every_epoch: bool = Field(default=False, description="Always save the model after each epoch.")

    @field_validator("device", mode="plain")
    @classmethod
    def set_device(cls: Type[TrainerConfig], v: str) -> torch.device:
        if v == "auto":
            return get_device()
        return torch.device(v)
