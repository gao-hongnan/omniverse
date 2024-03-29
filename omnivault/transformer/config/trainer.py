from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Type, Union

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from omnivault.transformer.utils.general_utils import PYTORCH_DTYPE_MAP
from omnivault.utils.torch_utils.device import get_device

__all__ = ["TrainerConfig"]


class TrainerConfig(BaseModel):
    device: torch.device = Field(default_factory=get_device, description="Device to use for training.")

    # general
    max_epochs: int = Field(default=2, description="Number of epochs to train for.")
    log_every_n_steps: int = Field(default=1, description="Log every n steps.")
    eval_every_n_steps: int = Field(default=1, description="Number of epochs between evaluations.")
    step_scheduler_on_batch_or_epoch: str = Field(
        default="epoch",
        description="Whether to step the scheduler on batch or epoch. "
        "If set to 'epoch', the scheduler will be stepped after each epoch. "
        "If set to 'batch', the scheduler will be stepped after each batch.",
    )

    # mixed precision training
    use_amp: bool = Field(default=False, description="Whether to use automatic mixed precision training.")
    autocast_config: Dict[str, Any] = Field(
        default={"enabled": False, "dtype": None, "cache_enabled": None},
        description="Autocast configuration, for details of the params, see `torch.cuda.amp.autocast`.",
    )
    scaler_config: Dict[str, Any] = Field(
        default={
            "enabled": False,
            "init_scale": 2.0**16,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
        description="Grad scaler configuration, for details of the params, see `torch.cuda.amp.GradScaler`. If not enabled, it is a no ops.",
    )

    # gradient accumulation
    gradient_accumulation_steps: int = Field(
        default=1, description="Number of gradient accumulation steps before performing a backward/update pass."
    )

    # training stability
    # 1. gradient clipping
    clip_grad_norm: Union[Dict[str, Any], None] = Field(
        default={"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": False, "foreach": None},
        description="Gradient clipping, for details of the params, see `torch.nn.utils.clip_grad_norm_`.",
    )

    # 2. weight decay on targetted parameter groups
    apply_weight_decay_to_different_param_groups: bool = Field(
        default=False, description="Whether to apply weight decay to different parameter groups."
    )

    # saving shenanigans
    save_dir: str = Field(default="checkpoints", description="Directory to save checkpoints to.")
    save_every_epoch: bool = Field(default=False, description="Always save the model after each epoch.")
    save_best_only: bool = Field(default=True, description="Only save the best model.")
    monitor: str = Field(
        default="valid_this_epoch_average_loss",
        description="The metric to monitor for saving best model.",
        examples=[
            "valid_this_epoch_average_loss",
            "valid_this_batch_average_loss",
            "train_this_epoch_average_loss",
            "train_this_batch_average_loss",
            "train_this_epoch_average_accuracy",
            "train_this_batch_average_accuracy",
            "valid_this_epoch_average_accuracy",
            "valid_this_batch_average_accuracy",
        ],
    )
    mode: str = Field(default="min", description="The mode to monitor for saving best model.", examples=["min", "max"])

    @field_validator("device", mode="before")
    @classmethod
    def set_device(cls: Type[TrainerConfig], v: str) -> torch.device:
        if v == "auto":
            return get_device()
        return torch.device(v)

    @field_validator("save_dir")
    @classmethod
    def set_and_create_timestamped_save_dir(cls: Type[TrainerConfig], v: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        v = f"{v}/{timestamp}"
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_autocast_config(self) -> TrainerConfig:
        use_amp = self.use_amp
        autocast_config = self.autocast_config
        if use_amp and not autocast_config["enabled"]:
            raise ValueError("If use_amp is True, autocast_config must be enabled.")

        for key, value in autocast_config.items():
            if key == "dtype" and value in PYTORCH_DTYPE_MAP:
                autocast_config[key] = PYTORCH_DTYPE_MAP[value]
        return self

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
