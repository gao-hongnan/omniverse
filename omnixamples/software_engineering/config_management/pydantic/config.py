from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated


class TransformConfig(BaseModel):
    image_size: int
    mean: List[float]
    std: List[float]


class ModelConfig(BaseModel):
    model_name: str
    pretrained: bool
    in_chans: Annotated[int, Field(strict=True, ge=1)]  # in_channels must be greater than or equal to 1
    num_classes: Annotated[int, Field(strict=True, ge=1)]  # num_classes must be greater than or equal to 1
    global_pool: str

    @field_validator("global_pool")
    @classmethod
    def validate_global_pool(cls: Type[ModelConfig], global_pool: str) -> str:
        """Validates global_pool is in ["avg", "max"]."""
        if global_pool not in ["avg", "max"]:
            raise ValueError("global_pool must be avg or max")
        return global_pool

    class Config:
        protected_namespaces = ()


class StoresConfig(BaseModel):
    project_name: str
    unique_id: str
    logs_dir: Path
    model_artifacts_dir: Path

    class Config:
        protected_namespaces = ()


class TrainConfig(BaseModel):
    device: str
    project_name: str
    debug: bool
    seed: int
    num_epochs: int
    num_classes: int = 3


class OptimizerConfig(BaseModel):
    optimizer_name: str
    optimizer_params: Dict[str, Any]


class DataConfig(BaseModel):
    data_dir: Path
    batch_size: int
    num_workers: int
    shuffle: bool = True


class Config(BaseModel):
    model: ModelConfig
    transform: TransformConfig
    datamodule: DataConfig
    optimizer: OptimizerConfig
    stores: StoresConfig
    train: TrainConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        """Creates Config object from a dictionary."""
        return cls(**config_dict)
