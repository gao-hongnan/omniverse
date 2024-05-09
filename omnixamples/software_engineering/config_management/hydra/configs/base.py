from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TransformConfig:
    image_size: int
    mean: List[float]
    std: List[float]


@dataclass
class ModelConfig:
    model_name: str
    pretrained: bool
    in_chans: int = field(metadata={"ge": 1})  # in_channels must be greater than or equal to 1
    num_classes: int = field(metadata={"ge": 1})  # num_classes must be greater than or equal to 1
    global_pool: str


@dataclass
class StoresConfig:
    project_name: str
    unique_id: str
    logs_dir: Path
    model_artifacts_dir: Path


@dataclass
class TrainConfig:
    device: str
    project_name: str
    debug: bool
    seed: int
    num_epochs: int
    num_classes: int = 3


@dataclass
class OptimizerConfig:
    optimizer: str
    optimizer_params: Dict[str, Any]


@dataclass
class DataConfig:
    data_dir: Path
    batch_size: int
    num_workers: int
    shuffle: bool = True


@dataclass
class Config:
    model: ModelConfig
    augmentations: TransformConfig
    datamodule: DataConfig
    optimizer: OptimizerConfig
    stores: StoresConfig
    train: TrainConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        return cls(
            model=ModelConfig(**config_dict["model"]),
            augmentations=TransformConfig(**config_dict["augmentations"]),
            datamodule=DataConfig(**config_dict["datamodule"]),
            optimizer=OptimizerConfig(**config_dict["optimizer"]),
            stores=StoresConfig(**config_dict["stores"]),
            train=TrainConfig(**config_dict["train"]),
        )
