from __future__ import annotations

import logging
from typing import Type

from pydantic import BaseModel, field_validator


class DistributedConfig(BaseModel):
    """Default settings, override if ddp, else stay as is for non-ddp."""

    log_dir: str = "logs_distributed"
    log_level: int = logging.INFO
    log_on_master_or_all: bool = True
    master_addr: str = "localhost"
    master_port: str = "29500"
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    world_size: int = 1
    backend: str = "gloo"
    init_method: str = "env://"

    @field_validator("master_port", mode="plain")
    @classmethod
    def coerce_str_to_int(cls: Type[DistributedConfig], v: str) -> int:
        return int(v)
