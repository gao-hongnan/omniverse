from __future__ import annotations

from typing import List

from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

__all__ = ["load_yaml_config", "parse_cli_args", "merge_configs"]


def load_yaml_config(yaml_path: str) -> DictConfig | ListConfig:
    """Load configuration from a YAML file."""
    try:
        with open(yaml_path) as f:
            return om.load(f)
    except OSError as err:
        raise RuntimeError(f"Error reading YAML file: {err}") from err


def parse_cli_args(args: List[str]) -> DictConfig:
    """Parse command line arguments."""
    return om.from_cli(args)


def merge_configs(yaml_cfg: DictConfig | ListConfig, args_list: List[str]) -> DictConfig | ListConfig:
    """Load and merge configurations from YAML file and command-line arguments."""
    cli_cfg = parse_cli_args(args_list)
    return om.merge(yaml_cfg, cli_cfg)
