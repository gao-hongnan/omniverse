import logging
from typing import Any, Dict

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from omnixamples.software_engineering.config_management.pydantic.config import Config
from omnixamples.software_engineering.config_management.train import train

LOGGER = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> Config:
    """Converts Hydra config to Pydantic config."""
    # use to_container to resolve
    config_dict: Dict[str, Any] = OmegaConf.to_object(config)  # type: ignore[assignment]
    return Config(**config_dict)


@hydra.main(version_base=None, config_path="../hydra/configs", config_name="config")
def run(config: DictConfig) -> None:
    """Run the main function."""
    LOGGER.info("Type of config is: %s", type(config))
    LOGGER.info("Merged Yaml:\n%s", OmegaConf.to_yaml(config))
    LOGGER.info(HydraConfig.get().job.name)

    config_pydantic = hydra_to_pydantic(config)
    pprint(config_pydantic)
    train(config_pydantic)


if __name__ == "__main__":
    run()
