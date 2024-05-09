import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from omnixamples.software_engineering.config_management.train import train

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(config: DictConfig) -> None:
    """Run the main function."""
    LOGGER.info("Type of config is: %s", type(config))
    LOGGER.info("Merged Yaml:\n%s", OmegaConf.to_yaml(config))
    LOGGER.info(HydraConfig.get().job.name)

    train(config)


if __name__ == "__main__":
    run()
