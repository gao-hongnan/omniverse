import logging

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from omnixamples.software_engineering.config_management.hydra.configs.base import (
    Config,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    StoresConfig,
    TrainConfig,
    TransformConfig,
)
from omnixamples.software_engineering.config_management.train import train

LOGGER = logging.getLogger(__name__)
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(name="model", node=ModelConfig)
cs.store(name="optimizer", node=OptimizerConfig)
cs.store(name="stores", node=StoresConfig)
cs.store(name="train", node=TrainConfig)
cs.store(name="transform", node=TransformConfig)
cs.store(name="datamodule", node=DataConfig)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(config: Config) -> None:
    """Run the main function."""
    LOGGER.info("Type of config is: %s", type(config))
    LOGGER.info("Merged Yaml:\n%s", OmegaConf.to_yaml(config))
    LOGGER.info(HydraConfig.get().job.name)

    config_obj = OmegaConf.to_object(config)
    LOGGER.info("Type of config is: %s", type(config_obj))
    train(config)


if __name__ == "__main__":
    run()
