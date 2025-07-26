---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Configuration Management

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)

```{contents}
```

With a well-structured and robust configuration system, you can centralize the
settings and hyperparameters that govern your application. This allows you to
avoid hard-coding values into your code, thereby improving its flexibility.
Furthermore, it allows you to perform changes and updates more easily, without
requiring modifications to the code itself. It is especially beneficial for
machine learning applications, where experiments often involve tuning a
multitude of hyperparameters.

## Composition Of Configurations

What does it mean to compose configurations? Let's have a look at an example:

```python
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
```

We call the above a **composition of sub-configs** where a well-organized config
should allow you to modularize and re-use components across different settings,
tasks, or stages of your system. You can have multiple sub-configs each
describing a part of your system, and a master config that includes these
sub-configs. Each sub-config can further be composed of even finer grains of
sub-configs. This not only promotes reusability, but also enhances readability
and maintainability of your configuration files. The only overhead is that you
need to define the structure of your config classes, but this is a one-time
effort that pays off in the long run. And for an initial poc, I don't suggest
you to nest too deep.

## Service Locator And Dependency Injection

In this configuration design pattern, we can see the influence of two other
underlying design patterns:

-   **Service Locator:** A pattern where an object (the service locator) knows
    how to get hold of all the services that an application might need. So
    instead of dependencies being pushed into a class, the class itself can pull
    in what it needs.

-   **Dependency Injection:** A pattern where dependencies are "injected" or
    passed into the dependent object. This can be done through various ways such
    as Constructor Injection (dependencies are provided through the class
    constructor), Method Injection (dependencies are provided via methods), or
    Property Injection (dependencies are set through properties).

Both of these patterns aim to separate the creation of objects from the behavior
of the system that uses them, thus promoting a cleaner and more modular
structure.

In fact, configuration management can serve as a mechanism for both service
location and dependency injection. Let's revisit the definition of these two
patterns but in the context of configuration management:

-   **Service Locator:** The configuration can be seen as a service locator
    because it holds the information needed to instantiate and retrieve various
    services or components throughout the application. For instance, if you have
    a database service configured in your application, the configuration file
    holds the necessary parameters (like connection string, credentials etc.)
    for your application to locate and connect to this service.

-   **Dependency Injection:** At the same time, your configuration management
    process can also enable dependency injection. The parameters from your
    configuration can be injected into the components that need them. For
    example, when a specific component of your application is created, the
    necessary parameters are injected from the configuration, instead of the
    component needing to fetch or know how to fetch those parameters.

By allowing the configuration to flow through your application, you're providing
a mechanism for each component to get its dependencies (dependency injection),
and you're also providing a way for components to locate the services they need
(service locator). This leads to a more decoupled and maintainable codebase.

Let's see a pseudo code example:

```python
State = Any

class Pipeline:
    def __init__(config: Config, state: State) -> None:
        self.config = config
        self.state = state

    def run_trainer(self) -> None:
        trainer.fit(
            self.config.trainer,
            self.config.dataloader,
            self.config.optimizer,
            self.config.scheduler,
            self.config.callbacks,
            state=self.state,
        )

    def run(self) -> None:
        self.run_trainer()
```

## Config For Different Stages/Evironments

Some people distinguish between different types of configurations based on the
stage or environment in which they are used. For example, you might have
different configurations for development, testing, and production environments.

However, one should know that production code is more or less a replica of the
development code, with exceptions like the database connection string, logging,
or environment variables. Often the config distinction is made at the deployment
level rather than at the code level (i.e. a different config file to deploy to
different environments).

## References And Further Readings

-   [PyTorch Lightning Pipeline](https://github.com/gao-hongnan/pytorch-lightning-pipeline/tree/main)
-   [Inversion of Control Containers and the Dependency Injection pattern](https://martinfowler.com/articles/injection.html)
