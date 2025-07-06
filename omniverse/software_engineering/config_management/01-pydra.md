---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Pydantic And Hydra

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/tree/4c886c6735b3164325bf9ebbd360ef42bbd71769/omnixamples/software_engineering/config_management)

```{contents}
```

This post is heavily inspired by
[Pydra - Pydantic and Hydra for configuration management of model training experiments](https://suneeta-mall.github.io/2022/03/15/hydra-pydantic-config-management-for-training-application.html)
which talks about combining Pydantic and Hydra for configuration management.

## Hydra

We won't go into the details of what [Hydra](https://hydra.cc/docs/intro/) is,
as the documentation covers the it very well. We will show a working example on
how to use Hydra for configuration management.

### YAML Driven Configuration

First we need to define the configuration files in `yaml` format.

````{tab} **model.yaml**
```yaml
model_name: "resnet18"
pretrained: True
in_chans: 3
num_classes: ${train.num_classes}
global_pool: "avg"
# You typically want _self_ somewhere after the schema (base_config)
```
````

````{tab} **optimizer.yaml**
```yaml
optimizer_name: "AdamW"
optimizer_params:
    lr: 0.0003  # bs: 32 -> lr = 3e-4
    betas: [0.9, 0.999]
    amsgrad: False
    weight_decay: 0.000001
    eps: 0.00000001
```
````

````{tab} **stores.yaml**
```yaml
project_name: ${train.project_name}
unique_id: ${now:%Y%m%d_%H%M%S} # in sync with hydra output dir
logs_dir: !!python/object/apply:pathlib.PosixPath ["./logs"]
model_artifacts_dir: !!python/object/apply:pathlib.PosixPath ["./artifacts"]
```
````

````{tab} **transform.yaml**
```yaml
image_size: 256
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]
```
````

````{tab} **datamodule.yaml**
```yaml
data_dir: !!python/object/apply:pathlib.PosixPath ["./data"]
batch_size: 32
num_workers: 0
shuffle: true
```
````

````{tab} **config.yaml**
```yaml
defaults:
  - _self_
  - model: base
  - datamodule: base
  - transform: base
  - stores: base
  - optimizer: base

train:
  num_classes: 10
  device: "cpu"
  project_name: "cifar10"
  debug: true
  seed: 1992
  num_epochs: 3

hydra:
  run:
    dir: "${stores.model_artifacts_dir}/${train.project_name}/${stores.unique_id}" # in sync with stores
```
````

Then we define an entrypoint to run, along with a simple `train` function.

````{tab} **main.py**
```python
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
```
````

````{tab} **train.py**
```python
from typing import Any

import torch
import torchvision  # type: ignore[import-untyped]
from rich.pretty import pprint
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore[import-untyped]
from tqdm import tqdm


def train(config: Any) -> None:
    """Run the training pipeline, however, the code below can be further
    modularized into functions for better readability and maintainability."""
    pprint(config)

    torch.manual_seed(config.train.seed)

    transform = transforms.Compose(
        [
            transforms.Resize((config.transform.image_size, config.transform.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.transform.mean, std=config.transform.std),
        ]
    )

    dataset = datasets.CIFAR10(root=config.datamodule.data_dir, train=True, transform=transform, download=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config.datamodule.batch_size,
        num_workers=config.datamodule.num_workers,
        shuffle=config.datamodule.shuffle,
    )

    model = getattr(torchvision.models, config.model.model_name)(pretrained=config.model.pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.model.num_classes)
    model = model.to(config.train.device)

    optimizer = getattr(torch.optim, config.optimizer.optimizer_name)(
        model.parameters(), **config.optimizer.optimizer_params
    )

    for epoch in range(config.train.num_epochs):
        model.train()
        with tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config.train.num_epochs}]", unit="batch") as tepoch:
            for images, labels in tepoch:
                images = images.to(config.train.device)
                labels = labels.to(config.train.device)

                optimizer.zero_grad()
                logits = model(images)
                loss = nn.CrossEntropyLoss()(logits, labels)
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

    model.eval()
    torch.save(model.state_dict(), config.stores.model_artifacts_dir / "model.pth")
```
````

The `@hydra.main` decorator is used to initialize the hydra application. We
specify `config_path` to tell hydra where to look for the base configuration
files. We also specify `config_name` to tell hydra which file is the main
controller of the hierarchy. In this case, it is `config.yaml`. Running the
`main.py` file will create an `artifacts` folder hosting useful information from
hydra. Note the `artifacts` folder is a configurable name, the default is
actually `outputs`. Let's see what we have inside this folder!

Running `tree -a artifacts/` will show the following:

```text
artifacts
└── cifar10
    └── 20240509_155331
        ├── .hydra
        │   ├── config.yaml
        │   ├── hydra.yaml
        │   └── overrides.yaml
        └── main.log

4 directories, 4 files
```

And the content is as follows:

````{tab} **.hydra/config.yaml**
```yaml
train:
    num_classes: 10
    device: cpu
    project_name: cifar10
    debug: true
    seed: 1992
    num_epochs: 3
model:
    model_name: resnet18
    pretrained: true
    in_chans: 3
    num_classes: ${train.num_classes}
    global_pool: avg
datamodule:
    data_dir: !!python/object/apply:pathlib.PosixPath
        - data
    batch_size: 32
    num_workers: 0
    shuffle: true
transform:
    image_size: 256
    mean:
        - 0.485
        - 0.456
        - 0.406
    std:
        - 0.229
        - 0.224
        - 0.225
stores:
    project_name: ${train.project_name}
    unique_id: ${now:%Y%m%d_%H%M%S}
    logs_dir: !!python/object/apply:pathlib.PosixPath
        - logs
    model_artifacts_dir: !!python/object/apply:pathlib.PosixPath
        - artifacts
optimizer:
    optimizer_name: AdamW
    optimizer_params:
        lr: 0.0003
        betas:
            - 0.9
            - 0.999
        amsgrad: false
        weight_decay: 1.0e-06
        eps: 1.0e-08
```
````

````{tab} **.hydra/hydra.yaml**
```yaml
hydra:
  run:
    dir: ${stores.model_artifacts_dir}/${train.project_name}/${stores.unique_id}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
  launcher:
    _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
  sweeper:
    _target_: hydra._internal.core_plugins.basic_sweeper.BasicSweeper
    max_batch_size: null
    params: null
  help:
    app_name: ${hydra.job.name}
    header: '${hydra.help.app_name} is powered by Hydra.

      '
    footer: 'Powered by Hydra (https://hydra.cc)

      Use --hydra-help to view Hydra specific help

      '
    template: '${hydra.help.header}

      == Configuration groups ==

      Compose your configuration from those groups (group=option)


      $APP_CONFIG_GROUPS


      == Config ==

      Override anything in the config (foo.bar=value)


      $CONFIG


      ${hydra.help.footer}

      '
  hydra_help:
    template: 'Hydra (${hydra.runtime.version})

      See https://hydra.cc for more info.


      == Flags ==

      $FLAGS_HELP


      == Configuration groups ==

      Compose your configuration from those groups (For example, append hydra/job_logging=disabled
      to command line)


      $HYDRA_CONFIG_GROUPS


      Use ''--cfg hydra'' to Show the Hydra config.

      '
    hydra_help: ???
  hydra_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][HYDRA] %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
    root:
      level: INFO
      handlers:
      - console
    loggers:
      logging_example:
        level: DEBUG
    disable_existing_loggers: false
  job_logging:
    version: 1
    formatters:
      simple:
        format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
    handlers:
      console:
        class: logging.StreamHandler
        formatter: simple
        stream: ext://sys.stdout
      file:
        class: logging.FileHandler
        formatter: simple
        filename: ${hydra.runtime.output_dir}/${hydra.job.name}.log
    root:
      level: INFO
      handlers:
      - console
      - file
    disable_existing_loggers: false
  env: {}
  mode: RUN
  searchpath: []
  callbacks: {}
  output_subdir: .hydra
  overrides:
    hydra:
    - hydra.mode=RUN
    task: []
  job:
    name: main
    chdir: null
    override_dirname: ''
    id: ???
    num: ???
    config_name: config
    env_set: {}
    env_copy: []
    config:
      override_dirname:
        kv_sep: '='
        item_sep: ','
        exclude_keys: []
  runtime:
    version: 1.3.2
    version_base: '1.3'
    cwd: /Users/gaohn/gaohn/omniverse
    config_sources:
    - path: hydra.conf
      schema: pkg
      provider: hydra
    - path: /Users/gaohn/gaohn/omniverse/omnixamples/software_engineering/config_management/hydra/configs
      schema: file
      provider: main
    - path: ''
      schema: structured
      provider: schema
    output_dir: /Users/gaohn/gaohn/omniverse/artifacts/cifar10/20240509_155331
    choices:
      optimizer: base
      stores: base
      transform: base
      datamodule: base
      model: base
      hydra/env: default
      hydra/callbacks: null
      hydra/job_logging: default
      hydra/hydra_logging: default
      hydra/hydra_help: default
      hydra/help: default
      hydra/sweeper: basic
      hydra/launcher: basic
      hydra/output: default
  verbose: false
```
````

````{tab} **.hydra/overrides.yaml**
```yaml
[]
```
````

````{tab} **main.log**
```text
[2024-05-09 15:53:31,713][__main__][INFO] - Type of config is: <class 'omegaconf.dictconfig.DictConfig'>
[2024-05-09 15:53:31,715][__main__][INFO] - Merged Yaml:
train:
  num_classes: 10
  device: cpu
  project_name: cifar10
  debug: true
  seed: 1992
  num_epochs: 3
model:
  model_name: resnet18
  pretrained: true
  in_chans: 3
  num_classes: ${train.num_classes}
  global_pool: avg
datamodule:
  data_dir: !!python/object/apply:pathlib.PosixPath
  - data
  batch_size: 32
  num_workers: 0
  shuffle: true
transform:
  image_size: 256
  mean:
  - 0.485
  - 0.456
  - 0.406
  std:
  - 0.229
  - 0.224
  - 0.225
stores:
  project_name: ${train.project_name}
  unique_id: ${now:%Y%m%d_%H%M%S}
  logs_dir: !!python/object/apply:pathlib.PosixPath
  - logs
  model_artifacts_dir: !!python/object/apply:pathlib.PosixPath
  - artifacts
optimizer:
  optimizer_name: AdamW
  optimizer_params:
    lr: 0.0003
    betas:
    - 0.9
    - 0.999
    amsgrad: false
    weight_decay: 1.0e-06
    eps: 1.0e-08

[2024-05-09 15:53:31,715][__main__][INFO] - main
```
````

One highlight is we can override this configuration file with command line
arguments. So if you tend to have many config folders, then this can come in
handy.

```bash
python main.py train.num_epochs=5
```

and now in `overrides.yaml` you will see

```yaml
- train.num_epochs=5
```

### Pros

1.  Once you load the configuration, you can access the configuration values
    using the dot (chain) notation. This is because `config` is loaded as
    [OmegaConf](https://github.com/omry/omegaconf)'s `DictConfig` object. They
    inherit from `dict` and `MutableMapping` so you can access the values using
    the dot notation access pattern. This is better than using
    `config["train"]["num_classes"]` because it is more readable and less error
    prone, and also allow you to manipulate the object. One other reason is
    using dictionary is a bit hard to do type checking.

2.  Allow command line overrides. For example, you can override the `model_name`
    from `model` by

    ```bash
    python main.py model.model_name=resnet50
    ```

    and the `model_name` will change during runtime. This is very useful when
    you want to run multiple experiments with different configurations. You
    won't need to create multiple config files for each experiment just to
    change a single/few value(s).

3.  A natural follow-up question is about persistence. If I can easily override
    the configuration, then how do I make sure the configuration is saved
    somewhere? Versioning the configs is just as important as versioning your
    code in Machine Learning. Usually I would dump these config to a registry or
    store in code.

    The highlight here is Hydra also saves the final configuration to an output
    folder. By default, it is stored in
    `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml`. Now you can revert back to
    this exact run by specifying the directory.

    ```bash
    python main.py --config_path outputs/2022-01-12/12-00-00/.hydra/config.yaml
    ```

    to recover your run. Here `YYYY-MM-DD/HH-MM-SS` is like your unique `run_id`
    for each run. You can also change it as follows in `config.yaml`:

    ```yaml title="configs/config.yaml" linenums="1"
    hydra:
    run:
        dir: "${stores.model_artifacts_dir}/${train.project_name}/${stores.unique_id}" # in sync with stores
    ```

4.  One more good to have feature is overriding hydra's own default settings,
    such as the `job` and `run` settings. You can either do it via `config.yaml`
    or manually create a folder called `hydra` and put the specifics inside. See
    [here](https://github.com/suneeta-mall/hydra_pydantic_config_management/tree/master/hydra/conf_custom_hydra/hydra)
    for an example.

5.  Multi-run. Hydra allows you to run multiple experiments in parallel. This is
    very useful when you want to run multiple experiments with different
    configurations. You can specify the number of runs and the configuration to
    use. See
    [here](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
    for more examples.

    ```bash
    python main.py model.model_name=resnet18,resnet50 --multirun
    ```

    This will run two experiments in parallel, one with `resnet18` and the other
    with `resnet50`. We see the true power of it if you want to do
    hyperparameter search, where you want to sweep over multiple values for a
    single parameter.

    Now in the same directory, an `multirun` folder will be created and inside
    it, you will see two runs, one with `resnet18` and the other with
    `resnet50`. Each process is indexed by an integer. For example, `resnet18`
    is indexed by `0` and `resnet50` is indexed by `1`.

    It is also worth noting that since the hydra's creator is from facebook, it
    is very easy to integrate with PyTorch's
    [distributed trainings](https://pytorch.org/tutorials/intermediate/dist_tuto.html).

-   Interpolation: This is an extremely highlighted feature of hydra. For
    example, we have `num_classes` defined under the `train` schema. However,
    this parameter is also used in the `model` schema, defining the number of
    output neurons. It can also be used in the `datamodule` schema (not shown
    here), where we need to know the number of classes to create certain `class`
    mapping. Repeatedly defining the same parameter over different config schema
    is prone to mistake. This is like hardcoding the same value `NUM_CLASSES` in
    different python scripts. Here we use the idea of **polymorphism** to define
    the `num_classes` in the `train` schema and use interpolation to **inject**
    this all around with `${train.num_classes}`.

### Structured Config

[Structured Config](https://hydra.cc/docs/tutorials/structured_config/schema/),
this is what Hydra call when your config is complex enough to warrant object
representations. For example, our `config.yaml` is currently just a Yaml
representation. But under the hood, you can think of it as **composed** of
`model`, `optimizer`, `store` and other schemas.

For each of these objects, you can define its own schema. This is very useful
when you want to validate the configuration.

In hydra, you can define the schema using
[dataclasses](https://docs.python.org/3/library/dataclasses.html).

````{tab} **base.py**
```python
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
```
````

````{tab} **main_structured.py**
```python
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
```
````

More details [here](https://hydra.cc/docs/tutorials/structured_config/schema/).
You can find how groups can be used to indicate inheritance.

The dependency injection here is merely a change of the type of the `config`
argument, from `DictConfig` to `Config`. The rest of the code remains the same.
The command line arguments are still the same.

Having dataclass representation also offers more flexibility from manipulating
the object to type hint. But validation still remains a problem as using
`__post_init__` method is
[not well supported](https://github.com/facebookresearch/hydra/issues/981) when
working with hydra. However, you can decouple the usage of dataclass from
hydras' dependency injection. For example, you can load the config from hydra
and instantiate through the dataclass without the use of `ConfigStore`. This
idea is made better when used together with Pydantic since it offers validation
and serialization. As we shall see later, Pydantic is a better version of
dataclass where it offers more features like pre-post validation, type checking,
constraints and better serialization. Just watch Jason Liu's
[Pydantic Is All You Need](https://www.youtube.com/watch?v=yj-wSRJwrrc) to get a
sense how good pydantic is. In fact, with the rise of LLM, we can see most big
libraries built around it are leveraging pydantic.

### Cons

-   Serializing/Deserializing canonical or complex python objects are not well
    supported. In earlier versions, objects like `pathlib.Path` are not
    supported.

-   Structured config is only limited to `dataclass`. This means that you cannot
    create your own custom abstraction. For example, you cannot create a `Model`
    class without invoking `dataclass` decorator, and still be able to interact
    with hydra.

-   No type checking. This is a big problem. You can define `num_classes` as an
    `int` but user passes in a `str` of `"10"` instead of `10`, but hydra will
    not complain. This means that you have to do type checking yourself (i.e. do
    checks all over the application/business logic code).

-   No validation support. This means that if your global pooling methods
    support only `avg` and `max`, you have to do the validation yourself. This
    is a big problem because you have to do the validation all over the place.

-   Interpolation is really good, but the inability to simple manipulation over
    it caused a lot of
    [complaints](https://github.com/omry/omegaconf/issues/91). Like if you
    defined a learning rate as `lr: 0.001` and you want to multiply it by 10 in
    another config file, you cannot do `10 * ${lr}`.

### Instantiating

You can also
[instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
objects with hydra.

### Composition Order

By default, Hydra 1.1 appends `_self_` to the end of the Defaults List. This
behavior is new in Hydra 1.1 and different from previous Hydra versions. As such
Hydra 1.1 issues a warning if `_self_` is not specified in the primary config,
asking you to add `_self_` and thus indicate the desired composition order. To
address the warning while maintaining the new behavior, append `_self_` to the
end of the Defaults List. Note that in some cases it may instead be desirable to
add `_self_` directly after the schema and before other Defaults List elements.

See
[Composition Order](https://hydra.cc/docs/advanced/defaults_list/#composition-order)
for more information.

## Pydantic

Pydantic is all you need and it solves all the aforementioned problems. We still
leverage yaml based configuration with easy command line overrides, but this
time, we also pass the compiled configuration from hydra to pydantic for
validation and serialization during runtime.

### Pydantic Schema

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

### Pros

-   Able to serialize and deserialize objects to and from DICT, JSON, YAML, and
    other formats. For example, the following code will serialize a `Dict`
    object to Pydantics' `Model` object. It can also convert back to `Dict`
    object.

    ```python
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

    model_config_dict = {
        "model_name": "resnet18",
        "pretrained": True,
        "in_chans": 3,
        "num_classes": 1000,
        "global_pool": "avg",
    }
    model = Model(**model_config_dict)
    assert model.model_dump() == model_config_dict
    ```

-   Validation of data types and values. For a large and complex configuration,
    you either validate the sanity of config at the config level, or check at
    the code level (i.e. sprinkled throughout your codebase). -
    [Constrained types](https://pydantic-docs.helpmanual.io/usage/types/#constrained-types)

    ```python
    model = Model(
        model_name="resnet18",
        pretrained=True,
        in_chans=0,
        num_classes=2,
        global_pool="avg",
    )
    ```

    This will raise an error because `in_chans` is less than 1. Pydantic offers
    a wide range of constrained types out of the box for you to use. If that is
    not enough, then the custom validators can be used to validate the data with
    custom needs.

-   [Custom Validators](https://pydantic-docs.helpmanual.io/usage/validators/#custom-validators)

    ```python
    model = Model(
        model_name="resnet18",
        pretrained=True,
        in_chans=3,
        num_classes=2,
        global_pool="average",
    )
    ```

    This will raise an error because `global_pool` is not `avg` or `max`. We
    implemented this custom checks in the `validate_global_pool` method where we
    decorated it with `@field_validator("global_pool")`.

There are many other good things like in-built type checking, and coercion. In
the next section we see how we combine Hydra and Pydantic together.

## Pydra

The provided code shows a way to merge Hydra and Pydantic in a machine learning
training pipeline, using Hydra for hierarchical configuration and command-line
interface, and Pydantic for data validation and type checking.

A Hydra-based application entry point is created using the `@hydra.main()`
decorator. This will use the Hydra library to manage configuration files and
command-line arguments. Hydra's `config_path` and `config_name` are specified to
tell Hydra where to find the configuration files.

This `hydra_to_pydantic` will take in a `hydra`'s `DictConfig` and convert it to
a pydantic's `Config` object.

```python
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
```

## References and Further Readings

-   [Hydra](https://hydra.cc/)
-   [Pydantic](https://pydantic-docs.helpmanual.io/)
-   [Pydra - Pydantic and Hydra for configuration management of model training experiments](https://suneeta-mall.github.io/2022/03/15/hydra-pydantic-config-management-for-training-application.html)
-   [hydra_pydantic_config_management](https://github.com/suneeta-mall/hydra_pydantic_config_management/tree/master)
-   [PyTorch Lightning Pipeline](https://github.com/gao-hongnan/pytorch-lightning-pipeline/tree/main)
-   [Inversion of Control Containers and the Dependency Injection pattern](https://martinfowler.com/articles/injection.html)
