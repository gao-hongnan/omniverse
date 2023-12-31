from __future__ import annotations

import sys
import time

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from rich.pretty import pprint

from omnivault._types._sentinel import MISSING
from omnivault.transformer.config.composer import Composer, DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.decoder import (
    AddNormConfig,
    DecoderBlockConfig,
    DecoderConfig,
    MultiHeadedAttentionConfig,
    PositionwiseFeedForwardConfig,
)
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY, AdamConfig, OptimizerConfig
from omnivault.transformer.core.dataset import AdderDataset, collate_fn, create_loader, split_dataset
from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.utils.config_utils import load_yaml_config, merge_configs
from omnivault.transformer.utils.reproducibility import seed_all

# TODO: I have a callable instead of _target_ field for me to use importlib to parse.
# so maybe consider using my own code base?


def main(cfg: DictConfig | ListConfig) -> None:
    """Main driver."""
    seed_all(42)

    constants = MaybeConstant(**cfg.constants)
    global_ = MaybeGlobal(**cfg.global_)
    data = DataConfig(**cfg.data)

    composer = Composer(constants=constants, global_=global_, data=data)
    pprint(composer)

    vocabulary = AdderVocabulary.from_tokens(tokens=constants.TOKENS, num_digits=constants.NUM_DIGITS)  # type: ignore[attr-defined]
    tokenizer = AdderTokenizer(vocabulary=vocabulary)
    # TODO: consider classmethod from file_path
    assert composer.data.dataset_path is not None
    with open(composer.data.dataset_path, "r") as file:
        sequences = [line.strip() for line in file]

    dataset = AdderDataset(data=sequences, tokenizer=tokenizer)
    if composer.data.split:
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset=dataset, split=composer.data.split, seed=composer.global_.seed
        )
    train_loader = create_loader(
        dataset=train_dataset,
        loader_config=composer.data.train_loader,
        collate_fn_config=composer.data.collate_fn,
    )

    val_loader = create_loader(
        dataset=val_dataset,
        loader_config=composer.data.val_loader,
        collate_fn_config=composer.data.collate_fn,
    )

    test_loader = create_loader(
        dataset=test_dataset,
        loader_config=composer.data.test_loader,
        collate_fn_config=composer.data.collate_fn,
    )
    time.sleep(1000)
    optimizer_config_cls = OPTIMIZER_REGISTRY[cfg.optimizer.name]
    optimizer_pydantic_config = optimizer_config_cls(**cfg.optimizer)

    # optimizer_config = OptimizerConfig(**cfg.optimizer)
    # pprint(optimizer_config)
    # optimizer_name = optimizer_config.name
    # optimizer_config_cls = OPTIMIZER_REGISTRY[optimizer_name]
    # optimizer_pydantic_config = optimizer_config_cls(**cfg.optimizer)
    # pprint(optimizer_pydantic_config)

    # Define a model with a non-linear activation function
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    optimizer = optimizer_pydantic_config.build(params=model.parameters())
    pprint(optimizer)
    # train
    time.sleep(1000)

    feed_forward_config = instantiate(cfg.feed_forward)
    pprint(feed_forward_config)
    pprint(type(feed_forward_config.activation))
    pprint(type(feed_forward_config))

    feed_forward_config = PositionwiseFeedForwardConfig(**feed_forward_config)
    pprint(feed_forward_config)
    # feed_forward_config.activation = instantiate(feed_forward_config.activation)
    # pprint(feed_forward_config)

    # attention_config =
    attention = instantiate(cfg.attention)
    pprint(attention)

    composer = Composer(data=data, optimizer=optimizer_pydantic_config)
    pprint(composer)
    if composer.optimizer is MISSING:
        print("optimizer is MISSING")
        # prob raise an error?

    # # Define a simple dataset
    # inputs = torch.randn(100, 2)
    # targets = torch.randn(100, 2)  # Targets should have some relationship to inputs

    # # Train the model
    # for epoch in range(100):
    #     optimizer.zero_grad()
    #     output = model(inputs)
    #     loss = torch.nn.functional.mse_loss(output, targets)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epoch} loss: {loss}")


if __name__ == "__main__":
    # python omnivault/transformer/projects/adder/main.py omnivault/transformer/projects/adder/config.yaml data.train_loader.batch_size=22
    yaml_path = sys.argv[1]
    args_list = sys.argv[2:]

    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)  # inplace ops
    pprint(cfg)

    main(cfg)
