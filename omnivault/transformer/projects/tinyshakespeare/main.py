# ruff: noqa
# type: ignore
from __future__ import annotations

import sys
import time

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from rich.pretty import pprint
from torch import nn

from omnivault._types._sentinel import MISSING
from omnivault.transformer.config.composer import Composer, DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.trainer import TrainerConfig
from omnivault.transformer.core.dataset import TextCharacterDataset, collate_fn, create_loader, split_dataset
from omnivault.transformer.core.tokenizer import TextCharacterTokenizer
from omnivault.transformer.core.trainer import Trainer
from omnivault.transformer.core.vocabulary import TextCharacterVocabulary
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.modules.attention.core import ScaledDotProductAttention
from omnivault.transformer.utils.config_utils import load_yaml_config, merge_configs
from omnivault.transformer.utils.reproducibility import seed_all

# TODO: I have a callable instead of _target_ field for me to use importlib to parse.
# so maybe consider using my own code base?


def main(cfg: DictConfig | ListConfig) -> None:
    """Main driver."""
    seed_all(cfg.global_.seed)

    constants = MaybeConstant(**cfg.constants) if cfg.constants is not None else MaybeConstant()
    global_ = MaybeGlobal(**cfg.global_)
    data = DataConfig(**cfg.data)
    trainer_config = TrainerConfig(**cfg.trainer)

    assert data.dataset_url is not None
    vocabulary = TextCharacterVocabulary.from_url(
        url=data.dataset_url, dataset_name=data.dataset_name, dest_folder=data.dataset_dir
    )
    tokenizer = TextCharacterTokenizer(vocabulary=vocabulary)

    # assign back model.vocab_size from ??? to vocabulary.vocab_size
    cfg.model.vocab_size = vocabulary.vocab_size
    model_config = instantiate(cfg.model)
    model_pydantic_config = DecoderConfig(**model_config)

    optimizer_config_cls = OPTIMIZER_REGISTRY[cfg.optimizer.name]
    optimizer_pydantic_config = optimizer_config_cls(**cfg.optimizer)

    criterion_config_cls = CRITERION_REGISTRY[cfg.criterion.name]
    criterion_pydantic_config = criterion_config_cls(**cfg.criterion)

    composer = Composer(
        constants=constants,
        global_=global_,
        data=data,
        model=model_pydantic_config,
        optimizer=optimizer_pydantic_config,
        criterion=criterion_pydantic_config,
        trainer=trainer_config,
    )
    assert composer.model is not MISSING
    assert composer.optimizer is not MISSING
    assert composer.criterion is not MISSING

    pprint(composer)

    # TODO: consider classmethod from file_path
    assert composer.data.dataset_path is not None
    with open(composer.data.dataset_path, "r") as file:
        corpus = file.read()

    dataset = TextCharacterDataset(corpus=corpus, context_length=composer.data.context_length, tokenizer=tokenizer)

    if composer.data.split:
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset=dataset, split=composer.data.split, seed=composer.global_.seed
        )
    else:
        train_dataset = dataset

    # you do these asserts to make sure that the loaders are not None
    # because create loader expects non-None loaders and collate_fn.
    # if you don't do these asserts, mypy cannot guarantee that the loaders are not None
    # so they cannot infer properly.
    assert composer.data.train_loader is not None

    train_loader = create_loader(
        dataset=train_dataset,
        loader_config=composer.data.train_loader,
        collate_fn_config=composer.data.collate_fn,
    )
    for batch in train_loader:
        x, y, padding_masks, future_masks = batch
        pprint(x)
        pprint(y)
        pprint(padding_masks)
        pprint(future_masks)
        time.sleep(1000)
        break

    # Create model
    model = GPTDecoder(model_pydantic_config).to(composer.trainer.device)
    model_size = model.total_trainable_parameters
    print(f"model_size: {model_size}, train_size: {len(train_dataset)}")

    # Create optimizer based on model parameters
    optimizer = optimizer_pydantic_config.build(params=model.parameters())
    pprint(optimizer)

    # Create criterion
    criterion = criterion_pydantic_config.create_instance()
    pprint(vocabulary.token_to_index)

    # Create Scheduler

    warmup_steps = 3 * len(train_loader)

    # lr first increases in the warmup steps, and then decays
    lr_fn = lambda step: composer.model.d_model ** (-0.5) * min(  # type: ignore[union-attr]
        [(step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)]
    )
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    scheduler = None

    # train
    device: torch.device = composer.trainer.device
    pprint(device)
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_norm_clip=1.0,
        device=device,
    )
    trained_model = trainer.fit(composer.trainer.num_epochs)
    torch.save(trained_model.state_dict(), "model_debug.pt")
    time.sleep(1000)


if __name__ == "__main__":
    # python omnivault/transformer/projects/tinyshakespeare/main.py omnivault/transformer/projects/tinyshakespeare/config.yaml
    yaml_path = sys.argv[1]
    args_list = sys.argv[2:]

    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)  # inplace ops
    pprint(cfg)

    main(cfg)
