from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om

from omnivault._types._alias import Missing
from omnivault._types._sentinel import MISSING
from omnivault.core.logger import RichLogger
from omnivault.transformer.config.composer import Composer, DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.config.generator import GeneratorConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.logger import LoggerConfig
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.scheduler import SCHEDULER_REGISTRY, LambdaLRConfig
from omnivault.transformer.config.trainer import TrainerConfig
from omnivault.transformer.core.dataset import AdderDataset, create_loader, split_dataset
from omnivault.transformer.core.optim import apply_weight_decay_to_different_param_groups
from omnivault.transformer.core.scheduler import noam_lr_decay
from omnivault.transformer.core.state import State
from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.core.trainer import Trainer, TrainerEvent
from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.projects.adder.main import evaluate_and_generate_on_valid_epoch_end
from omnivault.transformer.utils.general_utils import create_directory, download_file
from omnivault.transformer.utils.visualization import save_plot_history
from omnivault.utils.config_management.omegaconf import load_yaml_config, merge_configs
from omnivault.utils.general.paths import find_root_dir
from omnivault.utils.reproducibility.seed import seed_all

warnings.filterwarnings("ignore", category=UserWarning)  # usually related to deterministic behavior of pytorch

# TODO: I have a callable instead of _target_ field for me to use importlib to parse.
# so maybe consider using my own code base


def main(cfg: DictConfig | ListConfig) -> None:
    """Main driver."""
    seed_all(cfg.global_.seed)

    constants = MaybeConstant(**cfg.constants) if cfg.constants is not None else MaybeConstant()
    logger_pydantic_config = LoggerConfig(**cfg.logger)
    global_ = MaybeGlobal(**cfg.global_)
    data = DataConfig(**cfg.data)
    trainer_config = TrainerConfig(**cfg.trainer)
    generator_config = GeneratorConfig(**cfg.generator)

    # logger
    logger = RichLogger(**logger_pydantic_config.model_dump(mode="python")).logger
    assert isinstance(logger, logging.Logger)

    create_directory(data.dataset_dir)
    download_file(url=data.dataset_url, output_path=data.dataset_path)

    vocabulary = AdderVocabulary.from_tokens(tokens=constants.TOKENS, num_digits=constants.NUM_DIGITS)  # type: ignore[attr-defined]
    tokenizer = AdderTokenizer(vocabulary=vocabulary)

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
        logger=logger_pydantic_config,
        global_=global_,
        data=data,
        model=model_pydantic_config,
        optimizer=optimizer_pydantic_config,
        criterion=criterion_pydantic_config,
        trainer=trainer_config,
        generator=generator_config,
    )
    assert composer.model is not MISSING and not isinstance(composer.model, Missing)
    assert composer.optimizer is not MISSING and not isinstance(composer.optimizer, Missing)
    assert composer.criterion is not MISSING and not isinstance(composer.criterion, Missing)

    # TODO: consider classmethod from file_path
    assert composer.data.dataset_path is not None
    with open(composer.data.dataset_path, "r") as file:
        sequences = [line.strip() for line in file]

    dataset = AdderDataset(data=sequences, tokenizer=tokenizer)
    if composer.data.split:
        train_dataset, valid_dataset, test_dataset = split_dataset(
            dataset=dataset, split=composer.data.split, seed=composer.global_.seed
        )
    else:
        # no need to cater to mypy as either Subset or Dataset is fine.
        train_dataset = dataset  # type: ignore[assignment]

    # you do these asserts to make sure that the loaders are not None
    # because create loader expects non-None loaders and collate_fn.
    # if you don't do these asserts, mypy cannot guarantee that the loaders are not None
    # so they cannot infer properly.
    assert composer.data.train_loader is not None
    assert composer.data.valid_loader is not None
    assert composer.data.test_loader is not None
    assert composer.data.collate_fn is not None

    train_loader = create_loader(
        dataset=train_dataset,
        loader_config=composer.data.train_loader,
        collate_fn_config=composer.data.collate_fn,
    )

    if valid_dataset is not None:
        valid_loader = create_loader(
            dataset=valid_dataset,
            loader_config=composer.data.valid_loader,
            collate_fn_config=composer.data.collate_fn,
        )

    if test_dataset is not None:
        test_loader = create_loader(  # noqa: F841
            dataset=test_dataset,
            loader_config=composer.data.test_loader,
            collate_fn_config=composer.data.collate_fn,
        )

    # Create model
    model = GPTDecoder(model_pydantic_config)
    model = model.to(device=composer.trainer.device, dtype=next(model.parameters()).dtype, non_blocking=True)

    # Create optimizer based on model parameters
    if composer.trainer.apply_weight_decay_to_different_param_groups:
        assert hasattr(composer.optimizer, "weight_decay")
        optimizer = optimizer_pydantic_config.build(
            params=apply_weight_decay_to_different_param_groups(
                model=model, weight_decay=composer.optimizer.weight_decay
            )
        )
    else:
        optimizer = optimizer_pydantic_config.build(params=model.parameters())

    # Create criterion
    criterion = criterion_pydantic_config.create_instance()
    assert criterion.ignore_index == vocabulary.token_to_index[vocabulary.PAD]

    # Create Scheduler noam
    # TODO: this part is hardcoded in a way since we are using LambdaLR.
    # I do not have time to make it more "automated" so this is anti-config-pattern.
    warmup_steps = 3 * len(train_loader)

    # lr first increases in the warmup steps, and then decays
    noam = lambda step: noam_lr_decay(step, d_model=composer.model.d_model, warmup_steps=warmup_steps)  # noqa: E731

    scheduler_config_cls = SCHEDULER_REGISTRY[cfg.scheduler.name]

    if issubclass(scheduler_config_cls, LambdaLRConfig):
        scheduler_pydantic_config = scheduler_config_cls(lr_lambda=noam, **cfg.scheduler)
    else:
        scheduler_pydantic_config = scheduler_config_cls(**cfg.scheduler)  # type: ignore[assignment]

    assert composer.scheduler is MISSING  # now it is MISSING for us to fill up.
    composer.scheduler = scheduler_pydantic_config
    scheduler = scheduler_pydantic_config.build(optimizer=optimizer)

    composer.pretty_print()
    time.sleep(1)

    device = composer.trainer.device

    root_dir = find_root_dir(current_path=Path(__file__), marker="omnivault")
    resume_from_state_path = (
        f"{root_dir}/omnivault/transformer/projects/adder/checkpoints/2024-05-16_13-48-32/model_checkpoint_epoch_8.pt"
    )
    resume_from_rng_state_path = (
        f"{root_dir}/omnivault/transformer/projects/adder/checkpoints/2024-05-16_13-48-32/rng_state_epoch_8.pt"
    )

    loaded_state = State.load_snapshots(
        filepath=resume_from_state_path,
        device=device,  # type: ignore[arg-type]
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # train
    trainer = Trainer(
        state=loaded_state,
        composer=composer,
        logger=logger,
        device=device,  # type: ignore[arg-type]
        resume_from_rng_path=resume_from_rng_state_path,
    )
    trainer.add_callback(
        TrainerEvent.ON_VALID_EPOCH_END,
        lambda trainer: evaluate_and_generate_on_valid_epoch_end(trainer, num_batches_to_eval=None),
    )
    _trained_state = trainer.fit(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader)
    _trained_state.pretty_print()

    history = _trained_state.history
    _ = save_plot_history(history, plot=False, save_path=f"{composer.trainer.save_dir}/history.png")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    args_list = sys.argv[2:]

    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)  # inplace ops

    main(cfg)
