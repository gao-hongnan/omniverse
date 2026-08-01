import sys
import time

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from reproducibility.seed import seed_all
from torch.utils.data import Subset

from omnivault._types._alias import Missing
from omnivault._types._sentinel import MISSING
from omnivault.core.logger import RichLogger
from omnivault.transformer.config.composer import Composer, DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.data import DatasetSource
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.config.generator import GeneratorConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.logger import LoggerConfig
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.scheduler import SCHEDULER_REGISTRY
from omnivault.transformer.config.trainer import TrainerConfig
from omnivault.transformer.core.callbacks import save_state
from omnivault.transformer.core.dataset import TextCharacterDataset, create_loader, split_dataset
from omnivault.transformer.core.optim import apply_weight_decay_to_different_param_groups
from omnivault.transformer.core.state import State
from omnivault.transformer.core.tokenizer import TextCharacterTokenizer
from omnivault.transformer.core.trainer import Trainer, TrainerEvent
from omnivault.transformer.core.vocabulary import TextCharacterVocabulary
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.projects.tinyshakespeare_char.callbacks import evaluate_generate_on_train_batch_end
from omnivault.transformer.utils.visualization import save_plot_history
from omnivault.utils.config_management.omegaconf import load_yaml_config, merge_configs


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

    source = DatasetSource.from_data_config(data)
    vocabulary = TextCharacterVocabulary.from_url(
        url=source.dataset_url, dataset_name=source.dataset_name, dest_folder=source.dataset_dir
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
        generator=generator_config,
    )
    if (
        isinstance(composer.model, Missing)
        or isinstance(composer.optimizer, Missing)
        or isinstance(composer.criterion, Missing)
    ):
        raise ValueError("model, optimizer and criterion configs are required")

    # TODO: consider classmethod from file_path
    with open(source.dataset_path, "r") as file:
        corpus = file.read()

    dataset = TextCharacterDataset(corpus=corpus, context_length=composer.data.context_length, tokenizer=tokenizer)

    if composer.global_.debug:
        debug_samples = composer.global_.debug_samples or 256
        dataset = Subset(dataset, indices=range(debug_samples))  # type: ignore[assignment]

    if composer.data.split:
        train_dataset, valid_dataset, test_dataset = split_dataset(
            dataset=dataset, split=composer.data.split, seed=composer.global_.seed
        )
    else:
        train_dataset = dataset  # type: ignore[assignment]
        valid_dataset = None
        test_dataset = None

    if composer.data.train_loader is None or composer.data.collate_fn is None:
        raise ValueError("tinyshakespeare training requires train_loader and collate_fn configs")

    train_loader = create_loader(
        dataset=train_dataset,
        loader_config=composer.data.train_loader,
        collate_fn_config=composer.data.collate_fn,
    )

    if valid_dataset is not None:
        if composer.data.valid_loader is None:
            raise ValueError("valid_loader config is required when a validation split exists")
        valid_loader = create_loader(  # noqa: F841
            dataset=valid_dataset,
            loader_config=composer.data.valid_loader,
            collate_fn_config=composer.data.collate_fn,
        )

    if test_dataset is not None:
        if composer.data.test_loader is None:
            raise ValueError("test_loader config is required when a test split exists")
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
                model=model,
                weight_decay=getattr(composer.optimizer, "weight_decay"),  # noqa: B009  # dynamic OptimizerConfig field not visible to pyright; getattr keeps it sound
            )
        )
    else:
        optimizer = optimizer_pydantic_config.build(params=model.parameters())

    # Create criterion
    criterion = criterion_pydantic_config.create_instance()
    assert criterion.ignore_index == -100

    # Create scheduler
    scheduler_config_cls = SCHEDULER_REGISTRY[cfg.scheduler.name]
    scheduler_pydantic_config = scheduler_config_cls(**cfg.scheduler)
    assert composer.scheduler is MISSING  # now it is MISSING for us to fill up.
    composer.scheduler = scheduler_pydantic_config
    scheduler = scheduler_pydantic_config.build(optimizer=optimizer)

    composer.pretty_print()
    time.sleep(1)

    state = State(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
    )
    state.pretty_print()
    time.sleep(1)

    device = composer.trainer.device

    # train
    trainer = Trainer(
        state=state,
        composer=composer,
        logger=logger,
        device=device,
    )
    trainer.add_callback(TrainerEvent.ON_TRAIN_BATCH_END, evaluate_generate_on_train_batch_end)
    trainer.add_callback(TrainerEvent.ON_TRAIN_EPOCH_END, save_state)

    _trained_state = trainer.fit(train_loader=train_loader)
    history = _trained_state.history
    _ = save_plot_history(history, plot=False, save_path=f"{composer.trainer.save_dir}/history.png")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    args_list = sys.argv[2:]

    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)  # inplace ops

    main(cfg)
