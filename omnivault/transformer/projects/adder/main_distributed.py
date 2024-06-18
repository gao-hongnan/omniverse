# mypy: ignore-errors
from __future__ import annotations

import copy
import os
import sys
import time
import warnings

import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from omnivault.distributed.core import find_free_port, is_free_port
from omnivault.transformer.config.composer import Composer, DataConfig
from omnivault.transformer.config.constants import MaybeConstant
from omnivault.transformer.config.criterion import CRITERION_REGISTRY
from omnivault.transformer.config.decoder import DecoderConfig
from omnivault.transformer.config.distributed import DistributedConfig
from omnivault.transformer.config.generator import GeneratorConfig
from omnivault.transformer.config.global_ import MaybeGlobal
from omnivault.transformer.config.logger import LoggerConfig
from omnivault.transformer.config.optim import OPTIMIZER_REGISTRY
from omnivault.transformer.config.scheduler import SCHEDULER_REGISTRY, LambdaLRConfig
from omnivault.transformer.config.trainer import TrainerConfig
from omnivault.transformer.core.callbacks import CallbackPriority, set_dataloader_epoch_for_ddp_on_epoch_start
from omnivault.transformer.core.dataset import AdderDataset, create_loader, split_dataset
from omnivault.transformer.core.optim import apply_weight_decay_to_different_param_groups
from omnivault.transformer.core.scheduler import noam_lr_decay
from omnivault.transformer.core.state import State
from omnivault.transformer.core.tokenizer import AdderTokenizer
from omnivault.transformer.core.trainer_distributed import Trainer, TrainerEvent
from omnivault.transformer.core.vocabulary import AdderVocabulary
from omnivault.transformer.decoder.core import GPTDecoder
from omnivault.transformer.projects.adder.main import evaluate_and_generate_on_valid_epoch_end
from omnivault.transformer.utils.general_utils import create_directory, download_file, validate_and_cleanup
from omnivault.transformer.utils.visualization import save_plot_history
from omnivault.utils.config_management.omegaconf import load_yaml_config, merge_configs
from omnivault.utils.reproducibility.seed import seed_all
from omnixamples.distributed.a_basic.a_setup import init_process

warnings.filterwarnings("ignore", category=UserWarning)  # usually related to deterministic behavior of pytorch


# NOTE: we are not using composer.trainer.device and in favor of device created
# inside because we are setting device to local ranks.
def main(local_rank: int, cfg: DictConfig | ListConfig) -> None:
    """Main driver."""
    distributed_config = DistributedConfig(**cfg.distributed)
    logger, dist_info_per_process = init_process(local_rank, args=distributed_config)
    logger.info(f"{dist_info_per_process.model_dump_json(indent=4)}")
    is_distributed = dist_info_per_process.world_size > 1  # guardrail to handle if not ddp mode

    # NOTE: seed offset is needed since we want a different seed in each process
    seed_all(cfg.global_.seed + dist_info_per_process.global_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device=device)

    constants = MaybeConstant(**cfg.constants) if cfg.constants is not None else MaybeConstant()
    logger_pydantic_config = LoggerConfig(**cfg.logger)
    global_ = MaybeGlobal(**cfg.global_)
    data = DataConfig(**cfg.data)
    trainer_config = TrainerConfig(**cfg.trainer)
    generator_config = GeneratorConfig(**cfg.generator)

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
        distributed=distributed_config,
    )

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

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=dist_info_per_process.world_size) if is_distributed else None
    )
    # NOTE: important that trainset need wrap with DistributedSampler but not
    # validation or test set. Why? Cause train set need to be uniquely distributed
    # as subsets to the different processes, but for valid and test set we want
    # all process to evaluate on one single set.

    # NOTE: DistributedSampler will do shuffle, so we set `shuffle=False`
    data.train_loader["sampler"] = train_sampler
    data.train_loader["shuffle"] = train_sampler is None  # Need false for ddp

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
    model = model.to(device, dtype=next(model.parameters()).dtype, non_blocking=True)
    if is_distributed:
        if device.type == "cuda":
            model = DistributedDataParallel(module=model, device_ids=[dist_info_per_process.local_rank])
        else:
            model = DistributedDataParallel(module=model)

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

    # train
    trainer = Trainer(
        state=state,
        composer=composer,
        logger=logger,
        device=device,  # type: ignore[arg-type]
    )
    trainer.add_callback(
        TrainerEvent.ON_VALID_EPOCH_END,
        lambda trainer: evaluate_and_generate_on_valid_epoch_end(trainer, num_batches_to_eval=None),
    )
    trainer.add_callback(
        TrainerEvent.ON_TRAIN_EPOCH_START,
        set_dataloader_epoch_for_ddp_on_epoch_start,
        priority=CallbackPriority.HIGHEST,
    )
    _trained_state = trainer.fit(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader)
    _trained_state.pretty_print()
    history = _trained_state.history
    _ = save_plot_history(history, plot=False, save_path=f"{composer.trainer.save_dir}/history.png")

    loaded_state = State.load_snapshots(
        filepath=trainer.best_checkpoint_path,
        device=device,  # type: ignore[arg-type]
        model=copy.deepcopy(model),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    validate_and_cleanup(
        state_1=_trained_state,
        state_2=loaded_state,
        objects=["model", "criterion", "optimizer", "scheduler", "tokenizer", "vocabulary", "trainer"],
        logger=None,
    )

    if is_distributed:
        destroy_process_group()


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    args_list = sys.argv[2:]

    yaml_cfg = load_yaml_config(yaml_path)
    cfg = merge_configs(yaml_cfg, args_list)
    om.resolve(cfg)  # inplace ops

    master_addr, master_port = cfg.distributed.master_addr, cfg.distributed.master_port
    if not is_free_port(int(master_port)):
        master_port = find_free_port()

    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    mp.spawn(
        main,
        # see args=(fn, i, args, error_queue) in start_processes
        # where i is the local rank which is derived from nprocs
        args=(cfg,),
        nprocs=cfg.distributed.nproc_per_node,
        join=True,
        daemon=False,
        start_method="spawn",
    )
