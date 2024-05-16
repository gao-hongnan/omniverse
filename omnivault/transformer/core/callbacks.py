from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from omnivault.transformer.core.trainer import Trainer

from enum import IntEnum

from omnivault.transformer.utils.format import format_lr
from omnivault.utils.reproducibility.rng import save_rng_state


class CallbackPriority(IntEnum):
    HIGHEST = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    LOWEST = 5


def update_state(trainer: Trainer) -> None:
    """Update the state of the trainer."""
    relevant_attrs = [
        "model",
        "criterion",
        "optimizer",
        "scheduler",
        "epoch_index",
        "train_batch_index",
        "step_index",
        "history",
        "tokens_per_iter",
    ]
    trainer.state.__dict__.update({attr: getattr(trainer, attr) for attr in relevant_attrs})


def save_state(trainer: Trainer) -> None:
    """Save the state of the trainer."""
    # Initialize best_monitored_score if it's not set
    if trainer.best_monitored_score is None:
        if trainer.mode == "min":
            trainer.best_monitored_score = float("inf")
        else:  # mode == "max"
            trainer.best_monitored_score = -float("inf")

    # Get the current metric
    current_monitored_score = trainer.metrics_dict[trainer.monitor]

    # Determine whether the current model is the best
    is_best = False
    if trainer.mode == "min":
        is_best = current_monitored_score < trainer.best_monitored_score
    elif trainer.mode == "max":
        is_best = current_monitored_score > trainer.best_monitored_score

    # Decide whether to save the model
    should_save = (trainer.save_best_only and is_best) or (not trainer.save_best_only and trainer.save_every_epoch)

    if should_save:
        # Update the best metric if the current model is the best
        if is_best:
            trainer.best_monitored_score = current_monitored_score

        save_path = f"{trainer.save_dir}/model_checkpoint_epoch_{trainer.epoch_index}.pt"
        trainer.best_checkpoint_path = save_path

        trainer.state.save_snapshots(filepath=save_path)
        trainer.logger.info("Saved checkpoint at epoch %s to %s", trainer.epoch_index, save_path)

    # TODO: put this in `State` class for modularity
    # NOTE: I save this every epoch because it's cheap and useful for resuming
    save_rng_state(save_dir=trainer.save_dir, epoch_index=trainer.epoch_index)


def log_on_fit_start(trainer: Trainer) -> None:
    # TODO: add torchinfo's summary
    model_or_module = trainer.model_or_module
    total_params = model_or_module.total_parameters
    trainable_params = model_or_module.total_trainable_parameters

    vocab_size = trainer.composer.model.vocab_size  # type: ignore[union-attr]
    context_length = trainer.composer.model.context_length  # type: ignore[union-attr]
    device = trainer.device

    tokens_per_iter = trainer.tokens_per_iter
    total_tokens = tokens_per_iter * trainer.max_epochs * len(trainer.composer.data.train_loader)

    initial_lr_or_lrs = trainer._get_current_lr_or_lrs()
    lr_str = format_lr(initial_lr_or_lrs, precision=9)

    labels = [
        "Total Parameters:",
        "Trainable Parameters:",
        "Vocabulary Size:",
        "Context Length:",
        "Device:",
        "Tokens per Iteration:",
        "Total Tokens Consumed In Training:",
        "Initial Learning Rate(s):",
    ]
    max_width = max(len(label) for label in labels) + 1  # +1 for the space after the label

    trainer.logger.info(f"%-{max_width}s %d", "Total Parameters:", total_params)
    trainer.logger.info(f"%-{max_width}s %d", "Trainable Parameters:", trainable_params)
    trainer.logger.info(f"%-{max_width}s %d", "Vocabulary Size:", vocab_size)
    trainer.logger.info(f"%-{max_width}s %d", "Context Length:", context_length)
    trainer.logger.info(f"%-{max_width}s %s", "Device:", str(device))
    trainer.logger.info(f"%-{max_width}s %d", "Tokens per Iteration:", tokens_per_iter)
    trainer.logger.info(f"%-{max_width}s %d", "Total Tokens Consumed In Training:", total_tokens)

    if isinstance(initial_lr_or_lrs, list):
        trainer.logger.info(f"%-{max_width}s %s", "Initial Learning Rate(s):", lr_str)
    else:
        trainer.logger.info(f"%-{max_width}s %.9f", "Initial Learning Rate:", initial_lr_or_lrs)

    trainer.logger.info("\n")


def log_on_train_epoch_start(trainer: Trainer, phase: Literal["train", "valid", "test"]) -> None:
    phase_capitalized = phase.capitalize()
    trainer.logger.info(
        "====================================================== Starting %s Epoch: %d/%d ======================================================",
        phase_capitalized,
        trainer.epoch_index,
        trainer.max_epochs,
    )
    max_width = 32  # hardcoded value from `log_on_epoch_end`
    if phase == "train":
        initial_lr_or_lrs = trainer._get_current_lr_or_lrs()
        lr_str = format_lr(initial_lr_or_lrs, precision=20)

        if isinstance(initial_lr_or_lrs, list):
            trainer.logger.info(f"%-{max_width}s %s", "Learning rates for each parameter group:", lr_str)
        else:
            trainer.logger.info(f"%-{max_width}s %s", "Learning rate:", lr_str)


# TODO: add `phase` so can support valid and test
def log_every_n_steps_on_batch_end(trainer: Trainer) -> None:
    if trainer.step_index % trainer.log_every_n_steps == 0:
        lr_info = f"LR: {trainer.scheduler.get_last_lr()[0]:.9f}" if trainer.scheduler else "LR: N/A"
        train_this_batch_average_loss = trainer.metrics_dict["train_this_batch_average_loss"]
        train_this_batch_average_perplexity = trainer.metrics_dict["train_this_batch_average_perplexity"]
        trainer.logger.info(
            "Epoch: %d, Step: %d, Avg Batch Loss: %.5f, Avg Batch Perplexity: %.5f, %s",
            trainer.epoch_index,
            trainer.step_index,
            train_this_batch_average_loss,
            train_this_batch_average_perplexity,
            lr_info,
        )


def log_on_epoch_end(trainer: Trainer, phase: Literal["train", "valid", "test"]) -> None:
    dataloader = getattr(trainer, f"{phase}_loader")
    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    average_loss = trainer.metrics_dict[f"{phase}_this_epoch_average_loss"]
    average_perplexity = trainer.metrics_dict[f"{phase}_this_epoch_average_perplexity"]

    phase_capitalized = phase.capitalize()

    labels = [
        "Total Samples:",
        "Total Batches:",
        f"Average Epoch {phase_capitalized} Loss:",
        f"Average Epoch {phase_capitalized} Perplexity:",
    ]
    max_width = max(len(label) for label in labels) + 1  # +1 for the space after the label
    trainer.logger.info(f"%-{max_width}s %d", "Total Samples:", total_samples)
    trainer.logger.info(f"%-{max_width}s %d", "Total Batches:", total_batches)
    trainer.logger.info(f"%-{max_width}s %.5f", f"Average Epoch {phase_capitalized} Loss:", average_loss)
    trainer.logger.info(f"%-{max_width}s %.5f", f"Average Epoch {phase_capitalized} Perplexity:", average_perplexity)
    trainer.logger.info("\n")


def set_dataloader_epoch_for_ddp_on_epoch_start(trainer: Trainer, phase: Literal["train", "valid", "test"]) -> None:
    """Call :meth:`DistributedSampler.set_epoch` before each epoch. See
    core pytorch utils."""

    if phase == "train":
        data_loader = trainer.train_loader
    elif phase == "valid":
        assert trainer.valid_loader is not None
        data_loader = trainer.valid_loader
    else:
        assert trainer.test_loader is not None
        data_loader = trainer.test_loader

    if hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(trainer.epoch_index)
