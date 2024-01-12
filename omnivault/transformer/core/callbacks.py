from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from omnivault.transformer.core.trainer import Trainer

from omnivault.transformer.utils.format import format_lr


def update_state(trainer: Trainer) -> None:
    """Update the state of the trainer."""
    trainer.state.model = trainer.model
    trainer.state.criterion = trainer.criterion
    trainer.state.optimizer = trainer.optimizer
    trainer.state.scheduler = trainer.scheduler
    trainer.state.epoch_index = trainer.epoch_index
    trainer.state.batch_index = trainer.batch_index


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

        trainer.state.save_snapshots(filepath=save_path)
        trainer.logger.info("Saved checkpoint at epoch %s to %s", trainer.epoch_index, save_path)


def log_on_fit_start(trainer: Trainer) -> None:
    initial_lr_or_lrs = trainer._get_current_lr_or_lrs()
    lr_str = format_lr(initial_lr_or_lrs, precision=9)

    if isinstance(initial_lr_or_lrs, list):
        trainer.logger.info("Initial learning rates for each parameter group: %s", lr_str)
    else:
        trainer.logger.info("Initial learning rate: %s", initial_lr_or_lrs)


def log_on_train_epoch_start(trainer: Trainer, phase: Literal["train", "valid", "test"]) -> None:
    phase_capitalized = phase.capitalize()
    trainer.logger.info(
        "\n=== Starting %s Epoch: %d/%d ===", phase_capitalized, trainer.epoch_index, trainer.max_epochs
    )


def log_on_epoch_end(trainer: Trainer, phase: Literal["train", "valid", "test"]) -> None:
    dataloader = getattr(trainer, f"{phase}_dataloader")
    total_batches = len(dataloader)
    total_samples = len(dataloader.dataset)
    average_loss = trainer.metrics_dict[f"{phase}_this_epoch_average_loss"]

    phase_capitalized = phase.capitalize()
    trainer.logger.info("%s - Total Samples: %d, Total Batches: %d", phase_capitalized, total_samples, total_batches)
    trainer.logger.info("Average Epoch %s Loss: %.5f", phase_capitalized, average_loss)


def log_on_fit_start_model_summary(trainer: Trainer) -> None:
    # TODO: add torchinfo's summary
    total_params = trainer.model.total_parameters
    trainable_params = trainer.model.total_trainable_parameters
    trainer.logger.info("Total Parameters: %d, Trainable Parameters: %d", total_params, trainable_params)
