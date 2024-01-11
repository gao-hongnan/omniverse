from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnivault.transformer.core.trainer import Trainer


def save_state(trainer: Trainer) -> None:
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
