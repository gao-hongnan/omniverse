# mypy: disable-error-code="no-untyped-call"
from __future__ import annotations

import inspect
import logging
import warnings
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Protocol, Tuple, no_type_check, runtime_checkable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
from tqdm import tqdm

from omnivault._types._alias import Loss
from omnivault.transformer.config.composer import Composer
from omnivault.transformer.core.callbacks import (
    log_every_n_steps_on_batch_end,
    log_on_epoch_end,
    log_on_fit_start,
    log_on_fit_start_model_summary,
    log_on_train_epoch_start,
    save_state,
    update_state,
)
from omnivault.transformer.core.dataset import DatasetYield
from omnivault.transformer.core.state import State
from omnivault.transformer.utils.general_utils import get_default_logger


@runtime_checkable
class TrainerCallback(Protocol):
    def __call__(self, trainer: Trainer, *args: Any, **kwargs: Any) -> None:
        ...


class TrainerEvent(Enum):
    """Callback events for the trainer."""

    ON_TRAIN_EPOCH_START = "on_train_epoch_start"
    ON_TRAIN_EPOCH_END = "on_train_epoch_end"
    ON_VALID_EPOCH_START = "on_valid_epoch_start"
    ON_VALID_EPOCH_END = "on_valid_epoch_end"
    ON_TRAIN_BATCH_START = "on_train_batch_start"
    ON_TRAIN_BATCH_END = "on_train_batch_end"
    ON_VALID_BATCH_START = "on_valid_batch_start"
    ON_VALID_BATCH_END = "on_valid_batch_end"
    ON_FIT_START = "on_fit_start"
    ON_FIT_END = "on_fit_end"


class TrainerPhase(Enum):
    """Phase of training."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@no_type_check
def move_to_device(batch: DatasetYield, device: torch.device) -> DatasetYield:
    """
    Moves the elements of a batch to the specified device.

    Parameters
    ----------
    batch : tuple
        A tuple containing the elements of the batch.
        Expected format: (inputs, targets, target_padding_masks, future_masks)

    device : torch.device or str
        The target device to move the batch elements to.

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The batch with all elements moved to the specified device.

    Note
    ----
    1. We usually set `pin_memory=True` and `non_blocking=True` together.
       This allows us to move them to GPU asynchronously.
    2. `mps` seems to be buggy with `non_blocking=True` so we set it to False.
    3. `cpu` we set to False too because it is not asynchronous?
    """
    if device.type in ["cpu", "mps"]:
        return tuple(tensor.to(device) for tensor in batch)
    else:
        batch_on_device = []
        for tensor in batch:
            try:
                if tensor.is_pinned():
                    batch_on_device.append(tensor.to(device, non_blocking=True))
                else:
                    batch_on_device.append(tensor.to(device))
            except RuntimeError as err:  # noqa: PERF203
                warnings.warn(err, stacklevel=2)
                batch_on_device.append(tensor.to(device))
        return tuple(batch_on_device)


class Trainer:
    def __init__(
        self,
        *,
        state: State,
        composer: Composer,
        logger: logging.Logger | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Super unsatisfying trainer class. If it was old me I would
        spend time to make it extremely modular...but I have learnt that
        not all scenarios demand such code."""
        # fmt: off
        self.state            = state
        self.composer         = composer

        self.model            = state.model
        self.model.to(device=device, dtype=next(state.model.parameters()).dtype, non_blocking=True)

        self.criterion        = state.criterion
        self.optimizer        = state.optimizer
        self.scheduler        = state.scheduler

        self.device: torch.device = composer.trainer.device if device is None else device # type: ignore[assignment]

        # logger
        self.logger = logger or get_default_logger()

        # general
        self.max_epochs = composer.trainer.max_epochs
        self.log_every_n_steps = composer.trainer.log_every_n_steps
        self.eval_every_n_steps = composer.trainer.eval_every_n_steps
        self.step_scheduler_on_batch_or_epoch = composer.trainer.step_scheduler_on_batch_or_epoch

        # mixed precision training
        self.use_amp = composer.trainer.use_amp
        self.autocast_config = composer.trainer.autocast_config
        self.context_manager = torch.autocast(device_type=self.device.type, **self.autocast_config) # no ops if not enabled
        self.scaler_config = composer.trainer.scaler_config
        self.scaler = torch.cuda.amp.GradScaler(**self.scaler_config)

        # gradient accumulation
        self.gradient_accumulation_steps = composer.trainer.gradient_accumulation_steps

        # training stability
        self.clip_grad_norm   = composer.trainer.clip_grad_norm
        self.apply_weight_decay_to_different_param_groups = composer.trainer.apply_weight_decay_to_different_param_groups # but this is applied outside, anti-pattern?

        # saving shenanigans
        self.save_dir = composer.trainer.save_dir
        self.save_every_epoch = composer.trainer.save_every_epoch
        self.save_best_only = composer.trainer.save_best_only
        self.mode = composer.trainer.mode
        self.monitor = composer.trainer.monitor
        self.best_monitored_score: None | float = None  # or -float('inf') if higher metric indicates better performance
        self.metrics_dict: Dict[str, float] = {}
        self.best_checkpoint_path: str = "" # NOTE: not in __init__ constructor and not in composer, set in callback
        self.history: Dict[str, List[float]] = defaultdict(list) # NOTE: not in __init__ constructor and not in composer, set in callback

        # attributes not in __init__ constructor
        self.epoch_index = 0
        self.train_batch_index = 0
        self.step_index = 0
        self.tokens_per_step = composer.data.train_loader["batch_size"] * composer.data.context_length # see nanogpt
        self.callbacks: Dict[str, List[TrainerCallback]] = defaultdict(list)

        # additional metrics, ideally metrics is implemented as callback and injected into trainer
        self.perplexity = Perplexity(ignore_index=state.criterion.ignore_index).to(device=self.device)

        # fmt: on
        self.add_callback(TrainerEvent.ON_VALID_EPOCH_END.value, save_state)
        self.add_callback(TrainerEvent.ON_TRAIN_EPOCH_END.value, update_state)
        self.add_callback(TrainerEvent.ON_FIT_START.value, log_on_fit_start)
        self.add_callback(TrainerEvent.ON_TRAIN_BATCH_END.value, log_every_n_steps_on_batch_end)
        self.add_callback(TrainerEvent.ON_TRAIN_EPOCH_START.value, log_on_train_epoch_start)
        self.add_callback(TrainerEvent.ON_VALID_EPOCH_START.value, log_on_train_epoch_start)
        self.add_callback(TrainerEvent.ON_TRAIN_EPOCH_END.value, log_on_epoch_end)
        self.add_callback(TrainerEvent.ON_VALID_EPOCH_END.value, log_on_epoch_end)
        self.add_callback(TrainerEvent.ON_FIT_START.value, log_on_fit_start_model_summary)

    def add_callback(self, event: str, callback: TrainerCallback) -> None:
        """Adds a callback to the list for a given event."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback: TrainerCallback) -> None:
        """Sets a callback to the list for a given event."""
        self.callbacks[event] = [callback]

    def remove_callback(self, event: str, callback: TrainerCallback) -> None:
        """Removes a callback from the list for a given event."""
        self.callbacks[event].remove(callback)

    def trigger_callbacks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Triggers all callbacks associated with a given event."""
        for callback in self.callbacks[event]:
            params = inspect.signature(callback).parameters
            if all(param in params for param in kwargs):
                callback(self, *args, **kwargs)
            else:
                callback(self)

    def update_metrics_and_history(
        self, metric_name_or_names: str | List[str], metric_value_or_values: float | List[float]
    ) -> None:
        metric_names = [metric_name_or_names] if isinstance(metric_name_or_names, str) else metric_name_or_names
        metric_values = (
            [metric_value_or_values] if isinstance(metric_value_or_values, float) else metric_value_or_values
        )

        for metric_name, metric_value in zip(metric_names, metric_values):
            self.metrics_dict[metric_name] = metric_value
            self.history[metric_name].append(metric_value)

    def _train_one_batch(self, batch: DatasetYield) -> Tuple[float, float, float]:
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_BATCH_START.value)
        inputs, targets, target_padding_masks, future_masks = move_to_device(batch, self.device)
        batch_size = inputs.size(0)

        with self.context_manager:  # no ops if not enabled
            logits: torch.FloatTensor = self.model(
                inputs, target_padding_masks=target_padding_masks, future_masks=future_masks
            )
            loss: torch.Tensor = self.criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())
            loss: torch.Tensor = loss / self.gradient_accumulation_steps  # type: ignore[no-redef] # NOTE: no ops if gradient_accumulation_steps=1

        self.scaler.scale(loss).backward()  # NOTE: no ops if scaler is not enabled

        this_batch_average_loss: float = loss.item()  # because reduction="mean"
        this_batch_total_loss: float = this_batch_average_loss * batch_size * self.gradient_accumulation_steps
        this_batch_average_perplexity: float = self.perplexity(
            logits, targets
        ).item()  # torch.exp(this_batch_average_loss)

        # if grad accum is 1 then this is our normal training because any integer
        # modulo 1 is 0 so this if loop will be executed after every batch!
        if (self.train_batch_index + 1) % self.gradient_accumulation_steps == 0:
            if self.clip_grad_norm and self.clip_grad_norm["max_norm"] != 0.0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            # model vs optimizer zero grad, the former is safer if you have >=2 optimizers
            self.model.zero_grad(set_to_none=True)

        # Step the scheduler after each batch if specified
        if self.scheduler and self.step_scheduler_on_batch_or_epoch == "batch":
            self.scheduler.step()

        self.train_batch_index += 1
        self.step_index += 1

        self.update_metrics_and_history(
            metric_name_or_names=["train_this_batch_average_loss", "train_this_batch_average_perplexity"],
            metric_value_or_values=[this_batch_average_loss, this_batch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_BATCH_END.value)
        return this_batch_average_loss, this_batch_total_loss, this_batch_average_perplexity

    def train_one_epoch(self, dataloader: DataLoader[DatasetYield]) -> Loss:
        """
        Variables
        ---------
        inputs: This is the input sequence (the EOS token is removed).
            shape: [B, S or T]
        targets: This is the input shifted by one time step to the right (the BOS token is removed).
            shape: [B, S or T]
        target_padding_masks:
            shape: [B, 1, S or T, S or T]
        future_masks:
            shape: [B, 1, S or T, S or T]
        logits:
            shape: [B, S or T, V]
            shape: [B, V, S or T] when passed in to loss as pytorch enforces the 2nd dim to be class/vocab.

        Parameters
        ----------
        model : nn.Module
            The model to be evaluated.
        dataloader : DataLoader
            The DataLoader containing the validation dataset.
        criterion : nn.Module
            The loss function used for evaluation.
        optimizer : Optimizer
            The optimizer used for training.
        scheduler : LRScheduler | None, optional
            The learning rate scheduler used for training, by default None.
        grad_norm_clip : float, optional
            The gradient norm clipping value, by default 1.0.
        device : torch.device
            The device to run the training on. We restrict to only accept
            torch.device instead of the overloaded variants such as str or int.

        Returns
        -------
        Loss
            The average loss over the training dataset.
        """
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_EPOCH_START.value, phase=TrainerPhase.TRAIN.value)
        self.model.train()

        total_samples: int = 0
        this_epoch_total_running_loss: float = 0.0
        num_batches: int = len(dataloader)
        progress_bar: tqdm[Tuple[int, DatasetYield]] = tqdm(
            enumerate(dataloader, start=1), total=num_batches, leave=False
        )

        self.train_batch_index = 0
        for _batch_index, batch in progress_bar:
            batch_size = batch[0].size(0)
            total_samples += batch_size

            this_batch_average_loss, this_batch_total_loss, this_batch_average_perplexity = self._train_one_batch(batch)
            this_epoch_total_running_loss += this_batch_total_loss

            progress_bar.set_description(f"Epoch: {self.epoch_index}, Step: {_batch_index}")
            progress_bar.set_postfix(
                {
                    "total_batch_loss": f"{this_batch_total_loss:.5f}",
                    "average_batch_loss": f"{this_batch_average_loss:.5f}",
                    "average_batch_perplexity": f"{this_batch_average_perplexity:.5f}",
                    "lr": f"{self._get_current_lr_or_lrs():.9f}",
                }
            )

            if self.scheduler and self.step_scheduler_on_batch_or_epoch == "epoch":
                self.scheduler.step()

        this_epoch_average_loss = this_epoch_total_running_loss / total_samples
        this_epoch_average_perplexity = torch.exp(torch.tensor(this_epoch_average_loss)).item()

        self.update_metrics_and_history(
            metric_name_or_names=["train_this_epoch_average_loss", "train_this_epoch_average_perplexity"],
            metric_value_or_values=[this_epoch_average_loss, this_epoch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_EPOCH_END.value, phase=TrainerPhase.TRAIN.value)
        return this_epoch_average_loss

    @torch.no_grad()
    def _valid_one_batch(self, batch: DatasetYield) -> Tuple[float, float, float]:
        self.trigger_callbacks(TrainerEvent.ON_VALID_BATCH_START.value)
        inputs, targets, target_padding_masks, future_masks = move_to_device(batch, self.device)
        batch_size = inputs.size(0)

        logits: torch.FloatTensor = self.model(
            inputs, target_padding_masks=target_padding_masks, future_masks=future_masks
        )

        # argmax_of_predicted_logits = torch.argmax(logits, dim=-1) # shape [B, S or V]
        # decoded_logits = batch_decode_equation(argmax_of_predicted_logits)

        loss: torch.nn.Module = self.criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())

        this_batch_average_loss: float = loss.item()
        this_batch_total_loss: float = this_batch_average_loss * batch_size
        this_batch_average_perplexity: float = self.perplexity(logits, targets).item()

        self.update_metrics_and_history(
            metric_name_or_names=["valid_this_batch_average_loss", "valid_this_batch_average_perplexity"],
            metric_value_or_values=[this_batch_average_loss, this_batch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_VALID_BATCH_END.value)
        return this_batch_average_loss, this_batch_total_loss, this_batch_average_perplexity

    def valid_one_epoch(self, dataloader: DataLoader[DatasetYield]) -> Loss:
        """
        Validates the model for one epoch on the given dataloader.

        Parameters
        ----------
        model : nn.Module
            The model to be evaluated.
        dataloader : DataLoader
            The DataLoader containing the validation dataset.
        criterion : nn.Module
            The loss function used for evaluation.
        device : str
            The device to run the evaluation on, defaults to 'cuda'.

        Returns
        -------
        Loss
            The average loss over the validation dataset.
        """
        self.trigger_callbacks(TrainerEvent.ON_VALID_EPOCH_START.value, phase=TrainerPhase.VALID.value)
        self.model.eval()

        total_samples: int = 0
        this_epoch_total_running_loss: float = 0.0
        num_batches = len(dataloader)
        progress_bar = tqdm(enumerate(dataloader, start=1), total=num_batches, leave=False)

        for _batch_index, batch in progress_bar:
            batch_size = batch[0].size(0)
            total_samples += batch_size
            this_batch_average_loss, this_batch_total_loss, this_batch_average_perplexity = self._valid_one_batch(batch)
            this_epoch_total_running_loss += this_batch_total_loss

            progress_bar.set_description(f"Epoch: {self.epoch_index}, Step: {_batch_index}")
            progress_bar.set_postfix(
                {
                    "total_batch_loss": f"{this_batch_total_loss:.5f}",
                    "average_batch_loss": f"{this_batch_average_loss:.5f}",
                    "average_batch_perplexity": f"{this_batch_average_perplexity:.5f}",
                }
            )

        # average loss for this epoch for each sample
        this_epoch_average_loss = this_epoch_total_running_loss / total_samples
        this_epoch_average_perplexity = torch.exp(torch.tensor(this_epoch_average_loss)).item()

        self.update_metrics_and_history(
            metric_name_or_names=["valid_this_epoch_average_loss", "valid_this_epoch_average_perplexity"],
            metric_value_or_values=[this_epoch_average_loss, this_epoch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_VALID_EPOCH_END.value, phase=TrainerPhase.VALID.value)
        return this_epoch_average_loss

    def test_one_epoch(self, dataloader: DataLoader[DatasetYield]) -> Loss:
        raise NotImplementedError(
            "The method `test_one_epoch` is not implemented. "
            "Please override this method in a subclass or use a custom callback."
        )

    def _get_current_lr_or_lrs(self) -> float | List[float]:
        """Get current learning rate."""
        if len(self.optimizer.param_groups) == 1:
            # we are sure the key "lr" should return a float
            return self.optimizer.param_groups[0]["lr"]  # type: ignore[no-any-return]

        lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]
        return lrs

    def fit(
        self,
        *,
        train_loader: DataLoader[DatasetYield],
        valid_loader: DataLoader[DatasetYield] | None = None,
        test_loader: DataLoader[DatasetYield] | None = None,
    ) -> State:
        # add as attributes purely for callback's Trainer to access it.
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # put callback here because depends on dataloader
        self.trigger_callbacks(TrainerEvent.ON_FIT_START.value)

        for _ in range(1, self.max_epochs + 1):
            self.epoch_index += 1
            self.train_loss = self.train_one_epoch(dataloader=train_loader)

            if valid_loader:
                self.valid_loss = self.valid_one_epoch(dataloader=valid_loader)

            if test_loader:
                try:
                    self.test_loss = self.test_one_epoch(dataloader=test_loader)
                except NotImplementedError as err:
                    self.logger.warning(err)

        self.state.history = self.history
        self.trigger_callbacks(TrainerEvent.ON_FIT_END.value)
        return self.state
