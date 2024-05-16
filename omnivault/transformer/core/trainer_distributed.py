# mypy: disable-error-code="no-untyped-call"
from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
from tqdm import tqdm

from omnivault._types._alias import Loss
from omnivault.transformer.config.composer import Composer
from omnivault.transformer.core.callbacks import (
    CallbackPriority,
    log_every_n_steps_on_batch_end,
    log_on_epoch_end,
    log_on_fit_start,
    log_on_train_epoch_start,
    save_state,
    update_state,
)
from omnivault.transformer.core.dataset import DatasetYield
from omnivault.transformer.core.state import State
from omnivault.transformer.core.trainer import MetricNames, TrainerCallback, TrainerEvent, TrainerPhase, move_to_device
from omnivault.transformer.utils.format import format_lr
from omnivault.transformer.utils.general_utils import get_default_logger
from omnivault.utils.reproducibility.rng import load_and_set_rng_state


class Trainer:
    def __init__(
        self,
        *,
        state: State,
        composer: Composer,
        logger: logging.Logger | None = None,
        device: torch.device | None = None,
        resume_from_rng_path: str | None = None,
    ) -> None:
        """Super unsatisfying trainer class. If it was old me I would
        spend time to make it extremely modular...but I have learnt that
        not all scenarios demand such code."""
        # fmt: off
        self.resume_from_rng_path = resume_from_rng_path # resume from rng state
        if resume_from_rng_path:
            self.rng_state = load_and_set_rng_state(rng_state_path=resume_from_rng_path) # set RNG globally first

        self.state            = state
        self.composer         = composer

        self.model            = state.model

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
        self.epoch_index = self.rng_state["epoch_index"] if resume_from_rng_path else 0
        self.train_batch_index = 0
        self.step_index = 0
        self.tokens_per_iter = composer.data.train_loader["batch_size"] * composer.data.context_length * self.gradient_accumulation_steps * self.composer.distributed.world_size
        self.callbacks: Dict[TrainerEvent, List[Tuple[TrainerCallback, CallbackPriority]]] = defaultdict(list)

        # additional metrics, ideally metrics is implemented as callback and injected into trainer
        self.perplexity = Perplexity(ignore_index=state.criterion.ignore_index).to(device=self.device)
        # fmt: on
        self.register_default_callbacks()

    def register_default_callbacks(self) -> None:
        default_callbacks = [
            # NOTE: save state is lower priority than saving metrics at valid epoch end
            (TrainerEvent.ON_VALID_EPOCH_END, save_state, CallbackPriority.LOW),
            (TrainerEvent.ON_TRAIN_EPOCH_END, update_state, CallbackPriority.NORMAL),
            (TrainerEvent.ON_FIT_START, log_on_fit_start, CallbackPriority.NORMAL),
            (TrainerEvent.ON_TRAIN_BATCH_END, log_every_n_steps_on_batch_end, CallbackPriority.NORMAL),
            (TrainerEvent.ON_TRAIN_EPOCH_START, log_on_train_epoch_start, CallbackPriority.NORMAL),
            (TrainerEvent.ON_VALID_EPOCH_START, log_on_train_epoch_start, CallbackPriority.NORMAL),
            (TrainerEvent.ON_TRAIN_EPOCH_END, log_on_epoch_end, CallbackPriority.NORMAL),
            (TrainerEvent.ON_VALID_EPOCH_END, log_on_epoch_end, CallbackPriority.NORMAL),
        ]
        for event, callback, priority in default_callbacks:
            self.add_callback(event, callback, priority)  # type: ignore[arg-type]

    def add_callback(
        self, event: TrainerEvent, callback: TrainerCallback, priority: CallbackPriority = CallbackPriority.NORMAL
    ) -> None:
        self.callbacks[event].append((callback, priority))

    def set_callback(self, event: TrainerEvent, callback: TrainerCallback, priority: CallbackPriority) -> None:
        """Sets a callback to the list for a given event with the specified priority."""
        self.callbacks[event] = [(callback, priority)]

    def remove_callback(self, event: TrainerEvent, callback: TrainerCallback) -> None:
        """Removes a callback from the list for a given event."""
        self.callbacks[event] = [(cb, prio) for cb, prio in self.callbacks[event] if cb != callback]

    def trigger_callbacks(self, event: TrainerEvent, *args: Any, **kwargs: Any) -> None:
        """Why do we need to sort callbacks? Well consider the event `ON_TRAIN_EPOCH_END`
        and has two operations.

        1. Calculate metrics;
        2. Save checkpoint if metrics is better;

        These are all tied with the same event, and if you don't sort by order
        of priority, what happens? Calculating metrics at end of epoch must
        occur before both Checkpointing because these two operations depend on
        the latest metrics computed from the recent epoch. If metrics are not
        updated first, then the checkpoint decisions might be made on outdated data.
        """

        callbacks = sorted(self.callbacks[event], key=lambda x: x[1].value)  # Sort by priority
        for callback, _ in callbacks:
            sig = inspect.signature(callback)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            callback(self, *args, **filtered_kwargs)  # type: ignore[arg-type]

    @property
    def model_or_module(self) -> nn.Module:
        """The model not wrapped by :class:`DistributedDataParallel`."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module  # type: ignore[no-any-return]
        return self.model

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
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_BATCH_START)
        inputs, targets, target_padding_masks, future_masks = move_to_device(batch, self.device)
        batch_size = inputs.size(0)

        with self.context_manager:  # no ops if not enabled
            logits: torch.FloatTensor = self.model(
                inputs, target_padding_masks=target_padding_masks, future_masks=future_masks
            )
            # TODO: I want to change the permute to flattening instead for readability
            loss: torch.Tensor = self.criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())
            loss: torch.Tensor = loss / self.gradient_accumulation_steps  # type: ignore[no-redef] # NOTE: no ops if gradient_accumulation_steps=1

        # fmt: off
        self.scaler.scale(loss).backward()              # NOTE: no ops if scaler is not enabled

        this_batch_average_loss: float = loss.item()    # because reduction="mean"
        this_batch_total_loss: float = this_batch_average_loss * batch_size * self.gradient_accumulation_steps
        this_batch_average_perplexity: float = self.perplexity(logits, targets).item() # torch.exp(this_batch_average_loss)
        # fmt: on

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
            metric_name_or_names=[
                MetricNames.TRAIN_THIS_BATCH_AVERAGE_LOSS,
                MetricNames.TRAIN_THIS_BATCH_AVERAGE_PERPLEXITY,
            ],
            metric_value_or_values=[this_batch_average_loss, this_batch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_BATCH_END)
        return this_batch_average_loss, this_batch_total_loss, this_batch_average_perplexity

    def train_one_epoch(self, dataloader: DataLoader[DatasetYield]) -> Loss:
        """Train the model for one epoch on the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader containing the dataset.

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
        """
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_EPOCH_START, phase=TrainerPhase.TRAIN)
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
                    "lr": f"{format_lr(self._get_current_lr_or_lrs(), precision=9)}",
                }
            )

            if self.scheduler and self.step_scheduler_on_batch_or_epoch == "epoch":
                self.scheduler.step()

        this_epoch_average_loss = this_epoch_total_running_loss / total_samples
        this_epoch_average_perplexity = torch.exp(torch.tensor(this_epoch_average_loss)).item()

        self.update_metrics_and_history(
            metric_name_or_names=[
                MetricNames.TRAIN_THIS_EPOCH_AVERAGE_LOSS,
                MetricNames.TRAIN_THIS_EPOCH_AVERAGE_PERPLEXITY,
            ],
            metric_value_or_values=[this_epoch_average_loss, this_epoch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_TRAIN_EPOCH_END, phase=TrainerPhase.TRAIN)
        return this_epoch_average_loss

    @torch.no_grad()
    def _valid_one_batch(self, batch: DatasetYield) -> Tuple[float, float, float]:
        self.trigger_callbacks(TrainerEvent.ON_VALID_BATCH_START)
        inputs, targets, target_padding_masks, future_masks = move_to_device(batch, self.device)
        batch_size = inputs.size(0)

        logits: torch.FloatTensor = self.model(
            inputs, target_padding_masks=target_padding_masks, future_masks=future_masks
        )

        # argmax_of_predicted_logits = torch.argmax(logits, dim=-1) # shape [B, S or V]
        # decoded_logits = batch_decode_equation(argmax_of_predicted_logits)

        loss: torch.Tensor = self.criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())

        this_batch_average_loss: float = loss.item()
        this_batch_total_loss: float = this_batch_average_loss * batch_size
        this_batch_average_perplexity: float = self.perplexity(logits, targets).item()

        self.update_metrics_and_history(
            metric_name_or_names=[
                MetricNames.VALID_THIS_BATCH_AVERAGE_LOSS,
                MetricNames.VALID_THIS_BATCH_AVERAGE_PERPLEXITY,
            ],
            metric_value_or_values=[this_batch_average_loss, this_batch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_VALID_BATCH_END)
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
        self.trigger_callbacks(TrainerEvent.ON_VALID_EPOCH_START, phase=TrainerPhase.VALID)
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
            metric_name_or_names=[
                MetricNames.VALID_THIS_EPOCH_AVERAGE_LOSS,
                MetricNames.VALID_THIS_EPOCH_AVERAGE_PERPLEXITY,
            ],
            metric_value_or_values=[this_epoch_average_loss, this_epoch_average_perplexity],
        )
        self.trigger_callbacks(TrainerEvent.ON_VALID_EPOCH_END, phase=TrainerPhase.VALID)
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
        self.trigger_callbacks(TrainerEvent.ON_FIT_START)

        for _ in range(1, self.max_epochs + 1):
            # fmt: off
            self.epoch_index += 1               # to match range(1, max_epochs + 1) because we start from 1
            torch.manual_seed(self.epoch_index) # TODO: to replace with the full `load_and_set_rng_state` function for even stronger reproducibility
            if torch.cuda.is_available() and torch.cuda.is_initialized(): # type: ignore[no-untyped-call]
                torch.cuda.manual_seed_all(self.epoch_index)
            # fmt: on

            self.train_loss = self.train_one_epoch(dataloader=train_loader)

            if valid_loader:
                self.valid_loss = self.valid_one_epoch(dataloader=valid_loader)

            if test_loader:
                try:
                    self.test_loss = self.test_one_epoch(dataloader=test_loader)
                except NotImplementedError as err:
                    self.logger.warning(err)

        # NOTE: note clean here to add state here instead of in callback, but
        # for now this is because history only update finish after fit end but
        # most state is updated in epoch end.
        self.state.history = self.history
        self.trigger_callbacks(TrainerEvent.ON_FIT_END)
        return self.state
