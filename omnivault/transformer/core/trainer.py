from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, no_type_check

import torch
from rich.console import Console
from rich.logging import RichHandler
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnivault._types._alias import Loss
from omnivault.transformer.config.composer import Composer
from omnivault.transformer.core.dataset import DatasetYield
from omnivault.transformer.core.state import State
from omnivault.transformer.utils.format import format_lr

# Create a Rich console instance
console = Console()

# Configure Python logging to use RichHandler
logging.basicConfig(
    level="INFO",  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(message)s",  # No need for time and level since Rich will provide this
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)

logger = logging.getLogger("rich")


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
    return tuple(tensor.pin_memory().to(device, non_blocking=True) for tensor in batch)


def train_one_epoch(
    *,
    composer: Composer,
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader[DatasetYield],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
) -> Loss:
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
    # TODO: this is inefficient ops because each epoch one needs to equip model to device.
    model.to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True)
    model.train()

    # fmt: off
    epoch_running_loss: float = 0.0
    num_batches       : int   = len(dataloader)
    progress_bar      : tqdm[Tuple[int, DatasetYield]] = tqdm(enumerate(dataloader, start=1), total=num_batches) # FIXME: Find the correct type for tqdm
    # fmt: on

    for _batch_index, batch in progress_bar:
        inputs, targets, target_padding_masks, future_masks = move_to_device(batch, device)

        batch_size = inputs.size(0)

        # fmt: off
        logits: torch.FloatTensor = model(inputs, target_padding_masks=target_padding_masks, future_masks=future_masks)
        loss: torch.nn.Module   = criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())
        # model vs optimizer zero grad, the former is safer if you have >=2 optimizers
        model.zero_grad(set_to_none=True)
        loss.backward()

        this_batch_loss: float = loss.item()
        batch_average_loss     = this_batch_loss / batch_size
        epoch_running_loss    += this_batch_loss
        # fmt: on

        if composer.trainer.clip_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), **composer.trainer.clip_grad_norm)

        optimizer.step()
        if scheduler:
            scheduler.step()

        if _batch_index > 0 and _batch_index % 50 == 0:
            lr_info = f"LR: {scheduler.get_last_lr()[0]:.9f}" if scheduler else "LR: N/A"
            progress_bar.set_description(  # FIXME: wrong epoch info
                f"Epoch: {scheduler.last_epoch // num_batches if scheduler else _batch_index // num_batches}, "
                f"This Batch Train Loss: {this_batch_loss:.3f}, "
                f"This Batch Average Train Loss: {batch_average_loss:.3f}, "
                f"LR: {lr_info}"
            )
    # average loss for this epoch for each sample
    epoch_average_loss = epoch_running_loss / num_batches
    return epoch_average_loss


def valid_one_epoch(
    model: nn.Module,
    device: torch.device,
    dataloader: DataLoader[DatasetYield],
    criterion: nn.Module,
) -> Loss:
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
    model.to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True)
    model.eval()

    epoch_running_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(enumerate(dataloader, start=1), total=num_batches)

    with torch.no_grad():  # Disable gradient computation
        for _batch_index, batch in progress_bar:
            inputs, targets, target_padding_masks, future_masks = move_to_device(batch, device)

            # fmt: off
            logits = model(inputs, target_padding_masks=target_padding_masks, future_masks=future_masks)
            #argmax_of_predicted_logits = torch.argmax(logits, dim=-1) # shape [B, S or V]

            #decoded_logits = batch_decode_equation(argmax_of_predicted_logits)
            #pprint(decoded_logits)

            loss   = criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())

            this_batch_loss     = loss.item()
            epoch_running_loss += this_batch_loss
            # fmt: on

            # Update the progress bar
            # progress_bar.set_description(
            #     f"Valid or Holdout Loss: {this_batch_loss:.3f}"
            # )

    # average loss for this epoch for each sample
    epoch_average_loss = epoch_running_loss / num_batches
    return epoch_average_loss


class Trainer:
    def __init__(
        self,
        state: State,
        composer: Composer,
        *,
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
        self.logger = logger

        # general
        self.max_epochs = composer.trainer.max_epochs
        self.log_every_n_steps = composer.trainer.log_every_n_steps
        self.eval_every_n_steps = composer.trainer.eval_every_n_steps

        # training stability
        self.clip_grad_norm   = composer.trainer.clip_grad_norm
        self.apply_weight_decay_to_different_param_groups = composer.trainer.apply_weight_decay_to_different_param_groups # but this is applied outside, anti-pattern?

        # saving shenanigans
        self.save_dir = composer.trainer.save_dir
        self.save_every_epoch = composer.trainer.save_every_epoch

        # attributes not in __init__ constructor
        self.epoch_index = 0
        self.callbacks: Dict[str, List[Callable[[Trainer], None]]] = defaultdict(list)
        # fmt: on

    def add_callback(self, event: str, callback: Callable[[Trainer], None]) -> None:
        """Adds a callback to the list for a given event."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback: Callable[[Trainer], None]) -> None:
        """Sets a callback to the list for a given event."""
        self.callbacks[event] = [callback]

    def remove_callback(self, event: str, callback: Callable[[Trainer], None]) -> None:
        """Removes a callback from the list for a given event."""
        self.callbacks[event].remove(callback)

    def trigger_callbacks(self, event: str) -> None:
        """Triggers all callbacks associated with a given event."""
        for callback in self.callbacks[event]:
            callback(self)

    def train_one_epoch(self, loader: DataLoader[DatasetYield]) -> Loss:
        self.model.train()

        # fmt: off
        epoch_running_loss: float = 0.0
        num_batches       : int   = len(loader)
        progress_bar      : tqdm[Tuple[int, DatasetYield]] = tqdm(enumerate(loader, start=1), total=num_batches) # FIXME: Find the correct type for tqdm
        # fmt: on
        for _batch_index, batch in progress_bar:
            inputs, targets, target_padding_masks, future_masks = move_to_device(batch, self.device)
            batch_size = inputs.size(0)

            # fmt: off
            logits: torch.FloatTensor = self.model(inputs, target_padding_masks=target_padding_masks, future_masks=future_masks)
            loss: torch.nn.Module   = self.criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())
            # model vs optimizer zero grad, the former is safer if you have >=2 optimizers
            self.model.zero_grad(set_to_none=True)
            loss.backward()

            this_batch_loss: float = loss.item()
            batch_average_loss     = this_batch_loss / batch_size
            epoch_running_loss    += this_batch_loss
            # fmt: on

            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if _batch_index % self.log_every_n_steps == 0:
                self.logger.info("Step %d - Loss: %.5f", _batch_index, this_batch_loss)
                lr_info = f"LR: {self.scheduler.get_last_lr()[0]:.9f}" if self.scheduler else "LR: N/A"
                self.logger.info(
                    "Epoch: %d, Step: %d, Batch Loss: %.3f, Avg Batch Loss: %.3f, %s",
                    self.epoch_index,  # assuming you have a variable 'epoch'
                    _batch_index,
                    this_batch_loss,
                    batch_average_loss,
                    lr_info,
                )

        epoch_average_loss = epoch_running_loss / num_batches
        self.epoch_index += 1
        self.trigger_callbacks("on_epoch_end")
        return epoch_average_loss

    def train_epoch(self) -> Loss:
        return train_one_epoch(
            composer=self.composer,
            model=self.model,
            dataloader=self.train_dataloader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

    def valid_epoch(self) -> Loss:
        assert self.valid_dataloader is not None, "Valid dataloader must be provided for validation."
        return valid_one_epoch(
            model=self.model,
            dataloader=self.valid_dataloader,
            criterion=self.criterion,
            device=self.device,
        )

    def test_epoch(self) -> Loss:
        """
        Evaluates the model on the holdout dataset.
        """
        assert self.test_dataloader is not None, "Test dataloader must be provided for testing."
        return valid_one_epoch(
            model=self.model,
            dataloader=self.test_dataloader,
            criterion=self.criterion,
            device=self.device,
        )

    # def _train_batch(self, batch: DatasetYield) -> Loss:
    #     self.optimizer.zero_grad(set_to_none=True)

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
    ) -> nn.Module:
        self.train_dataloader = train_loader
        self.valid_dataloader = valid_loader
        self.test_dataloader = test_loader

        initial_lr_or_lrs = self._get_current_lr_or_lrs()
        lr_str = format_lr(initial_lr_or_lrs, precision=9)
        # Log the initial learning rate(s)
        if isinstance(initial_lr_or_lrs, list):
            self.logger.info("Initial learning rates for each parameter group: %s", lr_str)
        else:
            self.logger.info("Initial learning rate: %s", initial_lr_or_lrs)

        for epoch in range(1, self.max_epochs + 1):
            self.logger.info("Epoch %d/%d", epoch, self.max_epochs)
            # import time

            # self.train_loss = self.train_epoch()
            self.train_loss = self.train_one_epoch(train_loader)
            if self.save_every_epoch:
                torch.save(self.model.state_dict(), f"model_{epoch}.pth")

            self.logger.info("Average Epoch Training Loss: %.5f", self.train_loss)

            if valid_loader:
                self.valid_loss = self.valid_epoch()
                self.logger.info("Average Epoch Validation Loss: %.5f", self.valid_loss)

            if test_loader:
                test_loss = self.test_epoch()
                self.logger.info("Average Epoch Test Loss: %.5f", test_loss)

        print("Training complete")
        return self.model
