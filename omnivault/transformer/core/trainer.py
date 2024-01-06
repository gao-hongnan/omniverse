from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Tuple, no_type_check

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnivault._types._alias import Loss
from omnivault.transformer.config.composer import Composer
from omnivault.transformer.core.dataset import DatasetYield
from omnivault.transformer.core.state import State


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
    grad_norm_clip: float = 1.0,
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
            nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

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


# FIXME: make valid dataloader optional
class Trainer:
    def __init__(
        self,
        state: State,
        composer: Composer,
        train_dataloader: DataLoader[DatasetYield],
        *,
        device: torch.device | None = None,
        valid_dataloader: DataLoader[DatasetYield] | None = None,
        test_dataloader: DataLoader[DatasetYield] | None = None,
    ) -> None:
        """Super unsatisfying trainer class. If it was old me I would
        spend time to make it extremely modular...but I have learnt that
        not all scenarios demand such code."""
        # fmt: off
        self.state            = state
        self.composer         = composer

        self.model            = state.model
        self.criterion        = state.criterion
        self.optimizer        = state.optimizer
        self.scheduler        = state.scheduler

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader  = test_dataloader

        # training stability
        self.clip_grad_norm   = composer.trainer.clip_grad_norm

        self.device: torch.device = composer.trainer.device if device is None else device # type: ignore[assignment]


        # attributes not in __init__ constructor
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

    def fit(self, num_epochs: int, save_every_epoch: bool = False) -> nn.Module:
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 10)

            self.train_loss = self.train_epoch()
            if save_every_epoch:
                torch.save(self.model.state_dict(), f"model_{epoch}.pth")

            print(f"Average Epoch Training Loss   : {self.train_loss:.5f}")

            if self.valid_dataloader:
                self.valid_loss = self.valid_epoch()
                print(f"Average Epoch Validation Loss : {self.valid_loss:.5f}")

            if self.test_dataloader:
                test_loss = self.test_epoch()
                print(f"Average Epoch Test Loss       : {test_loss:.5f}")

        print("Training complete")
        return self.model
