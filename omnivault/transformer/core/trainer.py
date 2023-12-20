from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnivault._types._alias import Loss
from omnivault.transformer.core.dataset import AdderDatasetYield


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[AdderDatasetYield],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler | None = None,
    grad_norm_clip: float = 1.0,
    device: int | torch.device | None = None,
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
    scheduler : _LRScheduler | None, optional
        The learning rate scheduler used for training, by default None.
    grad_norm_clip : float, optional
        The gradient norm clipping value, by default 1.0.
    device : str
        The device to run the evaluation on, defaults to 'cuda'.

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
    progress_bar      : tqdm[Any] = tqdm(enumerate(dataloader, start=1), total=num_batches) # FIXME: Find the correct type for tqdm
    # fmt: on

    for _batch_index, batch in progress_bar:
        (
            inputs,
            targets,
            target_padding_masks,
            future_masks,
        ) = batch  # construct_batches(batch)
        inputs, targets, target_padding_masks, future_masks = (
            inputs.to(device),
            targets.to(device),
            target_padding_masks.to(device),
            future_masks.to(device),
        )
        batch_size = inputs.size(0)

        # fmt: off
        logits = model(inputs, target_padding_masks=target_padding_masks, future_masks=future_masks)
        loss   = criterion(logits.permute(0, 2, 1).contiguous(), targets.contiguous())
        # model vs optimizer zero grad, the former is safer if you have >=2 optimizers
        model.zero_grad(set_to_none=True)
        loss.backward()

        this_batch_loss: float = loss.item()
        batch_average_loss     = this_batch_loss / batch_size
        epoch_running_loss    += this_batch_loss
        # fmt: on

        nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        if _batch_index > 0 and _batch_index % 50 == 0:
            progress_bar.set_description(
                f"Epoch: {scheduler.last_epoch // num_batches if scheduler else _batch_index // num_batches}, "
                f"Total Train Loss: {this_batch_loss:.3f}, "
                f"Average Train Loss: {batch_average_loss:.3f}, "
                f"LR: {scheduler.get_last_lr()[0]:.5f}"
                if scheduler
                else "N/A"
            )
    # average loss for this epoch for each sample
    epoch_average_loss = epoch_running_loss / num_batches
    return epoch_average_loss


def valid_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[AdderDatasetYield],
    criterion: nn.Module,
    device: int | torch.device | None = None,
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
            # decoded_equations: List[str] = batch_decode_equation(x)
            # pprint(decoded_equations)

            (
                inputs,
                targets,
                target_padding_masks,
                future_masks,
            ) = batch  # construct_batches(batch)
            inputs, targets, target_padding_masks, future_masks = (
                inputs.to(device),
                targets.to(device),
                target_padding_masks.to(device),
                future_masks.to(device),
            )

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
        model: nn.Module,
        train_dataloader: DataLoader[AdderDatasetYield],
        valid_dataloader: DataLoader[AdderDatasetYield],
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler | None = None,
        grad_norm_clip: float = 1.0,
        device: int | torch.device | None = None,
        *,
        test_dataloader: DataLoader[AdderDatasetYield] | None = None,
    ) -> None:
        """Super unsatisfying trainer class. If it was old me I would
        spend time to make it extremely modular...but I have learnt that
        not all scenarios demand such code."""
        # fmt: off
        self.model            = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion        = criterion
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.grad_norm_clip   = grad_norm_clip
        self.device           = device
        self.test_dataloader  = test_dataloader

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
            model=self.model,
            dataloader=self.train_dataloader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            grad_norm_clip=self.grad_norm_clip,
            device=self.device,
        )

    def valid_epoch(self) -> Loss:
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

    def fit(self, num_epochs: int) -> nn.Module:
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")
            print("-" * 10)

            self.train_loss = self.train_epoch()
            self.valid_loss = self.valid_epoch()

            print(f"Training Loss   : {self.train_loss:.5f}")
            print(f"Validation Loss : {self.valid_loss:.5f}")

            if self.test_dataloader:
                test_loss = self.test_epoch()
                print(f"Test Loss       : {test_loss:.5f}")

        print("Training complete")
        return self.model
