"""State...Metadata...See how composer does it, quite elegant I'd say."""
from __future__ import annotations

from typing import Type, Union

import torch
from pydantic import BaseModel, Field
from rich.pretty import pprint
from torch import nn

from omnivault.transformer.core.tokenizer import Tokenizers
from omnivault.transformer.core.vocabulary import Vocabularies


def compare_models(model_a: nn.Module, model_b: nn.Module) -> bool:
    """
    Compare two PyTorch models to check if they have identical parameters.

    Parameters
    ----------
    model_a : nn.Module
        The first model to compare.
    model_b : nn.Module
        The second model to compare.

    Returns
    -------
    bool
        Returns True if both models have identical parameters, False otherwise.
    """
    return all(
        torch.equal(param_a[1], param_b[1])
        for param_a, param_b in zip(model_a.state_dict().items(), model_b.state_dict().items())
    )


class State(BaseModel):
    """Poor man's state, don't even compare it to Composer's implementation.
    We typically do not include the dataloaders here, because they do not have
    `state` in a sense of how model or optimizers have. However, we can inherit
    `State` with `Serializable` and force the dataloaders to be included."""

    model: nn.Module = Field(default=None, description="Model.")

    criterion: nn.Module = Field(default=None, description="Loss function.")
    optimizer: torch.optim.Optimizer = Field(default=None, description="Optimizer.")
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = Field(default=None, description="Scheduler.")

    epoch_index: int = Field(default=0, description="Current epoch index.")
    batch_index: int = Field(default=0, description="Current batch index.")

    vocabulary: Vocabularies = Field(default=None, description="Vocabulary.")
    tokenizer: Tokenizers = Field(default=None, description="Tokenizer.")

    def __eq__(self, other: object) -> bool:
        """Check if two State instances are equal."""
        assert isinstance(other, State), "Can only compare State instances."

        models_equal = compare_models(self.model, other.model)
        return models_equal

        # return (
        #     (self.model.state_dict() if self.model else None) == (other.model.state_dict() if other.model else None)
        #     and (self.criterion.state_dict() if self.criterion else None)
        #     == (other.criterion.state_dict() if other.criterion else None)
        #     and (self.optimizer.state_dict() if self.optimizer else None)
        #     == (other.optimizer.state_dict() if other.optimizer else None)
        #     and (self.scheduler.state_dict() if self.scheduler else None)
        #     == (other.scheduler.state_dict() if other.scheduler else None)
        #     and self.epoch_index == other.epoch_index
        #     and self.batch_index == other.batch_index
        # )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def pretty_print(self) -> None:
        """Pretty print the config."""
        pprint(self)

    def save_snapshots(self, filepath: str) -> None:
        """Save the state dictionaries of the components to a file."""
        state = {
            "model": self.model.state_dict() if self.model else None,
            "criterion": self.criterion.state_dict() if self.criterion else None,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch_index": self.epoch_index,
            "batch_index": self.batch_index,
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
        }
        torch.save(state, filepath)

    @classmethod
    def load_snapshots(
        cls: Type[State],
        filepath: str,
        device: torch.device,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> State:
        """Load state dictionaries from a file and return a new State instance."""
        state = torch.load(filepath, map_location=device)

        epoch_index = state["epoch_index"]
        batch_index = state["batch_index"]
        # Create a new instance of State with loaded state
        new_state = cls(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_index=epoch_index,
            batch_index=batch_index,
        )

        # Load state dicts into the model, criterion, etc., if they exist
        if new_state.model and "model" in state:
            new_state.model.load_state_dict(state["model"])
        if new_state.criterion and "criterion" in state:
            new_state.criterion.load_state_dict(state["criterion"])
        if new_state.optimizer and "optimizer" in state:
            new_state.optimizer.load_state_dict(state["optimizer"])
        if new_state.scheduler and "scheduler" in state:
            new_state.scheduler.load_state_dict(state["scheduler"])

        return new_state
