"""State...Metadata...See how composer does it, quite elegant I'd say."""

from typing import Annotated, Any, Self, override

import torch
from pydantic import BaseModel, ConfigDict, Field
from rich.pretty import pprint
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from omnivault.utils.torch_utils.model_utils import compare_models


class State(BaseModel):
    """Poor man's state, don't even compare it to Composer's implementation.
    We typically do not include the dataloaders here, because they do not have
    `state` in a sense of how model or optimizers have. However, we can inherit
    `State` with `Serializable` and force the dataloaders to be included."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Annotated[nn.Module, Field(description="Model.")]

    criterion: Annotated[nn.Module, Field(description="Loss function.")]
    optimizer: Annotated[torch.optim.Optimizer, Field(description="Optimizer.")]
    scheduler: Annotated[torch.optim.lr_scheduler.LRScheduler | None, Field(description="Scheduler.")] = None

    epoch_index: Annotated[int, Field(description="Current epoch index.")] = 0
    train_batch_index: Annotated[
        int, Field(description="Current batch index and is only referring to the training batch index.")
    ] = 0
    step_index: Annotated[
        int,
        Field(
            description="We do not add prefix train because it is understood and implied that the step number is the train due to how many gradients been stepped. Current step index and is only referring to the training step index. What is the difference between step and batch? In general, they coincide for when the epoch number is 1, but after the first epoch, we usually reset the batch index to 0, while the step index keeps increasing to the next epoch."
        ),
    ] = 0
    history: dict[str, list[float]] = Field(default_factory=dict, description="History of metrics.")

    # FIXME: loosen `Vocabularies` and `Tokenizers` to `Any` for now as it is too strict.
    vocabulary: Annotated[Any, Field(description="Vocabulary.")] = None
    tokenizer: Annotated[Any, Field(description="Tokenizer.")] = None

    tokens_per_iter: Annotated[int | None, Field(description="Tokens per iter/step.")] = None

    @override
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
        #     and self.train_batch_index == other.train_batch_index
        # )

    @property
    def model_or_module(self) -> nn.Module:
        """The model not wrapped by :class:`DistributedDataParallel`."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module  # type: ignore[no-any-return]
        return self.model

    def pretty_print(self) -> None:
        """Pretty print the config."""
        pprint(self)

    def save_snapshots(self, filepath: str) -> None:
        """Save the state dictionaries of the components to a file."""
        state = {
            "model": self.model_or_module.state_dict() if self.model_or_module else None,
            "criterion": self.criterion.state_dict() if self.criterion else None,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch_index": self.epoch_index,
            "train_batch_index": self.train_batch_index,
            "step_index": self.step_index,
            "history": self.history,
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
        }
        torch.save(state, filepath)

    @classmethod
    def load_snapshots(
        cls: type[Self],
        filepath: str,
        device: torch.device,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> Self:
        """Load state dictionaries from a file and return a new State instance.

        Warnings
        --------
        This unpickles arbitrary Python objects and must only be pointed at a
        checkpoint produced by :meth:`save_snapshots`, never at an untrusted file.
        ``weights_only=True`` (torch>=2.6's default) cannot be used here because a
        snapshot deliberately stores non-tensor state -- ``history`` is a
        ``defaultdict`` and ``vocabulary``/``tokenizer`` are project-specific
        classes -- none of which are allowlisted by the safe unpickler.
        """
        state = torch.load(filepath, map_location=device, weights_only=False)  # nosec B614  # trusted local checkpoint produced by this repo

        epoch_index = state["epoch_index"]
        train_batch_index = state["train_batch_index"]
        step_index = state["step_index"]
        history = state["history"]
        vocabulary = state["vocabulary"]
        tokenizer = state["tokenizer"]

        # Create a new instance of State with loaded state
        new_state = cls(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_index=epoch_index,
            train_batch_index=train_batch_index,
            step_index=step_index,
            history=history,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
        )

        # Load state dicts into the model, criterion, etc., if they exist
        if new_state.model and "model" in state:
            new_state.model_or_module.load_state_dict(state["model"])
        if new_state.criterion and "criterion" in state:
            new_state.criterion.load_state_dict(state["criterion"])
        if new_state.optimizer and "optimizer" in state:
            new_state.optimizer.load_state_dict(state["optimizer"])
        if new_state.scheduler and "scheduler" in state:
            new_state.scheduler.load_state_dict(state["scheduler"])

        return new_state
