"""State...Metadata...See how composer does it, quite elegant I'd say."""
import torch
from pydantic import BaseModel, Field
from rich.pretty import pprint
from torch import nn


class State(BaseModel):
    """Poor man's state, don't even compare it to Composer's implementation.
    We typically do not include the dataloaders here, because they do not have
    `state` in a sense of how model or optimizers have. However, we can inherit
    `State` with `Serializable` and force the dataloaders to be included."""

    model: nn.Module = Field(default=None, description="Model.")

    criterion: nn.Module = Field(default=None, description="Loss function.")
    optimizer: torch.optim.Optimizer = Field(default=None, description="Optimizer.")
    scheduler: torch.optim.lr_scheduler.LRScheduler = Field(default=None, description="Scheduler.")

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
        }
        torch.save(state, filepath)

    def load_snapshots(self, filepath: str, device: torch.device) -> None:
        """Load state dictionaries from a file."""
        state = torch.load(filepath, map_location=device)
        if self.model and "model" in state:
            self.model.load_state_dict(state["model"])
        if self.criterion and "criterion" in state:
            self.criterion.load_state_dict(state["criterion"])
        if self.optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
