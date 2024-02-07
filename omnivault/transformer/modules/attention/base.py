from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn


# FIXME: consider removing ABC since nn.Module ensures forward method to be implemented.
class Attention(ABC, nn.Module):
    """
    Base class for attention mechanisms.

    This abstract class provides a scaffold for attention mechanisms, with a
    dropout layer for regularization included. Subclasses are expected to
    implement the `forward` method.

    Attributes
    ----------
    dropout : The dropout layer applied to the attention scores.
        type: nn.Dropout

    Note
    ----
    ABC method might be redundant since inheritance from nn.Module ensures
    forward method to be implemented.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass for the attention mechanism."""
        raise NotImplementedError("The `forward` method must be implemented by the subclass.")
