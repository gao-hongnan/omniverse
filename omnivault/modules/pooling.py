from __future__ import annotations

from typing import overload

import torch
from torch import nn

__all__ = ["LastTokenPooling"]


class LastTokenPooling(nn.Module):
    """Last token pooling layer - specifically for decoder only models to do
    fine-tuning on sequence classification tasks."""

    def __init__(self, pre_head_pooling: bool = True) -> None:
        super().__init__()
        self.pre_head_pooling = pre_head_pooling

    @overload
    def forward(self, last_hidden_state: torch.Tensor, logits: None = None) -> torch.Tensor:
        ...

    @overload
    def forward(self, last_hidden_state: None, logits: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self, last_hidden_state: torch.Tensor | None = None, logits: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the pooling layer.

        Parameters
        ----------
        last_hidden_state:  Hidden state of the last layer.
                            type:  torch.Tensor
                            shape: (B, T, D)
        logits:             Logits from the last layer.
                            type:  torch.Tensor
                            shape: (B, T, C)

        Notes
        -----
        In both cases, we will slice the `T` dimension to get the last token's
        hidden state or logits. For example, if `last_hidden_state` is provided,
        then we have `[B, T, D] -> [B, D]` and if `logits` is provided, then we
        have `[B, T, C] -> [B, C]`.
        """
        if self.pre_head_pooling:
            assert last_hidden_state is not None, "last_hidden_state must be provided when pre_head is True"
            pooled_hidden_state = last_hidden_state[:, -1, :]
            return pooled_hidden_state
        else:
            assert logits is not None, "logits must be provided when pre_head is False"
            pooled_logits = logits[:, -1, :]
            return pooled_logits
