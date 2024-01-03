from typing import Optional

import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """
    Implements a Position-wise FeedForward Network (FFN) used in Transformer models.

    This module applies two linear transformations with a non-linear activation
    in between. It is often used after the multi-head self-attention layer
    in Transformer models.

    The naming convention for the linear layers ('context_fc' and 'context_projection') is inspired by
    the functionality within the Transformer architecture:

    - 'context_fc' (context fully connected): This layer expands the dimensionality
    of the input features, creating a richer representation. The expansion factor
    is often 4 in Transformer models, meaning the intermediate size is 4 times the
    size of the input/output dimensions.

    - 'context_projection' (context projection): This layer projects the expanded
    features back down to the original dimension, synthesizing the information
    processed by the 'context_fc' layer.

    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # fmt: off
        if d_ff is None:
            d_ff = 4 * d_model # typical value for d_ff in Transformer models

        self.ffn = nn.ModuleDict({
            'context_fc': nn.Linear(d_model, d_ff, bias=bias),
            'activation': activation,
            'context_projection': nn.Linear(d_ff, d_model, bias=bias),
            'dropout': nn.Dropout(p=dropout, inplace=False),

        })

        # self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters of the linear layers."""
        nn.init.xavier_uniform_(self.ffn["context_fc"].weight)
        if self.ffn["context_fc"].bias is not None:
            nn.init.constant_(self.ffn["context_fc"].bias, 0)

        nn.init.xavier_uniform_(self.ffn["context_projection"].weight)
        if self.ffn["context_projection"].bias is not None:
            nn.init.constant_(self.ffn["context_projection"].bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.ffn["context_fc"](z)
        z = self.ffn["activation"](z)
        z = self.ffn["dropout"](z)
        z = self.ffn["context_projection"](z)
        return z
