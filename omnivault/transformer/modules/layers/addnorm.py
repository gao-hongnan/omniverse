from typing import Callable

import torch
from torch import nn

from omnivault.transformer.modules.layers.normalization import LayerNorm


class AddNorm(nn.Module):
    """AddNorm is apt since the diagram uses Add + Norm.
    Some call it SubLayer connection (Harvard) some call it
    residual connection: x + dropout(sublayer(layernorm(x)))

    If stay true, then we apply residual then layer norm. So
    we adopt the d2l method."""

    def __init__(self, feature_dim: int, dropout: float) -> None:
        super().__init__()
        # fmt: off
        self.dropout    = nn.Dropout(p=dropout, inplace=False)
        self.layer_norm = LayerNorm(normalized_shape=feature_dim, eps=1e-5, elementwise_affine=True)
        # fmt: on

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """The formulation is x + dropout(sublayer(layernorm(x)))."""
        # NOTE: GPT-2 should be from self.layer_norm(x + sublayer(self.dropout(x))) to x + self.dropout(sublayer(self.layer_norm(x)))
        output: torch.Tensor = x + self.dropout(sublayer(self.layer_norm(x)))
        return output
