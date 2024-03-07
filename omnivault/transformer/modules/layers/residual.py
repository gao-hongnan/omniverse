from typing import Callable

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        return x + sublayer(x)
