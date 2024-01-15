from abc import ABC, abstractmethod

import torch
from torch import nn


class PositionalEncoding(ABC, nn.Module):
    def __init__(self, d_model: int, context_length: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Sinusoid(PositionalEncoding):
    P: torch.Tensor

    def __init__(self, d_model: int, context_length: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, context_length, dropout)

        P = self._init_positional_encoding()
        self.register_buffer("P", P, persistent=True)  # with this no need requires_grad=False

    def _init_positional_encoding(self) -> torch.Tensor:
        """Initialize the positional encoding tensor."""
        P = torch.zeros((1, self.context_length, self.d_model))
        position = self._get_position_vector()
        div_term = self._get_div_term_vector()
        P[:, :, 0::2] = torch.sin(position / div_term)
        P[:, :, 1::2] = torch.cos(position / div_term)
        return P

    def _get_position_vector(self) -> torch.Tensor:
        """Return a vector representing the position of each token in a sequence."""
        return torch.arange(self.context_length, dtype=torch.float32).reshape(-1, 1)

    def _get_div_term_vector(self) -> torch.Tensor:
        """Return a vector representing the divisor term for positional encoding."""
        return torch.pow(
            10000,
            torch.arange(0, self.d_model, 2, dtype=torch.float32) / self.d_model,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self._add_positional_encoding(z)
        z = self.dropout(z)
        return z

    def _add_positional_encoding(self, z: torch.Tensor) -> torch.Tensor:
        """Add the positional encoding tensor to the input tensor."""
        return z + self.P[:, : z.shape[1], :]
