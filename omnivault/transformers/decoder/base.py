"""Base classes for decoders in transformer-like architectures.
Template Design Pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseDecoderBlock(nn.Module, ABC):
    """
    Abstract base class for a decoder block in a transformer-like architecture.
    """

    @abstractmethod
    def forward(self, z: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Abstract method for forward pass to be implemented by subclasses.
        """


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for a decoder in a transformer-like architecture.
    """

    @abstractmethod
    def forward(self, z: torch.Tensor, **kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Abstract method for forward pass to be implemented by subclasses.
        """

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
