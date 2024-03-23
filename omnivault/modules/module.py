from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Activation(ABC):
    """
    Base class for activation functions.
    """

    @abstractmethod
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of the activation function.
        """
        raise NotImplementedError

    def gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the activation function.
        """
        raise NotImplementedError
