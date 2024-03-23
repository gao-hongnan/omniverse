from __future__ import annotations

import torch

from omnivault.modules.module import Activation


class Softmax(Activation):
    """
    Softmax activation function.
    """

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the softmax function for a given input.
        """
        numerator = torch.exp(z)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        g = numerator / denominator
        return g

    def gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the softmax function with respect to its input.
        """
        g = self.__call__(z)
        g = g.unsqueeze(-1)  # add an extra dimension
        eye = torch.eye(g.shape[1], device=z.device)[None, :]  # identity matrix for each sample
        dg_dz = g * (eye - g.transpose(-1, -2))
        return dg_dz.sum(dim=1)