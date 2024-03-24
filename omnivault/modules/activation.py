from __future__ import annotations

import torch

from omnivault.modules.module import Activation


class Softmax(Activation):
    """
    Softmax activation function.

    Example
    -------
    Let `z` be of shape `(B, K)` where `B` is the batch size and `K` is the
    number of classes. Then if we set `self.dim = 1`, this means that the
    softmax function will be applied along the second dimension of `z`. The
    output `g` will be of shape `(B, K)` and each row will sum to 1.
    In reality, we can have more than 2 dimension (i.e. softmax in transformer
    model). Furthermore, softmax can be applied on feature dimension as well,
    using class is just convenient for this example.
    """

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the softmax function for a given input.
        """
        numerator = torch.exp(z)
        denominator = torch.sum(numerator, dim=self.dim, keepdim=True)
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
