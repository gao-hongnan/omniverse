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

    Stable
    -------
    ```python
    def stable_softmax(z: torch.Tensor) -> torch.Tensor:
        max_z = torch.max(z, dim=1, keepdim=True).values
        numerator = torch.exp(z - max_z)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        g = numerator / denominator
        return g
    ```
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
        Use the Jacobian matrix to compute the gradient. For usage, see
        `omniverse/playbook/softmax_preserves_order_translation_invariant_not_invariant_scaling.md`.
        """
        S = self.__call__(z)
        diag_S = torch.diag_embed(S)
        outer_S = torch.matmul(S.unsqueeze(2), S.unsqueeze(1))
        gradient = diag_S - outer_S
        return gradient.squeeze()
