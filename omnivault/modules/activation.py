from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from omnivault.modules.module import Activation


class Softmax(Activation):
    r"""Softmax activation function.

    Notes
    -----
    Let :math:`z` be of shape :math:`(B, K)` where :math:`B` is the batch size
    and :math:`K` is the number of classes. Then if we set :math:`\text{self.dim} = 1`,
    this means that the softmax function will be applied along the second dimension
    of :math:`z`. The output :math:`g` will be of shape :math:`(B, K)` and each row
    will sum to 1.

    In reality, we can have more than 2 dimensions (i.e., softmax in transformer
    model). Furthermore, softmax can be applied on feature dimensions as well
    , using a class is just convenient for this example.

    Stable Version
    --------------
    .. code-block:: python

        def stable_softmax(z: torch.Tensor) -> torch.Tensor:
            max_z = torch.max(z, dim=self.dim, keepdim=True).values
            numerator = torch.exp(z - max_z)
            denominator = torch.sum(numerator, dim=self.dim, keepdim=True)
            g = numerator / denominator
            return g

    Examples
    --------
    We can easily see that given a tensor :math:`\mathbf{Z}` of shape :math:`(B, K)= (2, 3)`,
    and if we specify :math:`\text{dim} = -1`, the softmax function will be applied
    along the last dimension of :math:`\mathbf{Z}`. In this case, the softmax will be applied
    to each row of :math:`\mathbf{Z}` and the output :math:`g` will be of shape :math:`(2, 3)`.
    We can easily verify that softmax is indeed applied independently to each row of :math:`\mathbf{Z}`
    by comparing the output of the softmax function applied to each row of :math:`\mathbf{Z}`
    with the output of the softmax function applied to the entire tensor :math:`\mathbf{Z}`.

    >>> softmax = Softmax(dim=-1)
    >>> Z = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    >>> result = softmax(Z)
    >>> result
    tensor([[0.0900, 0.2447, 0.6652],
            [0.6652, 0.2447, 0.0900]])
    >>> individual_results = torch.vstack([softmax(z[i].unsqueeze(0)) for i in range(z.shape[0])])
    >>> torch.testing.assert_close(result, individual_results)
    """

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the softmax function for a given input.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the softmax function.
        """
        numerator = torch.exp(z)
        denominator = torch.sum(numerator, dim=self.dim, keepdim=True)
        g = numerator / denominator
        return g

    def gradient(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Use the Jacobian matrix to compute the gradient.

        For usage, see:
        :file:`omniverse/playbook/softmax_preserves_order_translation_invariant_not_invariant_scaling.md`.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Gradient of the softmax function.
        """
        S = self.__call__(z)
        diag_S = torch.diag_embed(S)
        outer_S = torch.matmul(S.unsqueeze(2), S.unsqueeze(1))
        gradient = diag_S - outer_S
        return gradient.squeeze()


class SoftmaxStable(Softmax):
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        max_z = torch.max(z, dim=self.dim, keepdim=True).values
        numerator = torch.exp(z - max_z)
        denominator = torch.sum(numerator, dim=self.dim, keepdim=True)
        g = numerator / denominator
        return g


class GELU(nn.Module):
    r"""A Gaussian Error Linear Unit (GELU) activation layer that applies the GELU function element-wise.

    The GELU activation function is defined as:

    .. math::
        \text{GELU}(x) = x \cdot \Phi(x)

    where :math:`\Phi(x)` is the cumulative distribution function of the standard Gaussian distribution.

    Parameters
    ----------
    approximate : {'tanh', None}, optional
        Specifies whether to use an approximation of the GELU function:
        - 'tanh': Uses a tanh-based approximation to compute GELU.
        - None: Uses the exact formula with the error function `erf`.
        The default is None, which uses the exact formulation.

    Notes
    -----
    The GELU function is mathematically represented as:

    .. math::
        \text{GELU}(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]

    When GELU (Gaussian Error Linear Unit) is applied to a tensor, it operates
    element-wise, meaning it is applied independently and identically (i.i.d.) to
    each element within the tensor. Given a tensor with shape
    :math:`[\mathcal{B}, T, D] = [2, 3, 4]`, which represents a batch size of 2,
    sequence length of 3, and feature dimension of 4, the GELU activation function
    would be applied 24 times in total, once for each element in the tensor.

    The `approximate='tanh'` option applies the following approximation:

    .. math::
        \text{GELU}_{\text{approx}}(x) = 0.5x \cdot \left(1 + \text{tanh}\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3 \right)\right)\right)

    This approximation is computationally faster and has been found to perform similarly to the exact form.

    Examples
    --------
    The GELU activation function is applied to a tensor :math:`\mathbf{Z}` of shape :math:`(2, 3)`
    elementwise, we can easily verify with an assert statement below to show that applying
    GELU on a tensor is equivalent to applying the GELU function elementwise to each element
    in the tensor.

    >>> import torch
    >>> Z = torch.tensor([[-1, 0, 1], [1, 0, -1]], dtype=torch.float32)
    >>> gelu = GELU(approximate=None)
    >>> gelu(Z)
    tensor([[-0.1587,  0.0000,  0.8413],
            [ 0.8413,  0.0000, -0.1587]])
    >>> for i, row in enumerate(Z):
    ...     for j, element in enumerate(row):
    ...         out = GELU()(element)
    ...         torch.testing.assert_close(out, result[i, j])
    """

    def __init__(self, approximate: Literal["tanh"] | None = None) -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == "tanh":
            x_out_BTD = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
            return x_out_BTD

        x_out_BTD = x * 0.5 * (1 + torch.erf(input=x / 2**0.5))
        return x_out_BTD
