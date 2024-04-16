from typing import Iterable

import torch
from torch import nn

__all__ = ["gradient_clipping"]


def gradient_clipping(parameters: Iterable[nn.Parameter], max_norm: float, epsilon: float = 1e-6) -> None:
    """
    Clips the gradients of the provided parameters to have a maximum L2 norm of `max_norm`.

    Parameters
    ----------
    parameters : Iterable[nn.Parameter]
        An iterable of PyTorch parameters.
    max_norm : float
        The maximum L2 norm for gradient clipping.
    epsilon : float, optional
        A small value added for numerical stability. Defaults to 1e-6.
    """
    parameters_with_grad = [p for p in parameters if p.grad is not None]

    # l2 norm of the gradients
    grad_norms = torch.stack([torch.norm(p.grad, 2) for p in parameters_with_grad])
    total_norm = torch.norm(grad_norms, 2)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + epsilon)
        for p in parameters_with_grad:
            assert p.grad is not None  # appease mypy
            p.grad.mul_(clip_coef)
