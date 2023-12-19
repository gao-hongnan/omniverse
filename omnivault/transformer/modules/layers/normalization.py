from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.types import _device, _dtype


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]

    normalized_shape: Union[int, Tuple[int, ...]]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[Union[_device, str, None]] = None,
        dtype: Optional[_dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))  # type: ignore[arg-type]
            self.beta = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))  # type: ignore[arg-type]
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        if self.elementwise_affine:
            return self.gamma * (x - mean) / (std + self.eps) + self.beta
        return (x - mean) / (std + self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(**self.__dict__)
