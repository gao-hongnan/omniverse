from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Literal, Tuple, Type

import torch
from torch import nn

from omnivault.utils.config_management.dynamic import DynamicClassFactory

RegisteredOptimizers = Literal["torch.optim.Adam", "torch.optim.AdamW", "torch.optim.SGD"]
OPTIMIZER_REGISTRY: Dict[RegisteredOptimizers, Type[OptimizerConfig]] = {}


def register_optimizer(name: RegisteredOptimizers) -> Callable[[Type[OptimizerConfig]], Type[OptimizerConfig]]:
    """
    Decorator factory for registering optimizer configurations.

    Parameters
    ----------
    name : RegisteredOptimizers
        The name to register the class under; if not provided, the class's 'name' attribute is used.

    Returns
    -------
    Callable[[Type[OptimizerConfig]], Type[OptimizerConfig]]
        A decorator that registers the optimizer configuration class.
    """

    def register_optimizer_cls(cls: Type[OptimizerConfig]) -> Type[OptimizerConfig]:
        if name in OPTIMIZER_REGISTRY:
            raise ValueError(f"Cannot register duplicate optimizer {name}")

        if not issubclass(cls, OptimizerConfig):
            raise ValueError(f"Optimizer (name={name}, class={cls.__name__}) must extend OptimizerConfig")

        OPTIMIZER_REGISTRY[name] = cls
        return cls

    return register_optimizer_cls


# NOTE: remember to give generic class a type here, which is
# aptly `torch.optim.Optimizer` in this case.
class OptimizerConfig(DynamicClassFactory[torch.optim.Optimizer]):
    """
    Configuration for creating PyTorch optimizer instances dynamically.

    This class extends `DynamicClassFactory` to specifically handle the
    instantiation of PyTorch optimizer classes based on provided configurations.
    The primary use case is to create optimizers with different settings
    for experimenting with model training.

    Parameters
    ----------
    name : str
        The fully qualified class name of the optimizer to instantiate.
        Defaults to 'torch.optim.Adam'.
    lr : float
        Learning rate for the optimizer. Defaults to 0.2.
    betas : Tuple[float, float]
        Coefficients used for computing running averages of gradient and its square.
        Defaults to (0.9, 0.98).
    eps : float
        Term added to the denominator to improve numerical stability.
        Defaults to 1e-9.

    Methods
    -------
    build(params: nn.ParameterList) -> torch.optim.Optimizer
        Creates and returns an optimizer instance with the specified parameters.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from omnivault.utils.config_management.dynamic import DynamicClassFactory
    >>> parameters = nn.ParameterList([nn.Parameter(torch.randn(2, 2, requires_grad=True))])
    >>> optimizer_config = OptimizerConfig(name="torch.optim.Adam", lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    >>> optimizer = optimizer_config.build(params=parameters)

    Below is an example that works but not recommended:
    >>> optimizer_config = OptimizerConfig(name="torch.optim.Adam", lr=0.1)
    >>> optimizer = optimizer_config.build(params=parameters, betas=(0.9, 0.999), eps=1e-8)

    This works but it is harder to maintain as we are passing in the "config" in
    `build` method instead of the constructor. The recommended way is to pass in
    the config in the constructor and only pass in the model params in the `build`
    method.
    """

    name: str
    lr: float  # assume all optimizers have this parameter

    # FIXME: how do we loosen this params type?
    def build(
        self,
        *,
        params: List[Dict[Literal["params", "weight_decay"], List[torch.nn.Parameter] | float]]
        | nn.ParameterList
        | Iterator[nn.Parameter],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """Builder method for creating an optimizer instance."""
        return self.create_instance(params=params, **kwargs)

    class Config:
        """Pydantic configuration for `OptimizerConfig`."""

        extra = "forbid"


@register_optimizer(name="torch.optim.Adam")
class AdamConfig(OptimizerConfig):
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.0


@register_optimizer(name="torch.optim.AdamW")
class AdamWConfig(OptimizerConfig):
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False


@register_optimizer(name="torch.optim.SGD")
class SGDConfig(OptimizerConfig):
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0


# At the end of this script, `OPTIMIZER_REGISTRY` will be populated with
# the following:
# OPTIMIZER_REGISTRY = {
#     "torch.optim.Adam": AdamConfig,
#     "torch.optim.SGD": SGDConfig,
# }
# This is because the `register_optimizer` decorator is called during
# runtime.
