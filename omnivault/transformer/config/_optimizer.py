import torch
from torch import nn
from omnivault.utils.config_management.dynamic import DynamicClassFactory
from typing import Tuple
from omnivault._types._generic import T


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

    lr: float # assume all optimizers have this parameter

    def build(self, *, params: nn.ParameterList, **kwargs: T) -> torch.optim.Optimizer:
        """Builder method for creating an optimizer instance."""
        return self.create_instance(params=params, **kwargs)

class AdamConfig(OptimizerConfig):
    name: str = "torch.optim.Adam"

    lr: float = 0.2
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9