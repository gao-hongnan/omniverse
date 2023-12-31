from __future__ import annotations

from typing import Callable, Dict, Literal, Type, Union

import torch
from torch import nn

from omnivault.utils.config_management.dynamic import DynamicClassFactory

RegisteredCriterion = Literal["torch.nn.CrossEntropyLoss", "torch.nn.MSELoss"]
CRITERION_REGISTRY: Dict[RegisteredCriterion, Type[CriterionConfig]] = {}


def register_criterion(name: RegisteredCriterion) -> Callable[[Type[CriterionConfig]], Type[CriterionConfig]]:
    """
    Decorator factory for registering criterion configurations.

    Parameters
    ----------
    name : RegisteredCriterion
        The name to register the class under.

    Returns
    -------
    Callable[[Type[CriterionConfig]], Type[CriterionConfig]]
        A decorator that registers the criterion configuration class.
    """

    def register_criterion_cls(cls: Type[CriterionConfig]) -> Type[CriterionConfig]:
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Cannot register duplicate criterion {name}")

        if not issubclass(cls, CriterionConfig):
            raise ValueError(f"Criterion (name={name}, class={cls.__name__}) must extend CriterionConfig")

        CRITERION_REGISTRY[name] = cls
        return cls

    return register_criterion_cls


class CriterionConfig(DynamicClassFactory[nn.Module]):
    """
    Configuration for creating PyTorch criterion instances dynamically.

    This class extends `DynamicClassFactory` to specifically handle the
    instantiation of PyTorch loss function classes based on provided configurations.

    Methods
    -------
    build() -> nn.Module
        Creates and returns a loss function instance with the specified parameters.
    """

    name: str

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


@register_criterion(name="torch.nn.CrossEntropyLoss")
class CrossEntropyLossConfig(CriterionConfig):
    weight: Union[torch.Tensor, None] = None
    size_average: Union[bool, None] = None
    ignore_index: int = -100
    reduction: str = "mean"
    label_smoothing: float = 0.0


@register_criterion(name="torch.nn.MSELoss")
class MSELossConfig(CriterionConfig):
    size_average: Union[bool, None] = None
    reduction: str = "mean"
