from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Type

import torch

from omnivault.utils.config_management.dynamic import DynamicClassFactory

RegisteredSchedulers = Literal[
    "torch.optim.lr_scheduler.StepLR", "torch.optim.lr_scheduler.CosineAnnealingLR", "torch.optim.lr_scheduler.LambdaLR"
]
SCHEDULER_REGISTRY: Dict[str, Type[SchedulerConfig]] = {}


def register_scheduler(name: str) -> Callable[[Type[SchedulerConfig]], Type[SchedulerConfig]]:
    def register_scheduler_cls(cls: Type[SchedulerConfig]) -> Type[SchedulerConfig]:
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"Cannot register duplicate scheduler {name}")
        if not issubclass(cls, SchedulerConfig):
            raise ValueError(f"Scheduler (name={name}, class={cls.__name__}) must extend SchedulerConfig")
        SCHEDULER_REGISTRY[name] = cls
        return cls

    return register_scheduler_cls


class SchedulerConfig(DynamicClassFactory[torch.optim.lr_scheduler.LRScheduler]):
    """
    Base class for creating PyTorch scheduler instances dynamically.

    This class extends `DynamicClassFactory` to specifically handle the
    instantiation of PyTorch scheduler classes based on provided configurations.

    Methods
    -------
    build(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler
        Creates and returns a scheduler instance with the specified optimizer.
    """

    name: str

    def build(self, optimizer: torch.optim.Optimizer, **kwargs: Any) -> torch.optim.lr_scheduler.LRScheduler:
        """Builder method for creating a scheduler instance."""
        return self.create_instance(optimizer=optimizer, **kwargs)

    class Config:
        extra = "forbid"


@register_scheduler("torch.optim.lr_scheduler.StepLR")
class StepLRConfig(SchedulerConfig):
    step_size: int
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@register_scheduler("torch.optim.lr_scheduler.CosineAnnealingLR")
class CosineAnnealingLRConfig(SchedulerConfig):
    T_max: int
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False


@register_scheduler("torch.optim.lr_scheduler.LambdaLR")
class LambdaLRConfig(SchedulerConfig):
    # The user must provide a lambda function for the scheduler
    lr_lambda: Callable[[int], float]  # we know this lr_lambda maps int (epoch) to float (some multiplier)
