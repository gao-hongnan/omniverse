"""Module for creating PyTorch scheduler instances dynamically with an enhanced Registry pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Type

import torch
from pydantic import BaseModel
from rich.pretty import pprint

RegisteredSchedulers = Literal[
    "StepLR",
    "CosineAnnealingLR",
    "LambdaLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
]


class SchedulerRegistry:
    _schedulers: Dict[str, Type[SchedulerConfig]] = {}

    @classmethod
    def register(cls: Type[SchedulerRegistry], name: str) -> Callable[[Type[SchedulerConfig]], Type[SchedulerConfig]]:
        def register_scheduler_cls(scheduler_cls: Type[SchedulerConfig]) -> Type[SchedulerConfig]:
            if name in cls._schedulers:
                raise ValueError(f"Cannot register duplicate scheduler {name}")
            if not issubclass(scheduler_cls, SchedulerConfig):
                raise ValueError(f"Scheduler (name={name}, class={scheduler_cls.__name__}) must extend SchedulerConfig")
            cls._schedulers[name] = scheduler_cls
            return scheduler_cls

        return register_scheduler_cls

    @classmethod
    def get_scheduler(cls: Type[SchedulerRegistry], name: str) -> Type[SchedulerConfig]:
        scheduler_cls = cls._schedulers.get(name)
        if not scheduler_cls:
            raise ValueError(f"Scheduler {name} not found in registry")
        return scheduler_cls

    @classmethod
    def create_scheduler(
        cls: Type[SchedulerRegistry], name: str, optimizer: torch.optim.Optimizer, **kwargs: Any
    ) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler_cls = cls.get_scheduler(name)
        scheduler_config = scheduler_cls(**kwargs)
        return scheduler_config.build(optimizer)


class SchedulerConfig(BaseModel, ABC):
    """Base class for creating PyTorch scheduler instances dynamically."""

    @abstractmethod
    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        """Builder method for creating a scheduler instance."""

    class Config:
        extra = "forbid"


@SchedulerRegistry.register("StepLR")
class StepLRConfig(SchedulerConfig):
    step_size: int
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma, last_epoch=self.last_epoch, verbose=self.verbose
        )


@SchedulerRegistry.register("CosineAnnealingLR")
class CosineAnnealingLRConfig(SchedulerConfig):
    T_max: int
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.eta_min, last_epoch=self.last_epoch, verbose=self.verbose
        )


@SchedulerRegistry.register("CosineAnnealingWarmRestarts")
class CosineAnnealingWarmRestartsConfig(SchedulerConfig):
    T_0: int
    T_mult: int = 1
    eta_min: float = 0
    last_epoch: int = -1
    verbose: bool = False

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch,
            verbose=self.verbose,
        )


@SchedulerRegistry.register("LambdaLR")
class LambdaLRConfig(SchedulerConfig):
    lr_lambda: Callable[[int], float]
    last_epoch: int = -1
    verbose: bool = False

    def build(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.lr_lambda, last_epoch=self.last_epoch, verbose=self.verbose
        )


if __name__ == "__main__":
    # Create a dummy optimizer for demonstration
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pprint(SchedulerRegistry._schedulers)

    # Create a StepLR scheduler
    step_lr = SchedulerRegistry.create_scheduler("StepLR", optimizer, step_size=30, gamma=0.1)
    print(f"Created StepLR scheduler: {step_lr}")

    # Create a CosineAnnealingLR scheduler
    cosine_lr = SchedulerRegistry.create_scheduler("CosineAnnealingLR", optimizer, T_max=100, eta_min=0.001)
    print(f"Created CosineAnnealingLR scheduler: {cosine_lr}")

    # Create a LambdaLR scheduler
    lambda_lr = SchedulerRegistry.create_scheduler("LambdaLR", optimizer, lr_lambda=lambda epoch: 0.95**epoch)
    print(f"Created LambdaLR scheduler: {lambda_lr}")
