import pytest
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from omnivault.transformer.config.scheduler import CosineAnnealingLRConfig, StepLRConfig


@pytest.fixture
def optimizer() -> torch.optim.SGD:
    parameter = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([parameter], lr=0.1)


def test_step_lr_config_build_returns_scheduler(optimizer: torch.optim.SGD) -> None:
    """Regression: torch 2.13 removed the ``verbose`` kwarg from LRScheduler
    constructors, so the config must not declare or forward it."""
    config = StepLRConfig.model_validate({"name": "torch.optim.lr_scheduler.StepLR", "step_size": 10, "gamma": 0.5})

    scheduler = config.build(optimizer=optimizer)

    assert isinstance(scheduler, StepLR)
    assert scheduler.step_size == 10


def test_cosine_annealing_lr_config_build_returns_scheduler(optimizer: torch.optim.SGD) -> None:
    """Regression companion to the ``StepLR`` case for the second config that
    declared the removed ``verbose`` field."""
    config = CosineAnnealingLRConfig.model_validate(
        {"name": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": 100}
    )

    scheduler = config.build(optimizer=optimizer)

    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 100
