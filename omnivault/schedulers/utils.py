from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["get_learning_rates", "plot_learning_rates"]


def get_learning_rates(optimizer: Optimizer, scheduler: _LRScheduler, steps: int) -> List[float]:
    """
    Retrieves a list of learning rates from an optimizer and scheduler over a
    specified number of steps.

    This function simulates 'steps' number of optimization steps, recording the
    learning rate at each step as dictated by the scheduler's policy.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer associated with the model parameters.
    scheduler : _LRScheduler
        The learning rate scheduler that adjusts the learning rate according to its policy.
    steps : int
        The number of steps to simulate and retrieve the learning rates for.

    Returns
    -------
    lrs : List[float]
        A list of learning rates corresponding to each step.

    Examples
    --------
    >>> from torch import nn
    >>> from torch.optim import Adam
    >>> from torch.optim.lr_scheduler import StepLR
    >>> dummy_model = nn.Linear(1, 1)
    >>> optimizer = Adam(dummy_model.parameters(), lr=1e-1)
    >>> scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    >>> lrs = get_learning_rates(optimizer, scheduler, steps=5)
    >>> print(lrs)
    [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    """
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def plot_learning_rates(
    lrs: List[float], title: str, marker: str = "o", ax: plt.Axes | None = None, **kwargs: Any
) -> None:
    """
    Plot learning rates on either a given Axes object or the current axes.

    Parameters
    ----------
    lrs : List[float]
        A list of learning rate values to plot.
    title : str
        The title of the plot.
    marker : str, optional
        The marker style for the plot, by default "o".
    ax : matplotlib.axes.Axes, optional
        The axes on which to draw the plot. If None, uses the current axes.
    **kwargs : dict
        Additional keyword arguments to pass to plt.plot.
    """
    ax = ax or plt.gca()

    ax.plot(lrs, label=title, marker=marker, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.legend()
