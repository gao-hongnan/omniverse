from __future__ import annotations

import logging
import math
from typing import Dict, List, Literal

import matplotlib.pyplot as plt
from rich.console import Console
from rich.logging import RichHandler
import seaborn as sns


def format_lr(lr_or_lrs: float | List[float], precision: int) -> str:
    format_str = f"%.{precision}f"
    if isinstance(lr_or_lrs, list):
        return ", ".join([format_str % lr for lr in lr_or_lrs])
    return format_str % lr_or_lrs



def plot_history(history: Dict[str, List[float]]) -> None:
    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    num_metrics = len(history.keys())
    num_cols = 2
    num_rows = math.ceil(num_metrics / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    axs = axs.flatten()

    for i, metric in enumerate(history.keys()):
        axs[i].plot(history[metric])
        axs[i].set_title(metric)
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Remove unused subplots
    for i in range(num_metrics, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

def get_default_logger(logger_type: Literal["rich"] | None = None) -> logging.Logger:
    """
    Sets up and returns a logger with RichHandler. If an existing logger is provided,
    it returns the same logger without modifying it.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : str, optional
        Logging level, by default "INFO".
    logger : Optional[logging.Logger], optional
        An existing logger instance, by default None.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name=logger_type)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # To avoid duplicate logs in parent loggers

    if logger_type == "rich":
        # Setup for Rich logging
        console = Console()
        rich_handler = RichHandler(console=console, level="INFO", show_time=True, show_path=False, show_level=True)
        logger.addHandler(rich_handler)
    else:
        # Setup for basic logging
        basic_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(basic_formatter)
        logger.addHandler(stream_handler)
    return logger
