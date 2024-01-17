from __future__ import annotations

import logging
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler


def format_lr(lr_or_lrs: float | List[float], precision: int) -> str:
    format_str = f"%.{precision}f"
    if isinstance(lr_or_lrs, list):
        return ", ".join([format_str % lr for lr in lr_or_lrs])
    return format_str % lr_or_lrs


def process_history(history: Dict[str, List[float]], plot: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    steps_df = pd.DataFrame({k: v for k, v in history.items() if "step" in k})
    epochs_df = pd.DataFrame({k: v for k, v in history.items() if "epoch" in k})

    if plot:
        fig, axes = plt.subplots(2)
        steps_df.plot(ax=axes[0], title="Steps")
        epochs_df.plot(ax=axes[1], title="Epochs")
        plt.tight_layout()
        plt.show()  # type: ignore[no-untyped-call]

    return steps_df, epochs_df


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
