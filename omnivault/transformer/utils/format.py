from __future__ import annotations

import logging
from typing import List

from rich.console import Console
from rich.logging import RichHandler


def format_lr(lr_or_lrs: float | List[float], precision: int) -> str:
    format_str = f"%.{precision}f"
    if isinstance(lr_or_lrs, list):
        return ", ".join([format_str % lr for lr in lr_or_lrs])
    return format_str % lr_or_lrs


def get_default_rich_logger(logger: logging.Logger | None = None) -> logging.Logger:
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
    if logger is None:
        console = Console()
        rich_handler = RichHandler(console=console, level="INFO", show_time=True, show_path=False, show_level=True)
        logger = logging.getLogger("rich")
        logger.setLevel(logging.INFO)
        logger.addHandler(rich_handler)
        logger.propagate = False  # To avoid duplicate logs in parent loggers
    return logger
