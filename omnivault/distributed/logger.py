from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def configure_logger(
    rank: int,
    log_dir: str | None = None,
    log_level: int = logging.INFO,
    log_on_master_or_all: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger for a given process rank.

    This function sets up a logger for the specified process rank, allowing for separate
    logging configurations based on the rank. The logger is configured to write logs to
    a file named `process_{rank}.log` and display logs with severity level INFO and above.

    Parameters
    ----------
    rank : int
        The rank of the process for which the logger is being configured.
    log_dir : str | None, optional
        The directory where the log files will be stored. If None, no log files will be created.
        Default is None.
    log_level : int, optional
        The log level for the logger. Default is logging.INFO.
    log_on_master_or_all : bool, optional
        Determines whether logs should be displayed only on the master rank (rank 0) or on all ranks.
        If True, logs will be displayed only on the master rank. If False, logs will be displayed
        on all ranks. Default is True.


    Returns
    -------
    logging.Logger
        Configured logger for the specified process rank.

    Notes
    -----
    -   The logger is configured to write logs to a file named `process_{rank}.log`
        in the specified `log_dir` directory. This allows for separate log files for
        each process rank.
    -   The reason for writing each rank's logs to a separate file is to avoid the
        non-deterministic ordering of log messages from different ranks in the same
        file.
    -   If `log_on_master_or_all` is True, logs will be displayed only on the master
        rank (rank 0). If False, logs will be displayed on all ranks.
    -   The file handler, if added, will always log messages from all ranks,
        regardless of the `log_on_master_or_all` setting.
    """
    logger = logging.getLogger(f"Process-{rank}")  # set name for each rank
    logger.setLevel(log_level)
    logger.propagate = False  # no propagation to parent

    logger.handlers = []  # clear existing handlers

    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # NOTE: # if log_on_master is False, it evaluates first part of conditional
    #         which will log on all ranks, else only on master rank.
    if not log_on_master_or_all or (log_on_master_or_all and rank == 0):
        console_handler = RichHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / Path(f"process_{rank}.log")
        file_handler = logging.FileHandler(filename=str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
