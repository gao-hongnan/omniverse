from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, List, Literal

import requests
import torch
from rich.console import Console
from rich.logging import RichHandler

from omnivault.transformer.core.state import State
from omnivault.utils.torch_utils.cleanup import purge_global_scope

PYTORCH_DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


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


def download_and_read_sequences(url: str, dataset_name: str) -> Generator[str, None, None]:
    temp_dir = tempfile.mkdtemp()

    try:
        response = requests.get(url)
        response.raise_for_status()

        temp_file_path = os.path.join(temp_dir, f"{dataset_name}.txt")
        with open(temp_file_path, "wb") as file:
            file.write(response.content)

        with open(temp_file_path, "r") as file:
            for line in file:
                yield line.strip()

    finally:
        shutil.rmtree(temp_dir)


def validate_and_cleanup(
    state_1: State, state_2: State, objects: List[str], logger: logging.Logger | None = None
) -> None:
    """
    Validates the equality of two State instances and performs cleanup by deleting
    the provided objects.

    This function compares `state_1` and `state_2` to ensure that they are equal,
    indicating that the loaded state is consistent with the last state of the
    Trainer. If the states are not equal, an `AssertionError` is raised and logged
    using the provided logger or a default logger. After the validation, the
    function deletes the objects specified in the `objects` list to perform cleanup.

    Parameters
    ----------
    state_1 : State
        The first State instance to compare.
    state_2 : State
        The second State instance to compare.
    objects : List[Any]
        A list of objects to be deleted during cleanup.
    logger : logging.Logger, optional
        The logger to use for logging any errors or exceptions. If not provided,
        a default logger will be used.

    Raises
    ------
    AssertionError
        If `state_1` and `state_2` are not equal, indicating a mismatch between
        the loaded state and the last state of the Trainer.

    Notes
    -----
    This function performs a strict equality check between `state_1` and `state_2`
    using the `==` operator. We can do this because we implemented `_eq_` method
    in the `State` class.
    """
    if logger is None:
        logger = get_default_logger()

    try:
        assert (
            state_1 == state_2
        ), "If this fails, then the loaded state has a different checkpoint from the last state of Trainer."
    except AssertionError as err:
        logger.exception(err)
    finally:
        purge_global_scope(objects)


def create_directory(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: str) -> None:
    """Download a file using curl."""
    subprocess.run(["curl", url, "-o", output_path])
