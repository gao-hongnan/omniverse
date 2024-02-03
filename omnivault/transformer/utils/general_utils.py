from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Generator, List, Literal

import requests
import torch
from rich.console import Console
from rich.logging import RichHandler

from omnivault.transformer.core.state import State

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


def cleanup(object_or_objects: Any | List[Any]) -> None:
    """
    Deletes the provided objects and performs cleanup.

    Parameters
    ----------
    objects: List[Any]
        The list of objects to be deleted.
    """
    if isinstance(object_or_objects, list):
        for obj in object_or_objects:
            del obj
    else:
        del object_or_objects

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_and_cleanup(
    state_1: State, state_2: State, objects: List[Any], logger: logging.Logger | None = None
) -> None:
    """
    Deletes the provided objects and performs cleanup.

    Parameters
    ----------
    state_1: State
    state_2: State
    objects: List[Any]
        The list of objects to be deleted.
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
        cleanup(objects)


def create_directory(path: str) -> None:
    """Create a directory if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: str) -> None:
    """Download a file using curl."""
    subprocess.run(["curl", url, "-o", output_path])
