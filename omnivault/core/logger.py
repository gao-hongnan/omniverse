"""Logger class for logging to console and file.

This module should be refactored one day.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from omnivault._types._sentinel import MISSING, Singleton

DEFAULT_CONSOLE = Console(
    theme=Theme(
        {
            "logging.level.debug": "magenta",
            "logging.level.info": "green",
            "logging.level.warning": "yellow",
            "logging.level.error": "red",
            "logging.level.critical": "bold red",
            # "logging.level.remark": "bold blue",
        }
    )
)


class CustomFormatter(logging.Formatter):
    """This class overrides logging.Formatter's pathname to be relative path."""

    def format(self, record: logging.LogRecord) -> str:
        record.pathname = os.path.relpath(record.pathname)
        return super().format(record)


# NOTE: quirks = https://github.com/Textualize/rich/issues/459
@dataclass(frozen=False)
class RichLogger(metaclass=Singleton):
    """
    Class for logger. Consider using singleton design to maintain only one
    instance of logger (i.e., shared logger).

    1. Factory Pattern: In terms of design patterns, this Logger class follows the 'Factory'
       pattern by creating and returning logging handlers and formatters.
       For more justification, just realize that factory pattern is a creational pattern
       which provides an interface for creating objects in a superclass, but
       allows subclasses to alter the type of objects that will be created.

       For more info, see my design pattern notes.

    Areas for improvement:

    1. Consider implementing a Singleton pattern to ensure only one
       instance of Logger is used throughout the application.

    2. Consider adding thread safety to ensure that logs from different
       threads don't interfere with each other.

    3. The logger could be further extended to support other types of
       logging, such as sending logs to an HTTP endpoint.


    Example:
        Logger(
            log_file="pipeline_training.log",
            module_name=__name__,
            level=logging.INFO,
            propagate=False,
            log_root_dir="/home/user/logs",
        )

        --> produces the below tree structure, note the log_root_dir is the root
            while the session_log_dir is the directory for the current session.

        /home/
        │
        └───user/
            │
            └───logs/                        # This is the log_root_dir
                │
                └───2023-06-14T10:20:30/     # This is the session_log_dir
                    │
                    └───pipeline_training.log


    Parameters
    ----------
    log_file : str | None, default=None
        The name of the log file. Logs will be written to this file if specified.
        It must be specified if `log_root_dir` is specified.
    module_name : str | None, default=None
        The name of the module. This is useful for multi-module logging.
    propagate : bool, default=False
        Whether to propagate the log message to parent loggers.
    log_root_dir : str | None, default=None
        The root directory for all logs. If specified, a subdirectory will be
        created in this directory for each logging session, and the log file will
        be created in the subdirectory. Must be specified if `log_file` is specified.

    Attributes
    ----------
    session_log_dir : str | Path | None
        The directory for the current logging session. This is a subdirectory
        within `log_root_dir` that is named with the timestamp of when the logger
        was created.
    logger : logging.Logger
        The logger instance.

    Raises
    ------
    AssertionError
        Both `log_file` and `log_root_dir` must be provided, or neither should be provided.
    """

    log_file: str | None = None
    module_name: str | None = None
    propagate: bool = False
    log_root_dir: str | None = None
    rich_handler_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "level": "INFO",  # logging.INFO,
            "console": MISSING,
            "show_level": True,
            "show_path": True,
            "show_time": True,
            "rich_tracebacks": True,
            "markup": True,
            "log_time_format": "[%Y-%m-%d %H:%M:%S]",
        }
    )

    session_log_dir: str | Path | None = field(default=None, init=False)
    logger: logging.Logger = field(init=False)

    _initialized: bool = field(init=False)

    def __post_init__(self) -> None:
        if bool(self.log_file) != bool(self.log_root_dir):
            raise AssertionError("Both log_file and log_root_dir must be provided, or neither should be provided.")

        if not hasattr(self, "_initialized"):  # no-ops if `_initialized`.
            self._initialized = True
            if not self.rich_handler_config.get("console") or self.rich_handler_config["console"] is MISSING:
                self.rich_handler_config["console"] = DEFAULT_CONSOLE
            self.logger = self._init_logger()

    def _create_log_output_dir(self) -> Path:
        try:
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            assert self.log_root_dir is not None
            session_log_dir = Path(self.log_root_dir) / current_time
            Path(session_log_dir).mkdir(parents=True, exist_ok=True)
            return session_log_dir
        except OSError as err:
            raise OSError(f"Failed to create log directory: {err}") from err

    def _get_log_file_path(self) -> Path | None:
        if self.log_root_dir is not None and self.log_file is not None:
            self.session_log_dir = self._create_log_output_dir()
            return self.session_log_dir / self.log_file
        return None

    def _create_stream_handler(self) -> RichHandler:
        stream_handler = RichHandler(**self.rich_handler_config)

        # FIXME: If you set custom formatter, it will duplicate level and time.
        # stream_handler.setFormatter(
        #     CustomFormatter(
        #         "%(asctime)s [%(levelname)s] %(pathname)s %(funcName)s L%(lineno)d: %(message)s",
        #         "%Y-%m-%d %H:%M:%S",
        #     )
        # )
        return stream_handler

    def _create_file_handler(self, log_file_path: Path) -> logging.FileHandler:
        file_handler = logging.FileHandler(filename=str(log_file_path))
        file_handler.setLevel(level=logging.DEBUG)
        file_handler.setFormatter(
            CustomFormatter(
                "%(asctime)s [%(levelname)s] %(pathname)s %(funcName)s L%(lineno)d: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )
        return file_handler

    def _init_logger(self) -> logging.Logger:
        # get module name, useful for multi-module logging
        logger = logging.getLogger(self.module_name or __name__)
        logger.setLevel(self.rich_handler_config["level"])  # set root level

        logger.addHandler(self._create_stream_handler())

        log_file_path = self._get_log_file_path()

        if log_file_path:
            logger.addHandler(self._create_file_handler(log_file_path))

        logger.propagate = self.propagate
        return logger
