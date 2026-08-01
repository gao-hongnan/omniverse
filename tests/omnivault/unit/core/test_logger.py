"""Tests for RichLogger.

``RichLogger`` is a ``Singleton`` (a second construction returns the first
instance, ignoring its arguments) and attaches handlers to process-global
``logging`` loggers — both are ambient state, so every test runs behind the
isolation fixture below.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from rich.logging import RichHandler

from omnivault._types._sentinel import Singleton
from omnivault.core.logger import RichLogger

_MODULE_LOGGER_NAME: Final = "omnivault-test-logger"
# Names of every process-global logger these tests attach handlers to;
# "omnivault.core.logger" is the fallback name when module_name is omitted.
_TOUCHED_LOGGER_NAMES: Final = (_MODULE_LOGGER_NAME, "omnivault.core.logger")


@pytest.fixture(autouse=True)
def _isolated_logging_state() -> Iterator[None]:
    """Reset the singleton registry and strip handlers the test attached.

    Without this, the first test's ``RichLogger`` instance (and its open file
    handlers on process-global loggers) leaks into every later test, making
    the file order-dependent and constructor arguments silently ignored.
    """
    registry = Singleton._instances  # type: ignore[misc]  # deliberate white-box access to the metaclass registry for isolation
    saved = dict(registry)
    registry.clear()
    yield
    registry.clear()
    registry.update(saved)
    for name in _TOUCHED_LOGGER_NAMES:
        touched_logger = logging.getLogger(name)
        for handler in touched_logger.handlers[:]:
            handler.close()
            touched_logger.removeHandler(handler)


def test_console_only_logger_configuration() -> None:
    """A logger built without file arguments logs to the console only."""
    rich_logger = RichLogger(module_name=_MODULE_LOGGER_NAME)

    assert rich_logger.logger.name == _MODULE_LOGGER_NAME
    assert rich_logger.logger.level == logging.INFO
    assert rich_logger.session_log_dir is None
    assert sum(isinstance(handler, RichHandler) for handler in rich_logger.logger.handlers) == 1
    assert not any(isinstance(handler, logging.FileHandler) for handler in rich_logger.logger.handlers)


@pytest.mark.parametrize("propagate", [pytest.param(True, id="propagate-on"), pytest.param(False, id="propagate-off")])
def test_propagate_flag_is_applied(propagate: bool) -> None:
    rich_logger = RichLogger(module_name=_MODULE_LOGGER_NAME, propagate=propagate)

    assert rich_logger.logger.propagate is propagate


def test_module_name_defaults_to_defining_module() -> None:
    rich_logger = RichLogger()

    assert rich_logger.logger.name == "omnivault.core.logger"


def test_rich_logger_is_a_singleton() -> None:
    first = RichLogger(module_name=_MODULE_LOGGER_NAME)

    second = RichLogger(module_name="ignored-on-second-construction")

    assert second is first
    assert second.logger.name == _MODULE_LOGGER_NAME


def test_file_logger_creates_session_dir_and_file(tmp_path: Path) -> None:
    """File logging creates a timestamped session dir under the root with the log file."""
    rich_logger = RichLogger(log_file="test_log.txt", module_name=_MODULE_LOGGER_NAME, log_root_dir=str(tmp_path))

    assert rich_logger.session_log_dir is not None
    session_dir = Path(rich_logger.session_log_dir)
    assert session_dir.is_dir()
    assert session_dir.parent == tmp_path
    assert (session_dir / "test_log.txt").is_file()
    assert sum(isinstance(handler, logging.FileHandler) for handler in rich_logger.logger.handlers) == 1


def test_logged_message_is_written_to_file(tmp_path: Path) -> None:
    rich_logger = RichLogger(log_file="test_log.txt", module_name=_MODULE_LOGGER_NAME, log_root_dir=str(tmp_path))

    rich_logger.logger.info("Message for the file handler")

    assert rich_logger.session_log_dir is not None
    log_content = (Path(rich_logger.session_log_dir) / "test_log.txt").read_text()
    assert "Message for the file handler" in log_content


def test_log_file_without_root_dir_raises() -> None:
    with pytest.raises(AssertionError, match="Both log_file and log_root_dir must be provided"):
        RichLogger(log_file="orphan.log", module_name=_MODULE_LOGGER_NAME)


def test_log_root_dir_without_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="Both log_file and log_root_dir must be provided"):
        RichLogger(log_root_dir=str(tmp_path), module_name=_MODULE_LOGGER_NAME)
