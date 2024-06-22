# from __future__ import annotations

# import logging
# import shutil
# from pathlib import Path
# from typing import Generator
# from unittest.mock import MagicMock, patch

# import pytest

# from omnivault.core.logger import RichLogger


# @pytest.fixture(scope="function")
# def log_dir() -> Generator[str, None, None]:
#     """
#     Fixture to create and remove a test log folder for tests.

#     Yields
#     ------
#     test_log_dir : str
#         The path of the test log folder.
#     """
#     test_log_dir: str = "test_outputs"
#     Path(test_log_dir).mkdir(parents=True, exist_ok=True)
#     yield test_log_dir
#     shutil.rmtree(test_log_dir)


# @pytest.mark.parametrize(
#     "module_name, propagate",
#     [
#         (None, False),
#         ("test_module", True),
#         ("test_module", False),
#     ],
# )
# def test_logger_init(log_dir: str, module_name: str | None, propagate: bool) -> None:
#     logger_obj: RichLogger = RichLogger(
#         log_file="test_log.txt",
#         module_name=module_name,
#         propagate=propagate,
#         log_root_dir=log_dir,
#     )

#     expected_level = logging.getLevelName(logger_obj.rich_handler_config["level"])
#     assert logger_obj.logger.level == expected_level
#     assert logger_obj.logger.propagate == propagate

#     with patch("omnivault.core.logger.__name__", "__main__"):
#         logger_obj = RichLogger(
#             log_file="test_log.txt",
#             module_name=module_name,
#             propagate=propagate,
#             log_root_dir=log_dir,
#         )

#     assert logger_obj.logger.name == (module_name or "__main__")

#     assert logger_obj.session_log_dir is not None
#     assert Path(logger_obj.session_log_dir).exists()
#     assert Path(logger_obj.session_log_dir).is_dir()
#     assert logger_obj.log_file is not None
#     log_file_path: Path = Path(logger_obj.session_log_dir) / Path(logger_obj.log_file)
#     assert log_file_path.exists()


# @pytest.mark.parametrize(
#     "message",
#     [
#         "Test info message",
#         "Test warning message",
#         "Test error message",
#         "Test critical message",
#     ],
# )
# def test_logger_messages(log_dir: str, message: str) -> None:
#     logger_obj: RichLogger = RichLogger(
#         log_file="test_log.txt",
#         module_name="test_module",
#         propagate=False,
#         log_root_dir=log_dir,
#     )

#     logger_obj.logger.log(logging.INFO, message)

#     assert logger_obj.session_log_dir is not None
#     assert logger_obj.log_file is not None

#     log_file_path: Path = Path(logger_obj.session_log_dir) / Path(logger_obj.log_file)
#     with log_file_path.open("r") as log_file:
#         log_content: str = log_file.read()
#         assert message in log_content
