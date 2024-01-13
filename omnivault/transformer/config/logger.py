from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type, Union

from pydantic import BaseModel, Field, field_validator

from omnivault._types._sentinel import MISSING


class LoggerConfig(BaseModel):
    """The data config."""

    log_file: Union[str, None] = None
    module_name: Union[str, None] = None
    propagate: bool = False
    log_root_dir: Union[str, None] = None
    rich_handler_config: Dict[str, Any] = Field(
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

    @field_validator("log_root_dir")
    @classmethod
    def check_log_root_dir(cls: Type[LoggerConfig], v: str) -> str:
        if v is not None:
            path = Path(v)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        return v
