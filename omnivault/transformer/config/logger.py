from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator

from omnivault._types._sentinel import MISSING


class LoggerConfig(BaseModel):
    """The data config."""

    log_file: str | None = None
    module_name: str | None = None
    propagate: bool = False
    log_root_dir: str | None = None
    rich_handler_config: dict[str, Any] = Field(
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
    def check_log_root_dir(cls: type[Self], v: str | None) -> str | None:
        if v is not None:
            path = Path(v)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        return v
