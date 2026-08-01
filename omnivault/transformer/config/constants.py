from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["MaybeConstant"]


class MaybeConstant(BaseModel):
    """The maybe constant config that allows arbitrary fields. Not type safe
    for sure! So have to use type ignore if mypy cannot locate dynamically
    generated fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, **arbitrary: Any) -> None:
        super().__init__(**arbitrary)
