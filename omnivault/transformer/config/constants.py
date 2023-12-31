from typing import Any

from pydantic import BaseModel

__all__ = ["MaybeConstant"]


class MaybeConstant(BaseModel):
    """The maybe constant config that allows arbitrary fields. Not type safe
    for sure! So have to use type ignore if mypy cannot locate dynamically
    generated fields."""

    def __init__(self, **arbitrary: Any) -> None:
        super().__init__(**arbitrary)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "allow"
