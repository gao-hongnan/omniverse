from typing import Any


class MaybeConstant:
    """The maybe constant config that allows arbitrary fields. Not type safe
    for sure! So have to use type ignore if mypy cannot locate dynamically
    generated fields."""

    def __init__(self, **arbitrary: Any) -> None:
        for key, value in arbitrary.items():
            setattr(self, key, value)
