from typing import Any


class ImmutableProxy:
    """Immutable proxy object for wrapping mutable objects.

    Example
    -------
    >>> from omnivault.utils.general import ImmutableProxy
    >>> obj = ImmutableProxy([1, 2, 3])
    >>> obj.append(4)
    Traceback (most recent call last):
        ...
        AttributeError: Attempting to set attribute append with <built-in method append of list object at xxx>. list object is immutable.
    """

    def __init__(self, obj: object) -> None:
        object.__setattr__(self, "_obj", obj)

    def __getattr__(self, item: str) -> Any:
        attr = getattr(self._obj, item)
        if callable(attr):

            def immutable_method(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
                raise AttributeError(
                    f"Attempting to modify object with method `{item}`. "
                    f"`{self._obj.__class__.__name__}` object is immutable."
                )

            return immutable_method
        return attr

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"Attempting to set attribute `{key}` with `{value}`. "
            f"`{self._obj.__class__.__name__}` object is immutable."
        ) from None
