import functools
import inspect
from collections.abc import Container, Iterable, Sized
from functools import lru_cache
from typing import Any, Generic, Iterator, Protocol, TypeVar, cast, overload

# NOTE: inside frostbound

T = TypeVar("T")
T_co = TypeVar("T_co", bound=Any, covariant=True)
U = TypeVar("U")


class SupportsContainer(Protocol):
    """Protocol for objects that support container operations."""

    def __iter__(self) -> Iterator[Any]: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: Any) -> bool: ...


class SupportsGetItem(Protocol):
    """Protocol for objects that support item access."""

    def __getitem__(self, key: Any) -> Any: ...


"""Methods that are considered safe (non-mutating) for common container types"""
_SAFE_LIST_METHODS = frozenset(["__iter__", "__len__", "__contains__", "__getitem__", "count", "index"])
_SAFE_DICT_METHODS = frozenset(
    [
        "__iter__",
        "__len__",
        "__contains__",
        "__getitem__",
        "get",
        "items",
        "keys",
        "values",
    ]
)
_SAFE_SET_METHODS = frozenset(["__iter__", "__len__", "__contains__", "issubset", "issuperset"])


class ImmutableT(Generic[T_co]):
    """Type for immutable proxy objects that wrap another object."""

    _obj: T_co


class ImmutableProxy(ImmutableT[T_co]):
    """Immutable proxy object for wrapping mutable objects.

    This class creates an immutable view of a mutable object by intercepting
    attribute access and method calls. Any attempt to modify the wrapped object
    will raise an AttributeError.

    Attributes returned from the proxy are themselves wrapped in an ImmutableProxy
    if they are mutable container types to prevent modification at any level.

    Example
    -------
    >>> obj = ImmutableProxy([1, 2, 3])
    >>> obj[0]  # Read access works
    1
    >>> obj.append(4)  # Raises AttributeError
    Traceback (most recent call last):
        ...
    AttributeError: Attempting to modify object with method `append`. `list` object is immutable.
    """

    def __init__(self, obj: T_co) -> None:
        """Initialize the immutable proxy with the object to wrap.

        Parameters
        ----------
        obj : T_co
            The object to wrap with an immutable proxy
        """
        object.__setattr__(self, "_obj", obj)

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> Any: ...

    def __getitem__(self, index: int | slice) -> Any:
        """Support indexing operations on the wrapped object.

        Parameters
        ----------
        index : Union[int, slice]
            The index to access

        Returns
        -------
        Any
            The value at the given index, possibly wrapped in an ImmutableProxy
        """
        if not hasattr(self._obj, "__getitem__"):
            raise TypeError(f"'{type(self._obj).__name__}' object is not subscriptable")

        obj = cast(SupportsGetItem, self._obj)
        result = obj[index]
        return self._wrap_if_needed(result)

    def __iter__(self) -> Iterator[Any]:
        """Support iteration over the wrapped object.

        Returns
        -------
        Iterable
            An iterator over the wrapped object
        """
        if not hasattr(self._obj, "__iter__"):
            raise TypeError(f"'{type(self._obj).__name__}' object is not iterable")

        obj = cast(Iterable, self._obj)
        for item in obj:
            yield self._wrap_if_needed(item)

    def __len__(self) -> int:
        """Support len() on the wrapped object.

        Returns
        -------
        int
            The length of the wrapped object
        """
        if not hasattr(self._obj, "__len__"):
            raise TypeError(f"object of type '{type(self._obj).__name__}' has no len()")

        obj = cast(Sized, self._obj)
        return len(obj)

    def __contains__(self, item: Any) -> bool:
        """Support 'in' operator on the wrapped object.

        Parameters
        ----------
        item : Any
            The item to check for containment

        Returns
        -------
        bool
            True if the item is in the wrapped object, False otherwise
        """
        if not hasattr(self._obj, "__contains__"):
            if hasattr(self._obj, "__iter__"):
                return any(val == item for val in cast(Iterable, self._obj))
            raise TypeError(f"'{type(self._obj).__name__}' object is not a container")

        obj = cast(Container, self._obj)
        return item in obj

    def __getattr__(self, item: str) -> Any:
        """Handle attribute access on the wrapped object.

        This method intercepts attribute access and method calls, blocking any
        potentially mutating operations and wrapping returned attributes if needed.

        Parameters
        ----------
        item : str
            The attribute name to access

        Returns
        -------
        Any
            The attribute value, possibly wrapped in an ImmutableProxy

        Raises
        ------
        AttributeError
            If attempting to access a mutating method
        """
        attr = getattr(self._obj, item)

        # NOTE: If it's a method, check if it's potentially mutating
        if callable(attr):
            obj_type = type(self._obj).__name__

            # NOTE: Check if the method is safe based on the object type
            is_safe = False
            if (
                (isinstance(self._obj, list) and item in _SAFE_LIST_METHODS)
                or (isinstance(self._obj, dict) and item in _SAFE_DICT_METHODS)
                or (isinstance(self._obj, set) and item in _SAFE_SET_METHODS)
            ):
                is_safe = True

            # NOTE: If the method is safe, wrap it to return immutable results
            if is_safe:

                @functools.wraps(attr)
                def safe_method(*args: Any, **kwargs: Any) -> Any:
                    result = attr(*args, **kwargs)
                    return self._wrap_if_needed(result)

                return safe_method

            # NOTE: Otherwise, block the potentially mutating method
            @functools.wraps(attr)
            def immutable_method(*_args: Any, **_kwargs: Any) -> None:
                raise AttributeError(
                    f"Attempting to modify object with method `{item}`. `{obj_type}` object is immutable."
                )

            return immutable_method

        # For non-callable attributes, wrap if needed
        return self._wrap_if_needed(attr)

    def __setattr__(self, key: str, value: Any) -> None:
        """Block attribute assignment.

        Parameters
        ----------
        key : str
            The attribute name
        value : Any
            The value to assign

        Raises
        ------
        AttributeError
            Always raised to prevent attribute modification
        """
        raise AttributeError(
            f"Attempting to set attribute `{key}` with `{value}`. `{type(self._obj).__name__}` object is immutable."
        )

    def __repr__(self) -> str:
        """Create a string representation of the immutable proxy.

        Returns
        -------
        str
            A string representation of the immutable proxy
        """
        return f"ImmutableProxy({self._obj!r})"

    def __str__(self) -> str:
        """Create a string representation of the wrapped object.

        Returns
        -------
        str
            A string representation of the wrapped object
        """
        return str(self._obj)

    def __eq__(self, other: object) -> bool:
        """Compare the wrapped object with another object.

        Parameters
        ----------
        other : Any
            The object to compare with

        Returns
        -------
        bool
            True if the wrapped object equals the other object, False otherwise
        """
        if isinstance(other, ImmutableProxy):
            return bool(self._obj == other._obj)
        return bool(self._obj == other)

    def __hash__(self) -> int:
        """Compute a hash value for the immutable proxy.

        Returns
        -------
        int
            A hash value for the immutable proxy

        Raises
        ------
        TypeError
            If the wrapped object is not hashable
        """
        return hash(self._obj)

    @classmethod
    @lru_cache(maxsize=128)
    def _is_mutable_container(cls, obj: Any) -> bool:
        """Check if an object is a mutable container type.

        Parameters
        ----------
        obj : Any
            The object to check

        Returns
        -------
        bool
            True if the object is a mutable container, False otherwise
        """
        if isinstance(obj, list | dict | set):
            return True

        if hasattr(obj, "__dict__") and not inspect.isclass(obj):
            mutable_method_patterns = [
                "add",
                "append",
                "extend",
                "insert",
                "remove",
                "pop",
                "clear",
                "update",
                "set",
                "delete",
            ]

            for name in dir(obj):
                if (name.startswith("__") and name.endswith("__")) or name.startswith("_"):
                    continue

                attr = getattr(obj, name)
                if callable(attr) and any(pattern in name for pattern in mutable_method_patterns):
                    return True

        return False

    def _wrap_if_needed(self, obj: Any) -> Any:
        """Recursively wrap mutable containers in ImmutableProxy.

        Parameters
        ----------
        obj : Any
            The object to potentially wrap

        Returns
        -------
        Any
            The original object or a wrapped version if it's mutable
        """
        if obj is None or isinstance(obj, int | float | str | bool | bytes | tuple | frozenset):
            return obj

        if self._is_mutable_container(obj):
            return ImmutableProxy(obj)

        return obj


def make_immutable(obj: T) -> ImmutableT[T]:
    """Create an immutable view of a mutable object.

    This is a convenience function for creating ImmutableProxy instances.

    Parameters
    ----------
    obj : T
        The object to make immutable

    Returns
    -------
    ImmutableT[T]
        An immutable proxy for the object
    """
    return ImmutableProxy(obj)
