"""
Reference:
- https://peps.python.org/pep-0661/
- https://github.com/taleinat/python-stdlib-sentinels/blob/main/sentinels/sentinels.py
"""

from __future__ import annotations

import sys as _sys
from threading import Lock as _Lock
from types import FrameType
from typing import Tuple, Type

__all__ = ["Sentinel"]

# Design and implementation decisions:
#
# The first implementations created a dedicated class for each instance.
# However, once it was decided to use Sentinel for type signatures, there
# was no longer a need for a dedicated class for each sentinel value on order
# to enable strict type signatures.  Since class objects consume a relatively
# large amount of memory, the implementation was changed to avoid this.
#
# With this change, the mechanism used for unpickling/copying objects needed
# to be changed too, since we could no longer count on each dedicated class
# simply returning its singleton instance as before.  __reduce__ can return
# a string, upon which an attribute with that name is looked up in the module
# and returned.  However, that would have meant that pickling/copying support
# would depend on the "name" argument being exactly the name of the variable
# used in the module, and simply wouldn't work for sentinels created in
# functions/methods.  Instead, a registry for sentinels was added, where all
# sentinel objects are stored keyed by their name + module name.  This is used
# to look up existing sentinels both during normal object creation and during
# copying/unpickling.


class Sentinel:
    """Create a unique sentinel object.

    *name* should be the fully-qualified name of the variable to which the
    return value shall be assigned.

    *repr*, if supplied, will be used for the repr of the sentinel object.
    If not provided, "<name>" will be used (with any leading class names
    removed).

    *module_name*, if supplied, will be used instead of inspecting the call
    stack to find the name of the module from which
    """

    _name: str
    _repr: str
    _module_name: str

    def __new__(  # noqa: PYI034
        cls: Type[Sentinel],
        name: str,
        repr: str | None = None,
        module_name: str | None = None,
    ) -> Sentinel:
        name = str(name)
        repr = str(repr) if repr else f'<{name.split(".")[-1]}>'
        if not module_name:
            parent_frame = _get_parent_frame()
            module_name = parent_frame.f_globals.get("__name__", "__main__") if parent_frame is not None else __name__

        # Include the class's module and fully qualified name in the
        # registry key to support sub-classing.
        registry_key = _sys.intern(f"{cls.__module__}-{cls.__qualname__}-{module_name}-{name}")
        sentinel = _registry.get(registry_key, None)
        if sentinel is not None:
            return sentinel
        sentinel = super().__new__(cls)
        sentinel._name = name
        sentinel._repr = repr
        sentinel._module_name = module_name
        with _lock:
            return _registry.setdefault(registry_key, sentinel)

    def __repr__(self) -> str:
        return self._repr

    def __reduce__(self) -> Tuple[Type[Sentinel], Tuple[str, str, str]]:
        return (
            self.__class__,
            (
                self._name,
                self._repr,
                self._module_name,
            ),
        )


_lock = _Lock()
_registry: dict[str, Sentinel] = {}


# The following implementation attempts to support Python
# implementations which don't support sys._getframe(2), such as
# Jython and IronPython.
#
# The version added to the stdlib may simply return sys._getframe(2),
# without the fallbacks.
#
# For reference, see the implementation of namedtuple:
# https://github.com/python/cpython/blob/67444902a0f10419a557d0a2d3b8675c31b075a9/Lib/collections/__init__.py#L503
def _get_parent_frame() -> FrameType | None:
    """Return the frame object for the caller's parent stack frame."""
    try:
        # Two frames up = the parent of the function which called this.
        return _sys._getframe(2)
    except (AttributeError, ValueError):
        global _get_parent_frame

        def _get_parent_frame() -> FrameType | None:
            """Return the frame object for the caller's parent stack frame."""
            try:
                raise Exception
            except Exception:
                try:
                    return _sys.exc_info()[2].tb_frame.f_back.f_back  # type: ignore[union-attr]
                except Exception:
                    global _get_parent_frame

                    def _get_parent_frame() -> None:
                        """Return the frame object for the caller's parent stack frame."""
                        return None

                    return _get_parent_frame()

        return _get_parent_frame()
