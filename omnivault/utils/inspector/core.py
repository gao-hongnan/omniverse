from __future__ import annotations

import inspect
from inspect import Parameter, Signature
from typing import Any, Callable, Dict, List, Set, Tuple, Type, get_type_hints, overload


@overload
def get_members_of_function_or_method(
    func_or_class: Type[object], predicate: Callable[[Any], bool] | None = None
) -> List[Tuple[str, Any]]:
    ...


@overload
def get_members_of_function_or_method(
    func_or_class: Callable[..., Any], predicate: Callable[[Any], bool] | None = None
) -> List[Tuple[str, Any]]:
    ...


def get_members_of_function_or_method(
    func_or_class: Type[object] | Callable[..., Any], predicate: Callable[[Any], bool] | None = None
) -> List[Tuple[str, Any]]:
    return inspect.getmembers(func_or_class, predicate)


def get_base_classes(cls: Type[Any], include_self: bool = False) -> Set[Type[Any]]:
    """
    Get the base classes of a class and all its base classes.
    """
    return set(cls.__mro__[0:-1] if include_self else cls.__mro__[1:-1])


def get_default(param: Parameter) -> Any:
    """Return the parameter's default value or None if not specified."""
    return param.default if param.default is not param.empty else None


def get_field_annotations(func_or_method: Callable[..., Any]) -> Tuple[List[Tuple[str, Any, Any]], Dict[str, Any]]:
    if not inspect.isroutine(func_or_method):
        raise ValueError("Expected a function or method")

    required_fields = []
    optional_fields = []
    annotations = {}

    try:
        sig: Signature = inspect.signature(func_or_method)
        type_hints: Dict[str, Any] = get_type_hints(func_or_method)
    except ValueError:
        raise ValueError("Object does not support signature or type hints extraction.") from None

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        type_hint = type_hints.get(name, Any)
        annotations[name] = type_hint
        if param.default is param.empty:
            required_fields.append((name, type_hint, Ellipsis))
        else:
            default_value = param.default
            optional_fields.append((name, type_hint, default_value))

    fields = required_fields + optional_fields
    return fields, annotations


# TODO: Tuple[str, str, Any, Any] should be Tuple[str, str, Any, ellipsis]
def get_constructor_field_annotations(
    cls: Type[Any], include_bases: bool = True
) -> Tuple[List[Tuple[str, str, Any, Any]], Dict[str, Any]]:
    fields = []
    annotations = {}

    classes_to_inspect = [cls] + list(get_base_classes(cls, include_self=False)) if include_bases else [cls]

    for c in reversed(classes_to_inspect):  # Reverse to respect MRO
        if hasattr(c, "__init__"):
            class_fields, class_annotations = get_field_annotations(c.__init__)
            # Update fields and annotations with those from the current class,
            # avoiding duplicates.
            class_name = f"{c.__module__}.{c.__name__}"
            annotations[class_name] = class_annotations

            for name, type_hint, default in class_fields:
                fields.append((class_name, name, type_hint, default))

    return fields, annotations
