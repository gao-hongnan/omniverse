from __future__ import annotations

import inspect
from inspect import Parameter, Signature
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, Field


# Sentinel type for empty parameter default
class _Empty:
    """Sentinel for empty parameter default."""

    pass


EMPTY = _Empty()


class FieldInfo[T](BaseModel):
    """Information about a function/method parameter with proper typing."""

    name: str
    # Annotated as `object` rather than `type[object]`: a type hint is not
    # necessarily a class. `Dict[str, Any]`, `int | None` and `Literal["a"]` are
    # all valid hints but none of them pass pydantic's `is_subclass_of` check,
    # so `type[object]` would reject exactly the signatures this module exists
    # to inspect.
    type_hint: object | None = None
    default: T | _Empty = Field(default=EMPTY)
    is_required: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True


class FunctionSchema(BaseModel):
    """Schema for function/method signature with proper typing."""

    fields: list[FieldInfo[Any]] = Field(default_factory=list)
    annotations: dict[str, object] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ClassFieldInfo(BaseModel):
    """Information about a class field including its source class."""

    class_name: str
    field: FieldInfo[Any]

    class Config:
        arbitrary_types_allowed = True


class ClassSchema(BaseModel):
    """Schema for class constructor signature including inheritance."""

    fields: list[ClassFieldInfo] = Field(default_factory=list)
    annotations: dict[str, dict[str, object]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


def get_base_classes[T](cls: type[T], include_self: bool = False) -> set[type[T]]:
    """Get the base classes of a class and all its base classes.

    Args:
        cls: The class to inspect
        include_self: Whether to include the class itself in the result

    Returns:
        Set of base classes
    """
    return set(cls.__mro__[0:-1] if include_self else cls.__mro__[1:-1])


def get_function_schema[P](func_or_method: Callable[..., P]) -> FunctionSchema:
    """Extract function/method schema using Pydantic models.

    Args:
        func_or_method: The function or method to inspect

    Returns:
        FunctionSchema containing field information and annotations

    Raises:
        ValueError: If the object is not a function or method
    """
    if not inspect.isroutine(func_or_method):
        raise ValueError("Expected a function or method")

    try:
        sig: Signature = inspect.signature(func_or_method)
        type_hints: dict[str, object] = get_type_hints(func_or_method)
    except (ValueError, TypeError, NameError) as e:
        raise ValueError(f"Object does not support signature or type hints extraction: {e}") from e

    fields: list[FieldInfo[Any]] = []
    annotations: dict[str, object] = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Get the type hint, defaulting to object if not specified
        type_hint = type_hints.get(name, object)
        annotations[name] = type_hint

        # Convert Parameter.empty to our EMPTY sentinel
        default_value = EMPTY if param.default is Parameter.empty else param.default

        field_info: FieldInfo[Any] = FieldInfo(
            name=name, type_hint=type_hint, default=default_value, is_required=(param.default is Parameter.empty)
        )
        fields.append(field_info)

    return FunctionSchema(fields=fields, annotations=annotations)


def get_class_schema[T](cls: type[T], include_bases: bool = True) -> ClassSchema:
    """Extract class constructor schema using Pydantic models.

    Args:
        cls: The class to inspect
        include_bases: Whether to include fields from base classes

    Returns:
        ClassSchema containing field information from the class and optionally its bases
    """
    fields: list[ClassFieldInfo] = []
    annotations: dict[str, dict[str, object]] = {}

    classes_to_inspect = [cls] + list(get_base_classes(cls, include_self=False)) if include_bases else [cls]

    for c in reversed(classes_to_inspect):  # Reverse to respect MRO
        if hasattr(c, "__init__"):
            func_schema = get_function_schema(c.__init__)
            class_name = f"{c.__module__}.{c.__name__}"
            annotations[class_name] = func_schema.annotations

            for field in func_schema.fields:
                class_field = ClassFieldInfo(class_name=class_name, field=field)
                fields.append(class_field)

    return ClassSchema(fields=fields, annotations=annotations)
