from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

ItemT = TypeVar("ItemT")
ItemT_co = TypeVar("ItemT_co", covariant=True)
ItemT_contra = TypeVar("ItemT_contra", contravariant=True)

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
MappingKeyT = TypeVar("MappingKeyT")
MappingValueT = TypeVar("MappingValueT")

ResultT = TypeVar("ResultT")

NodeT = TypeVar("NodeT")
GraphT = TypeVar("GraphT")


type Real = float | int
type SearchResult = int
type Numeric = float | int
type HungarianAssignment = list[tuple[int, int]]


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...
    def __le__(self, other: Any, /) -> bool: ...
    def __gt__(self, other: Any, /) -> bool: ...
    def __ge__(self, other: Any, /) -> bool: ...


def _comparable_pydantic_schema(cls: type, source_type: object, handler: object) -> object:  # noqa: ARG001
    from pydantic_core import core_schema  # noqa: PLC0415

    return core_schema.any_schema()


Comparable.__get_pydantic_core_schema__ = classmethod(_comparable_pydantic_schema)  # type: ignore[attr-defined]


ComparableT = TypeVar("ComparableT", bound=Comparable)
KeyT_co = TypeVar("KeyT_co", bound=Comparable, covariant=True)
ValueT_co = TypeVar("ValueT_co", covariant=True)
PriorityT = TypeVar("PriorityT", bound=Comparable)

NumericT = TypeVar("NumericT", int, float)

VertexT = TypeVar("VertexT")
VertexT_contra = TypeVar("VertexT_contra", contravariant=True)
EdgeWeightT = TypeVar("EdgeWeightT")
