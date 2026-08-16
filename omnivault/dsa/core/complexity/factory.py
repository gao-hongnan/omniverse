from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ...containers.linear.linked_list import SinglyLinkedList
from ..errors import UnsupportedDataType

if TYPE_CHECKING:
    from .protocols import DataStructure

    SupportedDataTypes = list[int] | dict[int, int] | str | DataStructure | None
else:
    SupportedDataTypes = list[int] | dict[int, int] | str | None


class DataFactory:
    @staticmethod
    def create(data_type: str | None, size: int) -> SupportedDataTypes:
        factories: dict[str | None, Callable[[int], SupportedDataTypes]] = {
            "string": lambda n: "a" * n,
            "array": lambda n: list(range(n)),
            "dict": lambda n: {i: i for i in range(n)},
            "singly_linked_list": lambda n: cast("SupportedDataTypes", SinglyLinkedList(list(range(n)))),
            None: lambda _: None,
        }

        if data_type not in factories:
            raise UnsupportedDataType(f"Unsupported data_type: {data_type}. Supported types: {list(factories.keys())}")

        return factories[data_type](size)
