from __future__ import annotations

from typing import Any, TypeVar

import pytest
from pydantic import BaseModel

T = TypeVar("T")


class TestData(BaseModel):
    integers: list[int]
    floats: list[float]
    strings: list[str]
    mixed: list[Any]
    empty: list[Any]


@pytest.fixture
def test_data() -> TestData:
    return TestData(
        integers=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
        floats=[1.1, 2.2, 3.3, 4.4, 5.5],
        strings=["apple", "banana", "cherry", "date", "elderberry"],
        mixed=[1, "two", 3.0, True, None],
        empty=[],
    )


@pytest.fixture
def large_dataset() -> list[int]:
    return list(range(10000))


@pytest.fixture
def random_ints() -> list[int]:
    import random

    random.seed(42)
    return [random.randint(1, 1000) for _ in range(100)]


@pytest.fixture
def sorted_ints() -> list[int]:
    return list(range(1, 101))


@pytest.fixture
def reverse_sorted_ints() -> list[int]:
    return list(range(100, 0, -1))


@pytest.fixture
def duplicate_ints() -> list[int]:
    return [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
