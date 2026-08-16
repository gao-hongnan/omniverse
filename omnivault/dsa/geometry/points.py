from __future__ import annotations

from typing import Generic, cast

from ..core.errors import IncompatibleSketch
from ..core.types import NumericT


class Point(Generic[NumericT]):  # noqa: UP046
    def __init__(self, coordinates: list[NumericT]) -> None:
        self.coordinates: list[NumericT] = coordinates
        self.dimension = len(coordinates)

    def __getitem__(self, index: int) -> NumericT:
        return self.coordinates[index]

    def __setitem__(self, index: int, value: NumericT) -> None:
        self.coordinates[index] = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Point):
            return self.coordinates == other.coordinates
        return False

    def __repr__(self) -> str:
        return f"Point({self.coordinates})"

    def distance_squared(self, other: Point[NumericT]) -> NumericT:
        if self.dimension != other.dimension:
            raise IncompatibleSketch("Points must have same dimension")

        total: NumericT = cast(NumericT, self.coordinates[0] * 0)
        for i in range(self.dimension):
            diff = self.coordinates[i] - other.coordinates[i]
            total = cast(NumericT, total + diff * diff)
        return total

    def distance(self, other: Point[NumericT]) -> float:
        return float(self.distance_squared(other) ** 0.5)


class Rectangle(Generic[NumericT]):  # noqa: UP046
    def __init__(self, min_point: Point[NumericT], max_point: Point[NumericT]) -> None:
        if min_point.dimension != max_point.dimension:
            raise IncompatibleSketch("Points must have same dimension")

        self.min_point: Point[NumericT] = min_point
        self.max_point: Point[NumericT] = max_point
        self.dimension = min_point.dimension

    def contains(self, point: Point[NumericT]) -> bool:
        if point.dimension != self.dimension:
            return False

        return all(not (point[i] < self.min_point[i] or point[i] > self.max_point[i]) for i in range(self.dimension))

    def intersects(self, other: Rectangle[NumericT]) -> bool:
        if self.dimension != other.dimension:
            return False

        for i in range(self.dimension):
            if self.max_point[i] < other.min_point[i] or self.min_point[i] > other.max_point[i]:
                return False
        return True

    def area(self) -> NumericT:
        if self.dimension != 2:
            raise IncompatibleSketch("Area calculation only supported for 2D rectangles")

        width = self.max_point[0] - self.min_point[0]
        height = self.max_point[1] - self.min_point[1]
        return cast(NumericT, width * height)

    def __repr__(self) -> str:
        return f"Rectangle({self.min_point}, {self.max_point})"
