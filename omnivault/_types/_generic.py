from typing import TypeVar

from omnivault.linear_algebra.vector import Vector

# fmt: off
T       = TypeVar("T", covariant=False, contravariant=False)
Real    = TypeVar("Real", int, float, covariant=False, contravariant=False)
Complex = TypeVar("Complex", int, float, complex, covariant=False, contravariant=False)
Vec     = TypeVar("Vec", bound=Vector, covariant=False, contravariant=False)
# fmt: on
