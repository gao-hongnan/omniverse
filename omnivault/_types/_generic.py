from typing import TypeVar

from omnivault.linear_algebra.vector import Vector

# fmt: off
T            = TypeVar("T", covariant=False, contravariant=False)
T_co         = TypeVar('T_co', covariant=True)
K            = TypeVar("K", covariant=False, contravariant=False)
V            = TypeVar("V", covariant=False, contravariant=False)
K_co         = TypeVar("K_co", covariant=True)
V_co         = TypeVar("V_co", covariant=True)
Real         = TypeVar("Real", int, float, covariant=False, contravariant=False)
Complex      = TypeVar("Complex", int, float, complex, covariant=False, contravariant=False)
Vec          = TypeVar("Vec", bound=Vector, covariant=False, contravariant=False)
DynamicClass = TypeVar("DynamicClass", covariant=False, contravariant=False) # bound=object?
# fmt: on
