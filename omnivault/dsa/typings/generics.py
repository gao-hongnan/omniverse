from typing import TypeVar

# fmt: off
T       = TypeVar("T", covariant=False, contravariant=False)
Real    = TypeVar("Real", int, float, covariant=False, contravariant=False)
Complex = TypeVar("Complex", int, float, complex, covariant=False, contravariant=False)
# fmt: on