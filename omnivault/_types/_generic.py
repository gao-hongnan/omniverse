from typing import Any, Generic, TypeVar

from numpy.typing import NDArray

from omnivault.linear_algebra.vector import Vector

# fmt: off
T            = TypeVar("T", covariant=False, contravariant=False)
_T           = TypeVar("_T", covariant=False, contravariant=False) # noqa: PYI018
T_co         = TypeVar('T_co', covariant=True)
T_obj        = TypeVar('T_obj', bound=object, covariant=False, contravariant=False)
K            = TypeVar("K", covariant=False, contravariant=False)
V            = TypeVar("V", covariant=False, contravariant=False)
K_co         = TypeVar("K_co", covariant=True)
V_co         = TypeVar("V_co", covariant=True)
Real         = TypeVar("Real", int, float, covariant=False, contravariant=False)
Complex      = TypeVar("Complex", int, float, complex, covariant=False, contravariant=False)
Vec          = TypeVar("Vec", bound=Vector, covariant=False, contravariant=False)
DynamicClass = TypeVar("DynamicClass", covariant=False, contravariant=False) # bound=object?
# fmt: on

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class Array(NDArray[Any], Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]

    Reference: https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
    """
