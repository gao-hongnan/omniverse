from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray

# indicates that the input type and the output type are the same
# i.e. predict(self, X: T) -> T means torch.Tensor -> torch.Tensor
NumpyTorch = TypeVar("NumpyTorch", NDArray[np.floating[Any]], torch.Tensor)


@runtime_checkable
class Fittable(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Fittable: ...


@runtime_checkable
class Predictable(Protocol):
    def predict(self, *args: Any, **kwargs: Any) -> Any: ...
