from typing import Any, Protocol, Self, runtime_checkable


@runtime_checkable
class Fittable(Protocol):
    def fit(self, *args: Any, **kwargs: Any) -> Self: ...


@runtime_checkable
class Predictable(Protocol):
    def predict(self, *args: Any, **kwargs: Any) -> Any: ...
