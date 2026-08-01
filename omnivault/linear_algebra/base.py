"""Base class for vector plotter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omnivault.linear_algebra.vector import Vector
from omnivault.utils.visualization.figure_manager import FigureManager


class VectorPlotter[Vec: Vector](FigureManager, ABC):
    @abstractmethod
    def plot(self, grid: bool = True, show_ticks: bool = False) -> None: ...

    @abstractmethod
    def add_vector(self, vector: Vec) -> None: ...

    @abstractmethod
    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        z: float | None = None,
        fontsize: int = 16,
        **kwargs: Any,
    ) -> None: ...
