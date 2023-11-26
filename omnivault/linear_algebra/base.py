from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

from omnivault._types._generic import Vec
from omnivault.utils.visualization.figure_manager import FigureManager


class VectorPlotter(Generic[Vec], FigureManager, ABC):
    @abstractmethod
    def plot(self, grid: bool = True, show_ticks: bool = False) -> None:
        ...
