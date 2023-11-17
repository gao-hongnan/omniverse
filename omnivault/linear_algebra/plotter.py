"""
This module provides utilities for creating quiver plots using Matplotlib.

A quiver plot is a type of plot that shows vector fields, which are useful for
visualizing velocity fields, gradients, or any other kind of directional data.
The plot consists of arrows (vectors) placed at various points in a 2D space.

Parameters for creating a quiver plot:
- X, Y: These arrays define the starting positions (origins) of the vectors.
  Each arrow in the plot originates from the coordinate (X[i], Y[i]). This
  represents the 'origin of the basis' for each vector in the field. X and Y
  should have the same shape and can be generated using meshgrid for regular
  grids or specified manually for irregular grids.
- U, V: These arrays represent the vector components at each point defined by X
  and Y. U[i], V[i] are the horizontal (x-direction) and vertical (y-direction)
  displacements from the origin (X[i], Y[i]). These determine the direction and
  magnitude (length) of each vector. The length of the vector is proportional
  to the magnitude of the displacement, calculated as sqrt(U[i]**2 + V[i]**2).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union

import matplotlib.pyplot as plt

from omnivault.utils.visualization.figure_manager import FigureManager


@dataclass
class Vector:
    origin: Tuple[float, float]
    direction: Tuple[float, float]
    label: Optional[str] = None
    color: Optional[str] = "black"


def add_vectors_to_plotter(plotter: VectorPlotter, vectors: List[Vector]):
    for vector in vectors:
        plotter.add_vector(vector)


def add_text_annotations(plotter: VectorPlotter, vectors: List[Vector]):
    # For first three vectors

    for vector in vectors:
        x_end = vector.origin[0] + vector.direction[0]
        y_end = vector.origin[1] + vector.direction[1]
        label = f"({x_end}, {y_end})"
        plotter.add_text(x=x_end, y=y_end, text=label, fontsize=16)

    # For vector operation labels
    labels = ["$u$", "$v$", "$u+v$"]
    for i, label in enumerate(labels):
        mid_point = (
            vectors[i].origin[0] + vectors[i].direction[0] / 2,
            vectors[i].origin[1] + vectors[i].direction[1] / 2,
        )
        plotter.add_text(x=mid_point[0], y=mid_point[1], text=label, fontsize=16)


class VectorPlotter(FigureManager):
    def __init__(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        ax_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        quiver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(fig, ax, ax_kwargs)

        self.quiver_kwargs = quiver_kwargs or {
            "angles": "xy",
            "scale_units": "xy",
            "scale": 1,
            "alpha": 0.6,
        }

        self.vectors = []
        self.colors = []

    def add_text(self, x, y, text, fontsize=16, **kwargs: Dict[str, Any]) -> None:
        self.ax.text(x, y, text, fontsize=fontsize, **kwargs)

    def annotate(self, x, y, text, arrow_props=None, **kwargs: Dict[str, Any]) -> None:
        self.ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y),
            arrowprops=arrow_props,
            fontsize=16,
            **kwargs,
        )

    def add_vector(self, vector: Vector) -> None:
        self.vectors.append(vector)

    def plot(self, grid: bool = True, show_ticks: bool = False) -> None:
        for vector in self.vectors:
            X, Y = vector.origin
            U, V = vector.direction
            self.ax.quiver(X, Y, U, V, color=vector.color, **self.quiver_kwargs)

        if grid:
            self.ax.grid()

        if not show_ticks:
            self.ax.tick_params(axis="both", which="both", length=0)

    def save(self, path: str, dpi: Union[float, str] = "figure") -> None:
        self.fig.savefig(path, dpi=dpi)
