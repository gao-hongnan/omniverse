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

from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt

from omnivault.linear_algebra.base import VectorPlotter
from omnivault.linear_algebra.vector import Vector, Vector2D, Vector3D


def add_vectors_to_plotter(plotter: VectorPlotter, vectors: List[Vector]) -> None:
    for vector in vectors:
        plotter.add_vector(vector)


def add_text_annotations(
    plotter: VectorPlotter,
    vectors: List[Vector],
    include_endpoint_label: bool = True,
    include_vector_label: bool = True,
    endpoint_kwargs: Optional[Dict[str, Any]] = None,
    vector_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    endpoint_kwargs = endpoint_kwargs or {"fontsize": 12}
    vector_kwargs = vector_kwargs or {"fontsize": 12}

    for vector in vectors:
        if include_endpoint_label:
            # Label with endpoint coordinates
            x_end = vector.origin[0] + vector.direction[0]
            y_end = vector.origin[1] + vector.direction[1]
            label = f"({x_end}, {y_end})"
            plotter.add_text(x=x_end, y=y_end, text=label, **endpoint_kwargs)

        if include_vector_label and vector.label:
            # Label with vector label
            mid_point = (
                vector.origin[0] + vector.direction[0] / 2,
                vector.origin[1] + vector.direction[1] / 2,
            )
            plotter.add_text(
                x=mid_point[0], y=mid_point[1], text=vector.label, **vector_kwargs
            )


class VectorPlotter2D(VectorPlotter[Vector2D]):
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

        self.vectors: List[Vector2D] = []
        self.colors: List[str] = []

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        fontsize: int = 16,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.ax.text(x, y, text, fontsize=fontsize, **kwargs)

    def annotate(
        self,
        x: float,
        y: float,
        text: str,
        arrow_props: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y),
            arrowprops=arrow_props,
            fontsize=16,
            **kwargs,  # type: ignore[arg-type]
        )

    def add_vector(self, vector: Vector2D) -> None:
        self.vectors.append(vector)

    def plot(self, grid: bool = True, show_ticks: bool = False) -> None:
        for vector in self.vectors:
            # fmt: off
            X, Y = vector.origin    # pylint: disable=invalid-name
            U, V = vector.direction # pylint: disable=invalid-name
            # fmt: on
            self.ax.quiver(X, Y, U, V, color=vector.color, **self.quiver_kwargs)

        if grid:
            self.ax.grid()

        if not show_ticks:
            self.ax.tick_params(axis="both", which="both", length=0)

    def save(
        self,
        path: str,
        *,
        dpi: Union[float, str] = "figure",
        format="svg",  # pylint: disable=redefined-builtin
        **kwargs: Dict[str, Any],
    ) -> None:
        self.fig.savefig(path, dpi=dpi, format=format, **kwargs)  # type: ignore[arg-type]

class VectorPlotter3D(VectorPlotter[Vector3D]):
    pass
    # def __init__(
    #     self,
    #     fig: Optional[plt.Figure] = None,
    #     ax: Optional[plt.Axes] = None,
    #     ax_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    #     quiver_kwargs: Optional[Dict[str, Any]] = None,
    # ) -> None:
    #     super().__init__(fig, ax, ax_kwargs)

    #     self.quiver_kwargs = quiver_kwargs or {
    #         "angles": "xy",
    #         "scale_units": "xy",
    #         "scale": 1,
    #         "alpha": 0.6,
    #     }

    #     self.vectors: List[Vector3D] = []
    #     self.colors: List[str] = []

    # def add_text(
    #     self,
    #     x: float,
    #     y: float,
    #     z: float,
    #     text: str,
    #     fontsize: int = 16,
    #     **kwargs: Dict[str, Any],
    # ) -> None:
    #     self.ax.text(x, y, z, text, fontsize=fontsize, **kwargs)

    # def annotate(
    #     self,
    #     x: float,
    #     y: float,
    #     z: float,
    #     text: str,
    #     arrow_props: Optional[Dict[str, Any]] = None,
    #     **kwargs: Dict[str, Any],
    # ) -> None:
    #     self.ax.annotate(
    #         text,
    #         xy=(x, y, z),
    #         xytext=(x, y, z),
    #         arrowprops=arrow_props,
    #         fontsize=16,
    #         **kwargs,  # type: ignore[arg-type]
    #     )

    # def add_vector(self, vector: Vector3D) -> None:
    #     self.vectors.append(vector)

    # def plot(self, grid: bool = True, show_ticks: bool = False) -> None:
    #     for vector in self.vectors:
    #         # fmt: off
    #         X, Y, Z = vector.origin    # pylint: disable=invalid-name
    #         U, V, W = vector.direction # pylint: disable=invalid-name
    #         # fmt: on
    #         self.ax.quiver(X, Y, Z, U, V, W, color=vector.color, **self.quiver_kwargs)

    #     if grid:
    #         self.ax.grid()

    #     if not show_ticks:
    #         self.ax.tick_params(axis="both", which="both", length=0)

    # def save(
    #     self, path: str, *, dpi: Union[float, str] = "figure", **kwargs: Dict[str, Any]
    # ) -> None:
    #     self.fig.savefig(path, dpi=dpi, **kwargs)
