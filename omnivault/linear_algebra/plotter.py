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

NOTE
----
All `quiver` calls only assumes that its takes in float values instead of
`1D` or `2D` array-like objects. This means that the `X`, `Y`, `U`, and `V`
are floats unpacked from our `Vector` objects. We then use a for loop to
iterate over the vectors and plot them one by one. Maybe consider FIXME
this to support vectorized plotting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt

from omnivault._types._generic import Vec
from omnivault.linear_algebra.base import VectorPlotter
from omnivault.linear_algebra.vector import Vector2D, Vector3D


def add_vectors_to_plotter(plotter: VectorPlotter[Vec], vectors: Sequence[Vec]) -> None:
    """Add vectors to a plotter.

    Type
    ----
    The reason we use `Vec` instead of `Vector` is because `Vec` is a type
    variable bounded to `Vector` and thus can represent `Vector` and any
    of its subclasses. This means `Vec` is an upper bound for `Vector` and
    its subclasses.
    """
    for vector in vectors:
        plotter.add_vector(vector)


def add_text_annotations(
    plotter: VectorPlotter[Vec],
    vectors: Sequence[Vec],
    include_endpoint_label: bool = True,
    include_vector_label: bool = True,
    endpoint_kwargs: Optional[Dict[str, Any]] = None,
    vector_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    endpoint_kwargs = endpoint_kwargs or {"fontsize": 12}
    vector_kwargs = vector_kwargs or {"fontsize": 12}

    for vector in vectors:
        dim = len(vector)  # Determine if the vector is 2D or 3D

        # endpoint = tuple(map(sum, zip(vector.origin, vector.direction)))
        if include_endpoint_label:
            # Label with endpoint coordinates
            x_end = vector.origin[0] + vector.direction[0]
            y_end = vector.origin[1] + vector.direction[1]
            if dim == 3:
                z_end = vector.origin[2] + vector.direction[2]
                label = f"({x_end}, {y_end}, {z_end})"
                plotter.add_text(x=x_end, y=y_end, z=z_end, text=label, **endpoint_kwargs)
            else:
                label = f"({x_end}, {y_end})"
                plotter.add_text(x=x_end, y=y_end, text=label, **endpoint_kwargs)

        if include_vector_label and vector.label:
            midpoint = tuple(origin + direction / 2 for origin, direction in zip(vector.origin, vector.direction))
            # Label with vector label
            if dim == 3:
                plotter.add_text(
                    x=midpoint[0],
                    y=midpoint[1],
                    z=midpoint[2],
                    text=vector.label,
                    **vector_kwargs,
                )
            else:  # dim == 2
                plotter.add_text(x=midpoint[0], y=midpoint[1], text=vector.label, **vector_kwargs)


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

        self._apply_ax_customizations()  # may not need since in FigureManager

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        z: Optional[float] = None,  # noqa: ARG002
        fontsize: int = 16,
        **kwargs: Any,
    ) -> None:
        self.ax.text(x, y, text, fontsize=fontsize, **kwargs)

    def annotate(
        self,
        x: float,
        y: float,
        text: str,
        arrow_props: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
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
        """Currently only works in notebooks, if in script, please add
        `plt.show()` after calling this method."""
        for vector in self.vectors:
            # fmt: off
            X, Y = vector.origin
            U, V = vector.direction
            # fmt: on
            self.ax.quiver(X, Y, U, V, color=vector.color, **self.quiver_kwargs)

        if grid:
            self.ax.grid()

        if not show_ticks:
            self.ax.tick_params(axis="both", which="both", length=0)


class VectorPlotter3D(VectorPlotter[Vector3D]):
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

    def __init__(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        ax_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        quiver_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(fig, ax, ax_kwargs)

        self.quiver_kwargs = quiver_kwargs or {
            "length": 1,
            "normalize": False,
            "alpha": 0.6,
            "arrow_length_ratio": 0.18,
            "pivot": "tail",
            "linestyles": "solid",
            "linewidths": 3,
        }

        self.vectors: List[Vector3D] = []
        self.colors: List[str] = []

        self._apply_ax_customizations()  # may not need since in FigureManager

    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        z: Optional[float] = None,
        fontsize: int = 16,
        **kwargs: Any,
    ) -> None:
        self.ax.text(x, y, z, text, fontsize=fontsize, **kwargs)  # type: ignore[arg-type]

    def annotate(
        self,
        x: float,
        y: float,
        z: float,
        text: str,
        arrow_props: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.ax.annotate(
            text,
            xy=(x, y, z),  # type: ignore[arg-type]
            xytext=(x, y, z),  # type: ignore[arg-type]
            arrowprops=arrow_props,
            fontsize=16,
            **kwargs,  # type: ignore[arg-type]
        )

    def add_vector(self, vector: Vector3D) -> None:
        self.vectors.append(vector)

    def plot(self, grid: bool = True, show_ticks: bool = False) -> None:
        for vector in self.vectors:
            # fmt: off
            X, Y, Z = vector.origin
            U, V, W = vector.direction
            # fmt: on
            self.ax.quiver(X, Y, Z, U, V, W, color=vector.color, **self.quiver_kwargs)

        if grid:
            self.ax.grid()

        if not show_ticks:
            self.ax.tick_params(axis="both", which="both", length=0)
