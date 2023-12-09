"""
TODO
----
1. Strategy Design Pattern?
2. Integrate with FigureManager.
"""

from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray


def plot_line(
    ax: plt.Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    **kwargs: Any,
) -> List[Line2D]:
    """Plot line."""
    return ax.plot(x, y, **kwargs)


def plot_quiver(
    ax: plt.Axes,  # This can be either a 2D or 3D Axes object
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    z: Optional[NDArray[np.float64]] = None,  # For 3D
    w: Optional[NDArray[np.float64]] = None,  # For 3D
    **kwargs: Any,
) -> Union[Quiver, mpl_toolkits.mplot3d.art3d.Line3DCollection]:
    """Plot quiver (2D or 3D depending on the ax type)."""
    if isinstance(ax, Axes3D):
        if z is None or w is None:
            raise ValueError("z and w must be provided for 3D quivers")
        return ax.quiver(x, y, z, u, v, w, **kwargs)
    if isinstance(ax, Axes):
        return ax.quiver(x, y, u, v, **kwargs)
    raise ValueError("Invalid Axes type for quiver plot")


def plot_hist(ax: plt.Axes, x: NDArray[np.float64], **kwargs: Any) -> BarContainer:
    """Plot histogram."""
    return ax.hist(x, **kwargs)  # type: ignore


def plot_bar(ax: plt.Axes, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> BarContainer:
    """Plot bar plot."""
    return ax.bar(x, y, **kwargs)


def plot_scatter(ax: plt.Axes, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> PathCollection:
    """Plot scatter plot."""

    return ax.scatter(x, y, **kwargs)
