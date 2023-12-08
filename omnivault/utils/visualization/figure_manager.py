"""
This module provides the FigureManager class, a utility for managing
matplotlib figures and axes. It's designed to abstract and simplify the
process of figure and axes creation and customization in matplotlib, making
it easier to create and manage plots in a reusable manner.
"""

from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt


class FigureManager:
    """
    A utility class for managing matplotlib figures and axes.

    The class provides an interface to create a figure and an axes object,
    either by using existing ones or creating new ones. It also allows for
    the application of custom settings to the axes object through a
    dictionary of attributes and their corresponding parameters.
    """

    def __init__(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        ax_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        # fmt: off
        self.fig       = fig or plt.gcf()
        self.ax        = ax or plt.gca()
        self.ax_kwargs = ax_kwargs or {}
        # fmt: on

        self._apply_ax_customizations()

    def _apply_ax_customizations(self) -> None:
        """
        Apply custom settings to the axes object.

        This method iterates through the `ax_kwargs` dictionary and applies
        each attribute and its parameters to the axes object.
        """
        for ax_attr, ax_params in self.ax_kwargs.items():
            getattr(self.ax, ax_attr)(**ax_params)

    def show(self) -> None:
        """Show the figure."""
        if self.fig:
            self.fig.show()
        else:
            plt.show()  # type: ignore[no-untyped-call]

    def save(
        self,
        path: str,
        *,
        dpi: Union[float, str] = "figure",
        format: str = "svg",
        **kwargs: Dict[str, Any],
    ) -> None:
        self.fig.savefig(path, dpi=dpi, format=format, **kwargs)  # type: ignore[arg-type]
