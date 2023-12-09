from matplotlib_inline import backend_inline  # type: ignore


def use_svg_display() -> None:
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats("svg")
