# mypy: disable-error-code="no-untyped-call, import-untyped"

from typing import List

import ipywidgets
from IPython.display import SVG, display


def create_tabbed_svg_viewer(image_paths: List[str], tab_titles: List[str]) -> ipywidgets.Tab:
    """
    Create a tabbed image viewer widget.

    Parameters
    ----------
    image_paths: List[str]
        A list containing the paths to the images to be displayed in the viewer.
    tab_titles : List[str]
        A list containing the titles for each tab in the viewer.

    Returns
    -------
    ipywidgets.Tab
        An IPython widget Tab instance with each tab containing an image from the
        provided paths and titled according to the `tab_titles` list.
    """
    if len(image_paths) != len(tab_titles):
        raise ValueError("image_paths and tab_titles must have the same length.")

    outputs = [ipywidgets.Output() for _ in image_paths]

    for output, image_path in zip(outputs, image_paths):
        with output:
            display(SVG(filename=image_path))

    tab = ipywidgets.Tab()
    tab.children = outputs

    for index, tab_title in enumerate(tab_titles):
        tab.set_title(index, str(tab_title))

    return tab
