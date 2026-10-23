"""From-scratch implementations of machine learning and deep learning concepts."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("omniverse")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
