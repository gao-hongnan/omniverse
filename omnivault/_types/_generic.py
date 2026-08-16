"""
This file contains the generic type variables shared across the book's code cells.

`T` is intentionally unconstrained and invariant: the DSA chapters use it to
parameterise containers that both accept and return the same element type.
"""

from typing import TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
