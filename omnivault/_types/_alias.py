"""
This file contains aliases for common types.
"""

from omnivault._types._sentinel import _Missing, _NotGiven, _Omit

type NonNegativeInt = int
type PositiveInt = int
type Loss = float
type Accuracy = float
type Token = str
NotGiven = _NotGiven
Missing = _Missing
Omit = _Omit
