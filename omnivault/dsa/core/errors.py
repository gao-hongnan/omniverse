from __future__ import annotations


class DSAError(Exception):
    pass


class KeyNotFound(DSAError, KeyError):
    pass


class EmptyContainer(DSAError, IndexError):
    pass


class CycleDetected(DSAError, RuntimeError):
    pass


class RectangularityViolation(DSAError, ValueError):
    pass


class IncompatibleSketch(DSAError, ValueError):
    pass


class InvalidProbability(DSAError, ValueError):
    pass


class InvalidConfiguration(DSAError, ValueError):
    pass


class UnsupportedDataType(DSAError, ValueError):
    pass
