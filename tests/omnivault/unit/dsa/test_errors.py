from __future__ import annotations

import pytest


def test_dsa_error_is_exception_subclass() -> None:
    from omnivault.dsa.core.errors import DSAError

    assert issubclass(DSAError, Exception)
    assert DSAError is not Exception


def test_key_not_found_co_inherits_key_error() -> None:
    from omnivault.dsa.core.errors import DSAError, KeyNotFound

    err = KeyNotFound("missing")
    assert isinstance(err, KeyError)
    assert isinstance(err, DSAError)
    assert isinstance(err, Exception)


def test_empty_container_co_inherits_index_error() -> None:
    from omnivault.dsa.core.errors import DSAError, EmptyContainer

    err = EmptyContainer("empty")
    assert isinstance(err, IndexError)
    assert isinstance(err, DSAError)


def test_cycle_detected_co_inherits_runtime_error() -> None:
    from omnivault.dsa.core.errors import CycleDetected, DSAError

    err = CycleDetected("cycle")
    assert isinstance(err, RuntimeError)
    assert isinstance(err, DSAError)


def test_value_error_family_co_inherits() -> None:
    from omnivault.dsa.core.errors import DSAError, IncompatibleSketch, InvalidProbability, RectangularityViolation

    for cls in (RectangularityViolation, IncompatibleSketch, InvalidProbability):
        err = cls("msg")
        assert isinstance(err, ValueError)
        assert isinstance(err, DSAError)


def test_pytest_raises_builtin_captures_subclass() -> None:
    from omnivault.dsa.core.errors import KeyNotFound

    with pytest.raises(KeyError):
        raise KeyNotFound("x")


def test_import_from_package_root() -> None:
    from omnivault.dsa import (
        CycleDetected,
        DSAError,
        EmptyContainer,
        IncompatibleSketch,
        InvalidProbability,
        KeyNotFound,
        RectangularityViolation,
    )

    assert all(
        isinstance(cls, type)
        for cls in (
            DSAError,
            KeyNotFound,
            EmptyContainer,
            CycleDetected,
            RectangularityViolation,
            IncompatibleSketch,
            InvalidProbability,
        )
    )
