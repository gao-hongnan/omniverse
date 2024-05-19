from typing import Any, Tuple

import numpy as np
import pytest
import scipy.spatial.distance  # type: ignore[import-untyped]
from _pytest.fixtures import FixtureRequest
from numpy.typing import NDArray

from omnivault.machine_learning.metrics.pairwise.distance import (
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)


@pytest.fixture
def vectors_1d() -> Tuple[NDArray[Any], NDArray[Any]]:
    return np.array([2, 3, 6]), np.array([1, 2, 3])


@pytest.fixture
def vectors_1d_with_origin() -> Tuple[NDArray[Any], NDArray[Any]]:
    return np.array([0, 0, 0]), np.array([1, 2, 3])


@pytest.fixture
def vectors_2d() -> Tuple[NDArray[Any], NDArray[Any]]:
    return np.array([[0, 0], [0, 0]]), np.array([[1, 2], [3, 4]])


@pytest.fixture
def vectors_3d() -> Tuple[NDArray[Any], NDArray[Any]]:
    return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


@pytest.mark.parametrize("fixture", ["vectors_1d"])
def test_manhattan_distance(request: FixtureRequest, fixture: str) -> None:
    x_1, x_2 = request.getfixturevalue(fixture)
    expected = scipy.spatial.distance.minkowski(x_1, x_2, p=1)
    assert manhattan_distance(x_1, x_2) == expected, f"{fixture} test failed"


@pytest.mark.parametrize("fixture,squared", [("vectors_1d", True), ("vectors_1d", False)])
def test_euclidean_distance_squared(request: FixtureRequest, fixture: str, squared: bool) -> None:
    x_1, x_2 = request.getfixturevalue(fixture)
    expected = scipy.spatial.distance.minkowski(x_1, x_2, p=2)
    if not squared:
        expected = expected**2

    assert euclidean_distance(x_1, x_2, squared) == expected, f"{fixture} test failed"


@pytest.mark.parametrize("fixture", ["vectors_1d"])
def test_cosine_distance(request: FixtureRequest, fixture: str) -> None:
    x_1, x_2 = request.getfixturevalue(fixture)
    expected = scipy.spatial.distance.cosine(x_1, x_2)
    assert cosine_distance(x_1, x_2) == expected, f"{fixture} test failed"


@pytest.mark.parametrize("fixture", ["vectors_1d"])
def test_cosine_similarity(request: FixtureRequest, fixture: str) -> None:
    x_1, x_2 = request.getfixturevalue(fixture)
    expected = 1 - scipy.spatial.distance.cosine(x_1, x_2)
    assert cosine_similarity(x_1, x_2) == expected, f"{fixture} test failed"


@pytest.mark.parametrize("fixture", ["vectors_1d_with_origin"])
def test_cosine_similarity_zero_vector_error(request: FixtureRequest, fixture: str) -> None:
    x_1, x_2 = request.getfixturevalue(fixture)
    with pytest.raises(ValueError) as exc_info:
        cosine_similarity(x_1, x_2)
    assert "Cosine similarity is undefined for zero-length vectors." in str(
        exc_info.value
    ), "Error message does not match expected text"
