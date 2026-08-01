import math

import numpy as np
import pytest
import scipy.spatial.distance
from numpy.typing import NDArray

from omnivault.machine_learning.metrics.pairwise.distance import (
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)

type _VectorPair = tuple[NDArray[np.float64], NDArray[np.float64]]


@pytest.fixture
def vectors_1d() -> _VectorPair:
    return np.array([2.0, 3.0, 6.0]), np.array([1.0, 2.0, 3.0])


@pytest.fixture
def vectors_1d_with_origin() -> _VectorPair:
    return np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0])


def test_manhattan_distance(vectors_1d: _VectorPair) -> None:
    x_1, x_2 = vectors_1d
    expected: float = scipy.spatial.distance.minkowski(x_1, x_2, p=1)

    assert manhattan_distance(x_1, x_2) == expected


@pytest.mark.parametrize(
    argnames=("squared", "expected"),
    argvalues=[
        pytest.param(False, math.sqrt(11.0), id="squared-false-returns-root-distance"),
        pytest.param(True, 11.0, id="squared-true-returns-squared-distance"),
    ],
)
def test_euclidean_distance_squared(vectors_1d: _VectorPair, squared: bool, expected: float) -> None:
    x_1, x_2 = vectors_1d

    assert euclidean_distance(x_1, x_2, squared) == expected


def test_cosine_distance(vectors_1d: _VectorPair) -> None:
    x_1, x_2 = vectors_1d
    expected: float = scipy.spatial.distance.cosine(x_1, x_2)

    assert cosine_distance(x_1, x_2) == expected


def test_cosine_similarity(vectors_1d: _VectorPair) -> None:
    x_1, x_2 = vectors_1d
    expected: float = 1 - scipy.spatial.distance.cosine(x_1, x_2)

    assert cosine_similarity(x_1, x_2) == expected


def test_cosine_similarity_zero_vector_error(vectors_1d_with_origin: _VectorPair) -> None:
    x_1, x_2 = vectors_1d_with_origin

    with pytest.raises(ValueError, match="Cosine similarity is undefined for zero-length vectors."):
        cosine_similarity(x_1, x_2)
