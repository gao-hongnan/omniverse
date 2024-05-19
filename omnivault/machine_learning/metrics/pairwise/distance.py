# NOTE: unit-tested
"""Even though the module is called pairwise, the functions are not pairwise.
This will be implemented in due time, so for now it just expects two 1D arrays."""
from typing import Any

import numpy as np
from numpy.typing import NDArray


def manhattan_distance(x_1: NDArray[np.floating[Any]], x_2: NDArray[np.floating[Any]]) -> float:
    r"""
    Calculate the Manhattan Distance between two data points.

    .. note::
        This is only 1D implementation of the Manhattan Distance and therefore
        is not pairwise. For example, if you have two data points X and Y
        with X = [[1, 2], [3, 4]] and Y = [[5, 6], [7, 8]], a pairwise Manhattan
        Distance would calculate the distance between the first data point in X
        and the first data point in Y, the distance between the first data point
        in X and the second data point in Y, the distance between the second data
        point in X and the first data point in Y, and the distance between the
        second data point in X and the second data point in Y.

    The Manhattan Distance is defined as the sum of the absolute differences
    between the coordinates of two vectors and it extends to N-dimensional
    Euclidean Space. It is formally defined as:

    .. math::

        L1(a, b) = \sum_{i=1}^{n} |a_i - b_i|

    Parameters
    ----------
    x_1 : NDArray[np.floating[Any]]
        A numpy array representing the first data point.
    x_2 : NDArray[np.floating[Any]]
        A numpy array representing the second data point.

    Returns
    -------
    float
        The Manhattan distance between the two data points.

    Examples
    --------
    >>> x_1 = np.array([1, 2, 3])
    >>> x_2 = np.array([2, 3, 5])
    >>> manhattan_distance(x_1, x_2)
    4
    """
    _manhattan_distance = np.sum(np.abs(x_1 - x_2))
    return float(_manhattan_distance)


def euclidean_distance(x_1: NDArray[np.floating[Any]], x_2: NDArray[np.floating[Any]], squared: bool = False) -> float:
    r"""
    Euclidean Distance measures the length of the line segment bewteen two points
    in the Euclidean space. This is also referred to as L2 distance or L2 vector norm.

    .. math::

        L2(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}


    This function generalizes to any number of dimensions:

    For example, for two points in 3D space (x1, y1, z1) and (x2, y2, z2),
    the Euclidean distance is:

    .. math::

        \sqrt{(x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2}

    Parameters
    ----------
    x_1 : NDArray[np.floating[Any]]
        An N-dimensional numpy array representing the first data point.
    x_2 : NDArray[np.floating[Any]]
        An N-dimensional numpy array representing the second data point.
    squared : bool, optional
        If True, return the squared Euclidean distance.

    Returns
    -------
    float
        The Euclidean distance (or squared Euclidean distance, if `squared` is True)
        between the two data points.

    Examples
    --------
    >>> x_1 = np.array([1, 2, 3])
    >>> x_2 = np.array([2, 3, 5])
    >>> euclidean_distance(x_1, x_2)
    2.23606797749979
    >>> euclidean_distance(x_1, x_2, squared=True)
    5.0
    """
    _euclidean_distance = np.sum(np.square(x_1 - x_2)) if not squared else np.sqrt(np.sum(np.square(x_1 - x_2)))
    return float(_euclidean_distance)


def cosine_similarity(x_1: NDArray[np.floating[Any]], x_2: NDArray[np.floating[Any]]) -> float:
    r"""
    Calculate the cosine similarity between two vectors.

    The cosine similarity measures the cosine of the angle between two vectors
    projected in a multi-dimensional space. The cosine similarity is particularly
    used in high-dimensional positive spaces. It is defined as:

    .. math::

        \text{cosine similarity} = \frac{\mathbf{x}_1 \cdot \mathbf{x}_2}
                                        {||\mathbf{x}_1|| \cdot ||\mathbf{x}_2||}

    where \(\mathbf{x}_1\) and \(\mathbf{x}_2\) are vector representations.

    Parameters
    ----------
    x_1 : NDArray[np.floating[Any]]
        An N-dimensional numpy array representing the first vector.
    x_2 : NDArray[np.floating[Any]]
        An N-dimensional numpy array representing the second vector.

    Returns
    -------
    float
        The cosine similarity between the two vectors.

    Raises
    ------
    ValueError
        If either of the vectors is zero-length.
    AssertionError
        If the norm of x_1 as calculated does not match the Euclidean distance
        from x_1 to the origin.

    Examples
    --------
    >>> x_1 = np.array([1, 2, 3])
    >>> x_2 = np.array([2, 3, 4])
    >>> cosine_similarity(x_1, x_2)
    0.9925833339709303
    """
    if np.linalg.norm(x_1) == 0 or np.linalg.norm(x_2) == 0:
        raise ValueError("Cosine similarity is undefined for zero-length vectors.")

    numerator = np.dot(x_1, x_2)
    origin = np.zeros(shape=(x_1.shape))  # origin is a vector of zeros
    norm_x1 = np.linalg.norm(x_1)
    norm_x2 = np.linalg.norm(x_2)

    np.testing.assert_allclose(norm_x1, euclidean_distance(x_1, origin, squared=True))

    denominator = norm_x1 * norm_x2
    _cosine_similarity = numerator / denominator
    return float(_cosine_similarity)


def cosine_distance(x_1: NDArray[np.floating[Any]], x_2: NDArray[np.floating[Any]]) -> float:
    r"""
    Calculate the cosine distance between two vectors.

    Cosine distance is defined as 1 minus the cosine similarity:

    .. math::

        \text{cosine distance} = 1 - \text{cosine similarity}(\mathbf{x}_1, \mathbf{x}_2)

    Parameters
    ----------
    x_1 : NDArray[np.floating[Any]]
        An N-dimensional numpy array representing the first vector.
    x_2 : NDArray[np.floating[Any]]
        An N-dimensional numpy array representing the second vector.

    Returns
    -------
    float
        The cosine distance between the two vectors.

    Examples
    --------
    >>> x_1 = np.array([1, 2, 3])
    >>> x_2 = np.array([2, 3, 4])
    >>> cosine_distance(x_1, x_2)
    0.007416666029069703
    """
    return 1 - cosine_similarity(x_1, x_2)
