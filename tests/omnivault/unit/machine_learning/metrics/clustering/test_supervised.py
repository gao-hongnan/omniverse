from typing import Any

import numpy as np
import pytest
import sklearn.metrics.cluster
from numpy.typing import NDArray

from omnivault.machine_learning.metrics.clustering.supervised import contingency_matrix, purity_score


@pytest.fixture
def y_trues() -> NDArray[np.integer[Any]]:
    return np.array([1, 1, 2, 2, 1, 1, 2, 2])


@pytest.fixture
def y_preds() -> NDArray[np.integer[Any]]:
    return np.array([1, 1, 1, 1, 2, 2, 2, 2])


def test_contingency_matrix(y_trues: NDArray[np.integer[Any]], y_preds: NDArray[np.integer[Any]]) -> None:
    actual = contingency_matrix(y_trues, y_preds, as_dataframe=False)
    expected = sklearn.metrics.cluster.contingency_matrix(y_trues, y_preds)
    np.testing.assert_array_equal(actual, expected, err_msg="The contingency matrix is incorrect.")


def test_purity_score_overall(y_trues: NDArray[np.integer[Any]], y_preds: NDArray[np.integer[Any]]) -> None:
    # Expected result computed manually
    # Cluster 1: True label 1 (2 points), label 2 (2 points) - max is 2
    # Cluster 2: True label 1 (2 points), label 2 (2 points) - max is 2
    # Purity = (2 + 2) / 8 = 0.5
    expected_purity = 0.5
    actual_purity = purity_score(y_trues, y_preds, per_cluster=False)
    assert actual_purity == expected_purity


def test_purity_score_per_cluster(y_trues: NDArray[np.integer[Any]], y_preds: NDArray[np.integer[Any]]) -> None:
    # Expected results computed manually
    # Cluster 1 purity: max(2, 2) / 4 = 0.5
    # Cluster 2 purity: max(2, 2) / 4 = 0.5
    expected_purities = np.array([0.5, 0.5])

    actual_purities = purity_score(y_trues, y_preds, per_cluster=True)
    np.testing.assert_array_equal(actual_purities, expected_purities, err_msg="The purity scores are incorrect.")
