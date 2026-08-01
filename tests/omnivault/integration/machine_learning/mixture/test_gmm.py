import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from omnivault.machine_learning.mixture.gmm import GaussianMixtureModel


@pytest.mark.parametrize(argnames="num_components", argvalues=[3])
def test_gmm_vs_sklearn(num_components: int) -> None:
    X_raw = make_blobs(
        n_samples=1000,
        centers=num_components,
        n_features=2,
        random_state=1992,
        cluster_std=1.5,
    )[0]
    X: NDArray[np.float64] = np.asarray(X_raw, dtype=np.float64)

    gmm = GaussianMixtureModel(num_components=num_components, init="random", max_iter=100, random_state=42)
    gmm.fit(X)

    sklearn_gmm = GaussianMixture(
        n_components=num_components, max_iter=100, init_params="random_from_data", random_state=42
    )
    sklearn_gmm.fit(X)

    # Fitted attributes are typed as `ndarray | None`; normalize to arrays for sorting.
    custom_means = np.asarray(gmm.means_)
    sklearn_means = np.asarray(sklearn_gmm.means_)
    assert np.allclose(np.sort(custom_means, axis=0), np.sort(sklearn_means, axis=0), atol=0.1), (
        f"Means are different: {np.sort(custom_means, axis=0)} vs {np.sort(sklearn_means, axis=0)}"
    )

    custom_covariances = np.asarray(gmm.covariances_)
    sklearn_covariances = np.asarray(sklearn_gmm.covariances_)
    assert np.allclose(np.sort(custom_covariances, axis=0), np.sort(sklearn_covariances, axis=0), atol=0.1), (
        f"Covariances are different: {np.sort(custom_covariances, axis=0)} vs {np.sort(sklearn_covariances, axis=0)}"
    )

    custom_weights = np.asarray(gmm.weights_)
    sklearn_weights = np.asarray(sklearn_gmm.weights_)
    assert np.allclose(np.sort(custom_weights), np.sort(sklearn_weights), atol=0.1), (
        f"Weights are different: {gmm.weights_} vs {sklearn_gmm.weights_}"
    )
