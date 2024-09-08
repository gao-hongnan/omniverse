import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from omnivault.machine_learning.mixture.gmm import GaussianMixtureModel


@pytest.mark.parametrize(argnames="num_components", argvalues=[3])
def test_gmm_vs_sklearn(num_components: int) -> None:
    X, y = make_blobs(
        n_samples=1000,
        centers=num_components,
        n_features=2,
        random_state=1992,
        cluster_std=1.5,
    )

    gmm = GaussianMixtureModel(num_components=num_components, init="random", max_iter=100, random_state=42)
    gmm.fit(X)

    sklearn_gmm = GaussianMixture(
        n_components=num_components, max_iter=100, init_params="random_from_data", random_state=42
    )
    sklearn_gmm.fit(X)

    assert np.allclose(
        np.sort(gmm.means_, axis=0), np.sort(sklearn_gmm.means_, axis=0), atol=0.1
    ), f"Means are different: {np.sort(gmm.means_, axis=0)} vs {np.sort(sklearn_gmm.means_, axis=0)}"
    assert np.allclose(
        np.sort(gmm.covariances_, axis=0), np.sort(sklearn_gmm.covariances_, axis=0), atol=0.1
    ), f"Covariances are different: {np.sort(gmm.covariances_, axis=0)} vs {np.sort(sklearn_gmm.covariances_, axis=0)}"
    assert np.allclose(
        np.sort(gmm.weights_), np.sort(sklearn_gmm.weights_), atol=0.1
    ), f"Weights are different: {gmm.weights_} vs {sklearn_gmm.weights_}"
