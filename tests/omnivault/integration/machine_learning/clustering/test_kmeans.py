import numpy as np
import pytest
import sklearn.cluster
from sklearn.datasets import make_blobs

from omnivault.machine_learning.clustering.kmeans import KMeansLloyd


@pytest.mark.parametrize("K", [3, 5])
def test_kmeans_vs_sklearn(K: int) -> None:
    # Generate synthetic data
    X, _ = make_blobs(n_samples=1000, centers=K, n_features=2, random_state=1992, cluster_std=1.5)

    kmeans_custom = KMeansLloyd(num_clusters=K, init="random", max_iter=500, random_state=1992)
    kmeans_custom.fit(X)

    kmeans_sklearn = sklearn.cluster.KMeans(n_clusters=K, init="random", n_init=10, random_state=1992)
    kmeans_sklearn.fit(X)

    # Check SSE values are close enough
    np.testing.assert_allclose(kmeans_custom.inertia, kmeans_sklearn.inertia_, rtol=1e-3)
