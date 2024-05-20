"""Main script to run the K-Means algorithm."""
from __future__ import annotations

import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
from rich import print
from rich.pretty import pprint
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_iris, make_blobs
from sklearn.model_selection import train_test_split

from omnivault.machine_learning.clustering.kmeans import KMeansLloyd, elbow_method, plot_kmeans
from omnivault.machine_learning.metrics.clustering.supervised import contingency_matrix, purity_score


# from sklearn.metrics.cluster import contingency_matrix
def plot_kmeans_clusters_and_elbow(K: int = 3, **kmeans_kwargs: Any) -> None:
    """Sanity check for K-Means implementation with diagram on 2D data."""
    # Generate 2D data points.
    X, y = make_blobs(  # pylint: disable=unbalanced-tuple-unpacking, unused-variable
        n_samples=1000,
        centers=K,
        n_features=2,
        random_state=1992,
        cluster_std=1.5,
    )

    # Run K-Means on the data.
    kmeans = KMeansLloyd(num_clusters=K, **kmeans_kwargs)
    kmeans.fit(X)

    plot_kmeans(X, kmeans.labels, kmeans.centroids)

    print(f"SSE1: {kmeans.inertia}")
    elbow_method(X)

    kmeans_sklearn = KMeans(n_clusters=3, random_state=1992, init="random", n_init=10)
    kmeans_sklearn.fit(X)

    print(f"SSE2: {kmeans_sklearn.inertia_}")


def display_results(
    kmeans: KMeansLloyd,
    sk_kmeans: KMeans,
    X: np.ndarray,
    y_preds: np.ndarray,
    sk_y_preds: np.ndarray,
) -> None:
    # Derive clusters for sklearn
    sk_clusters = {i: X[sk_kmeans.labels_ == i] for i in range(sk_kmeans.n_clusters)}

    result_dict = {
        "Attribute": [
            "Number of Clusters",
            "Centroids",
            "Labels",
            "Inertia",
            "Clusters",
            "Predicted Labels for New Data",
        ],
        "Mine": [
            kmeans.num_clusters,
            kmeans.centroids,
            kmeans.labels,
            kmeans.inertia,
            kmeans.clusters,
            y_preds,
        ],
        "Sklearn": [
            sk_kmeans.n_clusters,
            sk_kmeans.cluster_centers_,
            sk_kmeans.labels_,
            sk_kmeans.inertia_,
            sk_clusters,
            sk_y_preds,
        ],
    }
    df = pd.DataFrame(result_dict)
    print(df.to_string(index=False))


def perform_kmeans_on_iris() -> Tuple[Any, Any]:
    """
    Perform k-means clustering on the Iris dataset using both custom KMeansLloyd
    and scikit-learn's KMeans.

    Parameters
    ----------
    num_clusters : int
        Number of clusters for k-means.
    init : str
        Initialization method for centroids.
    max_iter : int
        Maximum number of iterations.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        Tuple containing results from custom KMeansLloyd and scikit-learn's KMeans.
    """

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    kmeans = KMeansLloyd(num_clusters=3, init="random", max_iter=500, random_state=2023)
    kmeans.fit(X_train)
    pprint(kmeans.labels)
    pprint(kmeans.inertia)

    y_preds = kmeans.predict(X_train)
    assert np.all(y_preds == kmeans.labels)
    contingency_matrix_ = contingency_matrix(y_train, y_preds)
    pprint(contingency_matrix_)
    # TODO: try K = 4

    purity = purity_score(y_train, y_preds)
    pprint(purity)
    purity_per_cluster = purity_score(y_train, y_preds, per_cluster=True)
    print("Purity per cluster: -------------------")
    pprint(purity_per_cluster)

    sk_kmeans = KMeans(
        n_clusters=3,
        random_state=2023,
        n_init=1,
        algorithm="lloyd",
        max_iter=500,
        init="random",
    )
    sk_kmeans.fit(X_train)
    pprint(sk_kmeans.labels_)
    pprint(sk_kmeans.inertia_)

    y_preds = sk_kmeans.predict(X_train)
    assert np.all(y_preds == sk_kmeans.labels_)
    contingency_matrix_ = contingency_matrix(y_train, y_preds)
    pprint(contingency_matrix_)

    purity = purity_score(y_train, y_preds)
    print("Purity Score: -------------------")
    pprint(purity)


def perform_kmeans_on_mnist() -> None:
    X, y = load_digits(return_X_y=True)
    (n_samples, n_features), n_digits = X.shape, np.unique(y).size

    pprint(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    kmeans = KMeansLloyd(num_clusters=n_digits, init="random", max_iter=50000, random_state=42)
    kmeans.fit(X_train)
    pprint(kmeans.labels)
    pprint(kmeans.inertia)

    y_preds = kmeans.predict(X_test)
    contingency_matrix_ = contingency_matrix(y_test, y_preds, as_dataframe=True)
    pprint(contingency_matrix_)

    purity = purity_score(y_test, y_preds)
    pprint(purity)

    purity_per_cluster = purity_score(y_test, y_preds, per_cluster=True)
    pprint(purity_per_cluster)

    sk_kmeans = KMeans(
        n_clusters=n_digits,
        random_state=42,
        n_init=1,
        max_iter=500,
        init="random",
        algorithm="lloyd",
    )
    sk_kmeans.fit(X_train)
    # Note that the labels can be permuted.
    pprint(sk_kmeans.labels_)
    pprint(sk_kmeans.inertia_)


if __name__ == "__main__":
    plot_kmeans_clusters_and_elbow(K=3, init="random", max_iter=500, random_state=1992)
    # time.sleep(500)
    # sklearn example
    X = np.array(
        [
            [1, 2],
            [1, 4],
            [1, 0],
            [10, 2],
            [10, 4],
            [10, 0],
        ]
    )
    kmeans = KMeansLloyd(num_clusters=2, init="random", max_iter=500, random_state=1992)
    kmeans.fit(X)

    y_preds = kmeans.predict([[0, 0], [12, 3]])

    sk_kmeans = KMeans(n_clusters=2, random_state=1992, n_init="auto", algorithm="lloyd", max_iter=500)
    sk_kmeans.fit(X)

    y_preds = kmeans.predict([[0, 0], [12, 3]])
    sk_y_preds = sk_kmeans.predict([[0, 0], [12, 3]])

    display_results(kmeans, sk_kmeans, X, y_preds, sk_y_preds)

    # ####### IRIS DATASET #######

    perform_kmeans_on_iris()

    # ####### MNIST DATASET #######
    perform_kmeans_on_mnist()
