"""K-Means Implementation.

References:
- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- https://github.com/rushter/MLAlgorithms/blob/master/mla/kmeans.py
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cluster
from numpy.typing import NDArray

from omnivault.machine_learning.estimator import BaseEstimator
from omnivault.machine_learning.metrics.pairwise.distance import euclidean_distance, manhattan_distance
from omnivault.utils.reproducibility.seed import seed_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class KMeansLloyd(BaseEstimator):
    """K-Means Lloyd's algorithm.

    Parameters
    ----------
    num_clusters : int, optional, default: 3
        The number of clusters to form as well as the number of
        centroids to generate.

    init : str, optional, default: "random"
        Method for initialization, either 'random' or 'k-means++'.

    max_iter : int, optional, default: 100
        Maximum number of iterations of the k-means algorithm for
        a single run.

    metric : str, optional, default: "euclidean"
        The distance metric used to calculate the distance between each
        sample and the centroid.

    tol : float, optional, default: 1e-8
        The tolerance with regards to the change in the within-cluster
        sum-squared-error to declare convergence.

    random_state : int, optional, default: 42
        Seed for the random number generator used during the
        initialization of the centroids.
    """

    def __init__(
        self,
        num_clusters: int = 3,
        init: Literal["random", "k-means++"] = "random",
        max_iter: int = 100,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        tol: float = 1e-8,
        random_state: int = 42,
    ) -> None:
        self._K = num_clusters  # K
        self.init = init  # random init
        self.max_iter = max_iter
        self.tol = tol

        self.metric = metric
        self.distance = self._get_distance_metric()  # get distance fn based on metric

        self.random_state = random_state
        seed_all(self.random_state, seed_torch=False, set_torch_deterministic=False)

        self._reset_clusters()  # initializes self._C = {C_1=[], C_2=[], ..., C_k=[]}

        self._C: Dict[int, List[int]]  # clusters
        self._N: int
        self._D: int
        self.t: int  # iteration counter
        self._labels: NDArray[np.floating[Any]]  # N labels np.zeros(shape=(self._N))
        self._centroids: NDArray[np.floating[Any]]  # np.zeros(shape=(self._K, self.num_features)) KxD matrix
        self._inertia: NDArray[np.floating[Any]]
        self._inertias: NDArray[np.floating[Any]]  # np.zeros(shape=(self._N)) N inertias

    @property
    def num_clusters(self) -> int:
        """Property to get the number of clusters K."""
        return self._K

    @property
    def num_features(self) -> int:
        """Property to get the number of features D."""
        return self._D

    @property
    def num_samples(self) -> int:
        """Property to get the number of samples N."""
        return self._N

    @property
    def clusters(self) -> Dict[int, List[int]]:
        """Property to get the clusters, this is our C."""
        return self._C

    @property
    def labels(self) -> NDArray[np.floating[Any]]:
        """Property to get the labels of the samples."""
        return self._labels.astype(int)

    @property
    def centroids(self) -> NDArray[np.floating[Any]]:
        """Property to get the centroids."""
        return self._centroids

    @property
    def inertia(self) -> NDArray[np.floating[Any]]:
        """Property to get the inertia."""
        return self._inertia

    def _reset_clusters(self) -> None:
        """Reset clusters."""
        self._C = {k: [] for k in range(self._K)}  # type: ignore[misc]

    def _reset_inertias(self) -> None:
        """
        Reset the inertias to zero for each sample.

        This method initializes `self._inertias`, a numpy array with
        shape (N,), to zero. It is used in the assignment step to store
        the minimum distances from each sample to its assigned centroid,
        which contributes to the cost function J, summed over all samples.
        """
        self._inertias = np.zeros(self._N)  # reset mechanism so don't accumulate

    def _reset_labels(self) -> None:
        """
        Reset the labels to zero for each sample.

        This method reinitializes `self._labels`, a numpy array with
        shape (N,), to zero, clearing previous cluster assignments.
        """
        self._labels = np.zeros(self._N)  # reset mechanism so don't accumulate

    def _init_centroids(self, X: NDArray[np.floating[Any]]) -> None:
        """
        Initialize the centroids for K-Means clustering.

        The method sets initial values for `self._centroids` based on the
        specified initialization strategy ('random' or 'k-means++').

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            A 2D array where each row is a data point.

        Raises
        ------
        ValueError
            If the `init` attribute is set to an unsupported value.
        """
        self._centroids = np.zeros(shape=(self._K, self._D))  # KxD matrix
        if self.init == "random":
            for k in range(self._K):
                self._centroids[k] = X[np.random.choice(range(self._N))]
        elif self.init == "k-means++":
            raise NotImplementedError("k-means++ initialization is not implemented.")
        else:
            raise ValueError(f"{self.init} is not supported.")

    def _get_distance_metric(self) -> Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]]], float]:
        """Get the distance metric based on the metric attribute.

        Returns
        -------
        Callable[[NDArray[np.floating[Any]], NDArray[np.floating[Any]]], float]
            The distance metric function.
        """
        if self.metric == "euclidean":
            return partial(euclidean_distance, squared=False)
        if self.metric == "manhattan":
            return manhattan_distance
        raise ValueError(f"{self.metric} is not supported. The metric must be 'euclidean' or 'manhattan'.")

    def _compute_argmin_assignment(
        self, x: NDArray[np.floating[Any]], centroids: NDArray[np.floating[Any]]
    ) -> Tuple[int, float]:
        """Compute the argmin assignment for a single sample x.

        In other words, for a single sample x, compute the distance between x and
        each centroid mu_1, mu_2, ..., mu_k. Then, return the index of the centroid
        that is closest to x, and the distance between x and that centroid.

        This step has O(K) complexity, where K is the number of clusters. However,
        to be more pedantic, the self.distance also has O(D) complexity,
        where D is the number of features, so it can be argued that this step has
        O(KD) complexity.

        Parameters
        ----------
        x : NDArray[np.floating[Any]]
            A single data point.
        centroids : NDArray[np.floating[Any]]
            Current centroids of the clusters.

        Returns
        -------
        Tuple[int, float]
            The index of the closest centroid and the distance to it.
        """
        min_index = -100  # some random number
        min_distance = np.inf
        for k, centroid in enumerate(centroids):
            distance = self.distance(x, centroid)
            if distance < min_distance:
                # now min_distance is the best distance between x and mu_k
                min_index = k
                min_distance = distance
        return min_index, min_distance

    def _assign_samples(self, X: NDArray[np.floating[Any]], centroids: NDArray[np.floating[Any]]) -> None:
        """Assignment step: assigns samples to clusters.

        This step has O(NK) complexity since we are looping over
        N samples, and for each sample, _compute_argmin_assignment requires
        O(K) complexity.

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            A 2D array where each row is a data point.
        centroids : NDArray[np.floating[Any]]
            Current centroids of the clusters.
        """
        self._reset_inertias()  # reset the inertias to [0, 0, ..., 0]
        self._reset_labels()  # reset the labels to [0, 0, ..., 0]
        for sample_index, x in enumerate(X):
            min_index, min_distance = self._compute_argmin_assignment(x, centroids)
            # fmt: off
            self._C[min_index].append(x) # here means append the data point x to the cluster C_k
            self._inertias[sample_index] = min_distance  # the cost of the sample x
            self._labels[sample_index] = int(min_index)  # the label of the sample x
            # fmt: on

    def _update_centroids(self, centroids: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Update step: update the centroid with new cluster mean.

        Parameters
        ----------
        centroids : NDArray[np.floating[Any]]
            Current centroids of the clusters.

        Returns
        -------
        NDArray[np.floating[Any]]
            Updated centroids of the clusters.
        """
        for k, cluster in self._C.items():  # for k and the corresponding cluster C_k
            mu_k = np.mean(cluster, axis=0)  # compute the mean of the cluster
            centroids[k] = mu_k  # update the centroid mu_k
        return centroids

    def _has_converged(
        self, old_centroids: NDArray[np.floating[Any]], updated_centroids: NDArray[np.floating[Any]]
    ) -> bool:
        """Checks for convergence.
        If the centroids are the same, then the algorithm has converged.
        You can also check convergence by comparing the SSE.

        Parameters
        ----------
        old_centroids : NDArray[np.floating[Any]]
            Old centroids of the clusters.
        updated_centroids : NDArray[np.floating[Any]]
            Updated centroids of the clusters.

        Returns
        -------
        bool
            True if the centroids have converged, False otherwise.
        """
        return np.allclose(updated_centroids, old_centroids, atol=self.tol)

    def fit(self, X: NDArray[np.floating[Any]]) -> KMeansLloyd:
        """
        Fit the K-Means clustering model to the data.

        This function iteratively assigns samples to clusters and updates
        centroids based on those assignments until the centroids do not
        change significantly or a maximum number of iterations is reached.

        Parameters
        ----------
        X : NDArray[np.floating[Any]]
            Data points to cluster.

        Returns
        -------
        KMeansLloyd
            The instance itself.
        """
        # fmt: off
        self._N, self._D = X.shape  # N=num_samples, D=num_features

        # step 1. Initialize self._centroids of shape KxD.
        self._init_centroids(X)

        # enter iteration of step 2-3 until convergence
        for t in range(self.max_iter):
            # copy the centroids and denote it as old_centroids
            # note that in t+1 iteration, we will update the old_centroids
            # to be the self._centroid in t iteration
            old_centroids = self._centroids.copy()

            # step 2: assignment step
            self._assign_samples(X, old_centroids)

            # step 3: update step
            # (careful of mutation without copy because I am doing centroids[k] = mu_k)
            # updated_centroids = self._update_centroids(old_centroids.copy())
            self._centroids = self._update_centroids(old_centroids.copy())

            # step 4: check convergence
            if self._has_converged(old_centroids, self._centroids):
                print(f"Converged at iteration {t}")
                self.t = t  # assign iteration index, used for logging
                break

            # do not put self._centroids here because if self._has_converged is True,
            # then this step is not updated, causing the final centroids to be the old_centroids
            # instead of the updated_centroids, especially problematic if tolerance>0
            # self._centroids = updated_centroids  # assign updated centroids to
            # self._centroids if not converged
            self._reset_clusters()  # reset the clusters if not converged

        # step 5: compute the final cost on the converged centroids
        self._inertia = np.sum(self._inertias, dtype=np.float64)
        # fmt: on
        return self

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Predict cluster labels for samples in X where X can be new data or training data."""
        y_preds = np.array([self._compute_argmin_assignment(x, self._centroids)[0] for x in X])
        return y_preds


def elbow_method(
    X: NDArray[np.floating[Any]], min_clusters: int = 1, max_clusters: int = 10, verbose: bool = True
) -> None:
    """Elbow method to find the optimal number of clusters.
    The optimal number of clusters is where the elbow occurs.
    The elbow is where the SSE starts to decrease in a linear fashion."""
    inertias = []
    for k in range(min_clusters, max_clusters + 1):
        if verbose:
            logging.info(f"Running KMeans with {k} clusters")
        kmeans = KMeansLloyd(num_clusters=k, init="random", max_iter=500, random_state=1992)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)
    plt.plot(range(min_clusters, max_clusters + 1), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Intertia/Distortion/SSE")
    plt.tight_layout()
    plt.show()


def plot_kmeans(
    X: NDArray[np.floating[Any]], cluster_assignments: NDArray[np.floating[Any]], centroids: NDArray[np.floating[Any]]
) -> None:
    """
    Plot the K-Means clustering results using matplotlib and seaborn.

    This function visualizes the clustering output of a K-Means algorithm,
    displaying both the data points and the centroids on a scatter plot. Each
    cluster is colored differently, and centroids are marked distinctly to
    highlight their position relative to the data points.

    Parameters
    ----------
    X : NDArray[np.floating[Any]]
        A 2D array with shape (N, D) where N is the number of data points
        and D is the dimensionality of each data point. This array contains
        the coordinates of the data points.
    cluster_assignments : NDArray[np.floating[Any]]
        A 1D array with shape (N,) that contains the cluster index to which
        each data point is assigned. Each element in this array should be
        an integer representing the cluster index.
    centroids : NDArray[np.floating[Any]]
        A 2D array with shape (K, D) where K is the number of clusters and
        D is the dimensionality of the centroids. This array contains the
        coordinates of the centroids.
    """
    # Ensure that seaborn is set to its default theme.
    sns.set_theme()

    # Create a figure with two subplots.
    _fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(X[:, 0], X[:, 1], s=50, alpha=0.6)
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].set_title("Original Scatter Plot")

    # Create a scatter plot of the data points, colored by their cluster assignments.
    axes[1].scatter(
        X[:, 0],
        X[:, 1],
        c=cluster_assignments,
        cmap="viridis",
        s=50,
        alpha=0.6,
    )

    # Plot the centroids as red 'X' markers.
    axes[1].scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, edgecolors="black")

    # Set the plot labels.
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].set_title("K-Means Clustering Results")

    # Display the plot.
    plt.show()


def kmeans_vectorized(
    X: NDArray[np.floating[Any]], num_clusters: int, max_iter: int = 500
) -> NDArray[np.floating[Any]]:
    """
    Perform K-Means clustering using vectorized operations.

    This function clusters data into a specified number of
    clusters using a vectorized implementation of the K-Means
    algorithm, which is generally more efficient than the
    iterative approach.

    Parameters
    ----------
    X : NDArray[np.floating[Any]]
        A 2D array where each row represents a data point
        and each column represents a dimension.
    num_clusters : int
        The number of clusters to form.
    max_iter : int, optional
        The maximum number of iterations of the K-Means
        clustering algorithm (default is 500).

    Returns
    -------
    NDArray[np.floating[Any]]
        A 2D array where each row is a cluster center.
    """
    indices = np.random.choice(X.shape[0], num_clusters, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        # Step 2: Assign points to the nearest cluster
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))  # Euclidean distance
        closest_cluster_ids = np.argmin(distances, axis=1)

        # Step 3: Update centroids
        new_centroids = np.array([X[closest_cluster_ids == k].mean(axis=0) for k in range(num_clusters)])

        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    return centroids  # type: ignore[no-any-return]


def display_results(
    kmeans: KMeansLloyd,
    sk_kmeans: sklearn.cluster.KMeans,
    X: NDArray[np.floating[Any]],
    y_preds: NDArray[np.integer[Any]],
    sk_y_preds: NDArray[np.integer[Any]],
) -> pd.DataFrame:
    """Display the results of the custom KMeansLloyd and scikit-learn's KMeans.

    Parameters
    ----------
    kmeans : KMeansLloyd
        The custom KMeansLloyd instance.
    sk_kmeans : sklearn.cluster.KMeans
        The scikit-learn KMeans instance.
    X : NDArray[np.floating[Any]]
        The input data.
    y_preds : NDArray[np.integer[Any]]
        The predicted labels from the custom KMeansLloyd.
    sk_y_preds : NDArray[np.integer[Any]]
        The predicted labels from scikit-learn's KMeans.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the custom KMeansLloyd and
        scikit-learn's KMeans.
    """
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
    return df
