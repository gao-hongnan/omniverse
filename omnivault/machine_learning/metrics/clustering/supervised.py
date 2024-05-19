# NOTE: unit-tested
"""Utilities to evaluate the clustering performance of models."""
from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def contingency_matrix(
    y_trues: NDArray[np.integer[Any]], y_preds: NDArray[np.integer[Any]], as_dataframe: bool = False
) -> pd.DataFrame | NDArray[np.integer[Any]]:
    """
    Generate a contingency matrix for clustering, analogous to a confusion matrix in
    classification.

    This matrix counts the number of times each unique element of the true dataset
    (rows) occurs together with each unique element of the predicted dataset
    (columns). This function is particularly useful for evaluating clustering by
    comparing the true labels and predicted labels.

    Parameters
    ----------
    y_trues : NDArray[np.integer[Any]]
        An array of ground truth (true) labels.
    y_preds : NDArray[np.integer[Any]]
        An array of predicted labels, typically from clustering.
    as_dataframe : bool, optional
        If True, returns the contingency matrix as a pandas DataFrame with labeled rows and columns;
        otherwise, returns it as a NumPy array.

    Returns
    -------
    Union[pd.DataFrame, NDArray[np.integer[Any]]]
        The contingency matrix as either a pandas DataFrame or a NumPy array, depending on the
        value of `as_dataframe`.

    Notes
    -----
    The contingency matrix does not ensure that each predicted cluster label is
    assigned only once to a true label. This might lead to potential ambiguities in
    cluster labeling, especially when the number of predicted clusters does not
    match the number of true classes.

    Examples
    --------
    >>> y_trues = np.array([1, 2, 1, 2, 1])
    >>> y_preds = np.array([0, 0, 1, 1, 1])
    >>> print(contingency_matrix(y_trues, y_preds))
    [[0 3]
     [2 1]]

    >>> print(contingency_matrix(y_trues, y_preds, as_dataframe=True))
           pred=0  pred=1
    true=1      0       3
    true=2      2       1
    """
    # fmt: off
    classes, class_idx        = np.unique(y_trues, return_inverse=True)     # get the unique classes and their indices
    clusters, cluster_idx     = np.unique(y_preds, return_inverse=True)  # get the unique clusters and their indices
    num_classes, num_clusters = classes.shape[0], clusters.shape[0]  # get the number of classes and clusters
    # fmt: on

    # initialize the contingency matrix with shape num_classes x num_clusters
    # exactly the same as the confusion matrix but in confusion matrix
    # the rows are the true labels and the columns are the predicted labels
    # and hence is num_classes x num_classes instead of num_classes x num_clusters
    # however in kmeans for example it is possible to have len(np.unique(y_true)) != len(np.unique(y_pred)
    # i.e. the number of clusters is not equal to the number of classes
    contingency_matrix = np.zeros((num_classes, num_clusters), dtype=np.int64)

    # note however y_true and y_pred are same sequence of samples
    for i in range(class_idx.shape[0]):
        # loop through each sample and increment the contingency matrix
        # at the row corresponding to the true label and column corresponding to the predicted label
        # so if the sample index is i = 0, and class_idx[i] = 1 and cluster_idx[i] = 2
        # this means the gt label is 1 and the predicted label is 2
        # so we increment the contingency matrix at row 1 and column 2 by 1
        # then for each row, which is the row for each gt label,
        # we see which cluster has the highest number of samples and that is the cluster
        # that the gt label is most likely to belong to.
        # in other words since kmeans permutes the labels, we can't just compare the labels
        # directly.
        contingency_matrix[class_idx[i], cluster_idx[i]] += 1

    # row is the true label and column is the predicted label
    if as_dataframe:
        return pd.DataFrame(
            contingency_matrix,
            index=[f"true={c}" for c in classes],
            columns=[f"pred={c}" for c in clusters],
        )
    return contingency_matrix


@overload
def purity_score(
    y_trues: NDArray[np.integer[Any]], y_preds: NDArray[np.integer[Any]], per_cluster: Literal[True]
) -> NDArray[np.floating[Any]]:
    ...


@overload
def purity_score(
    y_trues: NDArray[np.integer[Any]], y_preds: NDArray[np.integer[Any]], per_cluster: Literal[False] = False
) -> float:
    ...


def purity_score(
    y_trues: NDArray[np.integer[Any]], y_preds: NDArray[np.integer[Any]], per_cluster: bool = False
) -> float | NDArray[np.floating[Any]]:
    """
    Compute the purity score for clustering evaluation.

    Purity is an external evaluation metric of clustering
    quality. It is the measure of the largest fraction of
    true labels in each cluster. The overall purity is the
    sum of purities of all clusters divided by the total
    number of samples.

    Parameters
    ----------
    y_trues : NDArray[np.integer[Any]]
        Array of ground truth (true) labels.
    y_preds : NDArray[np.integer[Any]]
        Array of predicted labels from clustering.
    per_cluster : bool, optional
        If True, returns the purity for each cluster
        instead of the overall purity score.

    Returns
    -------
    Union[float, NDArray[np.floating[Any]]]
        If `per_cluster` is False, returns a single float
        value representing the overall purity of the
        clustering. If True, returns an array of purity
        scores for each cluster.

    Notes
    -----
    - Purity can be misleading in cases where cluster size
      varies significantly. A large cluster that is only
      moderately pure can disproportionately influence the
      overall purity score.
    - Purity does not penalize having many clusters, so
      it should not be used in isolation.
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix_ = contingency_matrix(y_trues, y_preds, as_dataframe=False)

    # total purity is the max value in each column divided by the sum of the matrix
    # this means for each cluster k, we find the gt label that has the most samples in that cluster
    # and then sum up all clusters and we divide by the total number of samples in all clusters

    # if per_cluster is True, we return the purity for each cluster
    # this means instead of sum up all clusters, we return the purity for each cluster.
    if per_cluster:
        return np.amax(contingency_matrix_, axis=0) / np.sum(contingency_matrix_, axis=0)  # type: ignore[no-any-return]
    # return purity which is the sum of the max values in each column divided by the sum of the matrix
    return float(np.sum(np.amax(contingency_matrix_, axis=0)) / np.sum(contingency_matrix_))
