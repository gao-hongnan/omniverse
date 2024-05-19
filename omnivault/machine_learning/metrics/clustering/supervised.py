"""Utilities to evaluate the clustering performance of models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def contingency_matrix(
    y_trues: NDArray[np.int64 | np.int32 | np.int16], y_preds: NDArray[np.int64 | np.int32 | np.int16], as_dataframe: bool = False
) -> pd.DataFrame | NDArray[np.int64 | np.int32 | np.int16]:
    """Contingency matrix for clustering. Similar to confusion matrix for classification.

    Note:
        One immediate problem is it does not ensure that each predicted cluster
        label is assigned only once to a true label.
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


def purity_score(
    y_trues: np.ndarray, y_preds: np.ndarray, per_cluster: bool = False
) -> float:
    """Computes the purity score for clustering.

    Note:
        Potentially misleading score just like accuracy.
        Imbalanced datasets will give high scores.
    """
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix_ = contingency_matrix(y_trues, y_preds, as_dataframe=False)

    # total purity is the max value in each column divided by the sum of the matrix
    # this means for each cluster k, we find the gt label that has the most samples in that cluster
    # and then sum up all clusters and we divide by the total number of samples in all clusters

    # if per_cluster is True, we return the purity for each cluster
    # this means instead of sum up all clusters, we return the purity for each cluster.
    if per_cluster:
        return np.amax(contingency_matrix_, axis=0) / np.sum(
            contingency_matrix_, axis=0
        )
    # return purity which is the sum of the max values in each column divided by the sum of the matrix
    return np.sum(np.amax(contingency_matrix_, axis=0)) / np.sum(contingency_matrix_)
