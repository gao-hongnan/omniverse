from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights_and_stats(labels: List[int]) -> Dict[str, Any]:
    """
    Calculate class counts, normalized class counts, class weights, and return relevant statistics
    for a given list of class labels in the training set.

    Parameters
    ----------
    labels : List[int]
        List of class labels in the training dataset.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - class_counts: Dictionary mapping each class to its count in `labels`.
        - normalized_class_counts: Dictionary mapping each class to its normalized count based on `labels`.
        - class_weights_dict: Dictionary mapping each class to its computed weight.
        - class_weights: List of computed class weights.
        - class_count_stats: Dictionary containing statistics about class counts:
            * total_samples: Total number of samples.
            * max_count: Maximum count of any class.
            * min_count: Minimum count of any class.
            * max_normalized: Maximum normalized count of any class.
            * min_normalized: Minimum normalized count of any class.

    Examples
    --------
    >>> labels = [1, 1, 2, 2, 2, 3]
    >>> stats = calculate_class_weights_and_stats(labels)
    >>> print(stats['class_count_stats'])
    {'total_samples': 6, 'max_count': 3, 'min_count': 1, 'max_normalized': 0.5, 'min_normalized': 0.16666666666666666}
    """
    classes, class_counts = np.unique(labels, return_counts=True)

    total_samples = sum(class_counts)
    normalized_class_counts = class_counts / total_samples

    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)

    class_count_stats = {
        "total_samples": total_samples,
        "max_count": np.max(class_counts),
        "min_count": np.min(class_counts),
        "max_normalized": np.max(normalized_class_counts),
        "min_normalized": np.min(normalized_class_counts),
    }

    result = {
        "class_counts": dict(zip(classes, class_counts)),
        "normalized_class_counts": dict(zip(classes, normalized_class_counts)),
        "class_weights_dict": dict(zip(classes, class_weights)),
        "class_weights": list(class_weights),
        "class_count_stats": class_count_stats,
    }

    return result
