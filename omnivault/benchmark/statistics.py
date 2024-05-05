from typing import Dict, List

import numpy as np


def calculate_statistics(data: List[float], suffix: str = "") -> Dict[str, float]:
    """
    Calculate statistics for a list of data.

    Parameters
    ----------
    data : List[float]
        List of 1D data such as a list of time taken in seconds on each rank.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the following statistics:
        - mean: The mean of the data.
        - median: The median of the data.
        - variance: The variance of the data.
        - standard_deviation: The standard deviation of the data.
    """
    mean = sum(data) / len(data)
    median = float(np.median(data))
    standard_deviation = float(np.std(data))
    variance = float(np.var(data))
    min_value, max_value = min(data), max(data)
    total = sum(data)

    return {
        "mean" + suffix: mean,
        "median" + suffix: median,
        "variance" + suffix: variance,
        "standard_deviation" + suffix: standard_deviation,
        "min" + suffix: min_value,
        "max" + suffix: max_value,
        "total" + suffix: total,
    }
