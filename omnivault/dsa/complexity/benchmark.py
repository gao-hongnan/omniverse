from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Protocol, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np

T = TypeVar("T")
R = TypeVar("R")


class DataStructure(Protocol):
    """Protocol defining the interface for data structures that can be used in timing analysis."""

    def __len__(self) -> int: ...


class TimingResult(Protocol):
    """Protocol for objects that can store timing results."""

    sizes: List[int]
    avg_times: List[float]
    median_times: List[float]
    best_times: List[float]
    worst_times: List[float]


@dataclass
class TimingMeasurement:
    """Data class to store timing measurement results."""

    sizes: List[int]
    avg_times: List[float]
    median_times: List[float]
    best_times: List[float]
    worst_times: List[float]

    def __post_init__(self) -> None:
        """Validate that all time lists have the same length as sizes."""
        lists = [self.avg_times, self.median_times, self.best_times, self.worst_times]
        if not all(len(lst) == len(self.sizes) for lst in lists):
            raise ValueError("All time lists must have the same length as sizes list")


SupportedDataTypes = Union[List[int], Dict[int, int], str, DataStructure, None]


class DataFactory:
    """Factory class for creating data structures used in timing analysis."""

    @staticmethod
    def create(data_type: str, size: int) -> SupportedDataTypes:
        """
        Create a data structure of specified type and size.

        Args:
            data_type: Type of data structure to create
            size: Size of the data structure

        Returns:
            Created data structure of specified type and size

        Raises:
            ValueError: If data_type is not supported
        """
        factories = {
            "string": lambda n: "a" * n,
            "array": lambda n: list(range(n)),
            "dict": lambda n: {i: i for i in range(n)},
            # "singly_linked_list": lambda n: SinglyLinkedList(list(range(n))),
            None: lambda _: None,
        }

        if data_type not in factories:
            raise ValueError(f"Unsupported data_type: {data_type}. Supported types: {list(factories.keys())}")

        return factories[data_type](size)


def time_complexity_analyzer(
    data_type: str | None = None,
    repeat: int = 1,
    plot: bool = False,
    plot_title: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., TimingMeasurement]]:
    """
    Decorator to analyze and optionally plot the time complexity of a function.

    Parameters
    ----------
    data_type: str, optional
        Type of data structure the function operates on
    repeat: int, optional
        Number of times to repeat timing measurements
    plot: bool, optional
        Whether to plot the timing results
    plot_title: str, optional
        Custom title for the plot

    Returns
    -------
    Callable[[Callable[..., Any]], Callable[..., TimingMeasurement]]
        Decorated function that performs timing analysis

    Raises
    ------
    ValueError: If repeat < 1
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")

    def decorator(func: Callable[..., Any]) -> Callable[..., TimingMeasurement]:
        @wraps(func)
        def wrapper(n_sizes: List[int], *args: Any, **kwargs: Any) -> TimingMeasurement:
            measurements: Dict[str, List[float]] = {
                "avg_times": [],
                "median_times": [],
                "best_times": [],
                "worst_times": [],
            }

            for n in n_sizes:
                # create a data structure of n elements
                # note data_structure is created before timing the function
                # because we want to time the function, not the creation.
                data_structure = DataFactory.create(data_type, n) if data_type else None

                runtimes = []
                for _ in range(repeat):
                    start_time = time.perf_counter()
                    if data_type:
                        func(n, data_structure, *args, **kwargs)
                    else:
                        func(n, *args, **kwargs)
                    end_time = time.perf_counter()
                    runtimes.append(end_time - start_time)

                measurements["avg_times"].append(float(np.mean(runtimes)))
                measurements["median_times"].append(float(np.median(runtimes)))
                measurements["best_times"].append(float(np.min(runtimes)))
                measurements["worst_times"].append(float(np.max(runtimes)))

            result = TimingMeasurement(sizes=n_sizes, **measurements)

            if plot:
                _plot_measurements(result, func.__name__ if plot_title is None else plot_title)

            return result

        return wrapper

    return decorator


def _plot_measurements(measurements: TimingMeasurement, title: str) -> None:
    """
    Plot timing measurements.

    Parameters
    ----------
    measurements: TimingMeasurement
        object containing the data to plot
    title: str
        Title for the plot
    """
    plt.figure(figsize=(10, 6))

    metrics = [("avg_times", "Average"), ("median_times", "Median"), ("best_times", "Best"), ("worst_times", "Worst")]

    for attr, label in metrics:
        plt.plot(measurements.sizes, getattr(measurements, attr), "o-", label=label)

    plt.xlabel("Size of Input (n)")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True)
    plt.title(f"Time Complexity of {title}")
    plt.show()
