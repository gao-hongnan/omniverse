from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np

from ..errors import InvalidConfiguration
from .factory import DataFactory


@dataclass
class TimingMeasurement:
    sizes: list[int]
    avg_times: list[float]
    median_times: list[float]
    best_times: list[float]
    worst_times: list[float]

    def __post_init__(self) -> None:
        lists = [self.avg_times, self.median_times, self.best_times, self.worst_times]
        if not all(len(lst) == len(self.sizes) for lst in lists):
            raise InvalidConfiguration("All time lists must have the same length as sizes list")


def time_complexity_analyzer(
    data_type: str | None = None,
    repeat: int = 1,
    plot: bool = False,
    plot_title: str | None = None,
) -> Callable[[Callable[..., object]], Callable[..., TimingMeasurement]]:
    if repeat < 1:
        raise InvalidConfiguration("repeat must be at least 1")

    def decorator(func: Callable[..., object]) -> Callable[..., TimingMeasurement]:
        @wraps(func)
        def wrapper(n_sizes: Sequence[int], *args: Any, **kwargs: Any) -> TimingMeasurement:
            measurements: dict[str, list[float]] = {
                "avg_times": [],
                "median_times": [],
                "best_times": [],
                "worst_times": [],
            }

            for n in n_sizes:
                data_structure = DataFactory.create(data_type, n) if data_type else None

                runtimes: list[float] = []
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

            result = TimingMeasurement(sizes=list(n_sizes), **measurements)

            if plot:
                _plot_measurements(result, func.__name__ if plot_title is None else plot_title)

            return result

        return wrapper

    return decorator


def _plot_measurements(measurements: TimingMeasurement, title: str) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415 - optional dependency, imported only when plotting is requested

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


def measure_recursive_complexity(
    func: Callable[..., object],
    test_inputs: Sequence[Any],
    repeat: int = 1,
    plot: bool = False,
    plot_title: str | None = None,
) -> TimingMeasurement:
    if repeat < 1:
        raise InvalidConfiguration("repeat must be at least 1")

    measurements: dict[str, list[float]] = {
        "avg_times": [],
        "median_times": [],
        "best_times": [],
        "worst_times": [],
    }
    sizes: list[int] = []

    for test_input in test_inputs:
        size = len(test_input) if hasattr(test_input, "__len__") else test_input
        sizes.append(size)

        runtimes: list[float] = []
        for _ in range(repeat):
            start_time = time.perf_counter()
            func(test_input)
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)

        measurements["avg_times"].append(float(np.mean(runtimes)))
        measurements["median_times"].append(float(np.median(runtimes)))
        measurements["best_times"].append(float(np.min(runtimes)))
        measurements["worst_times"].append(float(np.max(runtimes)))

    result = TimingMeasurement(sizes=sizes, **measurements)

    if plot:
        _plot_measurements(result, func.__name__ if plot_title is None else plot_title)

    return result
