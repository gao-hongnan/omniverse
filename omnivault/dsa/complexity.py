# from dataclasses import dataclass
# from enum import Enum
# from typing import Any, Callable, List, Optional, Protocol, Union
# import time
# from statistics import mean, median
# import matplotlib.pyplot as plt
# from functools import wraps


# class DataStructureType(Enum):
#     """Enumeration of supported data structure types."""

#     ARRAY = "array"
#     DICT = "dict"
#     NONE = "none"


# @dataclass
# class TimingResult:
#     """Container for timing measurement results."""

#     sizes: List[int]
#     average_times: List[float]
#     median_times: List[float]
#     best_times: List[float]
#     worst_times: List[float]

#     def plot(self, function_name: str) -> None:
#         """Plot timing results."""
#         plt.figure(figsize=(10, 6))

#         metrics = {
#             "Average": self.average_times,
#             "Median": self.median_times,
#             "Best": self.best_times,
#             "Worst": self.worst_times,
#         }

#         for label, times in metrics.items():
#             plt.plot(self.sizes, times, "o-", label=label)

#         plt.xlabel("Size of Input (n)")
#         plt.ylabel("Execution Time (s)")
#         plt.legend()
#         plt.grid(True)
#         plt.title(f"Time Complexity of {function_name}")
#         plt.show()


# class TimingError(Exception):
#     """Custom exception for timing-related errors."""

#     pass


# class DataStructureGenerator(Protocol):
#     """Protocol defining the interface for data structure generation."""

#     def __call__(self, size: int) -> Any: ...


# class ComplexityAnalyzer:
#     """Class to analyze time complexity of functions."""

#     def __init__(
#         self, min_size: int = 100, max_size: int = 1000, step_size: int = 100, repeats: int = 3, warmup_runs: int = 1
#     ):
#         """
#         Initialize the complexity analyzer.

#         Parameters:
#             min_size: Minimum input size to test
#             max_size: Maximum input size to test
#             step_size: Step size between input sizes
#             repeats: Number of times to repeat each measurement
#             warmup_runs: Number of warmup runs before timing
#         """
#         self.validate_parameters(min_size, max_size, step_size, repeats, warmup_runs)
#         self.min_size = min_size
#         self.max_size = max_size
#         self.step_size = step_size
#         self.repeats = repeats
#         self.warmup_runs = warmup_runs

#     @staticmethod
#     def validate_parameters(min_size: int, max_size: int, step_size: int, repeats: int, warmup_runs: int) -> None:
#         """Validate initialization parameters."""
#         if min_size <= 0 or max_size <= 0 or step_size <= 0:
#             raise ValueError("Size parameters must be positive")
#         if min_size >= max_size:
#             raise ValueError("min_size must be less than max_size")
#         if repeats <= 0 or warmup_runs < 0:
#             raise ValueError("Invalid repeats or warmup_runs value")

#     def measure_execution_time(self, func: Callable[..., Any], data_structure: Any, *args: Any, **kwargs: Any) -> float:
#         """Measure execution time of a single function call."""
#         # Perform warmup runs
#         for _ in range(self.warmup_runs):
#             func(data_structure, *args, **kwargs)

#         # Actual timing
#         start_time = time.perf_counter()
#         func(data_structure, *args, **kwargs)
#         end_time = time.perf_counter()

#         return end_time - start_time

#     def analyze(
#         self,
#         func: Callable[..., Any],
#         data_type: Union[DataStructureType, str],
#         generator: Optional[DataStructureGenerator] = None,
#         plot: bool = True,
#     ) -> TimingResult:
#         """
#         Analyze the time complexity of a function.

#         Parameters:
#             func: Function to analyze
#             data_type: Type of data structure to use
#             generator: Optional custom data structure generator
#             plot: Whether to plot the results

#         Returns:
#             TimingResult object containing the analysis results
#         """
#         if isinstance(data_type, str):
#             try:
#                 data_type = DataStructureType(data_type.lower())
#             except ValueError:
#                 raise ValueError(f"Invalid data_type: {data_type}")

#         sizes = range(self.min_size, self.max_size + 1, self.step_size)
#         results = TimingResult([], [], [], [], [])

#         for size in sizes:
#             runtimes = []
#             data = generator(size) if generator else self._generate_data_structure(data_type, size)

#             for _ in range(self.repeats):
#                 try:
#                     runtime = self.measure_execution_time(func, data)
#                     runtimes.append(runtime)
#                 except Exception as e:
#                     raise TimingError(f"Error measuring function: {str(e)}")

#             results.sizes.append(size)
#             results.average_times.append(mean(runtimes))
#             results.median_times.append(median(runtimes))
#             results.best_times.append(min(runtimes))
#             results.worst_times.append(max(runtimes))

#         if plot:
#             results.plot(func.__name__)

#         return results

#     @staticmethod
#     def _generate_data_structure(data_type: DataStructureType, size: int) -> Any:
#         """Generate data structure of specified type and size."""
#         if data_type == DataStructureType.ARRAY:
#             return list(range(size))
#         elif data_type == DataStructureType.DICT:
#             return {i: i for i in range(size)}
#         elif data_type == DataStructureType.NONE:
#             return size
#         else:
#             raise ValueError(f"Unsupported data structure type: {data_type}")


# def analyze_complexity(
#     data_type: Union[DataStructureType, str],
#     min_size: int = 100,
#     max_size: int = 1000,
#     step_size: int = 100,
#     repeats: int = 3,
#     plot: bool = True,
# ) -> Callable[[Callable[..., Any]], Callable[..., TimingResult]]:
#     """
#     Decorator for analyzing time complexity of functions.

#     Parameters:
#         data_type: Type of data structure to use
#         min_size: Minimum input size
#         max_size: Maximum input size
#         step_size: Step size between input sizes
#         repeats: Number of measurement repeats
#         plot: Whether to plot results
#     """

#     def decorator(func: Callable[..., Any]) -> Callable[..., TimingResult]:
#         @wraps(func)
#         def wrapper(*args: Any, **kwargs: Any) -> TimingResult:
#             analyzer = ComplexityAnalyzer(min_size=min_size, max_size=max_size, step_size=step_size, repeats=repeats)
#             return analyzer.analyze(func, data_type, plot=plot)

#         return wrapper

#     return decorator
