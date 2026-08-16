import pytest

from omnivault.dsa.core.complexity import (
    DataFactory,
    TimingMeasurement,
    measure_recursive_complexity,
    time_complexity_analyzer,
)


class TestDataFactory:
    def test_create_string(self) -> None:
        result = DataFactory.create("string", 10)
        assert isinstance(result, str)
        assert result == "a" * 10
        assert len(result) == 10

    def test_create_array(self) -> None:
        result = DataFactory.create("array", 5)
        assert isinstance(result, list)
        assert result == [0, 1, 2, 3, 4]
        assert len(result) == 5

    def test_create_dict(self) -> None:
        result = DataFactory.create("dict", 3)
        assert isinstance(result, dict)
        assert result == {0: 0, 1: 1, 2: 2}
        assert len(result) == 3

    def test_create_singly_linked_list(self) -> None:
        from omnivault.dsa.containers.linear.linked_list import SinglyLinkedList

        result = DataFactory.create("singly_linked_list", 4)
        assert isinstance(result, SinglyLinkedList)
        assert len(result) == 4
        assert list(result) == [0, 1, 2, 3]

    def test_create_none(self) -> None:
        result = DataFactory.create(None, 100)
        assert result is None

    def test_unsupported_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported data_type"):
            DataFactory.create("unsupported", 10)


class TestTimingMeasurement:
    def test_valid_measurement(self) -> None:
        measurement = TimingMeasurement(
            sizes=[10, 20, 30],
            avg_times=[0.1, 0.2, 0.3],
            median_times=[0.1, 0.2, 0.3],
            best_times=[0.09, 0.19, 0.29],
            worst_times=[0.11, 0.21, 0.31],
        )
        assert measurement.sizes == [10, 20, 30]
        assert measurement.avg_times == [0.1, 0.2, 0.3]

    def test_invalid_measurement_different_lengths(self) -> None:
        with pytest.raises(ValueError, match="All time lists must have the same length"):
            TimingMeasurement(
                sizes=[10, 20],
                avg_times=[0.1],
                median_times=[0.1, 0.2],
                best_times=[0.09, 0.19],
                worst_times=[0.11, 0.21],
            )


class TestTimeComplexityAnalyzer:
    def test_basic_function_no_data_structure(self) -> None:
        @time_complexity_analyzer(repeat=2)
        def simple_loop(n: int) -> None:
            total = 0
            for i in range(n):
                total += i

        result = simple_loop([100, 200, 300])
        assert isinstance(result, TimingMeasurement)
        assert result.sizes == [100, 200, 300]
        assert len(result.avg_times) == 3
        assert all(t > 0 for t in result.avg_times)
        assert all(result.best_times[i] <= result.avg_times[i] <= result.worst_times[i] for i in range(3))

    def test_function_with_array(self) -> None:
        @time_complexity_analyzer(data_type="array", repeat=2)
        def sum_array(_n: int, arr: list[int]) -> int:
            return sum(arr)

        result = sum_array([10, 20, 30])
        assert isinstance(result, TimingMeasurement)
        assert result.sizes == [10, 20, 30]

    def test_invalid_repeat(self) -> None:
        with pytest.raises(ValueError, match="repeat must be at least 1"):

            @time_complexity_analyzer(repeat=0)
            def dummy_func(n: int) -> None:
                pass


class TestMeasureRecursiveComplexity:
    def test_fibonacci(self) -> None:
        def fibonacci(n: int) -> int:
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        result = measure_recursive_complexity(fibonacci, [10, 20, 30], repeat=2)
        assert isinstance(result, TimingMeasurement)
        assert result.sizes == [10, 20, 30]
        assert len(result.avg_times) == 3
        assert all(t > 0 for t in result.avg_times)
        assert result.avg_times[2] > result.avg_times[0]  # fib(30) is O(2^30) vs fib(10) = O(2^10)

    def test_with_sequence_inputs(self) -> None:
        def process_list(lst: list[int]) -> int:
            if not lst:
                return 0
            return lst[0] + process_list(lst[1:])

        test_inputs = [[1, 2], [1, 2, 3], [1, 2, 3, 4]]
        result = measure_recursive_complexity(process_list, test_inputs, repeat=1)
        assert result.sizes == [2, 3, 4]

    def test_invalid_repeat(self) -> None:
        def dummy_func(n: int) -> int:
            return n

        with pytest.raises(ValueError, match="repeat must be at least 1"):
            measure_recursive_complexity(dummy_func, [1, 2, 3], repeat=0)
