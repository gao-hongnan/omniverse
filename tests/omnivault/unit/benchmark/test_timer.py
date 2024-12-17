# pytest tests/omnivault/unit/benchmark/test_timer.py -v
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Final

import psutil
import pytest
from pydantic import ValidationError

from omnivault.benchmark.timer import TimedExecution, TimedExecutionMetadata, TimerDecorator, timer


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Fixture providing sample metadata for testing."""
    cpu_times = psutil._common.pcputimes(
        user=0.0,
        system=0.0,
        children_user=0.0,
        children_system=0.0,
    )
    return {
        "thread_id": 123,
        "process_id": 456,
        "initial_memory_usage": 1000,
        "initial_cpu_time": cpu_times,
        "caller_function_name": "test_function",
        "caller_module_name": "test_module",
        "caller_module_path": "/path/to/test.py",
    }


class TestTimedExecutionMetadata:
    """Test suite for TimedExecutionMetadata model."""

    def test_valid_metadata_creation(self, sample_metadata: Dict[str, Any]) -> None:
        """Test creating TimedExecutionMetadata with valid data."""
        metadata = TimedExecutionMetadata(**sample_metadata)
        assert metadata.thread_id == 123
        assert metadata.process_id == 456
        assert metadata.initial_memory_usage == 1000
        assert metadata.caller_function_name == "test_function"
        assert metadata.tags == {}

    def test_metadata_with_optional_fields(self, sample_metadata: Dict[str, Any]) -> None:
        """Test creating metadata with optional fields."""
        sample_metadata.update(
            {"start_time": 1234.5678, "end_time": 1235.5678, "execution_time": 1.0, "tags": {"env": "test"}}
        )
        metadata = TimedExecutionMetadata(**sample_metadata)
        assert metadata.start_time == 1234.5678
        assert metadata.end_time == 1235.5678
        assert metadata.execution_time == 1.0
        assert metadata.tags == {"env": "test"}

    def test_invalid_metadata_creation(self) -> None:
        """Test that invalid metadata raises ValidationError."""
        with pytest.raises(ValidationError):
            TimedExecutionMetadata()  # type: ignore[call-arg]

    def test_metadata_with_exception(self, sample_metadata: Dict[str, Any]) -> None:
        """Test metadata with exception information."""
        sample_metadata["exception"] = "ValueError: Test error"
        metadata = TimedExecutionMetadata(**sample_metadata)
        assert metadata.exception == "ValueError: Test error"


class TestTimedExecution:
    """Test suite for TimedExecution context manager."""

    def test_sync_context_manager(self) -> None:
        """Test synchronous context manager functionality."""
        with TimedExecution(pretty_print=False) as timer:
            time.sleep(1.2)

        assert isinstance(timer.metadata["start_time"], float)
        assert isinstance(timer.metadata["end_time"], float)
        assert timer.metadata["execution_time"] >= 1.2
        assert timer.metadata["execution_time"] <= 1.3
        assert "exception" not in timer.metadata

    def test_sync_context_manager_with_exception(self) -> None:
        """Test synchronous context manager with exception."""
        with pytest.raises(ValueError), TimedExecution(pretty_print=False) as timer:
            raise ValueError("Test error")

        assert "exception" in timer.metadata
        assert "ValueError: Test error" in timer.metadata["exception"]

    def test_custom_tags(self) -> None:
        """Test adding custom tags to metadata."""
        tags: Final[Dict[str, str]] = {"environment": "test", "version": "1.0"}
        with TimedExecution(tags=tags, pretty_print=False) as timer:
            pass

        assert timer.metadata["environment"] == "test"
        assert timer.metadata["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test asynchronous context manager functionality."""
        async with TimedExecution(pretty_print=False) as timer:
            await asyncio.sleep(1.2)

        assert isinstance(timer.metadata["start_time"], float)
        assert isinstance(timer.metadata["end_time"], float)
        assert timer.metadata["execution_time"] >= 1.2
        assert timer.metadata["execution_time"] <= 1.3
        assert "exception" not in timer.metadata

    @pytest.mark.asyncio
    async def test_async_context_manager_with_exception(self) -> None:
        """Test asynchronous context manager with exception."""
        with pytest.raises(ValueError), TimedExecution(pretty_print=False) as timer:
            raise ValueError("Test error")

        assert "exception" in timer.metadata
        assert "ValueError: Test error" in timer.metadata["exception"]

    def test_metadata_collection(self) -> None:
        """Test that all required metadata fields are collected."""
        with TimedExecution(pretty_print=False) as timer:
            pass

        required_fields: Final[set[str]] = {
            "start_time",
            "start_datetime",
            "end_time",
            "end_datetime",
            "execution_time",
            "thread_id",
            "process_id",
            "initial_memory_usage",
            "final_memory_usage",
            "memory_usage_change",
            "initial_cpu_time",
            "final_cpu_time",
            "caller_function_name",
            "caller_module_name",
            "caller_module_path",
        }

        assert all(field in timer.metadata for field in required_fields)

    def test_class_method_detection(self) -> None:
        """Test detection of class method calls."""

        class TestClass:
            def test_method(self) -> None:
                with TimedExecution(pretty_print=False) as timer:
                    self.timer = timer

        test_instance = TestClass()
        test_instance.test_method()

        assert test_instance.timer.metadata["caller_class_name"] == "TestClass"
        assert test_instance.timer.metadata["caller_method_name"] == "test_method"


@timer
def sync_function(sleep_time: float = 0.1) -> str:
    time.sleep(sleep_time)
    return "sync done"


@timer
async def async_function(sleep_time: float = 0.1) -> str:
    await asyncio.sleep(sleep_time)
    return "async done"


class TestTimerDecorator:
    """Test the timer decorator implementations."""

    def test_sync_function(self) -> None:
        result: str = sync_function(0.3)
        assert result == "sync done"

    @pytest.mark.asyncio
    async def test_async_function(self) -> None:
        result: str = await async_function(0.3)
        assert result == "async done"


class TestTimerDecoratorClass:
    """Test the TimerDecorator class implementation."""

    @TimerDecorator
    def sync_method(self, sleep_time: float = 0.1) -> str:
        time.sleep(sleep_time)
        return "sync done"

    @TimerDecorator
    async def async_method(self, sleep_time: float = 0.1) -> str:
        await asyncio.sleep(sleep_time)
        return "async done"

    def test_sync_method(self) -> None:
        result: str = self.sync_method(0.1)  # type: ignore[call-overload]
        assert result == "sync done"

    @pytest.mark.asyncio
    async def test_async_method(self) -> None:
        result: str = await self.async_method(0.1)  # type: ignore[call-overload]
        assert result == "async done"
