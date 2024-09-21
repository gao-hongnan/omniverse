# Reference: https://gist.github.com/anatoly-kussul/f2d7444443399e51e2f83a76f112364d
# Reference: https://stackoverflow.com/questions/44169998/how-to-create-a-python-decorator-that-can-wrap-either-coroutine-or-function

from __future__ import annotations

import functools
import inspect
import logging
import os
import sys
import threading
import time
import timeit
import types
from contextlib import contextmanager
from typing import Any, Callable, Coroutine, Dict, Generator, Generic, Optional, Tuple, Type, TypeVar, cast, overload

import psutil
from pydantic import BaseModel
from rich.pretty import pprint
from typing_extensions import ParamSpec, Self

T = TypeVar("T")
P = ParamSpec("P")
F = Callable[..., Any]


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class SyncAsyncDecoratorFactory(Generic[P, T]):
    """
    Factory creates decorator which can wrap either a coroutine or function.
    To return something from wrapper use self._return
    If you need to modify args or kwargs, you can yield them from wrapper
    """

    class ReturnValue(Exception):
        def __init__(self, return_value: Any) -> None:
            self.return_value = return_value

    @overload
    def __new__(cls: Type[SyncAsyncDecoratorFactory[P, T]]) -> SyncAsyncDecoratorFactory[P, T]:
        ...

    @overload
    def __new__(cls: Type[SyncAsyncDecoratorFactory[P, T]], func: Callable[P, T]) -> SyncAsyncDecoratorFactory[P, T]:
        ...

    def __new__(
        cls: Type[SyncAsyncDecoratorFactory[P, T]], *args: Any, **kwargs: Any
    ) -> SyncAsyncDecoratorFactory[P, T]:
        instance = super().__new__(cls)
        if len(args) == 1 and not kwargs and (inspect.iscoroutinefunction(args[0]) or inspect.isfunction(args[0])):
            instance.__init__()  # type: ignore[misc]
            return cast(SyncAsyncDecoratorFactory[P, T], instance(args[0]))
        return cast(SyncAsyncDecoratorFactory[P, T], instance)

    @contextmanager
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> Generator[Tuple[P.args, P.kwargs] | None, None, None]:
        raise NotImplementedError

    @classmethod
    def _return(cls, value: Any) -> None:
        raise cls.ReturnValue(value)

    @overload
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:  # type: ignore[misc]
        ...

    @overload
    def __call__(self, func: Callable[P, Coroutine[Any, Any, T]]) -> Callable[P, Coroutine[Any, Any, T]]:
        ...

    def __call__(
        self, func: Callable[P, T] | Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, T] | Callable[P, Coroutine[Any, Any, T]]:
        self.func = func

        @functools.wraps(func)
        def call_sync(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                with self.wrapper(*args, **kwargs) as new_args:
                    if new_args:
                        args, kwargs = new_args
                    result = self.func(*args, **kwargs)
                    return cast(T, result)
            except self.ReturnValue as r:
                return cast(T, r.return_value)

        @functools.wraps(func)
        async def call_async(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                with self.wrapper(*args, **kwargs) as new_args:
                    if new_args:
                        args, kwargs = new_args
                    result = await cast(Coroutine[Any, Any, T], self.func(*args, **kwargs))
                    return result
            except self.ReturnValue as r:
                return cast(T, r.return_value)

        return call_async if inspect.iscoroutinefunction(func) else call_sync


class TimerDecorator(SyncAsyncDecoratorFactory[P, T]):
    """This decorator might be too complicated, for learning purposes only."""

    @contextmanager
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> Generator[None, None, None]:
        start_time = timeit.default_timer()
        yield
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{self.func.__name__}' took {execution_time:.4f} seconds to execute with args {args} and kwargs {kwargs}"
        )


def timer(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute."
        )  # with args {args} and kwargs {kwargs}"

        return result

    @functools.wraps(func)
    async def awrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = timeit.default_timer()
        result = await func(*args, **kwargs)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute."
        )  # with args {args} and kwargs {kwargs}"

        return result

    if inspect.iscoroutinefunction(func):
        return awrapper
    else:
        return wrapper


class TimedExecutionMetadata(BaseModel):
    start_time: Optional[float] = None
    start_datetime: Optional[str] = None
    end_time: Optional[float] = None
    end_datetime: Optional[str] = None
    execution_time: Optional[float] = None
    thread_id: int
    process_id: int
    initial_memory_usage: int
    final_memory_usage: Optional[int] = None
    memory_usage_change: Optional[int] = None
    initial_cpu_time: psutil._common.pcputimes
    final_cpu_time: Optional[psutil._common.pcputimes]
    caller_function_name: str
    caller_module_name: str
    caller_module_path: str
    caller_class_name: Optional[str] = None
    caller_method_name: Optional[str] = None
    exception: Optional[str] = None
    tags: Dict[str, str] = {}


class TimedExecution:
    """Time profiler.

    Examples
    --------
    >>> def my_function() -> List[int]:
    ...     with TimedExecution(pretty_print=True) as timer:
    ...         a = [1] * 1000000000
    ...         return a
    ...
    >>> def my_error_function() -> None:
    ...     with TimedExecution(pretty_print=True) as timer:
    ...         a = 1
    ...         if a == 1:
    ...             raise ValueError("Error!")
    ...
    >>> class MyClass:
    ...     def __init__(self) -> None:
    ...         pass
    ...
    ...     def my_method(self) -> None:
    ...         with TimedExecution(pretty_print=True) as timer:
    ...             time.sleep(5)
    ...
    >>> my_function()
    >>> MyClass().my_method()
    >>> my_error_function()
    """

    def __init__(
        self: Self,
        metadata: Dict[str, Any] | None = None,
        tags: Dict[str, str] | None = None,
        pretty_print: bool = True,
    ) -> None:
        self.metadata = metadata or {}
        self.tags = tags or {}
        self.pretty_print = pretty_print
        self.process = psutil.Process(os.getpid())

    def __enter__(self: Self) -> Self:
        self.start_time = timeit.default_timer()
        self.metadata["start_time"] = self.start_time
        self.metadata["start_datetime"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["thread_id"] = threading.get_ident()
        self.metadata["process_id"] = os.getpid()
        self.metadata["initial_memory_usage"] = self.process.memory_info().rss
        self.metadata["initial_cpu_time"] = self.process.cpu_times()
        self.metadata.update(self.tags)

        stack = inspect.stack()
        caller_frame = stack[1]
        frame_info = inspect.getframeinfo(caller_frame.frame)

        self.metadata["caller_function_name"] = caller_frame.function
        self.metadata["caller_module_name"] = caller_frame.frame.f_globals["__name__"]
        self.metadata["caller_module_path"] = frame_info.filename

        if "self" in caller_frame.frame.f_locals:
            self_instance = caller_frame.frame.f_locals["self"]
            self.metadata["caller_class_name"] = self_instance.__class__.__name__
            self.metadata["caller_method_name"] = caller_frame.function

        return self

    def __exit__(
        self: Self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.end_time = timeit.default_timer()
        self.metadata["end_time"] = self.end_time
        self.metadata["end_datetime"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["execution_time"] = self.end_time - self.start_time
        self.metadata["final_memory_usage"] = self.process.memory_info().rss
        self.metadata["memory_usage_change"] = (
            self.metadata["final_memory_usage"] - self.metadata["initial_memory_usage"]
        )
        self.metadata["final_cpu_time"] = self.process.cpu_times()
        if exc_type and exc_val and exc_tb:
            self.metadata["exception"] = f"{exc_type.__name__}: {exc_val} traceback: {exc_tb}"

        if self.pretty_print:
            pprint(TimedExecutionMetadata(**self.metadata))
