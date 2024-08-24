# Reference: https://gist.github.com/anatoly-kussul/f2d7444443399e51e2f83a76f112364d
# Reference: https://stackoverflow.com/questions/44169998/how-to-create-a-python-decorator-that-can-wrap-either-coroutine-or-function

from __future__ import annotations

import functools
import inspect
import logging
import sys
import timeit
from contextlib import contextmanager
from typing import Any, Callable, Coroutine, Generator, Generic, Tuple, Type, TypeVar, cast, overload

from typing_extensions import ParamSpec

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
            f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute with args {args} and kwargs {kwargs}"
        )
        return result

    @functools.wraps(func)
    async def awrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = timeit.default_timer()
        result = await func(*args, **kwargs)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        logger.info(
            f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute with args {args} and kwargs {kwargs}"
        )
        return result

    if inspect.iscoroutinefunction(func):
        return awrapper
    else:
        return wrapper
