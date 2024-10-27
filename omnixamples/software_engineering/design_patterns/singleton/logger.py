from __future__ import annotations

import threading
from typing import Type


class Logger:
    _instance: Logger | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls: Type[Logger]) -> Logger:  # noqa: PYI034
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """The initialized flag is used to prevent the __init__ method from
        being called more than once.
        """
        if not hasattr(self, "initialized"):
            self.initialized = True
            self.log(f"{self.__class__.__name__} initialized with id={id(self)}")
        else:
            self.log(f"{self.__class__.__name__} __init__ called again with id={id(self)}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Logger):
            return NotImplemented
        return id(self) == id(other)

    def log(self, message: str) -> None:
        print(f"LOG: {message}")
