---
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Singleton

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
```

````{tab} **logger.py**
```python
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Logger):
            return NotImplemented
        return id(self) == id(other)

    def log(self, message: str) -> None:
        print(f"LOG: {message}")
```
````

````{tab} **core.py**
```python
from omnixamples.software_engineering.design_patterns.singleton.logger import Logger

logger1 = Logger()
logger2 = Logger()

assert logger1 is logger2
assert logger1 == logger2
print(f"logger1 id={id(logger1)} | logger2 id={id(logger2)}")
```
````

## Project Structure Overview

To contextualize the implementation, let's visualize the directory structure of
the project:

```text
omnixamples/
└── software_engineering/
    └── design_patterns/
        └── singleton/
            ├── __init__.py
            ├── logger.py
            └── core.py
```

-   **`omnixamples/`**: Root directory of the project.
-   **`logger.py`**: Contains the `Logger` Singleton class implementation.
-   **`core.py`**: Demonstrates how to use the `Logger` Singleton.

## What is the Singleton Pattern?

The **Singleton** is a
[**creational design pattern**](https://refactoring.guru/design-patterns/singleton)
that ensures a class has **only one instance** and provides a **global point of
access** to that instance. This is particularly useful when exactly one object
is needed to coordinate actions across the system, such as:

-   **Logging Systems**
-   **Configuration Managers**

## Why Use a Singleton?

Consider a global settings object `Settings` hosting your platform definitions,
and you really want to ensure a single source of truth here. Same idea goes for
logger, so technically you can also instantiate the logger with the desired
configurations inside, say `__init__.py` and then use it anywhere in the
project.

## The Singleton Pattern in Detail

### Class Variables

The key to the singleton pattern is the use of class variables because they are
shared across all instances of the class - like a global variable.

```python
_instance: Logger | None = None
_lock: threading.Lock = threading.Lock()
```

-   **`_instance: Logger | None = None`:**

    -   **Purpose:** Holds the singleton instance of the `Logger` class.
    -   **Type Annotation:** Indicates that `_instance` can be either a `Logger`
        instance or `None`.
    -   **Initial Value:** Set to `None`, meaning no instance exists at the
        start.

-   **`_lock: threading.Lock = threading.Lock()`:**
    -   **Purpose:** A thread lock to ensure that only one thread can create an
        instance at a time.
    -   **Type Annotation:** Specifies that `_lock` is of type `threading.Lock`.
    -   **Initialization:** Creates a new lock instance.

### Overriding the `__new__` Method

```python
def __new__(cls: Type[Logger]) -> Logger:  # noqa: PYI034
    if cls._instance is None:
        with cls._lock:
            if cls._instance is None:
                cls._instance = object.__new__(cls)
    return cls._instance
```

-   **Purpose of `__new__`:**

    -   `__new__` is a **special method** responsible for **creating a new
        instance** of a class.
    -   It is called **before** `__init__`.
    -   Suitable for implementing singletons since it controls the instantiation
        process.

-   **Implementation Details:**

    1. **Check for Existing Instance:**
        - `if cls._instance is None:`: Determines if an instance already exists.
    2. **Acquire Lock for Thread Safety:**
        - `with cls._lock:`: Ensures that only one thread can execute the block
          at a time.
    3. **Double-Checked Locking:**
        - Inside the locked block, it **rechecks** `if cls._instance is None:`
          to prevent race conditions where multiple threads might have passed
          the first check simultaneously.
    4. **Create New Instance:**
        - `cls._instance = object.__new__(cls)`: Calls the base class (`object`)
          `__new__` method to create a new instance and assigns it to
          `_instance`.
    5. **Return the Instance:**
        - Regardless of whether a new instance was created or an existing one is
          used, `cls._instance` is returned.

### **e. Overriding the `__init__` Method**

```python
def __init__(self) -> None:
    """The initialized flag is used to prevent the __init__ method from
    being called more than once.
    """
    if not hasattr(self, "initialized"):
        self.initialized = True
        self.log(f"{self.__class__.__name__} initialized with id={id(self)}")
```

-   **Purpose of `__init__`:**

    -   `__init__` initializes the instance after it has been created by
        `__new__`.
    -   In the singleton pattern, it's crucial to prevent re-initialization if
        the instance already exists. Why so? Because the `__init__` method is
        called every time an instance is created, and we only want to run it
        once at some cases to prevent mutation of instance variables.

## Running Core

```python
LOG: Logger initialized with id=4379880944
LOG: Logger __init__ called again with id=4379880944
logger1 id=4379880944 | logger2 id=4379880944
```

We see that the memory address of the two logger instances are the same, which
indicates that they are the same instance - hence the singleton pattern is
working as expected.

## Thread Safety Considerations

-   **Race Conditions:** In multithreaded environments, multiple threads might
    attempt to create an instance simultaneously.
-   **Double-Checked Locking:** Ensures that only one thread can create the
    instance, preventing multiple instances from being created.

## References And Further Readings

-   [Singleton Design Pattern](https://refactoring.guru/design-patterns/singleton)
-   [Making Singletons in Python](https://www.pythonmorsels.com/making-singletons/)
