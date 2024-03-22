---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Sentinel Types

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from __future__ import annotations

from typing import Any, Literal, Type

from typing_extensions import override
```

## Motivation

Inspired by OpenAI's approaches in software development and type handling, this
discussion explores the concept of sentinel types, a powerful technique in for
representing unique default values or states. Sentinel types are particularly
useful in scenarios where `None` might be a valid input value, necessitating a
distinct marker for "no value given" cases. In fact, `None` is in itself a form
of sentinel value, a singleton object that represents the absence of a value.

While this exploration introduces the core idea as formalized in
[PEP 661 – Sentinel Values](https://peps.python.org/pep-0661/), which advocates
for a standardized approach to sentinel values in Python, we'll steer clear of
the more formalized specifics. Readers interested in a deeper dive into the
formal aspects of sentinel types are encouraged to consult the original
[PEP 661 – Sentinel Values](https://peps.python.org/pep-0661/) for comprehensive
insights and technical details.

## `NotGiven`

OpenAI's own implementation of a sentinel type, `NotGiven`, was introduced
[here in OpenAI's GitHub repository](https://github.com/openai/openai-python/blob/7367256070a975921ed4430f55d17dc0a9319f21/src/openai/_types.py#L273).

```{code-cell} ipython3
class _NotGiven:

    _instance: _NotGiven | None = None

    def __new__(cls: Type[_NotGiven]) -> _NotGiven:  # noqa: PYI034
        if cls._instance is None:
            cls._instance = super(_NotGiven, cls).__new__(cls)  # noqa: UP008
        return cls._instance

    def __bool__(self) -> Literal[False]:
        """
        This method is used to define the boolean value of an instance of `_NotGiven`.
        By returning `False`, it allows `_NotGiven` to be used in boolean contexts (like
        `if` statements) to signify the absence of a value. This is especially useful
        for checking if an argument was provided or not in a function.
        """
        return False

    @override
    def __repr__(self) -> Literal["NOT_GIVEN"]:
        return "NOT_GIVEN"

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"{self.__class__.__name__} instances are immutable")

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"{self.__class__.__name__} instances are immutable")


NOT_GIVEN = _NotGiven()
```

### Purpose and Behavior

The use of such sentinels indicates that a parameter was not provided at all.
It's used to distinguish between a parameter being explicitly set to `None` and
not being provided. Structurally, both `None` and `NotGiven` are singleton
instances and are considered "falsy" in boolean contexts. This allows them to be
used in conditional statements to check if a value was provided or not. The
singleton property ensures that all instances of `NotGiven` are equal to each
other.

### Use Case 1. Timeouts in HTTP Requests

The sentinel pattern like `NotGiven` is common in APIs where default behavior is
triggered when a parameter is not given, but `None` might be a valid, meaningful
input. For example, `None` might mean "disable timeout", while `NotGiven` means
"use a default timeout".

Consider you implemented a function `get` (not to be confused with the method
`get` from the `requests` library) to make HTTP requests, the `timeout`
parameter specifies the maximum number of seconds to wait for a response. If
`timeout` is set to `None`, it means that there is no timeout limit for the
request. In other words, the request will wait indefinitely until the server
responds or the connection is closed.

Here, we will use a relatively simple example to illustrate. Consider the
following function call `get` that takes in a argument `timeout` that defines
how many seconds to wait before raising a `TimeoutError`. If user specifies
`None`, it means that this program should have no timeout, and therefore should
run indefinitely until a server or something responds to halt.

```{code-cell} ipython3
import time

def get(timeout: int | None = 2) -> int | float:
    if timeout is None:
        actual_timeout = float("inf")
    else:
        actual_timeout = timeout
    return actual_timeout

print(f"Use default timeout: {get()}")
print(f"Use 2 seconds timeout: {get(timeout=2)}")
print(f"Use 3 seconds timeout: {get(timeout=3)}")
print(f"Use no timeout: {get(timeout=None)}")
```

What is the issue here? Not much. But one quirk is that the program has no
elegant way to distinguish whether a user passed in a default value or not.

```{code-cell} ipython3
print(f"Use default timeout: {get()}")
print(f"Use 2 seconds timeout: {get(timeout=2)}")
```

The above two will yield the same result, because the `timeout` has a default
value of `2`, so when the function is called without specifying `timeout`, it
automatically takes the value of `2` - which is the standard behaviour for
default values.

This approach does not disinguish between an user not providing the argument at
all and an user explicitly setting the argument to its default value.

Why does it matter? Besides the reason of expressing user intent and
explicitness, we can argue that we want more fine-grained behaviour control of
our program. If user pass in their own values, we may want to check whether that
value is within bounds, or in other words, legitimate.

The key motivations for using a singleton sentinel class are primarily centered
around distinguishing between different states of function arguments, especially
in the context of **default** values and **optional** arguments.

1. **Differentiating Between 'None', 'Default Values' and 'Not Provided':** In
   Python, `None` is often used as a default value for function arguments.
   However, there are situations where `None` is a meaningful value distinct
   from the absence of a value. The `NotGiven` singleton allows you to
   differentiate between a user explicitly passing `None` (which might have a
   specific intended behavior) and not passing any value at all.
2. **Default Behavior Control:** By using a sentinel like `NotGiven`, we can
   implement a default behavior that is only triggered when an argument is
   **not** provided. This is different from setting a default value in the
   function definition, as it allows the function to check if the user has
   explicitly set the argument, even if it's set to `None`.
3. **Semantic Clarity:** In complex APIs or libraries, using a sentinel value
   can provide clearer semantics. It makes the intention of the code more
   explicit, both for the developer and for users of the API. It indicates that
   thought has been given to the different states an argument can be in, and
   different behaviors are intentionally designed for each state.

```{code-cell} ipython3
def get_with_not_given(timeout: int | _NotGiven | None = NOT_GIVEN) -> int | float:
    actual_timeout: int | float
    if timeout is NOT_GIVEN:
        actual_timeout = 2
    elif timeout is None:
        actual_timeout = float("inf")
    else:
        assert isinstance(timeout, int)
        actual_timeout = timeout
    return actual_timeout

print(f"Use default timeout: {get_with_not_given()}")
print(f"Use 2 seconds timeout: {get_with_not_given(timeout=2)}")
print(f"Use 3 seconds timeout: {get_with_not_given(timeout=3)}")
print(f"Use no timeout: {get_with_not_given(timeout=None)}")
```

## Missing

Another common sentinel type is `MISSING`, which is used to represent a missing
value in data structures or configurations (for e.g. in
[Dataclasses](https://github.com/python/cpython/blob/b4dd31409cf6f23dcd8113d10f74a2e41b8cb1ed/Lib/dataclasses.py#L186)).

```{code-cell} ipython3
class _Missing:
    """
    -   **Primary Use:** `MISSING` is more common in data structures,
        configurations, or APIs where you need to signify that a value hasn't been
        set or provided, and it's expected to be present or filled in later.
    -   **Semantics:** It indicates the absence of a value in a more passive sense,
        as in "not yet provided" or "awaiting assignment."
    -   **Example:** In a configuration object, `None` might be used to disable an
        option, whereas `MISSING` would indicate that the user has not yet made a
        decision about that option.
    """

    _instance: _Missing | None = None

    def __new__(cls: Type[_Missing]) -> _Missing:  # noqa: PYI034
        if cls._instance is None:
            cls._instance = super(_Missing, cls).__new__(cls)  # noqa: UP008
        return cls._instance

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["MISSING"]:
        return "MISSING"

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(f"{self.__class__.__name__} instances are immutable")

    def __delattr__(self, key: str) -> None:
        raise AttributeError(f"{self.__class__.__name__} instances are immutable")


MISSING = _Missing()
```

### Purpose and Behavior

The typical use case of `MISSING` is often used in data structures or
configurations to indicate that a value is missing or has not been set. It's
particularly useful in contexts like dictionaries, APIs, or data processing
where you need to differentiate between a value that is intentionally set to
`None` and a value that is not provided at all.

For example, in a configuration dictionary where each key is supposed to map to
a specific value, `MISSING` could be used to represent keys that have not been
assigned a value yet. It signals that the value is expected but not available,
which is different from being intentionally set to `None`.

```{code-cell} ipython3
config = {
    "timeout": 30,
    "mode": MISSING,  # Indicates that the mode setting is yet to be configured
}
if config["mode"] is MISSING:
    print("Mode is not yet configured")
```

### `NotGiven` vs. `MISSING`

-   Use `NOTGIVEN` to explicitly indicate that no value has been provided for a
    parameter, especially when `None` is a valid input with a specific meaning.
-   Use `MISSING` to represent an absent or unassigned value in data structures
    or configurations, where you need to differentiate between an unassigned
    state and a value explicitly set to `None`.

## References and Further Readings

-   [PEP 661 – Sentinel Values](https://peps.python.org/pep-0661/)
-   [OpenAI's `NotGiven` Implementation](https://github.com/openai/openai-python/blob/7367256070a975921ed4430f55d17dc0a9319f21/src/openai/_types.py#L273)
-   [Dataclasses `MISSING` Implementation](https://github.com/python/cpython/blob/b4dd31409cf6f23dcd8113d10f74a2e41b8cb1ed/Lib/dataclasses.py#L186)
