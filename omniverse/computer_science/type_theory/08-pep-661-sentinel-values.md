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
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-MaybeWrong-red)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, List, Union, Any, Generic, Literal, TypeVar, Dict, Tuple, TYPE_CHECKING, Iterable, Iterator, Sequence, overload, Optional

from rich.pretty import pprint
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

OpenAI's own implementation of a sentinel type, `NotGiven`, exemplifies
practical application of this concept and can be seen
[here in OpenAI's GitHub repository](https://github.com/openai/openai-python/blob/7367256070a975921ed4430f55d17dc0a9319f21/src/openai/_types.py#L273).

-   **Purpose**: Indicates that a parameter was not provided at all. It's used
    to distinguish between a parameter being explicitly set to `None` and not
    being provided.
-   **Use Case**: Common in APIs where default behavior is triggered when a
    parameter is not given, but `None` might be a valid, meaningful input. For
    example, `None` might mean "disable timeout", while `NotGiven` means "use a
    default timeout".
-   Other example usage is if you want to assign a default empty list or dict
    but it is mutable, so you assign this type but not None since None don't
    make sense.
-   **Behavior**: Functions can check for `NotGiven` to apply default behavior.

## `Omit`

-   **Purpose**: Used to explicitly remove or omit a default value that would
    otherwise be applied. It's not just about a value being absent, but rather
    about actively removing a pre-existing default.
-   **Use Case**: Useful in situations where the default behavior or value needs
    to be explicitly overridden or disabled, and where `None` is not a suitable
    option. For example, removing a default HTTP header.
-   **Behavior**: Functions can check for `Omit` to actively remove or ignore a
    default setting or value.

### Comparison

-   **Similarity**: Both are used to signal special cases in the absence of
    normal parameter values.
-   **Difference**: `NotGiven` is about the absence of a value where a default
    may apply, while `Omit` is about actively overriding a default. """

In Python, when using the `requests` library to make HTTP requests, the
`timeout` parameter specifies the maximum number of seconds to wait for a
response. If `timeout` is set to `None`, it means that there is no timeout limit
for the request. In other words, the request will wait indefinitely until the
server responds or the connection is closed.

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

print(get())
print(get(timeout=2))
print(get(timeout=3))
print(get(timeout=None))
```

What is the issue here? Not much. But one quirk is that the program has no
elegant way to distinguish whether a user passed in a default value or not.

```python
print(get())
print(get(timeout=2))
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
def get_with_not_given(timeout: int | NotGiven | None = NOT_GIVEN) -> int | float:
    actual_timeout: Union[int, float]
    if timeout is NOT_GIVEN:
        actual_timeout = 2
    elif timeout is None:
        actual_timeout = float("inf")
    else:
        assert isinstance(timeout, int)
        actual_timeout = timeout
    return actual_timeout

print(get_with_not_given())
print(get_with_not_given(timeout=2))
print(get_with_not_given(timeout=3))
print(get_with_not_given(timeout=None))
```

### A More Practical Example: Database Query with Optional Filters

Consider a function that constructs a database query. In this scenario, the
function might accept several optional parameters that act as filters for the
query. The use of `NOT_GIVEN` allows us to differentiate between a filter being
intentionally set to `None` (indicating the desire to include records where the
field is `NULL`), a filter being set to a specific value, or the filter not
being used at all.

#### Example: `query_database`

```python
class NOT_GIVEN:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "NOT_GIVEN()"

def query_database(name_filter: Union[str, None, NOT_GIVEN] = NOT_GIVEN(),
                   age_filter: Union[int, None, NOT_GIVEN] = NOT_GIVEN()):
    query = "SELECT * FROM users"

    where_clauses = []
    if name_filter is not NOT_GIVEN:
        if name_filter is None:
            where_clauses.append("name IS NULL")
        else:
            where_clauses.append(f"name = '{name_filter}'")

    if age_filter is not NOT_GIVEN:
        if age_filter is None:
            where_clauses.append("age IS NULL")
        else:
            where_clauses.append(f"age = {age_filter}")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    return query

# Usage examples
print(query_database(name_filter="Alice"))       # Filter by name 'Alice'
print(query_database(age_filter=None))           # Filter by age being NULL
print(query_database())                          # No filters applied
print(query_database(name_filter=None, age_filter=30))  # Filter by name being NULL and age 30
```

#### Explanation

-   **Specific Value Provided:** If a specific value is provided (like `"Alice"`
    for `name_filter`), the function includes this in the query as a filter.
-   **`None` Value Provided:** If `None` is passed (like `None` for
    `age_filter`), the function interprets this as a requirement to include
    records where the corresponding field is `NULL`.
-   **No Value Provided (`NOT_GIVEN`):** If no value is provided, the function
    does not include the corresponding filter in the query. This is different
    from filtering where the field is `NULL`.

This example showcases a scenario where the distinction made by `NOT_GIVEN`
significantly alters the behavior of the function, demonstrating its practical
utility in a real-world context.

### Threading Example

```{code-cell} ipython3
import time
import threading

def perform_task(timeout: int | None, thread_id: int):
    try:
        print(f"Thread {thread_id}: Starting a task...")
        if timeout is None:
            # Simulate a task that runs indefinitely
            while True:
                time.sleep(1)
        else:
            time.sleep(timeout)
        print(f"Thread {thread_id}: Task completed successfully.")
    except TimeoutError:
        print(f"Thread {thread_id}: Task timed out and was terminated.")

def get(timeout: int | None = 2) -> None:
    thread_id = threading.get_ident()
    task_thread = threading.Thread(target=perform_task, args=(timeout, thread_id))
    task_thread.start()

    if timeout is not None:
        buffer_time = 0.1
        task_thread.join(timeout + buffer_time)
        if task_thread.is_alive():
            print(f"Thread {thread_id}: Raising TimeoutError.")
            raise TimeoutError
    else:
        task_thread.join()


# Usage examples
get(10)   # Task with a 5-second timeout
#time.sleep(3)
get(None) # Task with no timeout (indefinite task)
```

### A More Practical Example: Database Query with Optional Filters

Consider a function that constructs a database query. In this scenario, the
function might accept several optional parameters that act as filters for the
query. The use of `NOT_GIVEN` allows us to differentiate between a filter being
intentionally set to `None` (indicating the desire to include records where the
field is `NULL`), a filter being set to a specific value, or the filter not
being used at all.

#### Example: `query_database`

```python
class NOT_GIVEN:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "NOT_GIVEN()"

def query_database(name_filter: Union[str, None, NOT_GIVEN] = NOT_GIVEN(),
                   age_filter: Union[int, None, NOT_GIVEN] = NOT_GIVEN()):
    query = "SELECT * FROM users"

    where_clauses = []
    if name_filter is not NOT_GIVEN:
        if name_filter is None:
            where_clauses.append("name IS NULL")
        else:
            where_clauses.append(f"name = '{name_filter}'")

    if age_filter is not NOT_GIVEN:
        if age_filter is None:
            where_clauses.append("age IS NULL")
        else:
            where_clauses.append(f"age = {age_filter}")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    return query

# Usage examples
print(query_database(name_filter="Alice"))       # Filter by name 'Alice'
print(query_database(age_filter=None))           # Filter by age being NULL
print(query_database())                          # No filters applied
print(query_database(name_filter=None, age_filter=30))  # Filter by name being NULL and age 30
```

#### Explanation

-   **Specific Value Provided:** If a specific value is provided (like `"Alice"`
    for `name_filter`), the function includes this in the query as a filter.
-   **`None` Value Provided:** If `None` is passed (like `None` for
    `age_filter`), the function interprets this as a requirement to include
    records where the corresponding field is `NULL`.
-   **No Value Provided (`NOT_GIVEN`):** If no value is provided, the function
    does not include the corresponding filter in the query. This is different
    from filtering where the field is `NULL`.

This example showcases a scenario where the distinction made by `NOT_GIVEN`
significantly alters the behavior of the function, demonstrating its practical
utility in a real-world context.

## NotGiven vs Missing

The choice between using a sentinel like `MISSING` versus `NOTGIVEN` often
depends on the specific context and semantics you want to convey in your code.
Let's explore the typical use cases for each to understand when one might be
more appropriate than the other.

### `NOTGIVEN`

-   **Typical Use Case:** `NOTGIVEN` is generally used to represent the absence
    of a value in scenarios where `None` is a valid and meaningful input. This
    is particularly relevant in function arguments where you need to distinguish
    between "no argument provided" and "argument explicitly set to None."

-   **Example Context:** Consider a function with an optional parameter where
    `None` has a specific semantic meaning (like turning off a feature or using
    a default setting). If you also need to implement a different default
    behavior when the user does not provide any value, `NOTGIVEN` can be used to
    make this distinction.

-   **Code Example:**
    ```python
    def configure(setting=None, flag=NOTGIVEN):
        if flag is NOTGIVEN:
            # Apply some default behavior
        elif flag is None:
            # Disable the feature
        else:
            # Use the provided flag value
    ```

### `MISSING`

-   **Typical Use Case:** `MISSING` is often used in data structures or
    configurations to indicate that a value is missing or has not been set. It's
    particularly useful in contexts like dictionaries, APIs, or data processing
    where you need to differentiate between a value that is intentionally set to
    `None` and a value that is not provided at all.

-   **Example Context:** In a configuration dictionary where each key is
    supposed to map to a specific value, `MISSING` could be used to represent
    keys that have not been assigned a value yet. It signals that the value is
    expected but not available, which is different from being intentionally set
    to `None`.

-   **Code Example:**
    ```python
    config = {
        "timeout": 30,
        "mode": MISSING,  # Indicates that the mode setting is yet to be configured
    }
    if config["mode"] is MISSING:
        # Handle the case where mode is not set
    ```

### Summary

-   Use `NOTGIVEN` to explicitly indicate that no value has been provided for a
    parameter, especially when `None` is a valid input with a specific meaning.
-   Use `MISSING` to represent an absent or unassigned value in data structures
    or configurations, where you need to differentiate between an unassigned
    state and a value explicitly set to `None`.

The choice depends on what you're trying to communicate: `NOTGIVEN` emphasizes
the behavior of function arguments, while `MISSING` emphasizes the state of data
or configuration.

## References and Further Readings

-   [PEP 661 – Sentinel Values](https://peps.python.org/pep-0661/)
