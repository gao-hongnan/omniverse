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

# Thread Safety

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
[![Code](https://img.shields.io/badge/View-Code-blue?style=flat-square&logo=github)](https://github.com/gao-hongnan/omniverse/blob/8ddcd6a345925e7fd550b74ce4577a0e2807aa5f/omnixamples/software_engineering/concurrency_parallelism_asynchronous/race_condition.py)

```{contents}
:local:
```

With pre-emptive multitasking, the OS can interrupt a thread at any time, even
in the middle of executing a single Python statement. This can lead to issues
like **race conditions**, where the program's behavior depends on the
unpredictable timing of threads.

Consider the below code:

```python
"""With reference to effective python book chapter 54.
Ref: https://github.com/bslatkin/effectivepython/blob/master/example_code/item_54.py
"""
import logging
import threading
from threading import Barrier
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_THREADS = 5
BARRIER = Barrier(NUM_THREADS)


class Counter:
    def __init__(self) -> None:
        self.count = 0

    def increment(self, offset: int) -> None:
        self.count += offset

def worker(thread_index: int, total_iterations: int, counter: Counter) -> None:
    """The barrier is used to synchronize the threads so that they all start counting
    at the same time. This makes it easier to get a race condition since we wait for
    the other threads to start else in the loop we always have an order that the
    first thread likely starts first and then the second and so on.
    """
    BARRIER.wait()
    logging.debug("Thread %s, starting", thread_index)
    for _ in range(total_iterations):
        counter.increment(1)


def thread_unsafe(total_iterations: int) -> None:
    counter = Counter()

    threads: List[threading.Thread] = []
    for index in range(NUM_THREADS):
        thread = threading.Thread(target=worker, args=(index, total_iterations, counter))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    expected = total_iterations * NUM_THREADS
    found = counter.count
    logging.info("Counter should be %s, got %s", expected, found)


if __name__ == "__main__":
    total_iterations = 10**6

    thread_unsafe(total_iterations)
```

There are a total of 5 threads, each incrementing the counter by 1 for $10^6$
times. So ideally the **_expected output_** should be
$5 \times 10^6 = 5,000,000$. However, it could be less, for example, we may get
$2,000,000$ or $3,000,000$ sometimes.

## Why Does This Happen?

The line `counter.increment(1)` seems atomic but is actually a shorthand for:

1. **Read:** Retrieve the current value of `counter`:
   `current_value = counter.count`
2. **Add:** Increment the value by 1: `new_value = current_value + 1`
3. **Write:** Store the new value back to `counter`: `counter.count = new_value`

If the OS pre-empts a thread **after** reading but **before** writing, another
thread might read the same initial value, leading to lost updates.

Let's pan out a scenario where this can happen:

1. Thread 1: `thread_1_current_value = counter.count = 0`
2. Thread 2: `thread_2_current_value = counter.count = 0`. This is called a
   _context switch_.
3. Thread 2: `thread_2_new_value = thread_2_current_value + 1 = 1`
4. Thread 2: `counter.count = thread_2_new_value = 1`.
5. Thread 1: `thread_1_new_value = thread_1_current_value + 1 = 1`
6. Thread 1: `counter.count = thread_1_new_value = 1`.

So we see thread 2 interrupted thread 1 after thread 1 read the value of
`counter` but before it could write the new value back. Then even though the
counter should be 2, it is 1 because thread 1 overwrote it.

## Lock

Python's threading module offers a suite of tools to prevent issues like data
races and data structure corruption in multithreaded environments. Among these,
the `Lock` class stands out as a particularly useful and straightforward option.
It implements a mutual-exclusion lock, commonly known as a mutex.

By incorporating a `Lock` into the `Counter` class, you can safeguard its
current value from concurrent access by multiple threads. The lock ensures that
only one thread can access the protected data at any given moment. To manage the
lock efficiently, you can utilize Python's with statement, which handles both
the acquisition and release of the lock automatically.

```python
"""With reference to effective python book chapter 54.
Ref: https://github.com/bslatkin/effectivepython/blob/master/example_code/item_54.py
"""
import logging
import threading
from threading import Barrier
from typing import List

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_THREADS = 5
BARRIER = Barrier(NUM_THREADS)

class CounterLock:
    def __init__(self) -> None:
        self.count = 0
        self.lock = threading.Lock()

    def increment(self, offset: int) -> None:
        with self.lock:
            self.count += offset


def worker(thread_index: int, total_iterations: int, counter: Counter) -> None:
    """The barrier is used to synchronize the threads so that they all start counting
    at the same time. This makes it easier to get a race condition since we wait for
    the other threads to start else in the loop we always have an order that the
    first thread likely starts first and then the second and so on.
    """
    BARRIER.wait()
    logging.debug("Thread %s, starting", thread_index)
    for _ in range(total_iterations):
        counter.increment(1)


def thread_safe(total_iterations: int) -> None:
    counter = CounterLock()

    threads: List[threading.Thread] = []
    for index in range(NUM_THREADS):
        thread = threading.Thread(target=worker, args=(index, total_iterations, counter))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    expected = total_iterations * NUM_THREADS
    found = counter.count

    logging.info("Counter should be %s, got %s", expected, found)


if __name__ == "__main__":
    total_iterations = 10**6

    thread_safe(total_iterations)
```

## References And Further Readings

-   [Chapter 54: Use Lock to Prevent Data Races in Threads - Effective Python](https://github.com/bslatkin/effectivepython/blob/master/example_code/item_54.py)
