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

# A Rudimentary Introduction to Generator and Yield in Python

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Brain_Dump-red)
![Tag](https://img.shields.io/badge/Level-Beginner-green)
![Tag](https://img.shields.io/badge/Tag-Generator-orange)
![Tag](https://img.shields.io/badge/Tag-Yield-orange)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from typing import Generator, List, Union, Any, Iterator, Literal, Sized, TypeVar
from rich.pretty import pprint
import inspect
import sys
import cProfile
from torch.utils.data import DataLoader, Dataset
import torch
```

**Generator functions**, which were introduced in
[Python Enhancement Proposal (PEP) 255](https://www.python.org/dev/peps/pep-0255),
are a unique type of function that yield a
[lazy iterator](https://en.wikipedia.org/wiki/Lazy_evaluation). This is an
object that can be iterated over, similar to a list. The key difference,
however, is that unlike lists, lazy iterators do not hold their contents in
memory. Instead, they generate their contents on the fly, as they are iterated
over.

## Reading Large Files, Generator vs Iterator

Let's consider the following example from
[How to Use Generators and yield in Python - Real Python](https://realpython.com/introduction-to-python-generators/)[^1].
Suppose we have a large text file that we want to read and iterate over, say, to
obtain the total number of rows.

### Reading a Big File into a List

We can use the following code to read the file first into _memory_ and then
iterate over it:

```{code-cell} ipython3
def file_reader_using_iterator(file_path: str) -> List[str]:
    file = open(file_path, "r", encoding="utf-8")
    print(f"Is file an generator? {inspect.isgenerator(file)}")
    print(f"Is file an iterator? {isinstance(file, Iterator)}")
    result = file.read().split("\n")
    return result

text = file_reader_using_iterator("./assets/sample.txt")
print(f"Is text an generator? {inspect.isgenerator(text)}")
print(f"Is text an iterator? {isinstance(text, Iterator)}")
pprint(text)

row_count = 0
for row in text:
    row_count += 1

print(f"Row count: {row_count}")
```

In `file_reader_using_iterator`, we read the entire file into memory and then
split it and return it as a list of strings (list is an iterator). Then we
_iterate_ over the list to count the number of rows. This approach is
straightforward and easy to understand, but it has a major drawback: it reads
the entire file into memory. This is not a problem for small files, but for
large files, it can be cause memory issues - the file itself being larger than
your system's available memory.

When you read a big file into a list, you're loading the entire content of the
file into memory at once. This is because a list in Python is an in-memory data
structure, and when you create a list containing every line of a file, each of
those lines is stored in memory. This can be highly inefficient for large files,
as it requires enough memory to hold the entire file content at once, which can
lead to `MemoryError` if the file size exceeds the available memory.

### Using a Generator Function

To overcome this issue, we can use a generator function,
`file_reader_using_generator`, which reads the file line by line, yielding each
line as it goes. This approach is memory-efficient because it only needs to hold
one line in memory at a time, not the entire file.

```{code-cell} ipython3
def file_reader_using_generator(file_path: str) -> Generator[str, None, None]:
    file = open(file_path, "r", encoding="utf-8")
    for row in file:
        yield row.rstrip("\n")

text_gen = file_reader_using_generator("./assets/sample.txt")
print(f"Is text_gen a generator? {inspect.isgenerator(text_gen)}")
print(f"Is text_gen an iterator? {isinstance(text_gen, Iterator)}")

row_count = 0
for row in text_gen:
    row_count += 1

print(f"Row count: {row_count}")
```

In `file_reader_using_generator`, we open the file and iterate over it line by
line. For each line, we yield the line, which means we produce a value that can
be iterated over, but we do not terminate the function. Instead, we pause it
until the next value is requested. This allows us to read large files
efficiently, even if they are larger than the available memory.

How does this work? On a high level, when we call `file_reader_using_generator`,
it returns a generator object. This object is an iterator, so we can iterate
over it using a `for` loop. When we do this, the function is executed until the
first `yield` statement, at which point the function is paused. The value of the
`yield` statement is returned to the caller, and the function is paused. When
the next value is requested, the function resumes from where it was paused,
until the next `yield` statement is encountered. This process continues until
the function terminates.

```{prf:remark} All Generators are Iterators
:label: software-engineering-concurrency-parallelism-asynchronous-generator-yield-remark

Note that all generators are iterators, but not all iterators are generators.
```

To recap, the main reason why the generator function does not hold the entire
file in memory is because it yields each line one by one, rather than returning
a list of all lines at once.

1. **Lazy Evaluation**: Generators are lazily evaluated. This means that they
   generate values on the fly as needed, rather than computing them all at once
   and storing them.

2. **Single Item in Memory**: At any point in time, only the current row being
   yielded by the generator is held in memory. Once the consumer of the
   generator moves to the next item, the previous item can be garbage collected
   if no longer referenced, keeping the memory footprint low.

3. **Stateful Iteration**: The generator function maintains its state between
   each yield. It knows where it left off (which line it yielded last) and
   resumes from that point the next time the next value is requested. This
   statefulness is managed without keeping the entire dataset in memory.

## Next Method

The `__next__` method is a fundamental part of the iterator protocol in Python.
It's used to get the next value in an iteration.

When you use a `for` loop, or the `next()` function, Python internally calls the
`__next__` method of the iterator object. This method should return the next
value for the iterable. When there are no more items to return, it should raise
`StopIteration`.

In the context of generators, each call to the generator's `__next__` method
resumes the generator function from where it left off and runs until the next
`yield` statement, at which point it returns the yielded value and pauses
execution.

Without the `__next__` method, we wouldn't be able to use Python's built-in
iteration mechanisms with our custom iterator or generator objects.

So let's see how the `__next__` method works with our generator function.

```{code-cell} ipython3
try:
    first_row = text_gen.__next__()
    print(f"First row: {first_row}")
except StopIteration:
    print("StopIteration: No more rows")
```

Oh what happened? We try to get the first row from the generator using the
`__next__` method, but it raises a `StopIteration` exception. This is because
the generator has already been exhausted by the `for` loop earlier when we
counted the number of rows. Unlike a list, a generator can only be iterated over
once. Once it's been exhausted, it can't be iterated over again and will raise a
`StopIteration` exception if you try to do so.

Let's create a new generator and see how the `__next__` method works.

```{code-cell} ipython3
text_gen = file_reader_using_generator("./assets/sample.txt")
first_row = text_gen.__next__()
print(f"First row: {first_row}")

second_row = text_gen.__next__()
print(f"Second row: {second_row}")
```

## Generator Expression

Similar to
[list comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions),
you can also create a generator using a generator expression (generator
comprehension) so that you can create a genereator without defining a function.

```{code-cell} ipython3
text_gen_comprehension = (row for row in open("./assets/sample.txt", "r", encoding="utf-8"))
print(f"Is text_gen_comprehension a generator? {inspect.isgenerator(text_gen_comprehension)}")
print(f"Is text_gen_comprehension an iterator? {isinstance(text_gen_comprehension, Iterator)}")
```

### How does Generator Work?

Generator functions are nearly indistinguishable from standard functions in
appearance and behavior, with one key distinction. They utilize the `yield`
keyword in place of `return`. Consider the generator function that yields the
next integer indefinitely:

```python
def infinite_sequence() -> Generator[int, None, None]:
    num = 0
    while True:
        yield num
        num += 1
```

This function might look familiar, but it's the `yield` statement that sets it
apart. `yield` serves to return a value to the caller **without exiting the
function**.

What's truly unique here is how the function's **state is preserved**. Upon each
subsequent call to `next()` on the generator object (whether done directly or
through a loop), the function picks up right where it left off, incrementing and
yielding `num` once more[^1].

## Profiling Generator Performance

Let's compare the performance of the generator function and the list
comprehension.

### Memory Efficiency

Let's create a list of squared numbers using a list comprehension and a
generator expression, and compare their memory usage.

```{code-cell} ipython3
N = 100000

nums_squared_list_comprehension = [num ** 2 for num in range(N)]
print(f"Size of nums_squared_list_comprehension: {sys.getsizeof(nums_squared_list_comprehension)} bytes")

nums_squared_generator = (num ** 2 for num in range(N))
print(f"Size of nums_squared_generator: {sys.getsizeof(nums_squared_generator)} bytes")
```

-   The size of `nums_squared_list_comprehension` is 800984 bytes.
-   The size of `nums_squared_generator` is 112 bytes.

The list comprehension (`nums_squared_list_comprehension`) creates a list of all
squared numbers at once. This means it needs to allocate enough memory to hold
all these numbers. This can be quite large for big sequences, as shown by the
`sys.getsizeof(nums_squared_list_comprehension)` call.

On the other hand, the generator expression (`nums_squared_generator`) doesn't
compute all the squared numbers at once. Instead, it computes them one at a
time, on-the-fly, as you iterate over the generator. This means it doesn't need
to allocate memory for the whole sequence, only for the current number. This is
why `sys.getsizeof(nums_squared_generator)` returns a much smaller number.

This demonstrates the main advantage of generators when it comes to memory
efficiency: they allow you to work with large sequences of data without needing
to load the entire sequence into memory. This can be a significant advantage
when working with large data sets, where loading the entire data set into memory
might not be feasible.

### Time Efficiency

We know that creating a very large list in memory can take time and potentially
hang the system. However, if our list is much smaller than the system memory, it
has been shown that it can much
[faster to evaluate](https://stackoverflow.com/questions/11964130/list-comprehension-vs-generator-expressions-weird-timeit-results/11964478#11964478)
than a generator expression[^1].

```{code-cell} ipython3
cProfile.run("sum([num ** 2 for num in range(N)])")
cProfile.run("sum(num ** 2 for num in range(N))")
```

This shows for a small list, the list comprehension is faster than the generator
expression.

Now, why are there 100005 function calls for N=100000?

-   `100001` calls are from the generator expression `<string>:1(<genexpr>)`.
    For each number in the `range(N)`, the generator expression is called once,
    hence 100000 calls. The extra 1 call is to raise the `StopIteration`
    exception when the generator is exhausted.

-   The remaining `4` calls are from the other functions:
    `<string>:1(<module>)`, `{built-in method builtins.exec}`,
    `{built-in method builtins.sum}`, and
    `{method 'disable' of '_lsprof.Profiler' objects}`. Each of these is called
    once, hence 4 calls.

So, in total, there are 100001 (from the generator expression) + 4 (from the
other functions) = 100005 function calls.

## Yield

The `yield` statement in Python is probably what defines the generator
functions, let's take a look.

### The `yield` Statement

A **generator function** is a **function** that, when called, returns a
**generator iterator**. This is achieved by including at least one `yield`
statement in the function definition. **Unlike a `return` statement, which
terminates a function entirely and sends a value back to the caller, `yield`
pauses the function, saving its state for continuation when next required.**

When a generator function calls `yield`, the function execution is **paused**,
and a value is sent to the caller. However, the function's **local variables and
execution state are saved internally**. The next time the generator is advanced
(using the `next()` function or a for loop, for example), execution resumes from
exactly where it was left off, immediately after the `yield` statement.

More concretely (and verbosely), upon encountering a `yield` statement, the
function's current state is preserved or "frozen". This means that local
variables, the instruction pointer, and even the state of the evaluation stack
are saved. Consequently, when `.next()` is called again, the function resumes
precisely from where it left off, as if `yield` were a pause in execution rather
than an interruption. This mechanism allows generator functions to produce a
sequence of values over time, providing an efficient way to work with data
streams or large datasets without requiring all data to be loaded into memory
simultaneously[^2].

Moreover, when a generator function is called, the actual arguments are bound to
function-local formal argument names in the usual way, but no code in the body
of the function is executed. Instead a generator-iterator object is returned;
this conforms to the iterator protocol, so in particular can be used in
for-loops in a natural way. Note that when the intent is clear from context, the
unqualified name "generator" may be used to refer either to a generator-function
or a generator-iterator[^2].

### An Example

Let's see a function `count_up_to` that just yield count up to a max number.

```{code-cell} ipython3
def count_up_to(max: int) -> Generator[int, None, None]:
    count = 1
    while count <= max:
        yield count
        count += 1

gen = count_up_to(3)
for number in gen:
    print(number)
```

First, let's add type hints to the function. The return type of a generator
function can be hinted using `Generator[YieldType, SendType, ReturnType]` from
the `typing` module. Since this generator yields integers and does not
explicitly return a value, we'll use `None` for both the `SendType` and
`ReturnType`.

Secondly, as we see it prints from `1` to `3` incrementally.

-   When `yield count` is executed, the generator pauses and returns control to
    the caller,
-   The caller then resumes the generator, which continues execution from where
    it left off,
-   So `count += 1` is executed, incrementing the counter,
-   Then the loop condition `count <= max` is checked before yielding again.

So the key thing is - the generator resumes and continues execution from where
it previously yielded each time `next()` or `send()` is called.

#### Adding a Return Statement

The `yield` statement allows the generator to produce a series of values, while
the `return` statement can be used to terminate the generator and, optionally,
to provide a value that is accessible through the `StopIteration` exception
raised when the generator is exhausted.

Let's modify the function to return a message when the count exceeds the
maximum:

```{code-cell} ipython3
def count_up_to(max: int) -> Generator[int, None, Literal["Completed"]]:
    count = 1
    while count <= max:
        yield count
        count += 1
    return "Completed!"

gen = count_up_to(3)
try:
    while True:
        print(next(gen))
except StopIteration as err:
    completion_status = err.value
    print(completion_status)  # Output: Completed!
```

#### Adding `send`

The `send()` method of a generator is used to send a value back into the
generator function. The value sent in is returned by the `yield` expression.
This can be used to modify the internal state of the generator. Let's adapt the
function to use `send()` to optionally reset the `count`.

```{code-cell} ipython3
def count_up_to(max: int) -> Generator[int, int, Literal["Completed"]]:
    count = 1
    while count <= max:
        received = yield count
        print(f"count: {count}, received: {received}")
        if received is not None:
            count = received
        else:
            count += 1
    return "Completed!"

gen = count_up_to(10)
print(gen.__next__())  # 1
print(gen.send(5))  # 6
for number in gen:
    print(number)  # Continues from 7 to 10
```

This example illustrates basic usage, including how to use `send()` to alter the
internal state of the generator. After initializing the generator:

-   printing the first value with `__next__()` which gives `1`,
-   then `send(5)` is called, `received = yield count` is called, and now
    `received = 5`. Then `count` is subsequently set to `5` as well.
-   The generator is resumed and hits the `yield count` statement again,
    yielding the current value of `count` to be `5`. So it will print out `5` on
    the next yield
-   The generator continues, yielding values from `6` to `10` as it iterates
    through the remaining loop cycles, with each value being printed in the for
    loop.

```{prf:remark} Yield is an expression and not a statement
:label: software-engineering-concurrency-parallelism-asynchronous-generator-yield-is-an-expression

How did the `received` become `None` after `send` is done?

The key to understanding the behavior of your `count_up_to` generator,
especially in relation to how `received` can be `None`, lies in how the
generator is advanced and interacts with the `.send()` method versus the
`.__next__()` method (or its equivalent, `next(gen)`).

When you first call `gen.__next__()` or `next(gen)`, the generator starts
executing up to the first `yield` statement, yielding the value of `count`
(which is `1`). At this point, since you're not using `.send()` to advance the
generator but `.__next__()` instead, the value received by the `yield`
expression is `None`. This is the default behavior when the generator is
advanced without explicitly sending a value. The generator then proceeds to the
`if received is not None:` check. Since `received` is `None`, the condition
fails, and execution moves to the `else:` clause, incrementing `count`.

However, when you call `gen.send(5)`, you're explicitly sending a value (`5`)
into the generator, which resumes execution right after the `yield` statement,
with `received` now being `5`. This means the `if received is not None:`
condition succeeds, and the code inside that block executes, setting `count` to
`5`.

To clarify, here's a step-by-step breakdown:

1. **Initial Call with `.__next__()`**:

    - The generator yields `1`, and `received` is implicitly `None` because no
      value was sent into the generator. The `else:` clause is executed,
      incrementing `count`.

2. **Call with `.send(5)`**:

    - `received` is set to `5`, so the `if received is not None:` condition is
      true and `count` is set to `5`.

3. **Subsequent Iteration in the For Loop**:
    - The for loop implicitly calls `.__next__()` on each iteration, not
      `.send()`, so no new value is sent into the generator. Therefore,
      `received` is `None` again for each iteration within the loop, and the
      generator simply increments `count` until it exceeds `max`.

This mechanism allows the generator to either accept new values from the outside
via `.send(value)` or continue its own internal logic, incrementing `count`,
when advanced with `.__next__()` or `next(gen)`, where no external value is
provided, and thus `received` is `None`.
```

What we see here is a coroutine, a generator function in which you can pass
data[^1].

#### Adding `throw` and `close`

Let's extend our example to demonstrate how to use the `.throw()` and `.close()`
methods with our generator function.

We'll continue with the modified `count_up_to` function that allows for
resetting the count via the `send()` method.

```{code-cell} ipython3
def count_up_to(max: int) -> Generator[int, None, Literal["Completed"]]:
    count = 1
    while count <= max:
        try:
            received = yield count
            if received is not None:
                count = received
            else:
                count += 1
        except ValueError as err:
            print(f"Exception caught inside generator: {err}")
            count = max  # Force the loop to end.
            yield "Exception processed"
    return "Completed!"
```

The `.throw()` method is used to throw exceptions from the calling scope into
the generator. When a generator encounters an exception thrown into it, it can
either handle the exception or let it propagate, terminating the generator.

```{code-cell} ipython3
gen = count_up_to(5)

print(next(gen))  # Starts the generator, prints 1

# Injecting an exception into the generator
try:
    gen.throw(ValueError, "Something went wrong")
except StopIteration as err:
    print("Generator returned:", err.value)
```

In this example, after starting the generator and advancing it to yield `1` and
`2`, we throw a `ValueError` into the generator using `.throw()`. The generator
function can catch this exception and yield a response or allow it to propagate,
leading to the generator's termination. Our function does not explicitly catch
`ValueError`, so it will terminate and raise `StopIteration`.

The `.close()` method is used to stop a generator. After calling `.close()`, if
the generator function is executing a `yield` expression, it will raise a
`GeneratorExit` inside the generator function. This can be used to perform any
cleanup actions before the generator stops.

```{code-cell} ipython3
gen = count_up_to(10)

print(next(gen))  # Output: 1
print(next(gen))  # Output: 2

# Close the generator
gen.close()

# Trying to advance the generator after closing it will raise StopIteration
try:
    print(next(gen))
except StopIteration:
    print("Generator has been closed.")
```

In this scenario, we start the generator, yield a couple of values, and then
close it using `.close()`. Any attempt to advance the generator after closing it
results in a `StopIteration` exception, indicating that the generator is
exhausted.

## DataLoaders, Streaming and Lazy Loading

Deep learning models, particularly those trained on large datasets, benefit
significantly from efficient data loading mechanisms. PyTorch, a popular deep
learning framework, provides a powerful abstraction for this purpose through its
`DataLoader` class, which under the hood can be understood as leveraging
Python's generator functionality for streaming data.

### Generators and Streaming Data

Generators in Python are a way to iterate over data without loading the entire
dataset into memory. This is especially useful in deep learning, where datasets
can be enormous. A generator-based DataLoader:

-   **Lazily Loads Data**: It loads data as needed, rather than all at once.
    This means that at any point, only a portion of the dataset is in memory,
    making it possible to work with datasets larger than the available system
    memory.
-   **Supports Parallel Data Processing**: PyTorch's `DataLoader` can prefetch
    batches of data using multiple worker processes. This is akin to a generator
    yielding batches of data in parallel, improving efficiency by overlapping
    data loading with model training computations.
-   **Enables Real-time Data Augmentation**: Data augmentation (e.g., random
    transformations of images) can be applied on-the-fly as each batch is
    loaded. This dynamic generation of training samples from a base dataset
    keeps memory use low and variation high.

Here's a simplified conceptual example of how a data loader might be implemented
using a generator pattern in PyTorch:

```{code-cell} ipython3
T_co = TypeVar('T_co', covariant=True)

class MyDataset(Dataset[T_co]):
    def __init__(self, data: Sized) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        sample = self.data[index]
        return sample

data = torch.randn(8, 3)  # 128 samples, 3 features each
dataset = MyDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
dataloader = iter(dataloader)

try:
    while True:
        _ = dataloader.__next__()
except StopIteration:
    print("StopIteration: No more data.")
```

### A Naive Implementation of DataLoader

```{code-cell} ipython3
def simple_data_loader(dataset: Dataset[T_co], batch_size: int = 1)-> Generator[List[T], None, None]:
    batch = []
    for idx in range(len(dataset)):
        batch.append(dataset[idx])
        if len(batch) == batch_size:
            yield batch
            batch = []
    # Yield any remaining data as the last batch
    if batch:
        yield batch

def simple_data_loader(
    dataset: Dataset[T_co], batch_size: int = 1
) -> Generator[List[T_co], None, None]:
    batch = []
    for idx in range(len(dataset)):
        batch.append(dataset[idx])
        if len(batch) == batch_size:
            yield batch
            batch = []
    # Yield any remaining data as the last batch
    if batch:
        yield batch


data = list(range(100))  # Simulated dataset of 100 integers
dataset = MyDataset(data)

# Create and use the data loader
batch_size = 10
dataloader = simple_data_loader(dataset, batch_size=batch_size)

try:
    while True:
        print(dataloader.__next__())
except StopIteration:
    print("StopIteration: No more data.")
```

## References and Further Readings

-   [Generator Tricks for Systems Programmers, v3.0](https://www.dabeaz.com/generators/index.html)
-   [PEP 255 – Simple Generators](https://peps.python.org/pep-0255/)
-   [How to Use Generators and yield in Python - Real Python](https://realpython.com/introduction-to-python-generators/)
-   [What can you use generator functions for?](https://stackoverflow.com/questions/102535/what-can-you-use-generator-functions-for)

[^1]:
    [How to Use Generators and yield in Python - Real Python](https://realpython.com/introduction-to-python-generators/)

[^2]: [PEP 255 – Simple Generators](https://peps.python.org/pep-0255/)
