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

from typing import Generator, List, Union, Any, Iterator
from rich.pretty import pprint
import inspect
import sys
import cProfile
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
:label: software-engineering-concurrency_parallelism_asynchronous-generator-yield-remark

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

[^1]:
    [How to Use Generators and yield in Python - Real Python](https://realpython.com/introduction-to-python-generators/)
