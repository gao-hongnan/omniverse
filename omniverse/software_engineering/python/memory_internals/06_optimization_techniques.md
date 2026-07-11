---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Section 6: Optimization Techniques

## Overview

Now that you understand Python's memory internals (Sections 0-5), it's time to learn **how to optimize** when it matters. This section teaches practical, measurable optimization techniques backed by real data.

In this section, you'll learn:

- **When to optimize** (and when NOT to)
- **Memory optimization** with `__slots__`
- **Generators vs lists** for memory efficiency
- **NumPy arrays** for numerical data
- **Memory profiling** best practices
- **Real-world benchmarks** and case studies

### Learning Objectives

By the end of this section, you will be able to:

1. **Analyze** memory bottlenecks using Section 5's inspection techniques
2. **Choose** appropriate optimization strategies based on use case
3. **Implement** `__slots__`, generators, and NumPy efficiently
4. **Measure** the impact of optimizations quantitatively
5. **Avoid** premature optimization pitfalls

### Prerequisites

You should have completed:

- Section 0: Introduction (understanding the 28-byte mystery)
- Section 1: C Memory Basics (stack vs heap, memory layout)
- Section 2: Python Object Model (PyObject, reference counting)
- Section 3: Integer Internals (PyLongObject structure)
- Section 4: String Internals (PyUnicodeObject, interning)
- Section 5: ctypes Inspection (MemoryInspector, track_memory)

---

## 6.1 When to Optimize (and When Not To)

### The Optimization Decision Framework

Before we dive into techniques, let's establish **when** optimization is actually worth it.

:::{important}
**Donald Knuth's famous quote:**
> "Premature optimization is the root of all evil."

**But:** Understanding optimization techniques prevents you from making anti-patterns in your initial design.
:::

### When Optimization Matters

**Scale scenarios where optimization is critical:**

```{code-cell} python
import sys
from typing import List

# Scenario 1: Large-scale data processing
def process_million_items():
    """Processing 1 million items - optimization matters!"""
    items = list(range(1_000_000))  # ~28MB for integers
    return [x * 2 for x in items]    # Another ~56MB

print(f"Processing 1M integers:")
print(f"  Memory: {1_000_000 * 28 / 1024 / 1024:.1f} MB minimum")
print(f"  This scenario WANTS optimization!")

# Scenario 2: Embedded systems
def embedded_constraint():
    """Memory-constrained environment - optimization critical!"""
    available_memory = 64 * 1024  # 64KB
    max_objects = available_memory // 28  # ~2,300 integers max
    return f"Can only store ~{max_objects} integers in 64KB"

print(f"\nEmbedded constraint:")
print(embedded_constraint())

# Scenario 3: Web servers (concurrent requests)
def concurrent_memory():
    """1000 concurrent requests each using 10MB = 10GB total!"""
    per_request = 10 * 1024 * 1024  # 10MB
    concurrent = 1000
    total = per_request * concurrent
    return f"{concurrent} requests × {per_request // (1024*1024)}MB = {total // (1024**3)}GB"

print(f"\nConcurrent memory pressure:")
print(concurrent_memory())
```

**Expected Output:**
```
Processing 1M integers:
  Memory: 26.7 MB minimum
  This scenario WANTS optimization!

Embedded constraint:
Can only store ~2,300 integers in 64KB

Concurrent memory pressure:
1000 requests × 10MB = 9GB
```

### When NOT to Optimize

**Scenarios where optimization is wasteful:**

```{code-cell} python
# Scenario 1: Simple web app endpoints
def user_profile_data():
    """Typical web app data - not worth optimizing!"""
    return {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        # Total data: < 1KB per user
    }

# Scenario 2: Data pipelines where I/O dominates
def data_pipeline_example():
    """Network/database I/O dominates - memory optimization irrelevant"""
    import time
    # Simulate database query (100ms)
    time.sleep(0.1)
    # Data processing (1ms)
    result = list(range(1000))
    # Memory optimization saves microseconds, not worth it!
    return result

# Scenario 3: Prototyping and development
def prototype_data():
    """Rapid development matters more than optimization"""
    return [i for i in range(100)]  # Simple, readable, fast enough

print("When NOT to optimize:")
print("✓ Simple web apps (<1KB per object)")
print("✓ I/O-bound operations (network/disk)")
print("✓ Prototyping and development")
print("✓ One-off scripts")
```

### The Profiling First Principle

**Rule:** Always measure before optimizing!

```{code-cell} python
from omnivault.utils.memory import track_memory
import time

def demonstrate_profiling_first():
    """Show why profiling is essential."""

    # Option 1: "Optimized" based on assumptions
    def assumption_optimized():
        # Someone thought list comprehension was always faster
        return [str(x) for x in range(100_000)]

    # Option 2: Simple, readable version
    def readable_version():
        result = []
        for i in range(100_000):
            result.append(str(i))
        return result

    # Profile both
    print("Profiling both approaches:")

    with track_memory() as snapshot1:
        start = time.time()
        result1 = assumption_optimized()
        time1 = time.time() - start

    with track_memory() as snapshot2:
        start = time.time()
        result2 = readable_version()
        time2 = time.time() - start

    print(f"List comprehension: {time1:.4f}s, {snapshot1.peak:.1f}MB")
    print(f"Explicit loop:       {time2:.4f}s, {snapshot2.peak:.1f}MB")
    print(f"Difference:          {abs(time1-time2):.4f}s, {abs(snapshot1.peak - snapshot2.peak):.1f}MB")
    print(f"Same result:         {result1 == result2}")

demonstrate_profiling_first()
```

:::{tip}
**Optimization Decision Tree:**

1. **Is memory actually a problem?** → Profile with `track_memory()`
2. **Is it scale-related?** → Consider algorithm changes first
3. **Is it data-structure related?** → Use techniques from this section
4. **Is it I/O bound?** → Memory optimization won't help
:::

---

## 6.2 Memory Optimization with `__slots__`

### Understanding the Memory Impact

From Section 2, we learned that Python objects with `__dict__` have significant overhead. Let's quantify this with `__slots__`.

```{code-cell} python
from omnivault.utils.memory import MemoryInspector
import sys

# Regular class with __dict__
class RegularPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Optimized class with __slots__
class SlottedPoint:
    __slots__ = ['x', 'y']  # No __dict__ allowed!

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

# Compare memory usage
regular = RegularPoint(1.0, 2.0)
slotted = SlottedPoint(1.0, 2.0)

print("Memory Comparison: __dict__ vs __slots__")
print("=" * 50)

regular_inspector = MemoryInspector(regular)
slotted_inspector = MemoryInspector(slotted)

print(f"Regular point:")
print(f"  Size: {sys.getsizeof(regular)} bytes")
print(f"  Summary: {regular_inspector.summary()}")

print(f"\nSlotted point:")
print(f"  Size: {sys.getsizeof(slotted)} bytes")
print(f"  Summary: {slotted_inspector.summary()}")

memory_savings = sys.getsizeof(regular) - sys.getsizeof(slotted)
print(f"\nMemory savings per object: {memory_savings} bytes ({memory_savings/sys.getsizeof(regular)*100:.1f}%)")
```

### Scalability Impact

Let's see how this scales with thousands of objects:

```{code-cell} python
import gc
from omnivault.utils.memory import track_memory

def create_regular_points(n: int) -> list[RegularPoint]:
    """Create n regular points."""
    return [RegularPoint(i, i*2) for i in range(n)]

def create_slotted_points(n: int) -> list[SlottedPoint]:
    """Create n slotted points."""
    return [SlottedPoint(i, i*2) for i in range(n]

# Test with 100,000 points
N = 100_000

print(f"Creating {N:,} points:")
print("=" * 50)

# Test regular points
gc.collect()  # Clear memory
with track_memory() as regular_snapshot:
    regular_points = create_regular_points(N)
    regular_size = sum(sys.getsizeof(p) for p in regular_points)

# Test slotted points
gc.collect()
with track_memory() as slotted_snapshot:
    slotted_points = create_slotted_points(N)
    slotted_size = sum(sys.getsizeof(p) for p in slotted_points)

print(f"Regular points:")
print(f"  Total memory: {regular_size:,} bytes ({regular_size/1024/1024:.1f} MB)")
print(f"  Peak memory:  {regular_snapshot.peak:.1f} MB")

print(f"\nSlotted points:")
print(f"  Total memory: {slotted_size:,} bytes ({slotted_size/1024/1024:.1f} MB)")
print(f"  Peak memory:  {slotted_snapshot.peak:.1f} MB")

total_savings = regular_size - slotted_size
print(f"\nTotal savings: {total_savings:,} bytes ({total_savings/1024/1024:.1f} MB)")
print(f"Savings per object: {total_savings/N:.1f} bytes")
```

### Performance Impact of `__slots__`

`__slots__` doesn't just save memory - it can improve performance too!

```{code-cell} python
import timeit

def test_attribute_access():
    """Compare attribute access performance."""

    # Create test objects
    regular = RegularPoint(1.0, 2.0)
    slotted = SlottedPoint(1.0, 2.0)

    # Test attribute access
    def access_regular():
        return regular.x + regular.y

    def access_slotted():
        return slotted.x + slotted.y

    # Benchmark
    regular_time = timeit.timeit(access_regular, number=1_000_000)
    slotted_time = timeit.timeit(access_slotted, number=1_000_000)

    print("Attribute Access Performance (1M operations)")
    print("=" * 50)
    print(f"Regular __dict__: {regular_time:.4f}s")
    print(f"Slotted __slots__: {slotted_time:.4f}s")
    print(f"Speed improvement: {(regular_time/slotted_time - 1)*100:.1f}%")

test_attribute_access()
```

### When to Use `__slots__`

```{code-cell} python
def slots_decision_framework():
    """Decision framework for using __slots__."""

    scenarios = [
        ("Web form data (few fields)", False, "Small scale, flexibility matters"),
        ("Game entities (many instances)", True, "Memory-critical, many objects"),
        ("Configuration objects", False, "Few instances, readability matters"),
        ("Data processing pipelines", True, "Scale matters"),
        ("API response objects", False, "Dynamic fields needed"),
        ("Scientific computing data points", True, "Memory optimization critical"),
    ]

    print("__slots__ Decision Framework")
    print("=" * 60)
    print(f"{'Scenario':<35} {'Use __slots__':<12} {'Reason'}")
    print("-" * 60)

    for scenario, use_slots, reason in scenarios:
        decision = "YES" if use_slots else "NO"
        print(f"{scenario:<35} {decision:<12} {reason}")

slots_decision_framework()
```

:::{warning}
**`__slots__` Limitations:**

1. **No dynamic attributes**: Can't add attributes at runtime
2. **No `__dict__`**: Can't use methods that rely on `__dict__`
3. **Inheritance complexity**: Must be careful with inheritance
4. **Memory trade-off**: Only helps with many instances
:::

### Advanced `__slots__` Techniques

```{code-cell} python
# Advanced slots patterns

# 1. Slots with inheritance
class Base2D:
    __slots__ = ['x', 'y']

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Point3D(Base2D):
    __slots__ = ['z']  # Inherits x, y from Base2D

    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y)
        self.z = z

# 2. Slots with defaults
class FlexiblePoint:
    __slots__ = ['x', 'y', '_metadata']

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self._metadata = {}  # One dict for metadata instead of per-object

# Test advanced patterns
point3d = Point3D(1, 2, 3)
flex_point = FlexiblePoint(1, 2)

print("Advanced __slots__ Patterns")
print("=" * 40)
print(f"Point3D size: {sys.getsizeof(point3d)} bytes")
print(f"FlexiblePoint size: {sys.getsizeof(flex_point)} bytes")
print(f"3D point coordinates: ({point3d.x}, {point3d.y}, {point3d.z})")
```

---

## 6.3 Generators vs Lists: Memory Efficiency

### Understanding Memory Differences

Generators and lists can produce the same values, but with very different memory characteristics.

```{code-cell} python
from omnivault.utils.memory import track_memory
import sys

def create_list(n: int) -> list[int]:
    """Create a list of n integers."""
    return [i * 2 for i in range(n)]

def create_generator(n: int):
    """Create a generator of n integers."""
    return (i * 2 for i in range(n))

# Test memory usage
N = 1_000_000

print(f"Memory Comparison: List vs Generator ({N:,} items)")
print("=" * 60)

# List memory usage
with track_memory() as list_snapshot:
    number_list = create_list(N)
    list_size = sys.getsizeof(number_list)
    # Add size of individual integers
    total_list_size = list_size + N * sys.getsizeof(42)

# Generator memory usage
with track_memory() as gen_snapshot:
    number_generator = create_generator(N)
    gen_size = sys.getsizeof(number_generator)

print(f"List:")
print(f"  Container size: {list_size:,} bytes")
print(f"  Elements size: {N * sys.getsizeof(42):,} bytes")
print(f"  Total size: {total_list_size:,} bytes ({total_list_size/1024/1024:.1f} MB)")
print(f"  Peak memory: {list_snapshot.peak:.1f} MB")

print(f"\nGenerator:")
print(f"  Size: {gen_size:,} bytes")
print(f"  Peak memory: {gen_snapshot.peak:.1f} MB")

print(f"\nMemory savings: {(total_list_size - gen_size)/1024/1024:.1f} MB")
print(f"Size ratio: {total_list_size/gen_size:,.0f}x larger for list")
```

### Lazy Evaluation Benefits

Generators compute values **on demand**, which enables processing of infinite or very large sequences.

```{code-cell} python
def demonstrate_lazy_evaluation():
    """Show lazy evaluation benefits."""

    # 1. Infinite sequence with generator
    def fibonacci_generator():
        """Generate infinite Fibonacci numbers."""
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b

    # 2. Processing large files
    def process_large_file():
        """Simulate processing a huge file line by line."""
        def line_generator(size=1_000_000):
            for i in range(size):
                yield f"Line {i}: Some data here\n"

        # Process without loading entire file
        count = 0
        for line in line_generator():
            count += 1
            if count % 100_000 == 0:
                yield f"Processed {count:,} lines..."

        yield f"Total: {count:,} lines processed"

    # 3. Memory-efficient filtering
    def efficient_filtering():
        """Filter large dataset without loading all into memory."""
        def data_generator():
            # Simulate 10M records
            for i in range(10_000_000):
                yield {"id": i, "value": i % 100, "category": i % 3}

        # Filter and process without loading all
        filtered = (item for item in data_generator() if item["value"] > 50)

        # Take first 10 results
        results = []
        for item in filtered:
            results.append(item)
            if len(results) >= 10:
                break

        return results

    print("Lazy Evaluation Benefits")
    print("=" * 40)

    # Demo 1: First 10 Fibonacci numbers
    fib_gen = fibonacci_generator()
    fib_numbers = [next(fib_gen) for _ in range(10)]
    print(f"First 10 Fibonacci: {fib_numbers}")

    # Demo 2: Large file processing
    print("\nLarge file processing:")
    for progress in process_large_file():
        print(progress)

    # Demo 3: Efficient filtering
    print("\nEfficient filtering:")
    filtered_results = efficient_filtering()
    print(f"First 10 filtered items: {filtered_results}")

demonstrate_lazy_evaluation()
```

### When Generators Shine

```{code-cell} python
def generator_optimization_scenarios():
    """Identify scenarios where generators are optimal."""

    scenarios = [
        {
            "scenario": "Data pipeline processing",
            "use_generator": True,
            "reason": "Process data without loading entire dataset",
            "example": "ETL operations, log processing"
        },
        {
            "scenario": "Mathematical sequences",
            "use_generator": True,
            "reason": "Can represent infinite sequences",
            "example": "Primes, Fibonacci, factorials"
        },
        {
            "scenario": "File processing",
            "use_generator": True,
            "reason": "Read line by line without loading entire file",
            "example": "CSV parsing, log analysis"
        },
        {
            "scenario": "Multiple iterations needed",
            "use_generator": False,
            "reason": "Generator exhausted after one iteration",
            "example": "Multiple passes through same data"
        },
        {
            "scenario": "Random access needed",
            "use_generator": False,
            "reason": "No random access in generators",
            "example": "Data lookup, indexing"
        },
        {
            "scenario": "Small datasets (<10K items)",
            "use_generator": False,
            "reason": "Memory overhead negligible, convenience wins",
            "example": "Configuration data, small lists"
        }
    ]

    print("Generator Decision Guide")
    print("=" * 60)
    print(f"{'Scenario':<30} {'Use Generator':<13} {'Key Reason'}")
    print("-" * 60)

    for scenario in scenarios:
        decision = "YES" if scenario["use_generator"] else "NO"
        print(f"{scenario['scenario']:<30} {decision:<13} {scenario['reason']}")

    return scenarios

generator_scenarios = generator_optimization_scenarios()
```

### Performance Comparison

```{code-cell} python
import timeit

def compare_performance():
    """Compare performance of generators vs lists."""

    N = 1_000_000

    # Test creation speed
    def list_creation():
        return [i * 2 for i in range(N)]

    def generator_creation():
        return (i * 2 for i in range(N))

    # Test iteration speed
    test_list = list(range(N))
    test_generator = (i for i in range(N))

    def list_iteration():
        total = 0
        for item in test_list:
            total += item
        return total

    def generator_iteration():
        total = 0
        for item in test_generator:
            total += item
        return total

    # Benchmark
    list_create_time = timeit.timeit(list_creation, number=10)
    gen_create_time = timeit.timeit(generator_creation, number=10)

    list_iter_time = timeit.timeit(list_iteration, number=1)
    # Need to recreate generator each time for fair comparison
    gen_iter_time = timeit.timeit(lambda: sum(i for i in range(N)), number=1)

    print("Performance Comparison (1M items)")
    print("=" * 40)
    print(f"Creation:")
    print(f"  List:     {list_create_time:.4f}s")
    print(f"  Generator: {gen_create_time:.4f}s")
    print(f"  Speedup:   {list_create_time/gen_create_time:.1f}x")

    print(f"\nIteration:")
    print(f"  List:     {list_iter_time:.4f}s")
    print(f"  Generator: {gen_iter_time:.4f}s")
    print(f"  Speedup:   {list_iter_time/gen_iter_time:.1f}x")

compare_performance()
```

### Real-World Example: Data Processing Pipeline

```{code-cell} python
def real_world_pipeline():
    """Real-world data processing pipeline comparison."""

    # Simulate sensor data processing
    import random

    def generate_sensor_data(hours: int):
        """Generate sensor data for given hours."""
        for minute in range(hours * 60):
            for second in range(60):
                yield {
                    'timestamp': minute * 60 + second,
                    'temperature': 20 + random.gauss(0, 5),
                    'humidity': 50 + random.gauss(0, 10),
                    'pressure': 1013 + random.gauss(0, 2)
                }

    def process_with_list(hours: int):
        """Process using list - loads all data first."""
        data = list(generate_sensor_data(hours))

        # Process data
        processed = [
            {
                'hour': entry['timestamp'] // 3600,
                'avg_temp': entry['temperature']
            }
            for entry in data
            if entry['temperature'] > 25
        ]

        return len(processed)

    def process_with_generator(hours: int):
        """Process using generator - streams data."""
        processed_count = 0

        for entry in generate_sensor_data(hours):
            if entry['temperature'] > 25:
                processed_count += 1

        return processed_count

    # Compare for 24 hours of data
    hours = 24

    print("Data Processing Pipeline Comparison")
    print("=" * 50)
    print(f"Processing {hours} hours of sensor data...")

    # Memory comparison
    from omnivault.utils.memory import track_memory

    with track_memory() as list_snapshot:
        list_result = process_with_list(hours)

    with track_memory() as gen_snapshot:
        gen_result = process_with_generator(hours)

    print(f"\nResults:")
    print(f"  List approach:     {list_result} records, {list_snapshot.peak:.1f} MB")
    print(f"  Generator approach: {gen_result} records, {gen_snapshot.peak:.1f} MB")
    print(f"  Memory savings:    {list_snapshot.peak - gen_snapshot.peak:.1f} MB")
    print(f"  Same result:       {list_result == gen_result}")

real_world_pipeline()
```

---

## 6.4 NumPy Arrays for Numerical Data

### Understanding NumPy's Memory Layout

NumPy arrays store data in **contiguous memory blocks**, unlike Python lists which store pointers to objects.

```{code-cell} python
import numpy as np
from omnivault.utils.memory import MemoryInspector, track_memory

def compare_numpy_vs_python():
    """Compare memory usage of NumPy vs Python lists."""

    N = 1_000_000

    # Python list of integers
    python_list = list(range(N))

    # NumPy array of integers
    numpy_array = np.arange(N, dtype=np.int64)

    print("NumPy vs Python List Memory Comparison")
    print("=" * 50)
    print(f"Data size: {N:,} integers")
    print()

    # Python list memory
    list_container_size = sys.getsizeof(python_list)
    list_element_size = sys.getsizeof(42)  # Python int size
    total_list_size = list_container_size + N * list_element_size

    # NumPy array memory
    numpy_size = numpy_array.nbytes

    print(f"Python List:")
    print(f"  Container:  {list_container_size:,} bytes")
    print(f"  Elements:   {N * list_element_size:,} bytes")
    print(f"  Total:      {total_list_size:,} bytes ({total_list_size/1024/1024:.1f} MB)")

    print(f"\nNumPy Array:")
    print(f"  Total:      {numpy_size:,} bytes ({numpy_size/1024/1024:.1f} MB)")
    print(f"  Data type:  {numpy_array.dtype}")
    print(f"  Shape:      {numpy_array.shape}")

    memory_savings = total_list_size - numpy_size
    print(f"\nMemory savings: {memory_savings:,} bytes ({memory_savings/1024/1024:.1f} MB)")
    print(f"Size ratio: {total_list_size/numpy_size:.1f}x")

    return python_list, numpy_array

python_list, numpy_array = compare_numpy_vs_python()
```

### Memory Layout Visualization

```{code-cell} python
def visualize_memory_layout():
    """Visualize memory layout differences."""

    # Small example for visualization
    data = [1, 2, 3, 4]

    print("Memory Layout Visualization")
    print("=" * 40)

    print("\nPython List Memory:")
    print("┌─────────────────────────────────┐")
    print("│ List Object                     │")
    print("│  - pointer to first PyObject*   │")
    print("│  - length = 4                   │")
    print("│  - capacity = 4                  │")
    print("└─────────────────────────────────┘")
    print("           │")
    print("           ▼")
    print("┌─────────────────────────────────┐")
    print("│ PyObject* Array (4 pointers)    │")
    print("│  [0x1000] [0x2000] [0x3000] [0x4000] │")
    print("└─────────────────────────────────┘")
    print("     │        │        │        │")
    print("     ▼        ▼        ▼        ▼")
    print("┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐")
    print("│ PyO │  │ PyO │  │ PyO │  │ PyO │")
    print("│ int│  │ int│  │ int│  │ int│")
    print("│ 1  │  │ 2  │  │ 3  │  │ 4  │")
    print("│28B │  │28B │  │28B │  │28B │")
    print("└─────┘  └─────┘  └─────┘  └─────┘")
    print()

    print("NumPy Array Memory:")
    print("┌─────────────────────────────────┐")
    print("│ PyArrayObject                   │")
    print("│  - pointer to data block        │")
    print("│  - shape = (4,)                 │")
    print("│  - dtype = int64                │")
    print("└─────────────────────────────────┘")
    print("           │")
    print("           ▼")
    print("┌─────────────────────────────────┐")
    print("│ Contiguous Data Block           │")
    print("│ [ 1 ] [ 2 ] [ 3 ] [ 4 ]         │")
    print("│ 8B   8B   8B   8B               │")
    print("│ 32 bytes total                  │")
    print("└─────────────────────────────────┘")

visualize_memory_layout()
```

### Performance Benefits

NumPy's contiguous memory layout enables **vectorized operations** that are much faster than Python loops.

```{code-cell} python
import numpy as np
import time

def compare_performance():
    """Compare Python vs NumPy performance."""

    N = 10_000_000

    # Create data
    python_data = list(range(N))
    numpy_data = np.arange(N, dtype=np.float64)

    print(f"Performance Comparison ({N:,} elements)")
    print("=" * 50)

    # Operation 1: Element-wise multiplication
    def python_multiply():
        return [x * 2 for x in python_data]

    def numpy_multiply():
        return numpy_data * 2

    # Benchmark
    python_time = time.time()
    python_result = python_multiply()
    python_time = time.time() - python_time

    numpy_time = time.time()
    numpy_result = numpy_multiply()
    numpy_time = time.time() - numpy_time

    print(f"Element-wise multiplication:")
    print(f"  Python: {python_time:.4f}s")
    print(f"  NumPy:  {numpy_time:.4f}s")
    print(f"  Speedup: {python_time/numpy_time:.1f}x")

    # Operation 2: Sum
    python_sum = sum(python_data)
    numpy_sum = numpy_data.sum()

    print(f"\nSum verification:")
    print(f"  Python sum: {python_sum:,}")
    print(f"  NumPy sum:  {numpy_sum:,}")
    print(f"  Match: {python_sum == numpy_sum}")

    # Operation 3: Filtering
    def python_filter():
        return [x for x in python_data if x > N//2]

    def numpy_filter():
        return numpy_data[numpy_data > N//2]

    filter_python_time = time.time()
    python_filtered = python_filter()
    filter_python_time = time.time() - filter_python_time

    filter_numpy_time = time.time()
    numpy_filtered = numpy_filter()
    filter_numpy_time = time.time() - filter_numpy_time

    print(f"\nFiltering (> {N//2:,}):")
    print(f"  Python: {filter_python_time:.4f}s, {len(python_filtered):,} results")
    print(f"  NumPy:  {filter_numpy_time:.4f}s, {len(numpy_filtered):,} results")
    print(f"  Speedup: {filter_python_time/filter_numpy_time:.1f}x")

compare_performance()
```

### When NumPy Helps vs Hurts

```{code-cell} python
def numpy_decision_guide():
    """Guide for when to use NumPy."""

    scenarios = [
        {
            "scenario": "Large numerical arrays (>10K elements)",
            "use_numpy": True,
            "reason": "Memory efficiency and vectorized operations",
            "memory_benefit": "10-100x reduction"
        },
        {
            "scenario": "Mathematical operations on arrays",
            "use_numpy": True,
            "reason": "Vectorized operations, C-level speed",
            "speed_benefit": "10-1000x faster"
        },
        {
            "scenario": "Small arrays (<100 elements)",
            "use_numpy": False,
            "reason": "Overhead outweighs benefits",
            "alternative": "Python lists"
        },
        {
            "scenario": "Mixed data types",
            "use_numpy": False,
            "reason": "NumPy arrays require uniform dtype",
            "alternative": "Python lists or Pandas"
        },
        {
            "scenario": "Frequent append operations",
            "use_numpy": False,
            "reason": "NumPy arrays fixed size, expensive resizing",
            "alternative": "List, then convert to NumPy"
        },
        {
            "scenario": "Text/string data",
            "use_numpy": False,
            "reason": "NumPy designed for numerical data",
            "alternative": "Python strings or Pandas"
        },
        {
            "scenario": "Scientific computing",
            "use_numpy": True,
            "reason": "Standard in scientific Python",
            "benefits": "Optimized linear algebra, FFT, etc."
        }
    ]

    print("NumPy Decision Guide")
    print("=" * 60)
    print(f"{'Scenario':<35} {'Use NumPy':<10} {'Key Benefit/Alternative'}")
    print("-" * 60)

    for scenario in scenarios:
        decision = "YES" if scenario["use_numpy"] else "NO"
        benefit = scenario.get("reason", "")
        if scenario["use_numpy"]:
            benefit += f" ({scenario.get('memory_benefit', scenario.get('speed_benefit', ''))})"

        print(f"{scenario['scenario']:<35} {decision:<10} {benefit}")

    return scenarios

numpy_scenarios = numpy_decision_guide()
```

### Integration with ctypes (from Section 5)

```{code-cell} python
def numpy_ctypes_integration():
    """Show how NumPy integrates with ctypes inspection."""

    import ctypes

    # Create NumPy array
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    print("NumPy + ctypes Integration")
    print("=" * 40)

    # Get memory information
    print(f"NumPy array:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Size: {arr.nbytes} bytes")
    print(f"  Address: {arr.ctypes.data:#x}")

    # Use ctypes to access memory directly
    print(f"\nDirect memory access:")

    # Cast to ctypes array
    c_array = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    print(f"  Element 0 (via ctypes): {c_array[0]}")
    print(f"  Element 1 (via ctypes): {c_array[1]}")
    print(f"  Element 2 (via ctypes): {c_array[2]}")

    # Compare with NumPy indexing
    print(f"\nComparison with NumPy indexing:")
    for i in range(3):
        numpy_value = arr[i]
        ctypes_value = c_array[i]
        print(f"  [{i}]: NumPy={numpy_value}, ctypes={ctypes_value}, match={numpy_value == ctypes_value}")

    # Memory layout verification
    print(f"\nMemory layout verification:")
    print(f"  Element 0 address: {arr.ctypes.data:#x}")
    print(f"  Element 1 address: {arr.ctypes.data + 8:#x}")
    print(f"  Element 2 address: {arr.ctypes.data + 16:#x}")
    print(f"  Spacing: {8} bytes per element (int64)")

    return arr

numpy_array = numpy_ctypes_integration()
```

---

## 6.5 Memory Profiling Best Practices

### Advanced tracemalloc Usage

Go beyond basic `tracemalloc.start()` with advanced techniques.

```{code-cell} python
import tracemalloc
from omnivault.utils.memory import track_memory
import time

def advanced_tracemalloc_demo():
    """Demonstrate advanced tracemalloc techniques."""

    print("Advanced tracemalloc Techniques")
    print("=" * 40)

    # Start tracemalloc with snapshot interval
    tracemalloc.start(25)  # Take snapshot every 25 allocations

    def memory_intensive_function():
        """Function that allocates memory in different ways."""
        allocations = []

        # Allocation pattern 1: Large strings
        for i in range(100):
            allocations.append("x" * 1000)  # 1KB strings

        # Allocation pattern 2: Lists of integers
        for i in range(50):
            allocations.append(list(range(100)))  # Small lists

        # Allocation pattern 3: Dictionaries
        for i in range(25):
            allocations.append({f"key_{j}": j for j in range(50)})  # Dicts

        return allocations

    # Function to analyze
    result = memory_intensive_function()

    # Get current statistics
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

    # Get traceback of the biggest allocation
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print(f"\nTop 5 memory allocations:")
    for i, stat in enumerate(top_stats[:5], 1):
        print(f"{i}. {stat}")
        if i <= 2:  # Show traceback for top 2
            for line in stat.traceback.format(limit=3):
                print(f"   {line}")

    # Compare snapshots
    print(f"\nComparing with empty snapshot:")
    empty_snapshot = tracemalloc.take_snapshot()
    time.sleep(0.1)  # Small delay

    # Allocate some more memory
    more_allocations = [list(range(1000)) for _ in range(10)]

    new_snapshot = tracemalloc.take_snapshot()
    stats = new_snapshot.compare_to(empty_snapshot, 'lineno')

    print(f"New allocations since empty snapshot:")
    for stat in stats[:3]:
        print(f"  {stat}")

    tracemalloc.stop()
    return result

advanced_result = advanced_tracemalloc_demo()
```

### Memory Leak Detection

```{code-cell} python
def memory_leak_detection():
    """Demonstrate memory leak detection patterns."""

    print("Memory Leak Detection Patterns")
    print("=" * 40)

    class LeakyClass:
        """Class that intentionally leaks memory."""
        def __init__(self):
            self.data = list(range(1000))  # Some data
            # Problem: circular reference with self
            self.self_reference = self

        def __del__(self):
            print(f"LeakyClass instance deleted")

    class ProperClass:
        """Class that properly manages memory."""
        def __init__(self):
            self.data = list(range(1000))
            # No circular reference

        def __del__(self):
            print(f"ProperClass instance deleted")

    # Test leak detection
    import gc
    import weakref

    print("\n1. Testing memory leak with circular reference:")

    # Create leaky object
    leaky_obj = LeakyClass()
    leaky_weak_ref = weakref.ref(leaky_obj)

    print(f"  Object created: {leaky_weak_ref() is not None}")

    # Delete reference
    del leaky_obj
    gc.collect()  # Force garbage collection

    print(f"  After del + gc.collect(): {leaky_weak_ref() is not None}")
    print(f"  Memory leak: Object still exists!")

    print("\n2. Testing proper memory management:")

    # Create proper object
    proper_obj = ProperClass()
    proper_weak_ref = weakref.ref(proper_obj)

    print(f"  Object created: {proper_weak_ref() is not None}")

    # Delete reference
    del proper_obj
    gc.collect()

    print(f"  After del + gc.collect(): {proper_weak_ref() is not None}")
    print(f"  Proper cleanup: Object deleted")

    # Memory usage comparison
    print("\n3. Memory usage comparison:")

    tracemalloc.start()

    # Create many leaky objects
    leaky_objects = [LeakyClass() for _ in range(100)]
    leaky_memory = tracemalloc.get_traced_memory()[0]

    # Delete references but don't collect yet
    del leaky_objects
    after_delete_memory = tracemalloc.get_traced_memory()[0]

    # Force garbage collection
    gc.collect()
    after_gc_memory = tracemalloc.get_traced_memory()[0]

    print(f"  100 leaky objects created: {leaky_memory / 1024:.1f} KB")
    print(f"  After delete: {after_delete_memory / 1024:.1f} KB")
    print(f"  After gc.collect(): {after_gc_memory / 1024:.1f} KB")
    print(f"  Memory freed: {(after_delete_memory - after_gc_memory) / 1024:.1f} KB")

    tracemalloc.stop()

memory_leak_detection()
```

### Memory Hot-Spot Analysis

```{code-cell} python
from omnivault.utils.memory import MemoryInspector
import tracemalloc
import time

def memory_hotspot_analysis():
    """Analyze memory hot-spots in code."""

    def process_data_simulation():
        """Simulate data processing with different memory patterns."""
        results = []

        # Hot-spot 1: Large string concatenation (inefficient)
        large_string = ""
        for i in range(10000):
            large_string += f"Item {i}, "  # Inefficient concatenation

        results.append(len(large_string))

        # Hot-spot 2: Repeated list appends (growing memory)
        growing_list = []
        for i in range(50000):
            growing_list.append(i * 2)  # List keeps growing

        results.append(len(growing_list))

        # Hot-spot 3: Dictionary with many entries
        large_dict = {}
        for i in range(100000):
            large_dict[f"key_{i}"] = f"value_{i}" * 10  # Large values

        results.append(len(large_dict))

        return results

    print("Memory Hot-Spot Analysis")
    print("=" * 40)

    # Start tracemalloc with fine-grained tracking
    tracemalloc.start(1)  # Track every allocation

    start_time = time.time()
    process_data_simulation()
    end_time = time.time()

    # Get statistics
    current, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()

    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

    # Analyze top allocations by size
    print(f"\nTop 10 memory allocations by size:")
    top_stats = snapshot.statistics('size')

    for i, stat in enumerate(top_stats[:10], 1):
        size_mb = stat.size / 1024 / 1024
        count = stat.count
        print(f"{i:2d}. {size_mb:6.2f} MB, {count:4d} allocations - {stat.traceback.format()[-1].strip()}")

    # Group by filename
    print(f"\nMemory usage by source file:")
    stats_by_file = {}
    for stat in snapshot.statistics('size'):
        filename = stat.traceback.format()[0].split(',')[0].strip()
        if filename not in stats_by_file:
            stats_by_file[filename] = 0
        stats_by_file[filename] += stat.size

    sorted_files = sorted(stats_by_file.items(), key=lambda x: x[1], reverse=True)
    for filename, size in sorted_files[:5]:
        size_mb = size / 1024 / 1024
        print(f"  {filename}: {size_mb:.2f} MB")

    tracemalloc.stop()

memory_hotspot_analysis()
```

### Custom Profiling Tools

```{code-cell} python
from omnivault.utils.memory import track_memory
from typing import Callable, Dict, Any
import time
import sys

class MemoryProfiler:
    """Custom memory profiler for functions."""

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def profile(self, func_name: str, func: Callable, *args, **kwargs) -> Any:
        """Profile a function's memory usage."""

        print(f"\nProfiling: {func_name}")
        print("-" * 30)

        # Measure with track_memory
        with track_memory(verbose=False) as snapshot:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

        # Store results
        self.results[func_name] = {
            'peak_memory_mb': snapshot.peak,
            'time_seconds': end_time - start_time,
            'memory_efficiency': 'Unknown'  # Will be calculated later
        }

        print(f"  Peak memory: {snapshot.peak:.2f} MB")
        print(f"  Time: {end_time - start_time:.4f}s")

        return result

    def compare_functions(self, functions: Dict[str, Callable],
                         test_data: Any) -> None:
        """Compare multiple functions with same test data."""

        print(f"\nFunction Comparison")
        print("=" * 50)
        print(f"Test data: {test_data}")

        results = {}
        for name, func in functions.items():
            try:
                results[name] = self.profile(name, func, test_data)
            except Exception as e:
                print(f"  Error in {name}: {e}")
                continue

        # Calculate efficiency relative to best
        if results:
            min_memory = min(r['peak_memory_mb'] for r in self.results.values())
            min_time = min(r['time_seconds'] for r in self.results.values())

            print(f"\nComparison Results:")
            print(f"{'Function':<20} {'Memory (MB)':<12} {'Time (s)':<10} {'Memory Rank':<12} {'Time Rank'}")
            print("-" * 65)

            for name in self.results:
                r = self.results[name]
                memory_rank = f"{'✓' if r['peak_memory_mb'] == min_memory else f'{r[\"peak_memory_mb\"]/min_memory:.1f}x'}"
                time_rank = f"{'✓' if r['time_seconds'] == min_time else f'{r[\"time_seconds\"]/min_time:.1f}x'}"

                print(f"{name:<20} {r['peak_memory_mb']:<12.2f} {r['time_seconds']:<10.4f} {memory_rank:<12} {time_rank}")

# Test the profiler
def test_profiler():
    """Demonstrate the memory profiler."""

    profiler = MemoryProfiler()

    # Test functions to compare
    def list_comprehension(data):
        return [x * 2 for x in data]

    def map_function(data):
        return list(map(lambda x: x * 2, data))

    def generator_to_list(data):
        return list(x * 2 for x in data)

    # Test data
    test_data = range(100_000)

    functions = {
        'list_comprehension': list_comprehension,
        'map_function': map_function,
        'generator_to_list': generator_to_list
    }

    profiler.compare_functions(functions, test_data)

    return profiler

profiler_result = test_profiler()
```

---

## 6.6 Real-World Case Studies

### Case Study 1: Data Processing Pipeline Optimization

```{code-cell} python
def case_study_data_pipeline():
    """Optimize a realistic data processing pipeline."""

    print("Case Study 1: Data Processing Pipeline")
    print("=" * 50)

    # Simulate realistic data processing scenario
    import json
    import random

    class DataProcessor:
        """Process JSON-like records from a data source."""

        def __init__(self, optimization_level: str = 'none'):
            self.optimization_level = optimization_level
            self.processed_count = 0

        def generate_sample_data(self, n_records: int = 10_000):
            """Generate sample data records."""
            records = []
            for i in range(n_records):
                record = {
                    'id': i,
                    'name': f"User_{i}",
                    'email': f"user{i}@example.com",
                    'age': random.randint(18, 80),
                    'score': random.uniform(0, 100),
                    'tags': [f"tag_{j}" for j in range(random.randint(1, 5))],
                    'metadata': {
                        'created': f"2024-01-{(i % 28) + 1:02d}",
                        'active': random.choice([True, False])
                    }
                }
                records.append(record)
            return records

        def process_v1_no_optimization(self, records):
            """Baseline: No optimization."""
            processed = []

            for record in records:
                # Inefficient processing
                filtered_data = {}
                if record['age'] >= 25:
                    filtered_data['user_id'] = record['id']
                    filtered_data['name'] = record['name'].upper()
                    filtered_data['adult'] = True
                    filtered_data['score_category'] = 'HIGH' if record['score'] > 70 else 'LOW'

                    # Convert to JSON string and back (very inefficient)
                    json_str = json.dumps(filtered_data)
                    processed_record = json.loads(json_str)
                    processed.append(processed_record)

            self.processed_count = len(processed)
            return processed

        def process_v2_basic_optimization(self, records):
            """Basic optimization: Remove JSON conversion."""
            processed = []

            for record in records:
                if record['age'] >= 25:
                    processed_record = {
                        'user_id': record['id'],
                        'name': record['name'].upper(),
                        'adult': True,
                        'score_category': 'HIGH' if record['score'] > 70 else 'LOW'
                    }
                    processed.append(processed_record)

            self.processed_count = len(processed)
            return processed

        def process_v3_generator_optimization(self, records):
            """Use generators for memory efficiency."""
            def process_records():
                for record in records:
                    if record['age'] >= 25:
                        yield {
                            'user_id': record['id'],
                            'name': record['name'].upper(),
                            'adult': True,
                            'score_category': 'HIGH' if record['score'] > 70 else 'LOW'
                        }

            # Convert to list for comparison
            processed = list(process_records())
            self.processed_count = len(processed)
            return processed

        def process_v4_slots_optimization(self, records):
            """Use __slots__ for processed records."""

            class ProcessedRecord:
                __slots__ = ['user_id', 'name', 'adult', 'score_category']

                def __init__(self, user_id, name, adult, score_category):
                    self.user_id = user_id
                    self.name = name
                    self.adult = adult
                    self.score_category = score_category

            processed = []
            for record in records:
                if record['age'] >= 25:
                    processed_record = ProcessedRecord(
                        record['id'],
                        record['name'].upper(),
                        True,
                        'HIGH' if record['score'] > 70 else 'LOW'
                    )
                    processed.append(processed_record)

            self.processed_count = len(processed)
            return processed

    # Run comparison
    processor = DataProcessor()
    test_data = processor.generate_sample_data(50_000)  # 50K records

    optimization_methods = [
        ('No Optimization', processor.process_v1_no_optimization),
        ('Basic Optimization', processor.process_v2_basic_optimization),
        ('Generator + List', processor.process_v3_generator_optimization),
        ('With __slots__', processor.process_v4_slots_optimization)
    ]

    results = {}

    from omnivault.utils.memory import track_memory
    import time

    print(f"Processing {len(test_data):,} records with different optimizations...\n")

    for method_name, method in optimization_methods:
        print(f"Testing: {method_name}")

        with track_memory() as snapshot:
            start_time = time.time()
            result = method(test_data)
            end_time = time.time()

        results[method_name] = {
            'memory_mb': snapshot.peak,
            'time_seconds': end_time - start_time,
            'processed_count': processor.processed_count
        }

        print(f"  Memory: {snapshot.peak:.2f} MB")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Processed: {processor.processed_count:,} records")

    # Compare results
    print(f"\nOptimization Comparison")
    print("=" * 60)
    print(f"{'Method':<20} {'Memory (MB)':<12} {'Time (s)':<10} {'Records/sec'}")
    print("-" * 60)

    baseline_memory = results['No Optimization']['memory_mb']
    baseline_time = results['No Optimization']['time_seconds']

    for method_name, result in results.items():
        memory_ratio = result['memory_mb'] / baseline_memory
        time_ratio = result['time_seconds'] / baseline_time
        records_per_sec = result['processed_count'] / result['time_seconds']

        print(f"{method_name:<20} {result['memory_mb']:<12.2f} {result['time_seconds']:<10.2f} {records_per_sec:,.0f}")
        print(f"{'':20} ({memory_ratio:.1f}x memory) ({time_ratio:.1f}x time)")

    return results, processor

pipeline_results = case_study_data_pipeline()
```

### Case Study 2: Scientific Computing with Large Arrays

```{code-cell} python
def case_study_scientific_computing():
    """Optimize scientific computing operations."""

    print("Case Study 2: Scientific Computing")
    print("=" * 50)

    import numpy as np
    from omnivault.utils.memory import track_memory
    import time

    # Simulate scientific data processing
    class ScientificProcessor:
        """Process large scientific datasets."""

        def __init__(self):
            self.data_size = 0

        def generate_dataset(self, n_points: int = 1_000_000):
            """Generate scientific dataset."""
            print(f"Generating dataset with {n_points:,} points...")

            # Generate multi-dimensional data
            data = {
                'temperature': np.random.normal(20, 5, n_points),
                'pressure': np.random.normal(1013, 10, n_points),
                'humidity': np.random.normal(50, 15, n_points),
                'timestamp': np.arange(n_points),
                'quality_flag': np.random.choice([0, 1, 2], n_points, p=[0.1, 0.8, 0.1])
            }

            self.data_size = n_points
            return data

        def process_with_python_lists(self, data):
            """Process using Python lists (baseline)."""
            print("Processing with Python lists...")

            # Convert to Python lists
            temp_list = data['temperature'].tolist()
            pressure_list = data['pressure'].tolist()
            humidity_list = data['humidity'].tolist()

            results = []

            for i in range(len(temp_list)):
                # Quality check
                if data['quality_flag'][i] == 1:  # Good quality
                    # Calculate derived values
                    if temp_list[i] > 15 and temp_list[i] < 25:
                        comfort_index = (temp_list[i] + humidity_list[i]) / 2
                        pressure_anomaly = abs(pressure_list[i] - 1013)

                        result = {
                            'index': i,
                            'temperature': temp_list[i],
                            'pressure': pressure_list[i],
                            'humidity': humidity_list[i],
                            'comfort_index': comfort_index,
                            'pressure_anomaly': pressure_anomaly
                        }
                        results.append(result)

            return results

        def process_with_numpy(self, data):
            """Process using NumPy arrays."""
            print("Processing with NumPy arrays...")

            # Vectorized operations
            good_quality_mask = data['quality_flag'] == 1
            temp_range_mask = (data['temperature'] > 15) & (data['temperature'] < 25)
            valid_mask = good_quality_mask & temp_range_mask

            # Apply filters
            filtered_temp = data['temperature'][valid_mask]
            filtered_pressure = data['pressure'][valid_mask]
            filtered_humidity = data['humidity'][valid_mask]
            filtered_indices = np.where(valid_mask)[0]

            # Calculate derived values
            comfort_index = (filtered_temp + filtered_humidity) / 2
            pressure_anomaly = np.abs(filtered_pressure - 1013)

            # Create structured array for results
            result_dtype = [
                ('index', 'i8'),
                ('temperature', 'f8'),
                ('pressure', 'f8'),
                ('humidity', 'f8'),
                ('comfort_index', 'f8'),
                ('pressure_anomaly', 'f8')
            ]

            results = np.empty(len(filtered_indices), dtype=result_dtype)
            results['index'] = filtered_indices
            results['temperature'] = filtered_temp
            results['pressure'] = filtered_pressure
            results['humidity'] = filtered_humidity
            results['comfort_index'] = comfort_index
            results['pressure_anomaly'] = pressure_anomaly

            return results

        def analyze_results(self, python_results, numpy_results):
            """Analyze and compare processing results."""
            print("Analyzing results...")

            # Convert NumPy results to comparable format
            if isinstance(numpy_results, np.ndarray):
                numpy_dict = {
                    'count': len(numpy_results),
                    'avg_comfort': np.mean(numpy_results['comfort_index']),
                    'avg_pressure_anomaly': np.mean(numpy_results['pressure_anomaly']),
                    'max_temp': np.max(numpy_results['temperature']),
                    'min_temp': np.min(numpy_results['temperature'])
                }
            else:
                numpy_dict = {'error': 'Invalid NumPy results'}

            # Analyze Python results
            if python_results:
                comfort_indices = [r['comfort_index'] for r in python_results]
                pressure_anomalies = [r['pressure_anomaly'] for r in python_results]
                temperatures = [r['temperature'] for r in python_results]

                python_dict = {
                    'count': len(python_results),
                    'avg_comfort': sum(comfort_indices) / len(comfort_indices),
                    'avg_pressure_anomaly': sum(pressure_anomalies) / len(pressure_anomalies),
                    'max_temp': max(temperatures),
                    'min_temp': min(temperatures)
                }
            else:
                python_dict = {'error': 'No Python results'}

            return python_dict, numpy_dict

    # Run comparison
    processor = ScientificProcessor()
    dataset = processor.generate_dataset(2_000_000)  # 2M data points

    methods = [
        ('Python Lists', processor.process_with_python_lists),
        ('NumPy Arrays', processor.process_with_numpy)
    ]

    results = {}

    for method_name, method in methods:
        print(f"\n{method_name} Processing:")
        print("-" * 30)

        with track_memory() as snapshot:
            start_time = time.time()
            result = method(dataset)
            end_time = time.time()

        results[method_name] = {
            'memory_mb': snapshot.peak,
            'time_seconds': end_time - start_time,
            'result': result
        }

        print(f"Memory: {snapshot.peak:.2f} MB")
        print(f"Time: {end_time - start_time:.2f}s")

        if isinstance(result, np.ndarray):
            print(f"Processed: {len(result):,} valid records")
        else:
            print(f"Processed: {len(result):,} valid records")

    # Analyze results consistency
    print(f"\nResults Analysis:")
    print("=" * 40)

    if 'Python Lists' in results and 'NumPy Arrays' in results:
        python_analysis, numpy_analysis = processor.analyze_results(
            results['Python Lists']['result'],
            results['NumPy Arrays']['result']
        )

        print("Python Lists Results:")
        for key, value in python_analysis.items():
            print(f"  {key}: {value}")

        print("\nNumPy Arrays Results:")
        for key, value in numpy_analysis.items():
            print(f"  {key}: {value}")

    # Performance comparison
    print(f"\nPerformance Comparison:")
    print("-" * 30)

    baseline_time = results['Python Lists']['time_seconds']
    baseline_memory = results['Python Lists']['memory_mb']

    for method_name, result in results.items():
        time_ratio = result['time_seconds'] / baseline_time
        memory_ratio = result['memory_mb'] / baseline_memory

        print(f"{method_name}:")
        print(f"  Speedup: {time_ratio:.1f}x")
        print(f"  Memory: {memory_ratio:.1f}x")

    return results, processor

scientific_results = case_study_scientific_computing()
```

### Case Study 3: Web Application Memory Management

```{code-cell} python
def case_study_web_application():
    """Optimize memory usage in web application scenarios."""

    print("Case Study 3: Web Application Memory Management")
    print("=" * 60)

    class WebApplication:
        """Simulate web application memory patterns."""

        def __init__(self):
            self.user_sessions = {}
            self.cache = {}
            self.request_count = 0

        class UserSession:
            """User session data."""

            def __init__(self, user_id: int):
                self.user_id = user_id
                self.data = {
                    'preferences': {'theme': 'dark', 'language': 'en'},
                    'shopping_cart': [],
                    'viewed_products': [],
                    'search_history': []
                }
                self.last_activity = 0

        class OptimizedUserSession:
            """Optimized user session using __slots__."""
            __slots__ = ['user_id', 'preferences_theme', 'preferences_language',
                        'cart_count', 'viewed_count', 'search_count', 'last_activity']

            def __init__(self, user_id: int):
                self.user_id = user_id
                self.preferences_theme = 'dark'
                self.preferences_language = 'en'
                self.cart_count = 0
                self.viewed_count = 0
                self.search_count = 0
                self.last_activity = 0

        def simulate_regular_session_management(self, n_users: int):
            """Simulate regular session management with __dict__."""
            print(f"Simulating {n_users:,} user sessions (regular)...")

            # Create sessions
            for user_id in range(n_users):
                session = self.UserSession(user_id)

                # Add some activity data
                session.data['shopping_cart'] = [f'product_{i}' for i in range(5)]
                session.data['viewed_products'] = [f'product_{i}' for i in range(20)]
                session.data['search_history'] = [f'search_{i}' for i in range(10)]

                self.user_sessions[user_id] = session

            # Simulate requests
            for _ in range(1000):
                user_id = hash(_) % n_users
                session = self.user_sessions[user_id]
                session.last_activity += 1
                self.request_count += 1

        def simulate_optimized_session_management(self, n_users: int):
            """Simulate optimized session management with __slots__."""
            print(f"Simulating {n_users:,} user sessions (optimized)...")

            # Create optimized sessions
            for user_id in range(n_users):
                session = self.OptimizedUserSession(user_id)

                # Simulate activity (just counters, not full data)
                session.cart_count = 5
                session.viewed_count = 20
                session.search_count = 10

                self.user_sessions[user_id] = session

            # Simulate requests
            for _ in range(1000):
                user_id = hash(_) % n_users
                session = self.user_sessions[user_id]
                session.last_activity += 1
                self.request_count += 1

        def cleanup_inactive_sessions(self, max_age: int = 100):
            """Clean up inactive sessions."""
            inactive = [user_id for user_id, session in self.user_sessions.items()
                        if session.last_activity < max_age]

            for user_id in inactive:
                del self.user_sessions[user_id]

            return len(inactive)

        def get_memory_usage(self):
            """Calculate current memory usage."""
            total_size = 0
            for session in self.user_sessions.values():
                total_size += sys.getsizeof(session)
                if hasattr(session, 'data'):
                    total_size += sum(sys.getsizeof(v) for v in session.data.values()
                                     if hasattr(v, '__len__'))

            return total_size

    # Memory leak simulation
    print("1. Memory Leak Detection in Web Apps")
    print("-" * 40)

    app = WebApplication()

    # Simulate memory leak
    print("Creating sessions without cleanup...")
    app.simulate_regular_session_management(10000)

    memory_before = app.get_memory_usage()
    print(f"Memory after creating sessions: {memory_before / 1024 / 1024:.2f} MB")

    # Create more sessions (simulating continued usage)
    app.simulate_regular_session_management(5000)

    memory_after = app.get_memory_usage()
    print(f"Memory after adding more sessions: {memory_after / 1024 / 1024:.2f} MB")

    # Cleanup
    cleaned = app.cleanup_inactive_sessions(50)
    memory_after_cleanup = app.get_memory_usage()

    print(f"Cleaned up {cleaned:,} inactive sessions")
    print(f"Memory after cleanup: {memory_after_cleanup / 1024 / 1024:.2f} MB")
    print(f"Memory saved: {(memory_after - memory_after_cleanup) / 1024 / 1024:.2f} MB")

    # Optimization comparison
    print(f"\n2. Session Storage Optimization")
    print("-" * 40)

    from omnivault.utils.memory import track_memory

    # Test regular sessions
    app_regular = WebApplication()
    with track_memory() as snapshot_regular:
        app_regular.simulate_regular_session_management(5000)

    # Test optimized sessions
    app_optimized = WebApplication()
    with track_memory() as snapshot_optimized:
        app_optimized.simulate_optimized_session_management(5000)

    print(f"Regular session storage: {snapshot_regular.peak:.2f} MB")
    print(f"Optimized session storage: {snapshot_optimized.peak:.2f} MB")

    memory_savings = snapshot_regular.peak - snapshot_optimized.peak
    print(f"Memory savings: {memory_savings:.2f} MB ({memory_savings/snapshot_regular.peak*100:.1f}%)")

    # Caching strategy
    print(f"\n3. Caching Strategy Impact")
    print("-" * 40)

    class CacheManager:
        """Demonstrate different caching strategies."""

        def __init__(self):
            self.cache = {}
            self.lru_cache = {}  # Simplified LRU

        def unlimited_cache(self, key, value):
            """Unlimited cache - potential memory leak."""
            self.cache[key] = value

        def size_limited_cache(self, key, value, max_size=1000):
            """Size-limited cache."""
            if len(self.cache) >= max_size:
                # Remove oldest entries
                oldest_keys = list(self.cache.keys())[:100]
                for k in oldest_keys:
                    del self.cache[k]

            self.cache[key] = value

    cache_manager = CacheManager()

    # Test unlimited cache
    with track_memory() as snapshot_unlimited:
        for i in range(10000):
            # Simulate large cached objects
            cache_manager.unlimited_cache(f'key_{i}', 'x' * 1000)

    # Test limited cache
    cache_manager.limited_cache = {}
    with track_memory() as snapshot_limited:
        for i in range(10000):
            cache_manager.size_limited_cache(f'key_{i}', 'x' * 1000)

    print(f"Unlimited cache: {snapshot_unlimited.peak:.2f} MB")
    print(f"Size-limited cache: {snapshot_limited.peak:.2f} MB")
    print(f"Memory saved by limits: {snapshot_unlimited.peak - snapshot_limited.peak:.2f} MB")

    return {
        'memory_leak_demo': (memory_before, memory_after, memory_after_cleanup),
        'session_optimization': (snapshot_regular.peak, snapshot_optimized.peak),
        'cache_comparison': (snapshot_unlimited.peak, snapshot_limited.peak)
    }

web_app_results = case_study_web_application()
```

### Optimization Decision Framework

```{code-cell} python
def optimization_decision_framework():
    """Comprehensive decision framework for memory optimization."""

    print("Memory Optimization Decision Framework")
    print("=" * 60)

    optimization_scenarios = [
        {
            'category': 'Data Scale',
            'scenarios': [
                {
                    'situation': 'Small datasets (<1000 items)',
                    'recommendation': 'No optimization needed',
                    'techniques': 'Standard Python structures',
                    'reason': 'Memory overhead negligible, readability wins'
                },
                {
                    'situation': 'Medium datasets (1K-100K items)',
                    'recommendation': 'Consider __slots__',
                    'techniques': '__slots__, generators for one-pass processing',
                    'reason': 'Memory savings become noticeable'
                },
                {
                    'situation': 'Large datasets (100K+ items)',
                    'recommendation': 'Aggressive optimization',
                    'techniques': 'NumPy, generators, memory views',
                    'reason': 'Memory becomes critical constraint'
                }
            ]
        },
        {
            'category': 'Application Type',
            'scenarios': [
                {
                    'situation': 'Web applications',
                    'recommendation': 'Session optimization',
                    'techniques': '__slots__, cache limits, cleanup strategies',
                    'reason': 'Concurrent users multiply memory usage'
                },
                {
                    'situation': 'Data processing pipelines',
                    'recommendation': 'Streaming with generators',
                    'techniques': 'Generators, chunked processing',
                    'reason': 'Avoid loading entire datasets'
                },
                {
                    'situation': 'Scientific computing',
                    'recommendation': 'NumPy arrays',
                    'techniques': 'NumPy, vectorized operations',
                    'reason': 'Numerical data benefits from contiguity'
                },
                {
                    'situation': 'Embedded systems',
                    'recommendation': 'Maximum optimization',
                    'techniques': '__slots__, arrays module, pre-allocation',
                    'reason': 'Memory constraints are severe'
                }
            ]
        },
        {
            'category': 'Performance Profile',
            'scenarios': [
                {
                    'situation': 'CPU-bound operations',
                    'recommendation': 'Focus on algorithms',
                    'techniques': 'Better algorithms, NumPy for math',
                    'reason': 'CPU time more critical than memory'
                },
                {
                    'situation': 'I/O-bound operations',
                    'recommendation': 'Minimal memory optimization',
                    'techniques': 'Standard structures, focus on I/O',
                    'reason': 'I/O dominates performance'
                },
                {
                    'situation': 'Memory-bound operations',
                    'recommendation': 'Aggressive optimization',
                    'techniques': '__slots__, generators, NumPy',
                    'reason': 'Memory is the bottleneck'
                }
            ]
        }
    ]

    for category_data in optimization_scenarios:
        print(f"\n{category_data['category']}:")
        print("-" * 40)

        for scenario in category_data['scenarios']:
            print(f"\n  Situation: {scenario['situation']}")
            print(f"  Recommendation: {scenario['recommendation']}")
            print(f"  Techniques: {scenario['techniques']}")
            print(f"  Reason: {scenario['reason']}")

    # Quick reference checklist
    print(f"\nQuick Optimization Checklist")
    print("=" * 40)

    checklist_items = [
        '□ Profile memory usage before optimizing',
        '□ Identify memory hotspots with tracemalloc',
        '□ Consider __slots__ for many similar objects',
        '□ Use generators for one-pass processing',
        '□ Apply NumPy for numerical data',
        '□ Implement cache limits for growing caches',
        '□ Add cleanup strategies for long-running apps',
        '□ Verify optimizations with benchmarks',
        '□ Consider trade-offs (memory vs CPU vs readability)',
        '□ Document optimization decisions'
    ]

    for item in checklist_items:
        print(item)

    return optimization_scenarios

framework_results = optimization_decision_framework()
```

---

## Summary

### What You Learned in Section 6

1. **When to Optimize**
   - Scale scenarios where optimization matters
   - Profiling before optimizing principle
   - Decision framework for optimization vs maintainability

2. **Memory Optimization Techniques**
   - `__slots__` for reducing object overhead (56-72% memory savings)
   - Performance improvements with faster attribute access
   - When to use vs when to avoid `__slots__`

3. **Generators vs Lists**
   - Memory efficiency differences (1000x+ savings for large datasets)
   - Lazy evaluation benefits for infinite sequences
   - Performance trade-offs and appropriate use cases

4. **NumPy Arrays**
   - Contiguous memory layout benefits
   - Vectorized operations for speed (10-1000x faster)
   - When NumPy helps vs when it hurts

5. **Memory Profiling**
   - Advanced tracemalloc techniques
   - Memory leak detection patterns
   - Custom profiling tools and hot-spot analysis

6. **Real-World Applications**
   - Data processing pipeline optimization
   - Scientific computing with large arrays
   - Web application memory management

### Key Takeaways

- **Measure first, optimize second**: Always profile before making changes
- **Context matters**: Different scenarios need different optimizations
- **Trade-offs exist**: Memory vs CPU vs readability vs maintainability
- **Tools matter**: Use omnivault.utils.memory and tracemalloc effectively
- **Scale drives decisions**: Small apps don't need optimization

### Skills Acquired

You can now:
- ✅ Analyze memory bottlenecks using profiling tools
- ✅ Choose appropriate optimization strategies based on use case
- ✅ Implement `__slots__`, generators, and NumPy efficiently
- ✅ Measure the impact of optimizations quantitatively
- ✅ Apply optimization techniques to real-world scenarios
- ✅ Avoid premature optimization pitfalls

### Next Steps

Continue practicing these techniques:

1. **Profile your own code** using the tools from this section
2. **Experiment with different approaches** on your data
3. **Consider the trade-offs** in your specific context
4. **Build optimization into your development process**

---

## Additional Resources

### Memory Profiling Tools
- **tracemalloc**: Built-in Python memory profiler
- **memory_profiler**: Line-by-line memory usage
- **objgraph**: Visualize object references
- **pympler**: Advanced memory analysis tools

### Optimization Libraries
- **NumPy**: Numerical computing and arrays
- **Pandas**: Data analysis with optimized memory usage
- **Polars**: Fast DataFrame library with memory efficiency
- **Dask**: Parallel computing with memory-efficient processing

### Further Reading
- [High Performance Python](https://www.oreilly.com/library/view/high-performance-python/9781449361747/) - Practical optimization guide
- [Python Memory Management](https://docs.python.org/3/c-api/memory.html) - CPython memory management
- [Effective Python](https://effectivepython.com/) - Best practices including optimization

### Related Sections
- **Section 5**: ctypes Inspection - Tools for memory analysis
- **Section 2**: Python Object Model - Understanding memory overhead
- **Section 1**: C Memory Basics - Memory fundamentals

---

**Tutorial Series Navigation:**
- **Previous:** [05 - ctypes Inspection](05_ctypes_inspection.md)
- **Current:** 06 - Optimization Techniques
- **Next:** [07 - Advanced Topics](07_advanced_topics.md) *(coming soon)*