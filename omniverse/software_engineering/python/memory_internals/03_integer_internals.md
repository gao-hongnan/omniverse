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

# Integer Internals: PyLongObject and Arbitrary Precision

## Overview

Finally, we can answer the original mystery: **Why does `sys.getsizeof(5)` return 28 bytes?**

In this section, you'll learn:
- How Python implements integers using PyLongObject
- Why Python integers never overflow
- The small integer cache (-5 to 256)
- How integer size grows with magnitude
- Base-2^30 arithmetic internals

## The Mystery Solved: 28 Bytes for Integer 5

Let's break down exactly where those 28 bytes come from:

```{code-cell} ipython3
from __future__ import annotations

import sys
import ctypes
from typing import ClassVar

num: int = 5
print(f"sys.getsizeof(5) = {sys.getsizeof(num)} bytes")
print("\nBreakdown:")
print("  ob_refcnt:  8 bytes (reference count)")
print("  ob_type:    8 bytes (pointer to int type)")
print("  ob_size:    8 bytes (number of digits)")
print("  ob_digit:   4 bytes (the actual value)")
print("  " + "=" * 40)
print(f"  Total:     28 bytes")
```

### PyLongObject Structure

All Python integers use the `PyLongObject` structure:

```{code-cell} ipython3
# Simplified PyLongObject from CPython source
class PyObject(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
    ]

class PyVarObject(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_base", PyObject),
        ("ob_size", ctypes.c_ssize_t),
    ]

# This is a simplified version - actual CPython uses flexible array member
class PyLongObject(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_base", PyVarObject),
        ("ob_digit", ctypes.c_uint32 * 1),  # Variable-length array
    ]

print(f"PyLongObject minimum size: {ctypes.sizeof(PyLongObject)} bytes")
```

### Memory Layout Visualization

```
PyLongObject for integer 5 (28 bytes):
┌──────────────────┐  ← Offset 0
│  ob_refcnt (8B)  │  Reference count
├──────────────────┤  ← Offset 8
│  ob_type (8B)    │  Pointer to 'int' type object
├──────────────────┤  ← Offset 16
│  ob_size (8B)    │  Number of digits (1 for small ints)
├──────────────────┤  ← Offset 24
│  ob_digit[0] (4B)│  The value: 5
└──────────────────┘  ← Offset 28

Compare to C int (4 bytes):
┌──────────────────┐
│  00 00 00 05     │  Just the value
└──────────────────┘
```

## Let's Inspect It!

```{code-cell} ipython3
from omnivault.utils.memory.inspector import MemoryInspector

# Inspect the integer 5
inspector = MemoryInspector(5)
print(inspector.summary())
print("\nMemory dump (first 32 bytes):")
print(inspector.dump_hex(32))

# Get PyVarObject header
var_header = inspector.get_var_header()
print(f"\nPyVarObject fields:")
print(f"  ob_refcnt: {var_header.ob_base.ob_refcnt}")
print(f"  ob_type: {var_header.ob_base.ob_type:#x}")
print(f"  ob_size: {var_header.ob_size}")
```

## C Integer vs Python Integer

The fundamental difference:

```{code-cell} ipython3
# C integer: Fixed size, can overflow
c_int: ctypes.c_int = ctypes.c_int(2_147_483_647)  # Max 32-bit int
print(f"C int max: {c_int.value}")
print(f"Size: {ctypes.sizeof(c_int)} bytes")

c_int.value += 1  # Overflow!
print(f"After +1: {c_int.value}")  # Wraps to negative
print(f"Size: {ctypes.sizeof(c_int)} bytes (same)")

print("\n" + "=" * 60)

# Python integer: Arbitrary precision, never overflows
py_int: int = 2_147_483_647
print(f"\nPython int max (same value): {py_int}")
print(f"Size: {sys.getsizeof(py_int)} bytes")

py_int += 1  # No overflow!
print(f"After +1: {py_int}")
print(f"Size: {sys.getsizeof(py_int)} bytes")

# Go even bigger!
huge: int = 10**100
print(f"\n10^100: {huge}")
print(f"Size: {sys.getsizeof(huge)} bytes")
print(f"This would be impossible in C!")
```

## How Integer Size Grows

Python integers grow as needed:

```{code-cell} ipython3
import math

# Test different magnitudes
test_values: list[int] = [
    0,
    1,
    255,
    256,
    2**30 - 1,    # Max single digit
    2**30,        # Needs two digits
    2**60 - 1,    # Max two digits
    2**60,        # Needs three digits
    10**100,      # Very large
]

print(f"{'Value':<25} {'Size (bytes)':<15} {'Growth'}")
print("=" * 60)

prev_size: int = 0
for val in test_values:
    size = sys.getsizeof(val)
    growth = f"+{size - prev_size}" if prev_size > 0 else "baseline"
    val_str = f"{val:.2e}" if val > 10**6 else str(val)
    print(f"{val_str:<25} {size:<15} {growth}")
    prev_size = size
```

### Size Growth Pattern

```{code-cell} ipython3
# Visualize the pattern
sizes: dict[str, int] = {}
for i in range(100):
    val = 2 ** (i * 6)  # Exponentially increasing values
    sizes[f"2^{i*6}"] = sys.getsizeof(val)

# Show size jumps
print("Integer size jumps at powers of 2^30:")
print(f"{'Range':<30} {'Size'}")
print("-" * 45)
print(f"0 to 2^30-1 (1 digit)      28 bytes")
print(f"2^30 to 2^60-1 (2 digits)  32 bytes")
print(f"2^60 to 2^90-1 (3 digits)  36 bytes")
print(f"2^90 to 2^120-1 (4 digits) 40 bytes")
print("\nPattern: 28 + 4 * (num_digits - 1) bytes")
```

## Base-2^30 Arithmetic

Python uses **base-2^30** internally (not base 10!):

### Why Base-2^30?

```{code-cell} ipython3
print("Why base-2^30?")
print("=" * 60)
print("1. Uses 30 bits of each 32-bit uint32")
print("2. Leaves 2 bits for overflow during multiplication")
print("3. Efficient for CPU operations")
print("4. Each 'digit' can hold values 0 to 2^30-1")
print(f"\nMax digit value: {2**30 - 1:,}")
```

### Digit Array Visualization

```
Number: 1,234,567,890,123,456,789 (fits in two base-2^30 digits)

In base-2^30:
┌────────────┬────────────┐
│  digit[1]  │  digit[0]  │
│ 1,150,999  │  234,000   │ (example values, not actual)
└────────────┴────────────┘
     High         Low

Actual value = digit[0] + digit[1] * (2^30)
```

### How Addition Works

```{code-cell} ipython3
# Simplified explanation of how Python adds large integers
def explain_addition() -> None:
    """Explain how Python adds large integers digit by digit."""
    a: int = 2**30 + 100  # Two digits needed
    b: int = 2**30 + 200

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"Size of a: {sys.getsizeof(a)} bytes")
    print(f"Size of b: {sys.getsizeof(b)} bytes")

    result = a + b
    print(f"\nResult = {result}")
    print(f"Size of result: {sys.getsizeof(result)} bytes")

    print("\nInternal process (simplified):")
    print("1. Allocate result object with enough digits")
    print("2. Add digit[0] + digit[0] with carry")
    print("3. Add digit[1] + digit[1] + carry")
    print("4. Continue for all digits")
    print("5. Grow array if needed for final carry")

explain_addition()
```

## The Small Integer Cache (-5 to 256)

Python **pre-allocates** integers from -5 to 256. This is a huge optimization!

```{code-cell} ipython3
# Small integers are cached (singleton pattern)
a: int = 100
b: int = 100
c: int = 100

print(f"a is b: {a is b}")  # True - same object!
print(f"a is c: {a is c}")  # True
print(f"id(a): {id(a):#x}")
print(f"id(b): {id(b):#x}")
print(f"id(c): {id(c):#x}")

print("\n" + "=" * 60)

# Large integers are NOT cached
x: int = 1000
y: int = 1000
z: int = 1000

print(f"x is y: {x is y}")  # False - different objects
print(f"x is z: {x is z}")  # False
print(f"id(x): {id(x):#x}")
print(f"id(y): {id(y):#x}")  # Different!
print(f"id(z): {id(z):#x}")  # Different!
```

### Visualizing the Cache

```
Small Integer Cache: -5 to 256
┌────┬────┬────┬─────┬─────┬──────┐
│ -5 │ -4 │... │  0  │ ... │ 256  │  Pre-allocated at startup
└────┴────┴────┴─────┴─────┴──────┘
  ↑                ↑           ↑
All references point to the same objects

Example:
a = 5    ──┐
b = 5    ──┼─→  Single PyLongObject(5)
c = 2+3  ──┘

vs.

Large Integers: Each is unique
a = 1000  → PyLongObject(1000) at 0x1234...
b = 1000  → PyLongObject(1000) at 0x5678...  (different object!)
```

### Cache Boundaries

```{code-cell} ipython3
from typing import Final

# Cache constants
SMALL_INT_MIN: Final[int] = -5
SMALL_INT_MAX: Final[int] = 256

def is_cached(n: int) -> bool:
    """Check if an integer is in the small int cache."""
    # Create two instances and check if they're the same object
    a = n
    b = n
    return a is b

# Test boundaries
test_values: list[int] = [-6, -5, -4, 0, 255, 256, 257, 1000]

print(f"{'Value':<10} {'Cached?':<10} {'id() same?'}")
print("=" * 40)
for val in test_values:
    cached = "Yes" if is_cached(val) else "No"
    a, b = val, val
    same_id = "Yes" if id(a) == id(b) else "No"
    print(f"{val:<10} {cached:<10} {same_id}")
```

## Why This Design?

Let's understand the trade-offs:

```{code-cell} ipython3
from omnivault.utils.memory.profiler import track_memory

# Scenario 1: Using small integers (cached)
with track_memory() as snapshot1:
    small_ints: list[int] = [i % 100 for i in range(10000)]  # Reuses cached integers

# Scenario 2: Using large integers (not cached)
with track_memory() as snapshot2:
    large_ints: list[int] = [i + 1000 for i in range(10000)]  # Creates new integers

print("Memory Usage Comparison:")
print(f"10,000 small integers (cached):     {snapshot1.peak_kb:.2f} KB")
print(f"10,000 large integers (not cached): {snapshot2.peak_kb:.2f} KB")
print(f"\nSavings from caching: {snapshot2.peak_kb - snapshot1.peak_kb:.2f} KB")
```

## Performance Implications

### Fast Operations (Small Integers)

```{code-cell} ipython3
import timeit

# Small integer arithmetic (cache hits)
time_small = timeit.timeit("x = 10; y = 20; z = x + y", number=1_000_000)
print(f"Small int operations: {time_small:.6f} seconds")

# Large integer arithmetic (no cache)
time_large = timeit.timeit("x = 10**100; y = 20**100; z = x + y", number=1_000_000)
print(f"Large int operations: {time_large:.6f} seconds")

print(f"\nRatio: {time_large / time_small:.1f}x slower for large integers")
```

### Memory Comparison Table

```{code-cell} ipython3
import pandas as pd  # type: ignore

# Create comparison data
data: list[dict[str, str | int]] = [
    {"Type": "C int32", "Size": 4, "Min": -2**31, "Max": 2**31-1, "Overflow": "Yes"},
    {"Type": "C int64", "Size": 8, "Min": -2**63, "Max": 2**63-1, "Overflow": "Yes"},
    {"Type": "Python int (small)", "Size": 28, "Min": "Unlimited", "Max": "Unlimited", "Overflow": "No"},
    {"Type": "Python int (large)", "Size": "28+4n", "Min": "Unlimited", "Max": "Unlimited", "Overflow": "No"},
]

# Format for display
print("Integer Type Comparison")
print("=" * 90)
print(f"{'Type':<25} {'Size (bytes)':<15} {'Range':<30} {'Overflow?'}")
print("-" * 90)
for row in data:
    type_name = row["Type"]  # type: ignore
    size = row["Size"]  # type: ignore
    overflow = row["Overflow"]  # type: ignore

    if isinstance(row["Min"], int):
        range_str = f"{row['Min']} to {row['Max']}"  # type: ignore
    else:
        range_str = f"{row['Min']}"  # type: ignore

    print(f"{type_name:<25} {str(size):<15} {range_str:<30} {overflow}")
```

## Advanced: Inspecting Internal Representation

```{code-cell} ipython3
# Let's look at how different integers are stored
def inspect_integer(n: int) -> None:
    """Detailed inspection of an integer's internal representation."""
    inspector = MemoryInspector(n)
    var_header = inspector.get_var_header()

    # Calculate number of base-2^30 digits needed
    if n == 0:
        num_digits = 0
    else:
        bit_length = n.bit_length()
        num_digits = (bit_length + 29) // 30  # Ceiling division

    print(f"\nInteger: {n}")
    print(f"  Size: {inspector.size} bytes")
    print(f"  ob_size: {var_header.ob_size}")
    print(f"  Bit length: {n.bit_length()} bits")
    print(f"  Estimated digits: ~{num_digits}")
    print(f"  Binary: {bin(n) if n < 1000 else bin(n)[:50] + '...'}")

# Test various integers
for val in [0, 5, 255, 2**30-1, 2**30, 2**60, 10**100]:
    inspect_integer(val)
```

## Comparing with Other Languages

```{code-cell} ipython3
print("Integer Implementation Comparison")
print("=" * 80)
print("\nC/C++:")
print("  - Fixed size: int (4B), long (8B), long long (8B)")
print("  - Overflow: Silent wraparound (dangerous!)")
print("  - Performance: Fastest (direct CPU instructions)")
print("")
print("Java:")
print("  - Fixed size: int (4B), long (8B)")
print("  - Overflow: Silent wraparound")
print("  - BigInteger class for arbitrary precision (like Python int)")
print("")
print("Python:")
print("  - Arbitrary precision: Grows as needed")
print("  - Overflow: Never (automatically grows)")
print("  - Performance: Slower, but safer and more convenient")
print("")
print("Go:")
print("  - Fixed size: int, int64, uint64")
print("  - Overflow: Explicit via math/big package")
print("")
print("Rust:")
print("  - Fixed size: i32, i64, u64")
print("  - Overflow: Checked, unchecked, or wrapping modes")
```

## Key Takeaways

1. **PyLongObject Structure (28 bytes minimum)**
   - ob_refcnt: 8 bytes
   - ob_type: 8 bytes
   - ob_size: 8 bytes
   - ob_digit[]: 4+ bytes (grows with number magnitude)

2. **Arbitrary Precision is a Feature**
   - No overflow errors
   - Limited only by available memory
   - Uses base-2^30 representation

3. **Small Integer Cache (-5 to 256)**
   - Massive memory savings for common values
   - Faster comparison (pointer equality)
   - Created at interpreter startup

4. **Size Growth is Predictable**
   - Base size: 28 bytes
   - Growth: +4 bytes per additional base-2^30 digit
   - Formula: 28 + 4 * (num_digits - 1)

5. **Trade-offs**
   - Python: Safe, convenient, memory-heavy
   - C: Fast, dangerous, memory-efficient
   - Choose based on your use case!

## Exercises

### Exercise 1: Find Size Transitions

Find the exact values where integer size increases:

```{code-cell} ipython3
def find_size_transitions(max_bits: int = 200) -> None:
    """Find where integer size increases."""
    prev_size: int = 0
    transitions: list[tuple[int, int]] = []

    for bits in range(1, max_bits):
        val = 2**bits
        size = sys.getsizeof(val)

        if size != prev_size and prev_size > 0:
            transitions.append((bits, size))
            print(f"Size increases to {size} bytes at 2^{bits}")

        prev_size = size

    return None

find_size_transitions(100)
```

### Exercise 2: Cache Verification

Verify the small integer cache:

```{code-cell} ipython3
def verify_cache() -> None:
    """Verify small integer cache boundaries."""
    # Test around boundaries
    for n in range(-10, 300):
        a = n
        b = n
        is_same = a is b

        if n in (-6, -5, 256, 257):  # Boundaries
            status = "CACHED" if is_same else "NOT CACHED"
            print(f"n={n:>4}: {status}")

verify_cache()
```

### Exercise 3: Memory Profiling

Profile memory usage for different integer patterns:

```{code-cell} ipython3
# Pattern 1: Consecutive small integers
with track_memory() as s1:
    pattern1: list[int] = list(range(10000))

# Pattern 2: Large consecutive integers
with track_memory() as s2:
    pattern2: list[int] = list(range(100000, 110000))

# Pattern 3: Random large integers
with track_memory() as s3:
    import random
    pattern3: list[int] = [random.randint(10**10, 10**11) for _ in range(10000)]

print(f"Small consecutive:  {s1.peak_mb:.2f} MB")
print(f"Large consecutive:  {s2.peak_mb:.2f} MB")
print(f"Random large:       {s3.peak_mb:.2f} MB")
```

## Next Steps

Now that you understand integer internals, let's explore strings!

**[Continue to Section 4: String Internals →](04_string_internals.md)**

In the next section, we'll see how Python stores strings using PyUnicodeObject and optimizes with interning.

---

**Tutorial Series Navigation:**
- **Previous:** [02 - Python Object Model](02_python_object_model.md)
- **Current:** 03 - Integer Internals
- **Next:** [04 - String Internals](04_string_internals.md)
