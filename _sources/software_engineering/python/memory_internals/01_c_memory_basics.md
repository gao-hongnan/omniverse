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

# C Memory Basics: The Foundation

## Overview

Before we can understand Python's memory model, we need to understand how C manages memory. C is Python's foundation—CPython (the standard Python interpreter) is written in C, and Python objects are built on top of C structures.

In this section, you'll learn:
- How C stores data in memory (stack vs heap)
- Primitive data types and their sizes
- Structs and memory layout
- Memory alignment and padding
- How to use Python's `ctypes` to work with C

## Stack vs Heap: Two Kinds of Memory

Every program has access to two main regions of memory:

### The Stack

The **stack** is fast, automatic memory used for:
- Local variables
- Function parameters
- Return addresses

```{code-cell} ipython3
from __future__ import annotations

import ctypes
from typing import Final

# In C, this would be a stack-allocated variable:
# int x = 42;
#
# In Python with ctypes, we can simulate it:
x: ctypes.c_int = ctypes.c_int(42)
print(f"Value: {x.value}")
print(f"Size: {ctypes.sizeof(x)} bytes")
print(f"Address: {ctypes.addressof(x):#x}")
```

**Stack Characteristics:**
- **LIFO (Last In, First Out)**: Like a stack of plates
- **Automatic cleanup**: Variables disappear when function returns
- **Fast allocation**: Just move a pointer
- **Limited size**: Typically 1-8 MB (causes "stack overflow" errors)
- **Thread-local**: Each thread has its own stack

### Stack Memory Visualization

```
Stack grows downward (high address → low address)
┌─────────────────┐  ← High Address (e.g., 0x7fff...)
│  main() frame   │
│  int result     │
├─────────────────┤
│  foo() frame    │
│  int x = 42     │  ← Current function
│  char c = 'A'   │
├─────────────────┤
│  Unused Stack   │
│      ...        │
└─────────────────┘  ← Stack Pointer (SP)
```

### The Heap

The **heap** is slower, manual memory used for:
- Dynamic allocations
- Objects that outlive their function
- Large data structures

In C:
```c
// Heap allocation in C
int* p = malloc(sizeof(int));  // Allocate
*p = 42;                        // Use
free(p);                        // Must free manually!
```

```{code-cell} ipython3
# Python's 'everything is an object' means ALL Python objects live on the heap
numbers: list[int] = [1, 2, 3, 4, 5]  # This list is on the heap
# Python automatically frees it when refcount reaches 0
```

**Heap Characteristics:**
- **Manual management**: You allocate and free (in C)
- **Persistent**: Lives until you free it
- **Slow allocation**: Complex allocator algorithms
- **Large size**: Limited only by system RAM
- **Shared**: All threads share the same heap

### Heap Memory Visualization

```
Heap grows upward (low address → high address)
┌─────────────────┐  ← Low Address (e.g., 0x0001...)
│  malloc'd block │
│  (int*) 4 bytes │
├─────────────────┤
│  malloc'd block │
│  (struct) 40B   │
├─────────────────┤
│  Free space     │
│      ...        │
└─────────────────┘  ← Heap Pointer (grows upward)
```

## C Primitive Types

C has **primitive types**—basic types directly supported by the hardware.

```{code-cell} ipython3
# Let's examine all C integer types using ctypes
integer_types: list[tuple[str, type[ctypes._CData]]] = [
    ("char", ctypes.c_char),
    ("signed char", ctypes.c_byte),
    ("unsigned char", ctypes.c_ubyte),
    ("short", ctypes.c_short),
    ("unsigned short", ctypes.c_ushort),
    ("int", ctypes.c_int),
    ("unsigned int", ctypes.c_uint),
    ("long", ctypes.c_long),
    ("unsigned long", ctypes.c_ulong),
    ("long long", ctypes.c_longlong),
    ("unsigned long long", ctypes.c_ulonglong),
]

print("C Integer Types on This Platform")
print("=" * 60)
print(f"{'Type':<25} {'Size (bytes)':<15} {'Bits'}")
print("-" * 60)

for type_name, c_type in integer_types:
    size = ctypes.sizeof(c_type)
    print(f"{type_name:<25} {size:<15} {size * 8}")
```

### Key Observations

1. **Fixed size**: Each type has a specific size
2. **Platform dependent**: `long` might be 4 or 8 bytes
3. **No overflow protection**: Overflow wraps around silently

```{code-cell} ipython3
# Demonstrate C integer overflow
c_byte: ctypes.c_byte = ctypes.c_byte(127)  # Max value for signed char
print(f"Value: {c_byte.value}")

c_byte.value = 128  # Overflow!
print(f"After overflow: {c_byte.value}")  # Wraps to -128

# In Python, this would just work:
py_int: int = 127
py_int += 1
print(f"Python int: {py_int}")  # No overflow
```

## C Structs: Grouping Data

A **struct** groups related data together. This is the foundation for Python objects!

### Defining a Struct in C

```c
// C code (for reference)
struct Point {
    int x;
    int y;
};
```

### Using ctypes.Structure in Python

```{code-cell} ipython3
from typing import ClassVar

class Point(ctypes.Structure):
    """A simple 2D point structure."""
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
    ]

# Create an instance
p: Point = Point(10, 20)
print(f"Point: ({p.x}, {p.y})")
print(f"Size: {ctypes.sizeof(Point)} bytes")
print(f"Address: {ctypes.addressof(p):#x}")
```

### Memory Layout of Point

```
Point struct (8 bytes total on 64-bit):
┌─────────────────┐  ← Offset 0
│  x (int, 4B)    │  Value: 10
├─────────────────┤  ← Offset 4
│  y (int, 4B)    │  Value: 20
└─────────────────┘  ← End (8 bytes)
```

### Accessing Struct Fields

```{code-cell} ipython3
# Field offsets
print(f"Offset of x: {Point.x.offset} bytes")
print(f"Offset of y: {Point.y.offset} bytes")

# Field sizes
print(f"Size of x: {ctypes.sizeof(ctypes.c_int)} bytes")
print(f"Size of y: {ctypes.sizeof(ctypes.c_int)} bytes")

# Memory inspection
from omnivault.utils.memory.inspector import inspect_structure_layout

layout = inspect_structure_layout(Point)
print("\nDetailed Layout:")
for field in layout:
    print(f"  {field.field_name}: offset={field.offset}, size={field.size}, type={field.type_name}")
```

## Memory Alignment and Padding

CPUs prefer to access memory at aligned addresses. This leads to **padding**—extra bytes inserted for efficiency.

### The Alignment Rule

> On a 64-bit system, an N-byte type should start at an address divisible by N (up to 8).

```{code-cell} ipython3
class UnalignedStruct(ctypes.Structure):
    """Struct with potential padding."""
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("a", ctypes.c_char),   # 1 byte
        ("b", ctypes.c_int),    # 4 bytes
        ("c", ctypes.c_char),   # 1 byte
    ]

print(f"Size: {ctypes.sizeof(UnalignedStruct)} bytes")
print("Expected: 1 + 4 + 1 = 6 bytes")
print(f"Actual: {ctypes.sizeof(UnalignedStruct)} bytes")
print("\nLayout:")
for field in inspect_structure_layout(UnalignedStruct):
    print(f"  {field.field_name}: offset={field.offset}, size={field.size}")
```

### Visualizing Padding

```
UnalignedStruct (12 bytes with padding):
┌────┬───┬───┬───┐
│ a  │ P │ P │ P │  1 byte + 3 bytes padding
├────┴───┴───┴───┤  ← Offset 0-3
│  b (int, 4B)   │  Must align to 4-byte boundary
├────┬───┬───┬───┤  ← Offset 4-7
│ c  │ P │ P │ P │  1 byte + 3 bytes padding
└────┴───┴───┴───┘  ← Offset 8-11 (total: 12 bytes)

P = Padding byte (wasted space)
```

### Optimizing Struct Layout

By reordering fields, we can reduce padding:

```{code-cell} ipython3
class OptimizedStruct(ctypes.Structure):
    """Struct with optimized field order."""
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("b", ctypes.c_int),    # 4 bytes (put largest first)
        ("a", ctypes.c_char),   # 1 byte
        ("c", ctypes.c_char),   # 1 byte
        # Only 2 bytes of padding needed at end
    ]

print(f"Unoptimized: {ctypes.sizeof(UnalignedStruct)} bytes")
print(f"Optimized:   {ctypes.sizeof(OptimizedStruct)} bytes")
print(f"Savings:     {ctypes.sizeof(UnalignedStruct) - ctypes.sizeof(OptimizedStruct)} bytes")

print("\nOptimized Layout:")
for field in inspect_structure_layout(OptimizedStruct):
    print(f"  {field.field_name}: offset={field.offset}, size={field.size}")
```

```
OptimizedStruct (8 bytes):
┌────────────────┐
│  b (int, 4B)   │  ← Offset 0-3
├────┬────┬──┬──┤
│ a  │ c  │P │P │  ← Offset 4-7
└────┴────┴──┴──┘  (total: 8 bytes, only 2 bytes padding)
```

## Pointers: Addresses in Memory

A **pointer** is a variable that stores a memory address.

```{code-cell} ipython3
# In C: int* p = &x;
# In Python with ctypes:

value: ctypes.c_int = ctypes.c_int(42)
pointer: ctypes.POINTER(ctypes.c_int) = ctypes.pointer(value)

print(f"Value: {value.value}")
print(f"Address of value: {ctypes.addressof(value):#x}")
print(f"Pointer contents: {ctypes.addressof(value):#x}")
print(f"Dereferenced pointer: {pointer.contents.value}")
```

### Pointer Size

On 64-bit systems, pointers are always 8 bytes:

```{code-cell} ipython3
# All pointers are same size regardless of what they point to
ptr_to_char: type[ctypes._CData] = ctypes.POINTER(ctypes.c_char)
ptr_to_int: type[ctypes._CData] = ctypes.POINTER(ctypes.c_int)
ptr_to_double: type[ctypes._CData] = ctypes.POINTER(ctypes.c_double)

print(f"Pointer to char:   {ctypes.sizeof(ptr_to_char)} bytes")
print(f"Pointer to int:    {ctypes.sizeof(ptr_to_int)} bytes")
print(f"Pointer to double: {ctypes.sizeof(ptr_to_double)} bytes")
print("\nAll pointers are 8 bytes on 64-bit systems!")
```

## Arrays in C

Arrays are contiguous blocks of memory:

```{code-cell} ipython3
# Create a C array of 10 integers
IntArray10: type[ctypes.Array[ctypes.c_int]] = ctypes.c_int * 10
arr: ctypes.Array[ctypes.c_int] = IntArray10(*range(10))

print(f"Array size: {ctypes.sizeof(arr)} bytes")
print(f"Elements: {[arr[i] for i in range(10)]}")
print(f"Size per element: {ctypes.sizeof(ctypes.c_int)} bytes")
print(f"Total: {10 * ctypes.sizeof(ctypes.c_int)} bytes")
```

### Array Memory Layout

```
int arr[10] (40 bytes):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
  0    4    8   12   16   20   24   28   32   36  (offsets)

Contiguous memory: No gaps, no metadata
```

## C Strings: Null-Terminated Character Arrays

In C, strings are just arrays of characters ending with `\0`:

```{code-cell} ipython3
# C string: char str[] = "Hello";
c_string: bytes = b"Hello\x00"  # Must include null terminator
print(f"String: {c_string}")
print(f"Length: {len(c_string)} bytes (including \\0)")

# Using ctypes
c_str: ctypes.c_char_p = ctypes.c_char_p(b"Hello")
print(f"C string value: {c_str.value}")
print(f"Pointer size: {ctypes.sizeof(c_str)} bytes")
```

### C String Memory Layout

```
char str[] = "Hello" (6 bytes):
┌───┬───┬───┬───┬───┬────┐
│'H'│'e'│'l'│'l'│'o'│ \0 │
└───┴───┴───┴───┴───┴────┘
  0   1   2   3   4    5  (offsets)

\0 = Null terminator (marks end of string)
```

## Comparing C and Python Memory

Let's build a comprehensive comparison:

```{code-cell} ipython3
import sys
from omnivault.utils.memory.inspector import MemoryInspector

# Integer comparison
print("INTEGER COMPARISON")
print("=" * 60)
c_int_val: ctypes.c_int = ctypes.c_int(42)
py_int_val: int = 42

print(f"C int:      {ctypes.sizeof(c_int_val):>3} bytes (just the value)")
print(f"Python int: {sys.getsizeof(py_int_val):>3} bytes (object with metadata)")
print()

# String comparison
print("STRING COMPARISON")
print("=" * 60)
c_str_val: ctypes.c_char_p = ctypes.c_char_p(b"Hello")
py_str_val: str = "Hello"

print(f"C string (pointer): {ctypes.sizeof(c_str_val):>3} bytes (8-byte pointer)")
print(f"C string (data):    {6:>3} bytes (5 chars + \\0)")
print(f"Python string:      {sys.getsizeof(py_str_val):>3} bytes (full object)")
print()

# Array comparison
print("ARRAY COMPARISON")
print("=" * 60)
c_arr: ctypes.Array[ctypes.c_int] = (ctypes.c_int * 10)(*range(10))
py_list: list[int] = list(range(10))

print(f"C array:    {ctypes.sizeof(c_arr):>3} bytes (10 × 4 bytes, contiguous)")
print(f"Python list: {sys.getsizeof(py_list):>3} bytes (list object + pointers)")
print("            + 28 bytes × 10 for each integer object")
```

## Key Takeaways

1. **Stack vs Heap**
   - Stack: Fast, automatic, limited
   - Heap: Slower, manual (in C), unlimited

2. **C Types Are Simple**
   - Fixed size
   - No metadata
   - Direct hardware support
   - Manual memory management

3. **Memory Alignment Matters**
   - Padding wastes space but improves performance
   - Field order affects struct size
   - Pointers are always same size (8 bytes on 64-bit)

4. **Python Is Different**
   - Everything on heap
   - Everything is an object (with metadata)
   - Automatic memory management
   - Higher overhead, but much safer

## Exercises

### Exercise 1: Calculate Struct Sizes

Predict the size of these structs (assume 64-bit system):

```{code-cell} ipython3
class Exercise1A(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("a", ctypes.c_char),
        ("b", ctypes.c_char),
        ("c", ctypes.c_int),
    ]

class Exercise1B(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("a", ctypes.c_char),
        ("b", ctypes.c_int),
        ("c", ctypes.c_char),
    ]

# Your predictions:
# Exercise1A: ? bytes
# Exercise1B: ? bytes

# Actual sizes:
print(f"Exercise1A: {ctypes.sizeof(Exercise1A)} bytes")
print(f"Exercise1B: {ctypes.sizeof(Exercise1B)} bytes")

# Visualize layouts:
print("\nExercise1A layout:")
for field in inspect_structure_layout(Exercise1A):
    print(f"  {field.field_name}: offset={field.offset}, size={field.size}")

print("\nExercise1B layout:")
for field in inspect_structure_layout(Exercise1B):
    print(f"  {field.field_name}: offset={field.offset}, size={field.size}")
```

### Exercise 2: Optimize a Struct

Reorder the fields to minimize padding:

```{code-cell} ipython3
class UnoptimizedStruct(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("a", ctypes.c_char),
        ("b", ctypes.c_double),  # 8 bytes
        ("c", ctypes.c_char),
        ("d", ctypes.c_int),
    ]

class OptimizedStruct(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        # TODO: Reorder these fields to minimize size
        ("b", ctypes.c_double),
        ("d", ctypes.c_int),
        ("a", ctypes.c_char),
        ("c", ctypes.c_char),
    ]

print(f"Unoptimized: {ctypes.sizeof(UnoptimizedStruct)} bytes")
print(f"Optimized:   {ctypes.sizeof(OptimizedStruct)} bytes")
```

### Exercise 3: Memory Comparison

Compare memory usage for storing 1000 integers:

```{code-cell} ipython3
from omnivault.utils.memory.profiler import track_memory

# C array approach
with track_memory() as snapshot_c:
    c_array: ctypes.Array[ctypes.c_int] = (ctypes.c_int * 1000)(*range(1000))

# Python list approach
with track_memory() as snapshot_py:
    py_list_ex: list[int] = list(range(1000))

print(f"C array memory:    {snapshot_c.peak_kb:.2f} KB")
print(f"Python list memory: {snapshot_py.peak_kb:.2f} KB")
print(f"Ratio: {snapshot_py.peak_kb / snapshot_c.peak_kb:.1f}x more for Python")
```

## Next Steps

Now that you understand C's memory model, you're ready to see how Python builds on top of it!

**[Continue to Section 2: Python Object Model →](02_python_object_model.md)**

In the next section, we'll see how Python wraps C structures to create its rich object system.

---

**Tutorial Series Navigation:**
- **Previous:** [00 - Introduction](00_introduction.md)
- **Current:** 01 - C Memory Basics
- **Next:** [02 - Python Object Model](02_python_object_model.md)
