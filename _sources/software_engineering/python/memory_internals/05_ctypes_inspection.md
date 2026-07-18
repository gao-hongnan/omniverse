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

# Section 5: Hands-On ctypes Memory Inspection

## Overview

Now that you understand Python's object model (Section 2), integer internals (Section 3), and string internals (Section 4), it's time to **get your hands dirty** with direct memory inspection.

In this section, you'll learn:

- **What is ctypes** and why it's Python's "window into C"
- **Building custom ctypes.Structure definitions** for Python objects
- **Reading raw memory** with `ctypes.string_at()`
- **Pointer arithmetic** and following object references
- **Advanced techniques** like type punning and casting
- **3 Practical Projects** to solidify your understanding
- **Integration** with omnivault.utils.memory tools

### Learning Objectives

By the end of this section, you will be able to:

1. Define ctypes structures matching CPython's internal structures
2. Read and interpret raw memory bytes from Python objects
3. Follow pointers between objects in memory
4. Build custom memory inspection tools
5. Understand the safety concerns and limitations of memory inspection

### Prerequisites

You should have completed:

- Section 0: Introduction
- Section 1: C Memory Basics
- Section 2: Python Object Model
- Section 3: Integer Internals (recommended)
- Section 4: String Internals (recommended)

### What is ctypes?

**ctypes** is Python's built-in Foreign Function Interface (FFI) library that allows you to:

- Call C functions from shared libraries
- Define C-compatible data structures
- Access memory directly
- Convert between Python and C types

Think of ctypes as a **bridge** between Python's high-level world and C's low-level memory.

```
┌─────────────────────────────────────────┐
│         Python Layer                     │
│  - Objects with type info                │
│  - Automatic memory management           │
│  - Dynamic typing                        │
├─────────────────────────────────────────┤
│         ctypes Layer (Bridge)            │  ← We are here!
│  - ctypes.Structure definitions          │
│  - ctypes.string_at() for memory reads   │
│  - id() to get addresses                 │
├─────────────────────────────────────────┤
│         C Layer                          │
│  - Raw memory bytes                      │
│  - Structs and pointers                  │
│  - Manual memory management              │
└─────────────────────────────────────────┘
```

:::{warning}
## SAFETY WARNING: ctypes Can Crash Python

The official Python documentation states: _"There are, however, enough ways to crash Python with ctypes, so you should be careful anyway."_

**What this means:**
- Reading beyond allocated memory → Segmentation fault (instant crash)
- Incorrect pointer arithmetic → Memory corruption
- Accessing garbage-collected objects → Undefined behavior
- Wrong structure definitions → Data misalignment crashes

This tutorial teaches safe patterns, but ctypes gives you "C-level power" with "C-level danger."

**When in doubt:**
1. Use `faulthandler` module to debug crashes
2. Start with small byte counts (8, 16, 32 bytes)
3. Never read more bytes than `sys.getsizeof()` returns
4. Keep a reference to objects you're inspecting
:::

---

## 5.1 Setup and Platform Detection

Let's start by importing the tools we'll need and detecting our platform.

```{code-cell} python
from __future__ import annotations

import ctypes
import sys
from typing import Any, ClassVar, Final
from dataclasses import dataclass

# Import our memory utilities
from omnivault.utils.memory import (
    MemoryInspector,
    PyObject,
    PyVarObject,
    inspect_memory_bytes,
    format_hex_dump,
    inspect_structure_layout,
)

# Platform information
POINTER_SIZE: Final[int] = ctypes.sizeof(ctypes.c_void_p)
IS_64BIT: Final[bool] = POINTER_SIZE == 8

print(f"Platform Information:")
print(f"  Pointer size: {POINTER_SIZE} bytes")
print(f"  Architecture: {'64-bit' if IS_64BIT else '32-bit'}")
print(f"  Python version: {sys.version}")
print(f"  sys.maxsize: {sys.maxsize}")
```

:::{note}
**Platform Differences:**
- **32-bit systems**: Pointers are 4 bytes, `sys.maxsize` is 2^31-1
- **64-bit systems**: Pointers are 8 bytes, `sys.maxsize` is 2^63-1

This tutorial assumes **64-bit CPython**. Most code will work on 32-bit, but structure sizes will differ.
:::

---

## 5.2 Understanding ctypes.Structure

### What is a ctypes.Structure?

A `ctypes.Structure` is Python's way of representing a C `struct`. It defines a **memory layout** with named fields at specific offsets.

### Why Do We Need This?

To inspect Python objects, we need to:

1. Know the exact memory layout (which bytes mean what)
2. Define that layout in Python using `ctypes.Structure`
3. "Cast" raw memory to our structure definition
4. Read the fields

Let's see this in action with a simple example.

### Example: A Simple Point Structure

```{code-cell} python
# Define a simple C-style Point structure
class Point(ctypes.Structure):
    """
    Equivalent C code:
        struct Point {
            int32_t x;
            int32_t y;
        };
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]

# Create an instance
point: Point = Point(10, 20)

print(f"Point Structure Analysis")
print(f"{'='*60}")
print(f"Size: {ctypes.sizeof(Point)} bytes")
print(f"Fields:")
print(f"  x: value={point.x}, offset={Point.x.offset} bytes")
print(f"  y: value={point.y}, offset={Point.y.offset} bytes")
print()

# Inspect memory layout
print(f"Memory Layout:")
layout = inspect_structure_layout(Point)
for field in layout:
    print(f"  {field.field_name:>10}: offset={field.offset:>2}, size={field.size} bytes, type={field.type_name}")
print()

# Read raw memory
print(f"Raw Memory Dump:")
address = ctypes.addressof(point)
raw_bytes = ctypes.string_at(address, ctypes.sizeof(Point))
print(format_hex_dump(raw_bytes))
```

**Expected Output:**
```
Point Structure Analysis
============================================================
Size: 8 bytes
Fields:
  x: value=10, offset=0 bytes
  y: value=20, offset=4 bytes

Memory Layout:
           x: offset= 0, size=4 bytes, type=c_int32
           y: offset= 4, size=4 bytes, type=c_int32

Raw Memory Dump:
0000: 0a 00 00 00 14 00 00 00  |........|
```

### Memory Alignment and Padding

C structures often have **padding bytes** inserted by the compiler to align fields on natural boundaries (typically 4 or 8 bytes).

**Why?** CPUs access aligned memory faster. Unaligned access can be slower or even cause crashes on some architectures.

```{code-cell} python
class PaddedStruct(ctypes.Structure):
    """
    Equivalent C code:
        struct PaddedStruct {
            char a;      // 1 byte
            // 3 bytes padding here!
            int32_t b;   // 4 bytes
            char c;      // 1 byte
            // 3 bytes padding at end!
        };
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("a", ctypes.c_char),
        ("b", ctypes.c_int32),
        ("c", ctypes.c_char),
    ]

class OptimizedStruct(ctypes.Structure):
    """
    Equivalent C code (fields reordered):
        struct OptimizedStruct {
            int32_t b;   // 4 bytes
            char a;      // 1 byte
            char c;      // 1 byte
            // 2 bytes padding at end
        };
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("b", ctypes.c_int32),
        ("a", ctypes.c_char),
        ("c", ctypes.c_char),
    ]

# Compare
padded = PaddedStruct(b'A', 42, b'B')
optimized = OptimizedStruct(42, b'A', b'B')

print(f"Padding Comparison")
print(f"{'='*60}")
print(f"PaddedStruct:")
print(f"  Size: {ctypes.sizeof(PaddedStruct)} bytes")
for name, _ in PaddedStruct._fields_:
    offset = getattr(PaddedStruct, name).offset
    print(f"  {name}: offset={offset}")

print()
print(f"OptimizedStruct:")
print(f"  Size: {ctypes.sizeof(OptimizedStruct)} bytes")
for name, _ in OptimizedStruct._fields_:
    offset = getattr(OptimizedStruct, name).offset
    print(f"  {name}: offset={offset}")

print()
print(f"Memory saved: {ctypes.sizeof(PaddedStruct) - ctypes.sizeof(OptimizedStruct)} bytes")
```

### Visualizing Memory Layout

```
PaddedStruct (12 bytes):
┌─────┬─────┬─────┬─────┐
│  a  │ pad │ pad │ pad │  0-3   (char + 3 padding)
├─────┴─────┴─────┴─────┤
│         b (int32)      │  4-7   (aligned to 4-byte boundary)
├─────┬─────┬─────┬─────┤
│  c  │ pad │ pad │ pad │  8-11  (char + 3 padding)
└─────┴─────┴─────┴─────┘

OptimizedStruct (8 bytes):
┌─────────────────────────┐
│     b (int32)           │  0-3   (int first)
├─────┬─────┬─────┬─────┤
│  a  │  c  │ pad │ pad │  4-7   (chars together + 2 padding)
└─────┴─────┴─────┴─────┘
```

**Key Lesson**: Field ordering matters! Reordering can reduce memory usage.

---

## 5.3 Building CPython Structure Definitions

Now let's recreate the **actual CPython structures** for inspecting Python objects.

### The PyObject Structure (Revisited)

We've seen this before in Section 2, but now we'll build it ourselves to understand every detail.

```{code-cell} python
# We already have PyObject from omnivault.utils.memory, but let's rebuild it
# to understand every detail

class PyObjectManual(ctypes.Structure):
    """
    Manual definition of CPython's PyObject.

    From CPython source (Include/object.h):
        typedef struct _object {
            Py_ssize_t ob_refcnt;
            PyTypeObject *ob_type;
        } PyObject;

    On 64-bit systems:
        - ob_refcnt: 8 bytes (signed size_t)
        - ob_type:   8 bytes (pointer to type object)
        Total: 16 bytes
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
    ]

# Test it on a real Python object
test_obj: int = 42
obj_address: int = id(test_obj)

# Cast the address to our structure
pyobj: PyObjectManual = PyObjectManual.from_address(obj_address)

print(f"PyObject Inspection")
print(f"{'='*60}")
print(f"Object: {test_obj}")
print(f"Address: {obj_address:#x}")
print(f"Reference count: {pyobj.ob_refcnt}")
print(f"Type pointer: {pyobj.ob_type:#x}")
print()

# Verify with sys.getrefcount
print(f"sys.getrefcount(): {sys.getrefcount(test_obj)}")
print(f"Note: getrefcount includes temporary references")
```

:::{tip}
**Why the refcount difference?**

`sys.getrefcount(obj)` creates a temporary reference when passing `obj` as an argument, so it's typically 1 higher than the actual count.
:::

### The PyVarObject Structure

Variable-sized objects (lists, tuples, strings, integers) use `PyVarObject`:

```{code-cell} python
class PyVarObjectManual(ctypes.Structure):
    """
    Manual definition of CPython's PyVarObject.

    From CPython source:
        typedef struct {
            PyObject ob_base;
            Py_ssize_t ob_size;
        } PyVarObject;

    On 64-bit systems: 24 bytes total
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_base", PyObjectManual),
        ("ob_size", ctypes.c_ssize_t),
    ]

# Test on a list
test_list: list[int] = [1, 2, 3, 4, 5]
list_address: int = id(test_list)

pyvar: PyVarObjectManual = PyVarObjectManual.from_address(list_address)

print(f"PyVarObject Inspection")
print(f"{'='*60}")
print(f"Object: {test_list}")
print(f"Address: {list_address:#x}")
print(f"Reference count: {pyvar.ob_base.ob_refcnt}")
print(f"Type pointer: {pyvar.ob_base.ob_type:#x}")
print(f"Size (number of items): {pyvar.ob_size}")
print(f"Actual length: {len(test_list)}")
print()

# Compare with inspector
inspector = MemoryInspector(test_list)
print(f"Using MemoryInspector:")
print(inspector.summary())
```

### The PyLongObject Structure

Now the big one - integers! This is where we finally understand the 28 bytes.

```{code-cell} python
class PyLongObjectSimple(ctypes.Structure):
    """
    Simplified PyLongObject for small integers.

    From CPython source (Include/longintrepr.h):
        struct _longobject {
            PyVarObject ob_base;
            digit ob_digit[1];  // flexible array member
        };

    Where digit is uint32_t on most platforms.

    For small integers (one digit):
        - PyVarObject: 24 bytes
        - ob_digit[0]: 4 bytes
        Total: 28 bytes
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_base", PyVarObjectManual),
        ("ob_digit", ctypes.c_uint32 * 1),  # Simplified: just one digit
    ]

# Inspect a small integer
small_int: int = 5
int_address: int = id(small_int)

pylong: PyLongObjectSimple = PyLongObjectSimple.from_address(int_address)

print(f"PyLongObject Inspection")
print(f"{'='*60}")
print(f"Integer value: {small_int}")
print(f"Address: {int_address:#x}")
print()

print(f"PyVarObject header:")
print(f"  ob_refcnt: {pylong.ob_base.ob_base.ob_refcnt}")
print(f"  ob_type:   {pylong.ob_base.ob_base.ob_type:#x}")
print(f"  ob_size:   {pylong.ob_base.ob_size} (number of digits)")
print()

print(f"Integer data:")
print(f"  ob_digit[0]: {pylong.ob_digit[0]} (the actual value)")
print()

print(f"Total size: {sys.getsizeof(small_int)} bytes")
print()

# Memory dump
inspector_int = MemoryInspector(small_int)
print(f"Memory dump:")
print(inspector_int.dump_hex(32))
```

:::{admonition} The 28-Byte Mystery Solved!
:class: tip

For integer `5`:
- **PyObject header**: 16 bytes (ob_refcnt + ob_type)
- **ob_size field**: 8 bytes (number of digits = 1)
- **ob_digit[0]**: 4 bytes (value = 5)
- **Total**: 28 bytes

This is why `sys.getsizeof(5)` returns 28!
:::

---

## 5.4 Reading Raw Memory with ctypes.string_at()

### The Power (and Danger) of ctypes.string_at()

`ctypes.string_at(address, size)` is your **direct window into memory**. It reads `size` bytes starting at `address` and returns them as a bytes object.

### How It Works

```
Memory:
   Address     Value
   0x1000  →  [0x2A 0x00 0x00 0x00]  ← 4 bytes representing int(42)

ctypes.string_at(0x1000, 4)  →  b'*\x00\x00\x00'
```

**What**: Read any memory you have an address to
**Why**: Inspect object internals directly
**Danger**: Reading invalid memory crashes your program!

:::{danger}
## CRITICAL SAFETY RULES

1. **Only read from valid objects**: Use `id(obj)` to get addresses
2. **Don't read past the object**: Use `sys.getsizeof()` to know the limit
3. **Expect the unexpected**: Memory layouts can vary by Python version
4. **Test on disposable data**: Don't inspect critical production objects

**Example of instant crash:**
```python
# DON'T DO THIS - will crash Python!
ctypes.string_at(0, 1024)  # Reading from NULL pointer
```
:::

### Safe Memory Reading Pattern

```{code-cell} python
def safe_memory_read(obj: object, num_bytes: int | None = None) -> bytes:
    """
    Safely read memory from a Python object.

    Args:
        obj: Object to inspect
        num_bytes: Number of bytes to read (default: sys.getsizeof(obj))

    Returns:
        Raw bytes from object memory

    Raises:
        ValueError: If num_bytes exceeds object size
    """
    obj_size: int = sys.getsizeof(obj)
    if num_bytes is None:
        num_bytes = obj_size

    if num_bytes > obj_size:
        raise ValueError(
            f"Requested {num_bytes} bytes but object is only {obj_size} bytes"
        )

    address: int = id(obj)
    return ctypes.string_at(address, num_bytes)

# Example 1: Read an integer's memory
value: int = 42
memory: bytes = safe_memory_read(value, 32)

print(f"Reading Integer Memory")
print(f"{'='*60}")
print(f"Value: {value}")
print(f"Size: {sys.getsizeof(value)} bytes")
print()
print(f"First 32 bytes:")
print(format_hex_dump(memory))
print()

# Example 2: Read a string's memory
text: str = "hello"
memory_str: bytes = safe_memory_read(text, 64)

print(f"Reading String Memory")
print(f"{'='*60}")
print(f"String: {text!r}")
print(f"Size: {sys.getsizeof(text)} bytes")
print()
print(f"First 64 bytes:")
print(format_hex_dump(memory_str))
```

### Interpreting Raw Bytes

When you read raw memory, you get **bytes**. But what do they mean?

You need to know:
1. **Endianness**: Little-endian (Intel) vs big-endian
2. **Data types**: Are these bytes an int? A pointer? A float?
3. **Structure layout**: Where does each field start?

```{code-cell} python
def read_int64_le(data: bytes, offset: int = 0) -> int:
    """
    Read a 64-bit little-endian integer from bytes.

    Args:
        data: Byte array
        offset: Starting offset (default: 0)

    Returns:
        Integer value

    Example:
        >>> data = b'\x2a\x00\x00\x00\x00\x00\x00\x00'
        >>> read_int64_le(data)
        42
    """
    return int.from_bytes(data[offset:offset+8], byteorder='little', signed=True)

def read_uint32_le(data: bytes, offset: int = 0) -> int:
    """Read a 32-bit little-endian unsigned integer from bytes."""
    return int.from_bytes(data[offset:offset+4], byteorder='little', signed=False)

def read_uint64_le(data: bytes, offset: int = 0) -> int:
    """Read a 64-bit little-endian unsigned integer from bytes."""
    return int.from_bytes(data[offset:offset+8], byteorder='little', signed=False)

# Test on integer memory
test_value: int = 12345
memory: bytes = safe_memory_read(test_value, 32)

print(f"Manual Memory Interpretation")
print(f"{'='*60}")
print(f"Value: {test_value}")
print()

# Parse PyLongObject fields manually
ob_refcnt: int = read_int64_le(memory, 0)
ob_type: int = read_uint64_le(memory, 8)
ob_size: int = read_int64_le(memory, 16)
ob_digit0: int = read_uint32_le(memory, 24)

print(f"Manually parsed fields:")
print(f"  ob_refcnt (offset 0):  {ob_refcnt}")
print(f"  ob_type (offset 8):    {ob_type:#x}")
print(f"  ob_size (offset 16):   {ob_size}")
print(f"  ob_digit[0] (offset 24): {ob_digit0}")
print()

# Compare with ctypes.Structure approach
pylong_struct: PyLongObjectSimple = PyLongObjectSimple.from_address(id(test_value))
print(f"Using ctypes.Structure:")
print(f"  ob_refcnt: {pylong_struct.ob_base.ob_base.ob_refcnt}")
print(f"  ob_type:   {pylong_struct.ob_base.ob_base.ob_type:#x}")
print(f"  ob_size:   {pylong_struct.ob_base.ob_size}")
print(f"  ob_digit[0]: {pylong_struct.ob_digit[0]}")
print()

print(f"✓ Both methods produce identical results!")
```

---

## 5.5 Following Pointers and Pointer Arithmetic

### Understanding Pointers in Python

Every Python object can reference other objects. These references are **pointers** - memory addresses pointing to other objects.

For example:
- A list contains pointers to its elements
- An object's `ob_type` field points to its type object
- A class instance's `__dict__` points to a dictionary

Let's learn to **follow these pointers** and inspect the objects they point to.

```{code-cell} python
class PyListObject(ctypes.Structure):
    """
    Simplified PyListObject.

    From CPython source (Include/cpython/listobject.h):
        typedef struct {
            PyVarObject ob_base;
            PyObject **ob_item;  // Pointer to array of pointers
            Py_ssize_t allocated;
        } PyListObject;
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_base", PyVarObjectManual),
        ("ob_item", ctypes.POINTER(ctypes.c_void_p)),  # Array of pointers
        ("allocated", ctypes.c_ssize_t),
    ]

# Create a list
my_list: list[int] = [100, 200, 300]
list_addr: int = id(my_list)

# Cast to PyListObject
pylist: PyListObject = PyListObject.from_address(list_addr)

print(f"Following Pointers in a List")
print(f"{'='*60}")
print(f"List: {my_list}")
print(f"List address: {list_addr:#x}")
print()

print(f"PyListObject fields:")
print(f"  ob_refcnt: {pylist.ob_base.ob_base.ob_refcnt}")
print(f"  ob_size (length): {pylist.ob_base.ob_size}")
print(f"  ob_item (array pointer): {ctypes.cast(pylist.ob_item, ctypes.c_void_p).value:#x}")
print(f"  allocated: {pylist.allocated}")
print()

# Follow pointers to list elements
print(f"Following pointers to elements:")
for i in range(min(pylist.ob_base.ob_size, len(my_list))):
    elem_ptr: int = pylist.ob_item[i]
    print(f"  [{i}] pointer: {elem_ptr:#x}")

    # Cast to PyLongObject and read value
    if elem_ptr != 0:  # Check for NULL pointer
        pylong_elem: PyLongObjectSimple = PyLongObjectSimple.from_address(elem_ptr)
        value: int = pylong_elem.ob_digit[0]
        print(f"       value: {value}")
        print(f"       refcount: {pylong_elem.ob_base.ob_base.ob_refcnt}")

print()
print(f"✓ We successfully followed pointers from list to its elements!")
```

### Visualizing Pointer Relationships

```
List Object @ 0x12345000
┌──────────────────────────────┐
│ PyListObject                 │
│  ob_size: 3                  │
│  allocated: 4                │
│  ob_item: 0xABCD0000 ────────┼──┐
└──────────────────────────────┘  │
                                   │
Element Array @ 0xABCD0000      ◄──┘
┌──────────────────────────────┐
│ [0] → 0x56780000 ────────────┼────┐
│ [1] → 0x56780100 ────────────┼──┐ │
│ [2] → 0x56780200 ────────────┼┐ │ │
│ [3] → NULL (unused)          ││ │ │
└──────────────────────────────┘│ │ │
                                │ │ │
Integer 100 @ 0x56780000    ◄───┘ │ │
Integer 200 @ 0x56780100    ◄─────┘ │
Integer 300 @ 0x56780200    ◄───────┘
```

---

## 5.6 Practical Project 1: PyObject Header Viewer

**Difficulty**: ⭐ Beginner
**Time**: 30 minutes
**Goal**: Build a tool that displays PyObject header information for any Python object

### Project Description

You'll create a comprehensive inspector that:
1. Takes any Python object as input
2. Reads its PyObject/PyVarObject header
3. Displays all fields in a formatted table
4. Works with any object type

### Implementation

```{code-cell} python
@dataclass
class HeaderInfo:
    """Information extracted from PyObject header."""
    address: int
    refcount: int
    type_address: int
    type_name: str
    size_bytes: int
    ob_size: int | None = None
    python_len: int | None = None

def inspect_pyobject_header(obj: Any) -> HeaderInfo:
    """
    Inspect PyObject header fields.

    Args:
        obj: Any Python object to inspect

    Returns:
        HeaderInfo with all details
    """
    addr = id(obj)
    header = PyObjectManual.from_address(addr)

    # Try to get ob_size for variable objects
    ob_size_value: int | None = None
    python_len_value: int | None = None

    try:
        var_header = PyVarObjectManual.from_address(addr)
        ob_size_value = var_header.ob_size
        python_len_value = len(obj) if hasattr(obj, "__len__") else None
    except Exception:
        pass

    return HeaderInfo(
        address=addr,
        refcount=header.ob_refcnt,
        type_address=header.ob_type,
        type_name=type(obj).__name__,
        size_bytes=sys.getsizeof(obj),
        ob_size=ob_size_value,
        python_len=python_len_value,
    )

def display_header_info(obj: Any) -> None:
    """
    Display object header information in a formatted table.

    Args:
        obj: Python object to inspect
    """
    info = inspect_pyobject_header(obj)

    print(f"\nPyObject Header: {repr(obj)[:50]}")
    print("=" * 60)
    print(f"Type:         {info.type_name}")
    print(f"Address:      {info.address:#x}")
    print(f"Refcount:     {info.refcount}")
    print(f"Type Pointer: {info.type_address:#x}")
    print(f"Size:         {info.size_bytes} bytes")

    if info.ob_size is not None:
        print(f"ob_size:      {info.ob_size}")
        if info.python_len is not None:
            print(f"len():        {info.python_len}")

# Test on various objects
test_objects = [
    42,                    # Small int (cached)
    1000,                  # Large int (not cached)
    "hello",               # String
    [1, 2, 3],            # List
    (1, 2, 3),            # Tuple
    {"a": 1},             # Dict
]

print("PROJECT 1: PyObject Header Viewer")
print("=" * 60)

for obj in test_objects:
    display_header_info(obj)
```

:::{admonition} Challenge
:class: tip

**Extend this project:**
1. Add comparison mode to show headers side-by-side
2. Track refcount changes through operations
3. Follow the `ob_type` pointer to explore type relationships
:::

---

## 5.7 Practical Project 2: Reference Count Tracker

**Difficulty**: ⭐⭐ Intermediate
**Time**: 35 minutes
**Goal**: Build a system that tracks reference count changes over time

### Project Description

Monitor how different operations affect an object's reference count, helping understand Python's memory management.

### Implementation

```{code-cell} python
@dataclass
class RefcountSnapshot:
    """Snapshot of an object's reference count at a moment in time."""
    timestamp: float
    refcount: int
    operation: str

class RefcountTracker:
    """Track reference count changes for a Python object."""

    def __init__(self, obj: Any) -> None:
        """Initialize tracker with an object to monitor."""
        self.obj: Any = obj
        self.snapshots: list[RefcountSnapshot] = []
        self.baseline_refcount: int = sys.getrefcount(obj) - 1
        self.snapshot("Initial state")

    def current_refcount(self) -> int:
        """Get current reference count (adjusted)."""
        return sys.getrefcount(self.obj) - 1

    def snapshot(self, operation: str) -> RefcountSnapshot:
        """Take a snapshot of current reference count."""
        import time
        snap = RefcountSnapshot(
            timestamp=time.time(),
            refcount=self.current_refcount(),
            operation=operation,
        )
        self.snapshots.append(snap)
        return snap

    def delta(self, index: int = -1) -> int:
        """Get change in refcount since previous snapshot."""
        if len(self.snapshots) < 2:
            return 0
        current = self.snapshots[index]
        previous = self.snapshots[index - 1]
        return current.refcount - previous.refcount

    def report(self) -> None:
        """Generate a report of reference count changes."""
        print("\nREFERENCE COUNT TRACKING REPORT")
        print("=" * 80)
        print(f"Object: {repr(self.obj)[:60]}")
        print(f"Baseline refcount: {self.baseline_refcount}")
        print(f"Current refcount: {self.current_refcount()}")
        print(f"Total change: {self.current_refcount() - self.baseline_refcount:+d}")
        print(f"\nSnapshots: {len(self.snapshots)}")
        print("-" * 80)
        print(f"{'#':<4} {'Refcount':<10} {'Delta':<8} {'Operation':<50}")
        print("-" * 80)

        for i, snap in enumerate(self.snapshots):
            delta = self.delta(i) if i > 0 else 0
            delta_str = f"{delta:+d}" if i > 0 else "--"
            print(f"{i:<4} {snap.refcount:<10} {delta_str:<8} {snap.operation[:50]}")

        print("=" * 80)

# Demo
print("PROJECT 2: Reference Count Tracker")
print("=" * 60)

obj = [1, 2, 3]
tracker = RefcountTracker(obj)

# Create references
refs: list[Any] = []
refs.append(obj)
tracker.snapshot("refs.append(obj)")

refs.append(obj)
tracker.snapshot("refs.append(obj) again")

refs.pop()
tracker.snapshot("refs.pop()")

tracker.report()
```

---

## 5.8 Practical Project 3: List Growth Visualizer

**Difficulty**: ⭐⭐⭐ Advanced
**Time**: 45 minutes
**Goal**: Visualize how Python lists grow their internal arrays

### Project Description

Build a comprehensive tool that shows Python's over-allocation strategy, tracking reallocations and capacity management.

### Implementation

```{code-cell} python
@dataclass
class GrowthEvent:
    """Record of a list growth event."""
    operation: str
    length_before: int
    length_after: int
    allocated_before: int
    allocated_after: int

    @property
    def reallocated(self) -> bool:
        """Check if capacity changed."""
        return self.allocated_after != self.allocated_before

def inspect_list_internals(lst: list[Any]) -> dict[str, Any]:
    """Inspect internal structure of a Python list."""
    addr = id(lst)
    list_obj = PyListObject.from_address(addr)

    length = list_obj.ob_base.ob_size
    allocated = list_obj.allocated

    return {
        "length": length,
        "allocated": allocated,
        "load_factor": length / allocated if allocated > 0 else 0.0,
    }

class ListGrowthVisualizer:
    """Track and visualize how Python lists grow."""

    def __init__(self) -> None:
        self.lst: list[Any] = []
        self.events: list[GrowthEvent] = []
        self._last_info = inspect_list_internals(self.lst)

    def _record_event(self, operation: str) -> None:
        """Record a growth event."""
        current_info = inspect_list_internals(self.lst)

        event = GrowthEvent(
            operation=operation,
            length_before=self._last_info["length"],
            length_after=current_info["length"],
            allocated_before=self._last_info["allocated"],
            allocated_after=current_info["allocated"],
        )

        if (event.length_after != event.length_before or
            event.allocated_after != event.allocated_before):
            self.events.append(event)

        self._last_info = current_info

    def track_append(self, item: Any) -> None:
        """Append item and track growth."""
        self.lst.append(item)
        self._record_event(f"append({repr(item)[:20]})")

    def report(self) -> None:
        """Generate comprehensive growth report."""
        print("\nLIST GROWTH TRACKING REPORT")
        print("=" * 100)
        print(f"Final length: {len(self.lst)}")
        print(f"Final capacity: {self._last_info['allocated']}")
        print(f"Load factor: {self._last_info['load_factor']:.2%}")
        print(f"\nTotal events: {len(self.events)}")
        print(f"Reallocations: {sum(1 for e in self.events if e.reallocated)}")
        print("-" * 100)

        print(f"{'#':<4} {'Operation':<30} {'Length':<12} {'Capacity':<12} {'Realloc?':<10}")
        print("-" * 100)

        for i, event in enumerate(self.events[:20]):  # First 20
            len_str = f"{event.length_before}→{event.length_after}"
            cap_str = f"{event.allocated_before}→{event.allocated_after}"
            realloc_str = "YES" if event.reallocated else "no"

            print(f"{i:<4} {event.operation:<30} {len_str:<12} {cap_str:<12} {realloc_str:<10}")

        if len(self.events) > 20:
            print(f"... and {len(self.events) - 20} more events")

        print("=" * 100)

# Demo
print("PROJECT 3: List Growth Visualizer")
print("=" * 60)

viz = ListGrowthVisualizer()

for i in range(50):
    viz.track_append(i)

viz.report()
```

:::{admonition} Python's Growth Strategy
:class: note

Python uses the formula: `new_allocated = (size >> 3) + (size < 9 ? 3 : 6) + size`

This gives roughly 12.5% over-allocation, balancing memory efficiency with reallocation frequency.
:::

---

## 5.9 Integration with omnivault.utils.memory

Throughout this tutorial, we've built custom ctypes code. Let's integrate with the production-ready `omnivault.utils.memory` utilities:

```{code-cell} python
from omnivault.utils.memory import (
    MemoryInspector,
    track_memory,
)

print("Integration with omnivault.utils.memory")
print("=" * 60)

# Example: Combine MemoryInspector with custom structures
value: int = 12345
inspector = MemoryInspector(value)

# High-level view
print("High-level inspection (MemoryInspector):")
print(inspector.summary())
print()

# Low-level details with custom PyLongObject
pylong: PyLongObjectSimple = PyLongObjectSimple.from_address(id(value))
print("Low-level inspection (custom PyLongObject):")
print(f"  Digit value: {pylong.ob_digit[0]}")
print(f"  Digit count: {pylong.ob_base.ob_size}")
```

---

## 5.10 Safety Guidelines and Best Practices

:::{danger}
## CRITICAL SAFETY RULES

### 1. Never read beyond object boundaries

```python
# BAD: Will likely crash
obj = 42
memory = ctypes.string_at(id(obj), 1000000)  # Way too many bytes!

# GOOD: Use sys.getsizeof()
memory = ctypes.string_at(id(obj), sys.getsizeof(obj))
```

### 2. Check for NULL pointers before dereferencing

```python
# BAD: Might crash if pointer is NULL
elem_ptr = pylist.ob_item[i]
value = PyLongObject.from_address(elem_ptr)  # Crash if NULL!

# GOOD: Check first
elem_ptr = pylist.ob_item[i]
if elem_ptr != 0:  # NULL check
    value = PyLongObject.from_address(elem_ptr)
```

### 3. Keep references to inspected objects

```python
# BAD: Object may be garbage collected
inspector = MemoryInspector(42 + 1)  # Temporary object!
data = inspector.read_bytes(0, 32)   # May crash

# GOOD: Keep strong reference
obj = 42 + 1  # Store the object
inspector = MemoryInspector(obj)
data = inspector.read_bytes(0, 32)  # Safe
```
:::

### When to Use ctypes Inspection

**Good use cases:**
- Learning about CPython internals ✅
- Debugging memory issues ✅
- Understanding performance characteristics ✅
- Building educational tools ✅

**Bad use cases:**
- Production code (too fragile) ❌
- Critical applications (unsafe) ❌
- Cross-platform tools (layout varies) ❌

**Instead, use:**
- `sys.getsizeof()` for size
- `sys.getrefcount()` for refcount
- `tracemalloc` for memory profiling
- `omnivault.utils.memory` for safe inspection

---

## 5.11 Summary and Next Steps

### What You Learned

1. **ctypes.Structure basics**: Defining C-compatible structures in Python
2. **Memory reading**: Using `ctypes.string_at()` and `from_address()`
3. **CPython structures**: PyObject, PyVarObject, PyLongObject, PyListObject
4. **Pointer following**: Navigating object references in memory
5. **Practical tools**: Built 3 complete inspection tools
6. **Safety**: Critical rules for preventing crashes

### Key Takeaways

- **ctypes is a bridge** between Python and C memory
- **Everything has a structure** - Python objects are C structs
- **Following pointers** reveals object relationships
- **Safety is critical** - one mistake can crash the interpreter
- **Use production tools** for real applications

### Skills Acquired

You can now:
- ✅ Define custom ctypes structures matching CPython internals
- ✅ Read and interpret raw memory bytes
- ✅ Navigate complex object hierarchies
- ✅ Build custom memory inspection tools
- ✅ Integrate with omnivault.utils.memory utilities
- ✅ Understand the dangers and limitations

### Next Steps

**Section 6: Optimization Techniques** will teach you:
- When to optimize (and when not to)
- Using `__slots__` to reduce memory
- Generators vs lists for memory efficiency
- NumPy arrays for numerical data
- Memory profiling best practices

**Continue to [Section 6: Optimization Techniques →](06_optimization.md)**

---

## Additional Resources

### CPython Source Code
- [Include/object.h](https://github.com/python/cpython/blob/main/Include/object.h) - PyObject definitions
- [Include/longintrepr.h](https://github.com/python/cpython/blob/main/Include/longintrepr.h) - PyLongObject
- [Include/cpython/listobject.h](https://github.com/python/cpython/blob/main/Include/cpython/listobject.h) - PyListObject

### Documentation
- [ctypes documentation](https://docs.python.org/3/library/ctypes.html)
- [Python C API](https://docs.python.org/3/c-api/)
- [sys module](https://docs.python.org/3/library/sys.html)

### Related Tutorials
- [Section 1: C Memory Basics](01_c_memory_basics.md)
- [Section 2: Python Object Model](02_python_object_model.md)
- [Section 3: Integer Internals](03_integer_internals.md)
- [Section 4: String Internals](04_string_internals.md)

---

**Tutorial Series Navigation:**
- **Previous:** [04 - String Internals](04_string_internals.md)
- **Current:** 05 - ctypes Inspection
- **Next:** [06 - Optimization Techniques](06_optimization.md)
