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

# Python Object Model: Everything is a PyObject

## Overview

In the previous section, we learned how C manages memory with simple types and structs. Now we'll see how Python uses C structs to build its "everything is an object" philosophy.

In this section, you'll learn:
- The PyObject structure that underlies all Python objects
- How reference counting works for automatic memory management
- The type system and metaclasses
- Object attributes and `__dict__`
- Memory optimization with `__slots__`

## The Fundamental Truth: Everything Is an Object

In Python, **everything** is an object:

```{code-cell} ipython3
from __future__ import annotations

import sys

# Numbers are objects
num: int = 42
print(f"42 is an object: {isinstance(num, object)}")
print(f"Type: {type(num)}")

# Even types are objects!
print(f"\nint is an object: {isinstance(int, object)}")
print(f"Type of int: {type(int)}")

# Functions are objects
def my_func() -> None:
    pass

print(f"\nFunction is an object: {isinstance(my_func, object)}")
print(f"Type: {type(my_func)}")
```

This is fundamentally different from C, where primitives like `int` are just patterns of bits.

## PyObject: The Base of Everything

Every Python object starts with a C struct called `PyObject`. Let's recreate it:

```{code-cell} ipython3
import ctypes
from typing import ClassVar

class PyObject(ctypes.Structure):
    """The base structure for all Python objects.

    This is the actual structure from CPython's Include/object.h (simplified).
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_refcnt", ctypes.c_ssize_t),  # Reference count
        ("ob_type", ctypes.c_void_p),     # Pointer to type object
    ]

# Size on 64-bit system:
print(f"PyObject size: {ctypes.sizeof(PyObject)} bytes")
print(f"  ob_refcnt: {ctypes.sizeof(ctypes.c_ssize_t)} bytes")
print(f"  ob_type:   {ctypes.sizeof(ctypes.c_void_p)} bytes")
```

### PyObject Memory Layout

```
PyObject (16 bytes on 64-bit):
┌─────────────────┐  ← Offset 0
│  ob_refcnt (8B) │  Reference count for garbage collection
├─────────────────┤  ← Offset 8
│  ob_type (8B)   │  Pointer to PyTypeObject (describes the type)
└─────────────────┘  ← End (16 bytes minimum overhead)
```

Every single Python object has at least these 16 bytes of overhead!

## Inspecting Real Python Objects

Let's use our memory inspector to see PyObject in real Python objects:

```{code-cell} ipython3
from omnivault.utils.memory.inspector import MemoryInspector

# Inspect a Python integer
num: int = 42
inspector = MemoryInspector(num)

# Get the PyObject header
header = inspector.get_header()
print(f"Object: {num}")
print(f"Size: {inspector.size} bytes")
print(f"Reference count: {header.ob_refcnt}")
print(f"Type pointer: {header.ob_type:#x}")
print(f"\nMemory dump (first 32 bytes):")
print(inspector.dump_hex(32))
```

### Understanding the Output

The hex dump shows:
- **First 8 bytes**: Reference count (how many variables point to this object)
- **Next 8 bytes**: Type pointer (points to the `int` type object)
- **Remaining bytes**: Type-specific data (the actual integer value)

## Reference Counting: Automatic Memory Management

Python uses **reference counting** to automatically free objects when they're no longer needed.

### How Reference Counting Works

```{code-cell} ipython3
# Create an object
a: list[int] = [1, 2, 3]
print(f"Initial refcount: {sys.getrefcount(a) - 1}")  # -1 for getrefcount's own ref

# Create another reference
b: list[int] = a
print(f"After b = a: {sys.getrefcount(a) - 1}")

# Create yet another reference
c: list[int] = a
print(f"After c = a: {sys.getrefcount(a) - 1}")

# Delete a reference
del b
print(f"After del b: {sys.getrefcount(a) - 1}")

# Delete all references
del c
del a
# Object is now freed (refcount reached 0)
```

### Reference Counting Rules

```{code-cell} ipython3
def demonstrate_refcount_changes(obj: object) -> None:
    """Show how different operations affect reference count."""
    print(f"Initial: {sys.getrefcount(obj) - 1}")

    # Assignment increments refcount
    ref1 = obj
    print(f"After assignment: {sys.getrefcount(obj) - 1}")

    # Container inclusion increments refcount
    container: list[object] = [obj]
    print(f"After adding to list: {sys.getrefcount(obj) - 1}")

    # Function argument increments refcount (temporarily)
    # (We're inside the function, so it's already incremented)

    # Cleanup
    del ref1
    del container
    print(f"After cleanup: {sys.getrefcount(obj) - 1}")

demonstrate_refcount_changes([1, 2, 3])
```

### Visualizing Reference Counting

```
Step 1: x = [1, 2, 3]
┌─────────┐
│ List    │
│ refcnt=1│  ← x points here
└─────────┘

Step 2: y = x
┌─────────┐
│ List    │
│ refcnt=2│  ← x and y point here
└─────────┘

Step 3: del x
┌─────────┐
│ List    │
│ refcnt=1│  ← only y points here
└─────────┘

Step 4: del y
┌─────────┐
│ List    │
│ refcnt=0│  → Object freed!
└─────────┘
```

## PyVarObject: Variable-Sized Objects

Some objects can vary in size (lists, strings, tuples). These use `PyVarObject`:

```{code-cell} ipython3
class PyVarObject(ctypes.Structure):
    """Structure for variable-sized objects.

    Adds ob_size to track the number of items.
    """
    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("ob_base", PyObject),        # Base PyObject fields
        ("ob_size", ctypes.c_ssize_t), # Number of items
    ]

print(f"PyVarObject size: {ctypes.sizeof(PyVarObject)} bytes")
print(f"Overhead: {ctypes.sizeof(PyVarObject)} bytes before any data!")
```

### PyVarObject Memory Layout

```
PyVarObject (24 bytes on 64-bit):
┌─────────────────┐  ← Offset 0
│  ob_refcnt (8B) │  Reference count
├─────────────────┤  ← Offset 8
│  ob_type (8B)   │  Pointer to type object
├─────────────────┤  ← Offset 16
│  ob_size (8B)   │  Number of items (for len())
└─────────────────┘  ← End (24 bytes overhead)
                     + Variable data follows
```

### Inspecting Variable-Sized Objects

```{code-cell} ipython3
# Lists use PyVarObject
my_list: list[int] = [1, 2, 3, 4, 5]
inspector = MemoryInspector(my_list)

var_header = inspector.get_var_header()
print(f"List: {my_list}")
print(f"ob_size (number of items): {var_header.ob_size}")
print(f"len(my_list): {len(my_list)}")
print(f"Total size: {inspector.size} bytes")
print(f"\nMemory dump:")
print(inspector.dump_hex(64))
```

## The Type System

The `ob_type` pointer points to a `PyTypeObject`, which describes the type:

```{code-cell} ipython3
# Every object knows its type
num: int = 42
string: str = "hello"
lst: list[int] = [1, 2, 3]

print(f"Type of {num}: {type(num)}")
print(f"Type of '{string}': {type(string)}")
print(f"Type of {lst}: {type(lst)}")

# Types themselves are objects!
print(f"\nType of int: {type(int)}")
print(f"Type of str: {type(str)}")
print(f"Type of list: {type(list)}")

# The ultimate metaclass
print(f"\nType of type: {type(type)}")  # type is its own type!
```

### Type Hierarchy

```
Everything is an object:
┌────────────────────────────────────┐
│           object                   │  (base of everything)
└────────────────────────────────────┘
         ↑           ↑           ↑
    ┌────┘           │           └────┐
    │                │                │
  ┌────┐          ┌────┐          ┌──────┐
  │int │          │str │          │list  │
  └────┘          └────┘          └──────┘
    ↑                ↑                ↑
    │                │                │
   42            "hello"          [1,2,3]

Type of types:
┌──────┐
│ type │ ← metaclass (type of all types)
└──────┘
    ↑
    │
  ┌─┴─┬─────┬──────┐
  │int│str  │list  │ (all inherit from type)
  └───┴─────┴──────┘
```

## Object Attributes and `__dict__`

Python objects store attributes in a dictionary called `__dict__`:

```{code-cell} ipython3
class Person:
    """Simple class to demonstrate __dict__."""
    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

person = Person("Alice", 30)
print(f"Object: {person}")
print(f"__dict__: {person.__dict__}")
print(f"Size of object: {sys.getsizeof(person)} bytes")
print(f"Size of __dict__: {sys.getsizeof(person.__dict__)} bytes")
```

### The Cost of `__dict__`

Every attribute access requires a dictionary lookup:

```{code-cell} ipython3
# Create many instances
persons: list[Person] = [Person(f"Person{i}", i) for i in range(1000)]

# Measure memory
from omnivault.utils.memory.profiler import track_memory

with track_memory() as snapshot:
    persons_with_dict: list[Person] = [Person(f"Person{i}", i) for i in range(10000)]

print(f"10,000 Person objects: {snapshot.peak_mb:.2f} MB")
print(f"Includes 10,000 __dict__ objects!")
```

## Memory Optimization: `__slots__`

You can eliminate `__dict__` by defining `__slots__`:

```{code-cell} ipython3
class PersonWithSlots:
    """Memory-optimized class using __slots__."""
    __slots__ = ("name", "age")

    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

# Compare sizes
regular_person = Person("Alice", 30)
slotted_person = PersonWithSlots("Alice", 30)

print(f"Regular Person: {sys.getsizeof(regular_person)} bytes")
print(f"Slotted Person: {sys.getsizeof(slotted_person)} bytes")
print(f"Savings: {sys.getsizeof(regular_person) - sys.getsizeof(slotted_person)} bytes")

# __dict__ doesn't exist with __slots__
try:
    print(slotted_person.__dict__)
except AttributeError as e:
    print(f"\nError: {e}")
```

### `__slots__` vs `__dict__` Comparison

```{code-cell} ipython3
# Memory comparison for many objects
with track_memory() as snapshot_dict:
    regular_persons: list[Person] = [Person(f"P{i}", i) for i in range(10000)]

with track_memory() as snapshot_slots:
    slotted_persons: list[PersonWithSlots] = [PersonWithSlots(f"P{i}", i) for i in range(10000)]

print(f"Regular (with __dict__): {snapshot_dict.peak_mb:.2f} MB")
print(f"Slotted (no __dict__):   {snapshot_slots.peak_mb:.2f} MB")
print(f"Savings: {(1 - snapshot_slots.peak_mb / snapshot_dict.peak_mb) * 100:.1f}%")
```

### When to Use `__slots__`

**Use `__slots__` when:**
- Creating many instances of the same class
- Memory is constrained
- Attribute set is fixed and known

**Don't use `__slots__` when:**
- You need dynamic attributes
- You're not creating many instances
- Premature optimization

## Object Identity vs Equality

Python distinguishes between **identity** (`is`) and **equality** (`==`):

```{code-cell} ipython3
# Identity checks memory address
a: list[int] = [1, 2, 3]
b: list[int] = a
c: list[int] = [1, 2, 3]

print(f"a is b: {a is b}")  # Same object
print(f"a is c: {a is c}")  # Different objects
print(f"a == c: {a == c}")  # But equal values

print(f"\nMemory addresses:")
print(f"id(a): {id(a):#x}")
print(f"id(b): {id(b):#x}")  # Same as a
print(f"id(c): {id(c):#x}")  # Different from a
```

### Identity Implementation

`id()` returns the memory address:

```{code-cell} ipython3
obj: int = 42
print(f"id(obj) = {id(obj):#x}")
print(f"Memory address = {id(obj):#x}")

# This is literally the pointer value!
inspector = MemoryInspector(obj)
print(f"Inspector address = {inspector.address:#x}")
```

## Circular References and Garbage Collection

Reference counting has a limitation: **circular references**

```{code-cell} ipython3
import gc

class Node:
    """Node in a linked structure."""
    def __init__(self, value: int) -> None:
        self.value = value
        self.next: Node | None = None

# Create a circular reference
a = Node(1)
b = Node(2)
a.next = b
b.next = a  # Circular!

print(f"a refcount: {sys.getrefcount(a) - 1}")
print(f"b refcount: {sys.getrefcount(b) - 1}")

# Delete the references
a_id: int = id(a)
b_id: int = id(b)
del a
del b

# Objects still exist (circular reference keeps refcount > 0)
# Garbage collector must detect and clean this up
collected = gc.collect()
print(f"\nObjects collected by GC: {collected}")
```

### Garbage Collection Architecture

Python uses two systems:
1. **Reference counting**: Fast, immediate cleanup (most objects)
2. **Cyclic GC**: Slower, periodic cleanup (circular references)

```
Reference Counting (immediate):
x = [1, 2, 3]
└→ refcount = 1

del x
└→ refcount = 0 → FREED IMMEDIATELY

Cyclic GC (periodic):
a.next = b
b.next = a
└→ refcount = 2 (even after del a, del b)
└→ GC scan finds cycle
└→ Both freed together
```

## Memory Layout Examples

Let's visualize complete object layouts:

```{code-cell} ipython3
from omnivault.utils.memory.inspector import compare_memory_layout

# Compare small vs large integer
print("INTEGER SIZE COMPARISON:")
print("=" * 120)
compare_result = compare_memory_layout(5, 10**100, 48)
print(compare_result)
```

## Key Takeaways

1. **PyObject is Universal**
   - Every Python object starts with 16 bytes (ob_refcnt + ob_type)
   - This overhead enables dynamic typing and automatic memory management

2. **Reference Counting Works**
   - Simple: count goes up/down with references
   - Fast: objects freed immediately when refcount = 0
   - Limited: can't handle circular references

3. **Variable-Sized Objects**
   - Add 8 more bytes for ob_size
   - Enables O(1) len() operation
   - Used by lists, tuples, strings, large integers

4. **Optimization Opportunities**
   - `__slots__` eliminates __dict__ overhead
   - Can save 50%+ memory for many objects
   - Trade flexibility for efficiency

5. **Everything Really Is an Object**
   - Types are objects
   - Functions are objects
   - Even `type` is an object (of type `type`!)

## Exercises

### Exercise 1: Reference Counting Investigation

Track reference counts through operations:

```{code-cell} ipython3
def track_refcounts() -> None:
    """Track reference counts through different operations."""
    obj: list[int] = [1, 2, 3]
    print(f"After creation: {sys.getrefcount(obj) - 1}")

    # TODO: Try these operations and predict refcount changes:
    # 1. ref2 = obj
    # 2. container = [obj, obj, obj]
    # 3. obj_tuple = (obj,)
    # 4. del ref2
    # 5. del container

track_refcounts()
```

### Exercise 2: `__slots__` Optimization

Create a memory-efficient Point class:

```{code-cell} ipython3
class PointWithDict:
    """Point with __dict__."""
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

class PointWithSlots:
    """Point with __slots__."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

# Compare memory for 10,000 points
with track_memory() as snapshot1:
    points_dict: list[PointWithDict] = [PointWithDict(i, i*2) for i in range(10000)]

with track_memory() as snapshot2:
    points_slots: list[PointWithSlots] = [PointWithSlots(i, i*2) for i in range(10000)]

print(f"With __dict__: {snapshot1.peak_mb:.2f} MB")
print(f"With __slots__: {snapshot2.peak_mb:.2f} MB")
print(f"Savings: {(1 - snapshot2.peak_mb / snapshot1.peak_mb) * 100:.1f}%")
```

### Exercise 3: Object Inspection

Inspect different object types:

```{code-cell} ipython3
objects_to_inspect: list[object] = [
    42,
    "hello",
    [1, 2, 3],
    (1, 2, 3),
    {"a": 1},
    lambda x: x,
]

for obj in objects_to_inspect:
    inspector = MemoryInspector(obj)
    header = inspector.get_header()
    print(f"{str(obj)[:30]:<30} | Size: {inspector.size:>4} | Refcount: {header.ob_refcnt}")
```

## Next Steps

Now that you understand Python's object model, let's dive into specific types!

**[Continue to Section 3: Integer Internals →](03_integer_internals.md)**

In the next section, we'll explore how Python implements arbitrary-precision integers using PyLongObject.

---

**Tutorial Series Navigation:**
- **Previous:** [01 - C Memory Basics](01_c_memory_basics.md)
- **Current:** 02 - Python Object Model
- **Next:** [03 - Integer Internals](03_integer_internals.md)
