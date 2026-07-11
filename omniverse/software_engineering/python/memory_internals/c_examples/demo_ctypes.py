#!/usr/bin/env python3
"""Demonstration of Python ctypes integration with C code.

This script shows how to:
1. Load C shared libraries
2. Define C structures in Python
3. Call C functions from Python
4. Compare C and Python memory usage

Usage:
    First compile the C library:
        make

    Then run this script:
        python demo_ctypes.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import ClassVar

# Add parent directory to path for omnivault imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from omnivault.utils.memory import MemoryInspector


# Define C structures using ctypes
class Point(ctypes.Structure):
    """C struct Point { int32_t x; int32_t y; }"""

    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
    ]


class UnalignedStruct(ctypes.Structure):
    """C struct with padding issues."""

    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("a", ctypes.c_char),
        ("b", ctypes.c_int32),
        ("c", ctypes.c_char),
    ]


class OptimizedStruct(ctypes.Structure):
    """C struct with optimized field order."""

    _fields_: ClassVar[list[tuple[str, type[ctypes._CData]]]] = [
        ("b", ctypes.c_int32),
        ("a", ctypes.c_char),
        ("c", ctypes.c_char),
    ]


def demonstrate_c_structures() -> None:
    """Demonstrate C structure memory layout."""
    print("C Structure Memory Layout")
    print("=" * 60)

    # Create instances
    point = Point(10, 20)
    unaligned = UnalignedStruct(b"A", 42, b"B")
    optimized = OptimizedStruct(42, b"A", b"B")

    # Compare sizes
    print(f"Point:            {ctypes.sizeof(Point):>3} bytes")
    print(f"UnalignedStruct:  {ctypes.sizeof(UnalignedStruct):>3} bytes")
    print(f"OptimizedStruct:  {ctypes.sizeof(OptimizedStruct):>3} bytes")
    print()

    # Show field offsets
    print("Field Offsets:")
    print("-" * 60)
    print(f"Point.x:              offset={Point.x.offset}")
    print(f"Point.y:              offset={Point.y.offset}")
    print(f"UnalignedStruct.a:    offset={UnalignedStruct.a.offset}")
    print(f"UnalignedStruct.b:    offset={UnalignedStruct.b.offset}")
    print(f"UnalignedStruct.c:    offset={UnalignedStruct.c.offset}")
    print(f"OptimizedStruct.b:    offset={OptimizedStruct.b.offset}")
    print(f"OptimizedStruct.a:    offset={OptimizedStruct.a.offset}")
    print(f"OptimizedStruct.c:    offset={OptimizedStruct.c.offset}")
    print()


def compare_c_python_integers() -> None:
    """Compare C and Python integer memory usage."""
    print("C vs Python Integer Comparison")
    print("=" * 60)

    value: int = 42

    # C integer
    c_int: ctypes.c_int32 = ctypes.c_int32(value)
    c_size = ctypes.sizeof(c_int)

    # Python integer
    py_size = sys.getsizeof(value)

    print(f"Value: {value}")
    print(f"C int32:      {c_size:>3} bytes")
    print(f"Python int:   {py_size:>3} bytes")
    print(f"Overhead:     {py_size - c_size:>3} bytes ({(py_size / c_size):.1f}x)")
    print()

    # Use memory inspector
    inspector = MemoryInspector(value)
    print("Python Integer Memory Dump:")
    print(inspector.dump_hex(32))
    print()


def demonstrate_type_sizes() -> None:
    """Demonstrate sizes of various C types."""
    print("C Type Sizes")
    print("=" * 60)

    types: list[tuple[str, type[ctypes._CData]]] = [
        ("char", ctypes.c_char),
        ("int8", ctypes.c_int8),
        ("int16", ctypes.c_int16),
        ("int32", ctypes.c_int32),
        ("int64", ctypes.c_int64),
        ("float", ctypes.c_float),
        ("double", ctypes.c_double),
        ("pointer", ctypes.c_void_p),
    ]

    for name, c_type in types:
        size = ctypes.sizeof(c_type)
        print(f"{name:<15} {size:>3} bytes ({size * 8:>3} bits)")

    print()


def demonstrate_arrays() -> None:
    """Demonstrate C arrays vs Python lists."""
    print("C Arrays vs Python Lists")
    print("=" * 60)

    # C array
    IntArray10: type[ctypes.Array[ctypes.c_int32]] = ctypes.c_int32 * 10
    c_array: ctypes.Array[ctypes.c_int32] = IntArray10(*range(10))
    c_size = ctypes.sizeof(c_array)

    # Python list
    py_list: list[int] = list(range(10))
    py_size = sys.getsizeof(py_list) + sum(sys.getsizeof(i) for i in py_list)

    print("10 integers:")
    print(f"C array:      {c_size:>5} bytes (contiguous)")
    print(f"Python list:  {py_size:>5} bytes (list + objects)")
    print(f"Ratio:        {py_size / c_size:.1f}x more for Python")
    print()


def try_load_shared_library() -> None:
    """Try to load and use the C shared library if available."""
    lib_path = Path(__file__).parent / "basic_types.so"

    if not lib_path.exists():
        print("Note: C shared library not found. Run 'make' to build it.")
        print()
        return

    print("Loading C Shared Library")
    print("=" * 60)

    try:
        lib = ctypes.CDLL(str(lib_path))

        # Call print_type_sizes function
        print("Calling C function: print_type_sizes()")
        lib.print_type_sizes()

        # Call print_struct_sizes function
        print("Calling C function: print_struct_sizes()")
        lib.print_struct_sizes()

        # Call stack_example function
        stack_result = lib.stack_example()
        print(f"C stack_example() returned: {stack_result}")
        print()

    except OSError as e:
        print(f"Error loading library: {e}")
        print()


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Python ctypes Integration with C")
    print("=" * 60 + "\n")

    demonstrate_type_sizes()
    demonstrate_c_structures()
    compare_c_python_integers()
    demonstrate_arrays()
    try_load_shared_library()

    print("=" * 60)
    print("Tutorial: See tutorial sections for more details")
    print("  - Section 1: C Memory Basics")
    print("  - Section 2: Python Object Model")
    print("  - Section 3: Integer Internals")
    print("=" * 60)


if __name__ == "__main__":
    main()
