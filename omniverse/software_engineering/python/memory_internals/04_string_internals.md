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

# String Internals: PyUnicodeObject and Memory

## Overview

Strings are fundamental to programming, yet their memory implementation is complex. In this section, you'll discover:
- Why `sys.getsizeof("hello")` returns more bytes than you expect
- How Python stores Unicode strings efficiently (PEP 393)
- String interning and when to use it
- The performance implications of string operations

## The String Memory Mystery

Let's start with a simple question:

```{code-cell} ipython3
from __future__ import annotations

import sys
import ctypes

s: str = "hello"
print(f"String: '{s}'")
print(f"Length: {len(s)} characters")
print(f"Memory: {sys.getsizeof(s)} bytes")
print(f"\nExpected: ~5 bytes")
print(f"Actual: {sys.getsizeof(s)} bytes")
print(f"Overhead: {sys.getsizeof(s) - 5} bytes!")
```

**Why does a 5-character string need 54+ bytes?**

This tutorial will answer this question completely.

## PyUnicodeObject: Python's String Implementation

Since Python 3.3 (PEP 393), strings use a flexible representation that balances memory and Unicode support.

### The Three Representations

Python automatically chooses the most memory-efficient representation:

```{code-cell} ipython3
from omnivault.utils.memory import MemoryInspector

# ASCII string (1 byte per character)
ascii_str: str = "hello"
inspector_ascii = MemoryInspector(ascii_str)

# String with accents (still 1 byte per character, Latin-1)
latin_str: str = "café"
inspector_latin = MemoryInspector(latin_str)

# String with emoji (4 bytes per character)
emoji_str: str = "hi👋"
inspector_emoji = MemoryInspector(emoji_str)

print(f"{'String':<15} {'Chars':<8} {'Size (bytes)':<15} {'Bytes/char'}")
print("=" * 55)
print(f"{repr(ascii_str):<15} {len(ascii_str):<8} {inspector_ascii.size:<15} "
      f"{(inspector_ascii.size - 49) / max(1, len(ascii_str)):.1f}")
print(f"{repr(latin_str):<15} {len(latin_str):<8} {inspector_latin.size:<15} "
      f"{(inspector_latin.size - 49) / max(1, len(latin_str)):.1f}")
print(f"{repr(emoji_str):<15} {len(emoji_str):<8} {inspector_emoji.size:<15} "
      f"{(inspector_emoji.size - 49) / max(1, len(emoji_str)):.1f}")
```

### String Memory Layout

```
PyUnicodeObject (Compact ASCII):
┌────────────────────┐  ← Offset 0
│  ob_refcnt (8B)    │  Reference count
├────────────────────┤  ← Offset 8
│  ob_type (8B)      │  Pointer to 'str' type
├────────────────────┤  ← Offset 16
│  length (8B)       │  Number of code points
├────────────────────┤  ← Offset 24
│  hash (8B)         │  Cached hash (-1 if not computed)
├────────────────────┤  ← Offset 32
│  state (4B)        │  kind, compact, ascii, ready flags
├────────────────────┤  ← Offset 36
│  [padding] (4B)    │  Alignment
├────────────────────┤  ← Offset 40
│  wstr* (8B)        │  Legacy pointer (usually NULL)
├────────────────────┤  ← Offset 48
│  data[] (variable) │  Character data (inline)
└────────────────────┘

For "hello": 48 bytes header + 6 bytes data = 54 bytes total
```

### Character Width Optimization

```{code-cell} ipython3
# Python picks the smallest representation that fits ALL characters

def analyze_string(s: str) -> None:
    """Analyze string encoding and memory."""
    from omnivault.utils.memory import MemoryInspector

    inspector = MemoryInspector(s)
    max_codepoint = max(ord(c) for c in s) if s else 0

    # Determine encoding
    if max_codepoint <= 0x7F:
        encoding = "ASCII (1 byte/char)"
    elif max_codepoint <= 0xFF:
        encoding = "Latin-1 (1 byte/char)"
    elif max_codepoint <= 0xFFFF:
        encoding = "BMP Unicode (2 bytes/char)"
    else:
        encoding = "Full Unicode (4 bytes/char)"

    print(f"String: {repr(s)}")
    print(f"  Max codepoint: U+{max_codepoint:04X}")
    print(f"  Encoding: {encoding}")
    print(f"  Size: {inspector.size} bytes")
    print()

# Test various strings
analyze_string("hello")              # ASCII
analyze_string("café")               # Latin-1
analyze_string("你好")               # BMP Unicode (Chinese)
analyze_string("𝕳𝖊𝖑𝖑𝖔")          # Full Unicode (Math bold)
analyze_string("Hi👋🌍")           # Emoji (4 bytes)
```

### Three Representation Types

```
1. Compact ASCII (kind=1): 1 byte/char, only ASCII (U+0000 to U+007F)
   Example: "hello"
   ┌───┬───┬───┬───┬───┬───┐
   │ h │ e │ l │ l │ o │\0 │  6 bytes
   └───┴───┴───┴───┴───┴───┘

2. Compact Unicode (kind=2): 2 bytes/char, BMP (U+0000 to U+FFFF)
   Example: "你好" (U+4F60, U+597D)
   ┌──────┬──────┬──────┐
   │ 4F60 │ 597D │ 0000 │  6 bytes (3 × 2)
   └──────┴──────┴──────┘

3. Compact Unicode (kind=4): 4 bytes/char, Full Unicode
   Example: "Hi👋" (U+0048, U+0069, U+1F44B)
   ┌────────┬────────┬────────┬────────┐
   │   48   │   69   │ 1F44B  │  0000  │  16 bytes (4 × 4)
   └────────┴────────┴────────┴────────┘
```

## C Strings vs Python Strings

### C String: Null-Terminated Array

```{code-cell} ipython3
# C string: char* (null-terminated)
c_string: bytes = b"hello\x00"

print("C String:")
print(f"  Representation: char* (null-terminated)")
print(f"  Size: {len(c_string)} bytes")
print(f"  Content: {c_string}")
print(f"  No length tracking (must scan for \\0)")
print(f"  No Unicode support")
print(f"  Mutable (can modify in place)")
```

### Python String: Immutable Unicode Object

```{code-cell} ipython3
# Python string: str (immutable Unicode)
py_string: str = "hello"

print("Python String:")
print(f"  Representation: PyUnicodeObject")
print(f"  Size: {sys.getsizeof(py_string)} bytes")
print(f"  Length: O(1) cached in header")
print(f"  Full Unicode support (1M+ codepoints)")
print(f"  Immutable (cannot modify)")
print(f"  Hash cached for fast dict lookups")

# Try to modify (will fail)
try:
    py_string[0] = 'H'  # type: ignore
except TypeError as e:
    print(f"\n  Immutability: {e}")
```

### Comparison Table

| Feature | C (char*) | Python (str) |
|---------|-----------|--------------|
| Base overhead | 0 bytes | 49 bytes minimum |
| Null terminator | Required | Not needed |
| Length operation | O(n) - strlen() | O(1) - cached |
| Unicode support | No (ASCII only) | Full (UTF-8/16/32) |
| Mutability | Mutable | Immutable |
| Memory safety | Unsafe (buffer overflows) | Safe (bounds checked) |
| Concatenation | Fast (in-place) | Slow (creates new object) |

## String Interning

Python automatically interns some strings to save memory.

### Automatic Interning

```{code-cell} ipython3
# String literals are automatically interned if they look like identifiers
a: str = "hello"
b: str = "hello"
print(f"a is b: {a is b}")  # True
print(f"id(a): {id(a):#x}")
print(f"id(b): {id(b):#x}")  # Same!

# But runtime-created strings aren't automatically interned
x: str = "hello world"
y: str = "hello" + " " + "world"
print(f"\nx is y: {x is y}")  # False
print(f"id(x): {id(x):#x}")
print(f"id(y): {id(y):#x}")  # Different!
```

### Manual Interning with sys.intern()

```{code-cell} ipython3
import sys
from omnivault.utils.memory import track_memory

# Manually intern strings
s1: str = "hello" * 100
s2: str = "hello" * 100

# Without interning: different objects
print("Without interning:")
print(f"  s1 is s2: {s1 is s2}")  # False
print(f"  Size s1: {sys.getsizeof(s1)} bytes")
print(f"  Size s2: {sys.getsizeof(s2)} bytes")
print(f"  Total: {sys.getsizeof(s1) + sys.getsizeof(s2)} bytes")

# With interning: same object
s3: str = sys.intern("hello" * 100)
s4: str = sys.intern("hello" * 100)

print("\nWith interning:")
print(f"  s3 is s4: {s3 is s4}")  # True
print(f"  Size: {sys.getsizeof(s3)} bytes")
print(f"  Savings: {sys.getsizeof(s1) + sys.getsizeof(s2) - sys.getsizeof(s3)} bytes")
```

### When to Intern Strings

```{code-cell} ipython3
# ✅ GOOD: Intern repeated identifiers/keys
column_names = ["name", "age", "email", "status"]
interned_columns = [sys.intern(col) for col in column_names]

# These will be used repeatedly in dict lookups
data_records = [
    {sys.intern("name"): "Alice", sys.intern("age"): 30},
    {sys.intern("name"): "Bob", sys.intern("age"): 25},
]

# ❌ BAD: Don't intern unique data
user_comments = [
    "This is my first comment",
    "Here's another unique comment",
    "Every user writes different text",
]
# DON'T DO THIS:
# bad_interned = [sys.intern(comment) for comment in user_comments]
```

### Connection to stringintern.py

Your repository includes `stringintern.py` which demonstrates string interning patterns:

```{code-cell} ipython3
# Simplified version of scenarios from stringintern.py

def demo_string_interning_memory() -> None:
    """Demonstrate memory implications of string interning."""
    from omnivault.utils.memory import track_memory

    # Scenario 1: High cardinality, no interning (GOOD)
    print("High cardinality strings (unique values):")
    with track_memory() as snapshot:
        cache: dict[str, int] = {}
        for i in range(10000):
            key = f"user_{i}_session_{i}"  # Unique keys
            cache[key] = i
    print(f"  Memory without interning: {snapshot.peak_mb:.2f} MB")
    print(f"  This is correct - interning would waste memory!")

    # Scenario 2: Low cardinality, with interning (GOOD)
    print("\nLow cardinality strings (repeated values):")
    with track_memory() as snapshot:
        status_cache: dict[str, list[int]] = {}
        statuses = [sys.intern(s) for s in ["active", "inactive", "pending"]]
        for i in range(10000):
            status = statuses[i % 3]  # Only 3 unique values
            status_cache.setdefault(status, []).append(i)
    print(f"  Memory with interning: {snapshot.peak_mb:.2f} MB")
    print(f"  Savings from reusing 3 objects for 10K references!")

demo_string_interning_memory()
```

## String Encoding and Memory

### How Python Chooses Encoding

```{code-cell} ipython3
def demonstrate_encoding_choice() -> None:
    """Show how Python picks the most efficient encoding."""

    strings_to_test = [
        ("ASCII only", "hello"),
        ("Latin-1 (café)", "café"),
        ("BMP Unicode", "你好世界"),
        ("Mixed: ASCII + Emoji", "Hello 👋"),
        ("Math symbols", "𝕳𝖊𝖑𝖑𝖔"),
    ]

    print(f"{'Description':<25} {'String':<20} {'Max CP':<10} {'Size':<8} {'Bytes/char'}")
    print("=" * 75)

    for desc, s in strings_to_test:
        max_cp = max(ord(c) for c in s)
        size = sys.getsizeof(s)
        overhead = 49  # Approximate header size
        bytes_per_char = (size - overhead) / len(s)

        print(f"{desc:<25} {s:<20} U+{max_cp:04X}     {size:<8} {bytes_per_char:.1f}")

demonstrate_encoding_choice()
```

### UTF-8 vs Python's Internal Encoding

```{code-cell} ipython3
# UTF-8: Variable-width encoding (1-4 bytes per character)
s: str = "Hello 你好 👋"

# UTF-8 encoding (for transmission/storage)
utf8_bytes = s.encode('utf-8')
print(f"String: {repr(s)}")
print(f"UTF-8 bytes: {len(utf8_bytes)} bytes")
print(f"Python object: {sys.getsizeof(s)} bytes")

# Python stores internally based on max codepoint
max_cp = max(ord(c) for c in s)
print(f"\nMax codepoint: U+{max_cp:04X}")
print(f"Python uses: 4 bytes/char (emoji present)")
print(f"Character data: {4 * len(s)} bytes")
print(f"Header: ~49 bytes")
print(f"Total: ~{49 + 4 * len(s)} bytes")
```

## String Concatenation Performance

### Why `s += "x"` is Slow

```{code-cell} ipython3
import timeit
from omnivault.utils.memory import track_memory

def concat_with_plus(n: int) -> str:
    """BAD: Concatenate using += (creates n new strings!)."""
    result: str = ""
    for i in range(n):
        result += "x"  # Creates new string each time
    return result

def concat_with_join(n: int) -> str:
    """GOOD: Concatenate using join (single allocation)."""
    parts: list[str] = []
    for i in range(n):
        parts.append("x")
    return "".join(parts)

# Benchmark
N = 10000
time_plus = timeit.timeit(lambda: concat_with_plus(N), number=10)
time_join = timeit.timeit(lambda: concat_with_join(N), number=10)

print(f"Concatenating {N} strings:")
print(f"  Using +=:   {time_plus:.4f} seconds")
print(f"  Using join: {time_join:.4f} seconds")
print(f"  Speedup: {time_plus/time_join:.1f}x faster with join!")

# Memory profiling
with track_memory() as s1:
    concat_with_plus(5000)

with track_memory() as s2:
    concat_with_join(5000)

print(f"\nMemory (5000 concatenations):")
print(f"  Using +=:   {s1.peak_mb:.2f} MB")
print(f"  Using join: {s2.peak_mb:.2f} MB")
```

### Best Practices for String Building

```{code-cell} ipython3
from typing import List

# ❌ BAD: Repeated concatenation
def bad_string_building(items: List[str]) -> str:
    result = ""
    for item in items:
        result += item + ", "  # Creates new string each iteration
    return result

# ✅ GOOD: Use join
def good_string_building(items: List[str]) -> str:
    return ", ".join(items)

# ✅ ALSO GOOD: F-strings for few items
def format_message(name: str, age: int, email: str) -> str:
    return f"Name: {name}, Age: {age}, Email: {email}"

# ✅ GOOD: StringIO for complex building
from io import StringIO

def complex_string_building(data: List[dict[str, Any]]) -> str:
    buffer = StringIO()
    for item in data:
        buffer.write(f"ID: {item['id']}, ")
        buffer.write(f"Value: {item['value']}\n")
    return buffer.getvalue()

# Benchmark
test_items = [f"item_{i}" for i in range(1000)]

time_bad = timeit.timeit(lambda: bad_string_building(test_items), number=100)
time_good = timeit.timeit(lambda: good_string_building(test_items), number=100)

print(f"String building with 1000 items:")
print(f"  Bad (+=):   {time_bad:.4f}s")
print(f"  Good (join): {time_good:.4f}s")
print(f"  Speedup: {time_bad/time_good:.1f}x")
```

## Key Takeaways

1. **PyUnicodeObject overhead is significant**
   - Minimum 49 bytes header before any characters
   - Worth it for: Unicode support, O(1) length, safety

2. **Python auto-optimizes encoding**
   - ASCII: 1 byte/char
   - BMP Unicode: 2 bytes/char
   - Full Unicode: 4 bytes/char
   - Chosen based on maximum codepoint

3. **String interning saves memory**
   - Automatic for identifiers
   - Use `sys.intern()` for repeated strings
   - Don't intern unique data

4. **String concatenation matters**
   - `+=` is O(n²) - avoid in loops
   - `join()` is O(n) - use for many parts
   - F-strings are fine for few parts

5. **Immutability is key**
   - Every modification creates a new string
   - Design with this in mind
   - Use StringBuilder patterns when needed

## Exercises

### Exercise 1: String Size Detective

```{code-cell} ipython3
# Find the size jump points for different Unicode ranges
def exercise_unicode_boundaries() -> None:
    """Explore when Python switches encoding."""

    # TODO: Create strings with max codepoints at boundaries:
    # - U+007F (last ASCII)
    # - U+0080 (first Latin-1)
    # - U+00FF (last Latin-1)
    # - U+0100 (first BMP)
    # - U+FFFF (last BMP)
    # - U+10000 (first Full Unicode)

    # TODO: Measure size for each
    # TODO: Calculate bytes per character
    # TODO: Identify encoding used

    pass

exercise_unicode_boundaries()
```

### Exercise 2: Interning Benchmark

```{code-cell} ipython3
# Benchmark dictionary lookup performance with interning
def exercise_interning_performance() -> None:
    """Measure the speed benefit of interning."""

    # TODO: Create a dict with 1000 string keys
    # TODO: Perform 100,000 lookups using non-interned strings
    # TODO: Repeat with interned keys
    # TODO: Compare lookup times
    # Hint: Use timeit.timeit()

    pass

exercise_interning_performance()
```

### Exercise 3: Concatenation Profiler

```{code-cell} ipython3
# Profile different concatenation methods
def exercise_concat_comparison() -> None:
    """Compare memory and time for different methods."""

    # TODO: Implement 4 versions:
    # 1. result += "x"
    # 2. "".join(parts)
    # 3. f-string in comprehension
    # 4. StringIO

    # TODO: Profile each with track_memory()
    # TODO: Time each with timeit
    # TODO: Create comparison table

    pass

exercise_concat_comparison()
```

## Next Steps

Now you understand string internals! Continue to:

**[Continue to Section 5: ctypes Inspection →](05_ctypes_inspection.ipynb)**

Learn hands-on techniques for inspecting Python objects at the memory level.

---

**Tutorial Series Navigation:**
- **Previous:** [03 - Integer Internals](03_integer_internals.md)
- **Current:** 04 - String Internals
- **Next:** [05 - ctypes Inspection](05_ctypes_inspection.ipynb)
