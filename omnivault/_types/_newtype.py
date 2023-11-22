"""
`NewType` is used to create distinct types for type checkers. It is helpful when
you want to differentiate between different kinds of data that are structurally
the same. For instance, if you want to distinguish between a regular integer and
a non-negative integer at the type-checking level, `NewType` would be
appropriate.

However, `NewType` does not enforce any runtime checks. It's essentially a cast
for the type checker, and at runtime, it's equivalent to the base type.
"""

from typing import NewType

# fmt: off
NonNegativeInt = NewType("NonNegativeInt", int) # Too strong typing lol need to cast everywhere, may not be worth it.
PositiveInt    = NewType("PositiveInt", int)
# fmt: on
