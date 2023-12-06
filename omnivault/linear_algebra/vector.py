from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class Vector:
    """Base class for vectors."""

    origin: Tuple[float, ...]
    direction: Tuple[float, ...]
    color: Optional[str] = "black"
    label: Optional[str] = None

    def __len__(self) -> int:
        """Return the dimension of the vector."""
        return len(self.origin)

    @property
    def dimension(self) -> int:
        """Return the dimension of the vector."""
        return len(self)

    @property
    def magnitude(self) -> float:
        """Return the magnitude of the vector."""
        return self.l2_norm  # TODO: consider setter/getter for magnitude

    @property
    def l1_norm(self) -> float:
        """Return the L1 norm of the vector."""
        return float(np.linalg.norm(self.direction, ord=1))

    @property
    def l2_norm(self) -> float:
        """Return the L2 norm of the vector."""
        return float(np.linalg.norm(self.direction, ord=2))

    def __add__(self, other: Vector) -> Vector:
        """Add two vectors."""
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension.")
        return Vector(
            origin=tuple(np.array(self.origin) + np.array(other.origin)),
            direction=tuple(np.array(self.direction) + np.array(other.direction)),
        )


# Vector2D and Vector3D inherit from Vector and remember Vec is bound to Vector
@dataclass
class Vector2D(Vector):
    origin: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    direction: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


@dataclass
class Vector3D(Vector):
    origin: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    direction: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
