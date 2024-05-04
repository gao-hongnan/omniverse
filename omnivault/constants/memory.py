from __future__ import annotations

from enum import IntEnum


class MemoryUnit(IntEnum):
    BIT = 2**0
    BYTE = 2**3

    # Binary Units In Bytes
    KiB = 2**10  # 1024 BYTES -> 1024 * 8 = 8192 BITs
    MiB = 2**20  # 1024 * KiB
    GiB = 2**30  # 1024 * MiB
    TiB = 2**40  # 1024 * GiB
    PiB = 2**50  # 1024 * TiB

    # Decimal Units
    KB = 10**3  # 1000 * BYTE
    MB = 10**6  # 1000 * KB
    GB = 10**9  # 1000 * MB
    TB = 10**12  # 1000 * GB
    PB = 10**15  # 1000 * TB

    @staticmethod
    def convert(value_in_bytes: int, from_unit: MemoryUnit, to_unit: MemoryUnit) -> float:
        """Converts memory unit from one to another.

        Example
        -------
        >>> value = 20480000
        >>> from_unit = MemoryUnit.BYTE
        >>> to_unit = MemoryUnit.GiB
        >>> result = MemoryUnit.convert(value, from_unit, to_unit)
        0.152587890625 # 20480000 bytes is 0.152587890625 GiB
        """
        total_bytes = value_in_bytes * from_unit
        return total_bytes / to_unit
