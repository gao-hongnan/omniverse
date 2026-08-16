from __future__ import annotations

import secrets

from ...core.errors import InvalidConfiguration
from .modular import gcd
from .primes import miller_rabin


def trial_division(n: int) -> list[int]:
    if n < 2:
        raise InvalidConfiguration(f"n must be >= 2, got {n}")
    factors: list[int] = []
    remaining = n
    while remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    candidate = 3
    while candidate * candidate <= remaining:
        while remaining % candidate == 0:
            factors.append(candidate)
            remaining //= candidate
        candidate += 2
    if remaining > 1:
        factors.append(remaining)
    return factors


def pollard_rho(n: int) -> int:
    if n < 2:
        raise InvalidConfiguration(f"n must be >= 2, got {n}")
    if n % 2 == 0:
        return 2
    if miller_rabin(n):
        return n
    while True:
        x = 2 + secrets.randbelow(max(n - 3, 1))
        y = x
        c = 1 + secrets.randbelow(max(n - 1, 1))
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = gcd(abs(x - y), n)
        if d != n:
            return d
