from __future__ import annotations

from ...core.errors import InvalidConfiguration


def gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def modular_inverse(a: int, m: int) -> int:
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise InvalidConfiguration(f"modular inverse does not exist: gcd({a}, {m}) = {g}")
    return x % m


def fast_exponentiation(base: int, exp: int, mod: int | None = None) -> int:
    if exp < 0:
        if mod is None:
            raise InvalidConfiguration("negative exponent requires a modulus")
        base = modular_inverse(base % mod, mod)
        exp = -exp
    result = 1
    current = base if mod is None else base % mod
    while exp > 0:
        if exp & 1:
            result = result * current if mod is None else (result * current) % mod
        current = current * current if mod is None else (current * current) % mod
        exp >>= 1
    return result
