from __future__ import annotations

import secrets

from ...core.errors import InvalidConfiguration

_DETERMINISTIC_WITNESSES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_DETERMINISTIC_BOUND: int = 3_317_044_064_679_887_385_961_981


def sieve_of_eratosthenes(limit: int) -> list[int]:
    if limit < 2:
        raise InvalidConfiguration(f"limit must be >= 2, got {limit}")
    composite = [False] * (limit + 1)
    composite[0] = composite[1] = True
    for i in range(2, int(limit**0.5) + 1):
        if composite[i]:
            continue
        for j in range(i * i, limit + 1, i):
            composite[j] = True
    return [i for i, c in enumerate(composite) if not c]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    bound = int(n**0.5)
    return all(n % candidate != 0 for candidate in range(3, bound + 1, 2))


def _decompose(n: int) -> tuple[int, int]:
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return d, s


def _composite_witness(witness: int, d: int, s: int, n: int) -> bool:
    x = pow(witness, d, n)
    if x in (1, n - 1):
        return False
    for _ in range(s - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return False
    return True


def miller_rabin(n: int, k: int = 10) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False

    d, s = _decompose(n)
    witnesses: tuple[int, ...]
    if n < _DETERMINISTIC_BOUND:
        witnesses = tuple(w for w in _DETERMINISTIC_WITNESSES if w < n)
    else:
        witnesses = tuple(2 + secrets.randbelow(n - 3) for _ in range(k))

    return not any(_composite_witness(w, d, s, n) for w in witnesses)
