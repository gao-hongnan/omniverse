from __future__ import annotations

from ...core.errors import InvalidConfiguration


def catalan(n: int) -> int:
    if n < 0:
        raise InvalidConfiguration(f"catalan requires n >= 0, got n={n}")
    dp: list[int] = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        total = 0
        for j in range(i):
            total += dp[j] * dp[i - 1 - j]
        dp[i] = total
    return dp[n]


def catalan_sequence(n: int) -> list[int]:
    if n < 0:
        raise InvalidConfiguration(f"catalan_sequence requires n >= 0, got n={n}")
    dp: list[int] = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        total = 0
        for j in range(i):
            total += dp[j] * dp[i - 1 - j]
        dp[i] = total
    return dp
