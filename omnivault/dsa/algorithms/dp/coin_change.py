from __future__ import annotations

from collections.abc import Sequence


def coin_change_count(coins: Sequence[int], amount: int) -> int:
    if amount < 0:
        return 0
    ways: list[int] = [0] * (amount + 1)
    ways[0] = 1
    for coin in coins:
        for value in range(coin, amount + 1):
            ways[value] += ways[value - coin]
    return ways[amount]


def coin_change_min_coins(coins: Sequence[int], amount: int) -> int:
    if amount < 0:
        return -1
    sentinel: int = amount + 1
    table: list[int] = [sentinel] * (amount + 1)
    table[0] = 0
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value and table[value - coin] + 1 < table[value]:
                table[value] = table[value - coin] + 1
    return -1 if table[amount] == sentinel else table[amount]
