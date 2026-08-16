from __future__ import annotations

from collections.abc import Sequence

from ...core.errors import InvalidConfiguration


def matrix_chain_order(dims: Sequence[int]) -> int:
    if len(dims) < 2:
        raise InvalidConfiguration("matrix_chain_order requires at least 2 dimensions")
    n: int = len(dims) - 1
    table: list[list[int]] = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j: int = i + length - 1
            table[i][j] = -1
            for k in range(i, j):
                cost: int = table[i][k] + table[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if table[i][j] == -1 or cost < table[i][j]:
                    table[i][j] = cost
    return table[0][n - 1]
