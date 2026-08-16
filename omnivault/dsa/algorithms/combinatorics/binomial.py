from __future__ import annotations

from ...core.errors import InvalidConfiguration


def binomial(n: int, k: int) -> int:
    if n < 0 or k < 0:
        raise InvalidConfiguration(f"binomial requires n >= 0 and k >= 0, got n={n}, k={k}")
    if k > n:
        return 0
    k = min(k, n - k)
    row: list[int] = [1]
    for i in range(1, k + 1):
        row.append(row[-1] * (n - i + 1) // i)
    return row[-1]


def permutations_count(n: int, k: int) -> int:
    if n < 0 or k < 0:
        raise InvalidConfiguration(f"permutations_count requires n >= 0 and k >= 0, got n={n}, k={k}")
    if k > n:
        return 0
    result = 1
    for i in range(n, n - k, -1):
        result *= i
    return result


def pascals_triangle(rows: int) -> list[list[int]]:
    if rows < 0:
        raise InvalidConfiguration(f"pascals_triangle requires rows >= 0, got rows={rows}")
    triangle: list[list[int]] = []
    for r in range(rows):
        row: list[int] = [1] * (r + 1)
        for c in range(1, r):
            row[c] = triangle[r - 1][c - 1] + triangle[r - 1][c]
        triangle.append(row)
    return triangle
