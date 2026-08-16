from __future__ import annotations

from ...core.errors import InvalidConfiguration


def solve_n_queens(n: int) -> list[list[str]]:
    if n < 1:
        raise InvalidConfiguration(f"n must be >= 1, got {n}")

    solutions: list[list[str]] = []
    queens: list[int] = [-1] * n
    cols: set[int] = set()
    diag1: set[int] = set()
    diag2: set[int] = set()

    def render() -> list[str]:
        board: list[str] = []
        for row in range(n):
            line = ["."] * n
            line[queens[row]] = "Q"
            board.append("".join(line))
        return board

    def backtrack(row: int) -> None:
        if row == n:
            solutions.append(render())
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            queens[row] = col
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return solutions
