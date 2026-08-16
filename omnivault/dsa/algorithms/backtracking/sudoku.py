from __future__ import annotations

from ...core.errors import InvalidConfiguration

_BOARD_SIZE = 9
_BOX_SIZE = 3
_EMPTY = "."
_DIGITS: tuple[str, ...] = tuple(str(d) for d in range(1, 10))


def solve_sudoku(board: list[list[str]]) -> None:
    if len(board) != _BOARD_SIZE or any(len(row) != _BOARD_SIZE for row in board):
        raise InvalidConfiguration("sudoku board must be 9x9")

    rows: list[set[str]] = [set() for _ in range(_BOARD_SIZE)]
    cols: list[set[str]] = [set() for _ in range(_BOARD_SIZE)]
    boxes: list[set[str]] = [set() for _ in range(_BOARD_SIZE)]

    for r in range(_BOARD_SIZE):
        for c in range(_BOARD_SIZE):
            value = board[r][c]
            if value == _EMPTY:
                continue
            rows[r].add(value)
            cols[c].add(value)
            boxes[(r // _BOX_SIZE) * _BOX_SIZE + c // _BOX_SIZE].add(value)

    def backtrack(r: int, c: int) -> bool:
        if r == _BOARD_SIZE:
            return True
        next_r, next_c = (r, c + 1) if c + 1 < _BOARD_SIZE else (r + 1, 0)
        if board[r][c] != _EMPTY:
            return backtrack(next_r, next_c)
        box = (r // _BOX_SIZE) * _BOX_SIZE + c // _BOX_SIZE
        for digit in _DIGITS:
            if digit in rows[r] or digit in cols[c] or digit in boxes[box]:
                continue
            board[r][c] = digit
            rows[r].add(digit)
            cols[c].add(digit)
            boxes[box].add(digit)
            if backtrack(next_r, next_c):
                return True
            board[r][c] = _EMPTY
            rows[r].remove(digit)
            cols[c].remove(digit)
            boxes[box].remove(digit)
        return False

    backtrack(0, 0)
