from __future__ import annotations

from .combinatorial import combinations, permutations, subsets
from .n_queens import solve_n_queens
from .sudoku import solve_sudoku

__all__ = ["combinations", "permutations", "solve_n_queens", "solve_sudoku", "subsets"]
