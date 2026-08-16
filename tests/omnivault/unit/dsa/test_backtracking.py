from __future__ import annotations

import pytest

from omnivault.dsa.algorithms.backtracking import combinations, permutations, solve_n_queens, solve_sudoku, subsets
from omnivault.dsa.core.errors import EmptyContainer, InvalidConfiguration


class TestCombinatorial:
    @pytest.mark.unit
    def test_permutations_count(self) -> None:
        assert len(list(permutations([1, 2, 3]))) == 6

    @pytest.mark.unit
    def test_permutations_exact_set(self) -> None:
        result = {tuple(p) for p in permutations([1, 2, 3])}
        expected = {(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}
        assert result == expected

    @pytest.mark.unit
    def test_permutations_empty_yields_single_empty(self) -> None:
        assert list(permutations([])) == [[]]

    @pytest.mark.unit
    def test_combinations_count(self) -> None:
        assert len(list(combinations([1, 2, 3, 4], 2))) == 6

    @pytest.mark.unit
    def test_combinations_exact_set(self) -> None:
        result = {tuple(c) for c in combinations([1, 2, 3, 4], 2)}
        assert result == {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}

    @pytest.mark.unit
    def test_combinations_k_zero_yields_single_empty(self) -> None:
        assert list(combinations([1, 2, 3], 0)) == [[]]

    @pytest.mark.unit
    def test_combinations_empty_positive_k_raises(self) -> None:
        with pytest.raises(EmptyContainer):
            list(combinations([], 1))

    @pytest.mark.unit
    def test_subsets_count(self) -> None:
        assert len(list(subsets([1, 2, 3]))) == 8

    @pytest.mark.unit
    def test_subsets_exact_set(self) -> None:
        result = {tuple(s) for s in subsets([1, 2, 3])}
        assert result == {(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)}


class TestNQueens:
    @pytest.mark.unit
    def test_n_queens_n1(self) -> None:
        assert solve_n_queens(1) == [["Q"]]

    @pytest.mark.unit
    def test_n_queens_n2_no_solution(self) -> None:
        assert solve_n_queens(2) == []

    @pytest.mark.unit
    def test_n_queens_n3_no_solution(self) -> None:
        assert solve_n_queens(3) == []

    @pytest.mark.unit
    def test_n_queens_n4_two_solutions(self) -> None:
        solutions = solve_n_queens(4)
        assert len(solutions) == 2
        expected = {(".Q..", "...Q", "Q...", "..Q."), ("..Q.", "Q...", "...Q", ".Q..")}
        assert {tuple(s) for s in solutions} == expected

    @pytest.mark.unit
    def test_n_queens_n8_count(self) -> None:
        assert len(solve_n_queens(8)) == 92

    @pytest.mark.unit
    def test_n_queens_invalid(self) -> None:
        with pytest.raises(InvalidConfiguration):
            solve_n_queens(0)


class TestSudoku:
    @pytest.fixture
    def puzzle(self) -> list[list[str]]:
        return [
            ["5", "3", ".", ".", "7", ".", ".", ".", "."],
            ["6", ".", ".", "1", "9", "5", ".", ".", "."],
            [".", "9", "8", ".", ".", ".", ".", "6", "."],
            ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
            ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
            ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
            [".", "6", ".", ".", ".", ".", "2", "8", "."],
            [".", ".", ".", "4", "1", "9", ".", ".", "5"],
            [".", ".", ".", ".", "8", ".", ".", "7", "9"],
        ]

    @pytest.fixture
    def solution(self) -> list[list[str]]:
        return [
            ["5", "3", "4", "6", "7", "8", "9", "1", "2"],
            ["6", "7", "2", "1", "9", "5", "3", "4", "8"],
            ["1", "9", "8", "3", "4", "2", "5", "6", "7"],
            ["8", "5", "9", "7", "6", "1", "4", "2", "3"],
            ["4", "2", "6", "8", "5", "3", "7", "9", "1"],
            ["7", "1", "3", "9", "2", "4", "8", "5", "6"],
            ["9", "6", "1", "5", "3", "7", "2", "8", "4"],
            ["2", "8", "7", "4", "1", "9", "6", "3", "5"],
            ["3", "4", "5", "2", "8", "6", "1", "7", "9"],
        ]

    @pytest.mark.unit
    def test_solve_sudoku_in_place(self, puzzle: list[list[str]], solution: list[list[str]]) -> None:
        solve_sudoku(puzzle)
        assert puzzle == solution

    @pytest.mark.unit
    def test_solve_sudoku_wrong_shape_raises(self) -> None:
        with pytest.raises(InvalidConfiguration):
            solve_sudoku([["."] * 8 for _ in range(9)])
