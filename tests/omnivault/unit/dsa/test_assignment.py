from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from omnivault.dsa.graphs.assignment import (
    assignment_problem_max,
    assignment_problem_min,
    hungarian_algorithm,
    maximum_bipartite_matching,
    stable_marriage,
)

if TYPE_CHECKING:
    pass


class TestHungarianAlgorithm:
    @pytest.mark.unit
    def test_hungarian_algorithm_basic(self) -> None:
        cost_matrix = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]

        result = hungarian_algorithm(cost_matrix)

        assert result.total_cost == 5
        assert len(result.assignment) == 3

        assigned_rows = {assignment[0] for assignment in result.assignment}
        assigned_cols = {assignment[1] for assignment in result.assignment}
        assert len(assigned_rows) == 3
        assert len(assigned_cols) == 3

    @pytest.mark.unit
    def test_hungarian_algorithm_rectangular(self) -> None:
        cost_matrix = [[9, 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8]]

        result = hungarian_algorithm(cost_matrix)

        assert len(result.assignment) == 3
        assert result.total_cost == 9

    @pytest.mark.unit
    def test_hungarian_algorithm_empty_matrix(self) -> None:
        result = hungarian_algorithm([])

        assert result.assignment == []
        assert result.total_cost == 0

    @pytest.mark.unit
    def test_assignment_problem_min_wrapper(self) -> None:
        cost_matrix = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]

        result = assignment_problem_min(cost_matrix)
        assert result.total_cost == 5

    @pytest.mark.unit
    def test_assignment_problem_max(self) -> None:
        profit_matrix = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]

        result = assignment_problem_max(profit_matrix)
        assert result.total_cost == 11

    @pytest.mark.unit
    def test_hungarian_algorithm_large_matrix(self) -> None:
        cost_matrix = [
            [7, 53, 183, 439, 863],
            [497, 383, 563, 79, 973],
            [287, 63, 343, 169, 583],
            [627, 343, 773, 959, 943],
            [767, 473, 103, 699, 303],
        ]

        result = hungarian_algorithm(cost_matrix)

        assert len(result.assignment) == 5
        assert result.total_cost > 0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("matrix", "expected_cost"),
        [
            ([[1, 2], [3, 4]], 5),
            ([[1]], 1),
            ([[5, 9, 1], [10, 3, 2], [8, 7, 4]], 12),
        ],
    )
    def test_hungarian_algorithm_parametrized(self, matrix: list[list[int]], expected_cost: int) -> None:
        result = hungarian_algorithm(matrix)
        assert result.total_cost == expected_cost


class TestBipartiteMatching:
    @pytest.mark.unit
    def test_maximum_bipartite_matching_basic(self) -> None:
        edges = [(0, 0), (0, 1), (1, 1), (2, 0), (2, 2)]
        left_size = 3
        right_size = 3

        matching = maximum_bipartite_matching(edges, left_size, right_size)

        assert len(matching) == 3
        left_matched = {m[0] for m in matching}
        right_matched = {m[1] for m in matching}
        assert len(left_matched) == 3
        assert len(right_matched) == 3

    @pytest.mark.unit
    def test_maximum_bipartite_matching_no_perfect_matching(self) -> None:
        edges = [(0, 0), (1, 0)]
        left_size = 3
        right_size = 2

        matching = maximum_bipartite_matching(edges, left_size, right_size)

        assert len(matching) <= min(left_size, right_size)

    @pytest.mark.unit
    def test_maximum_bipartite_matching_empty_edges(self) -> None:
        edges: list[tuple[int, int]] = []
        left_size = 2
        right_size = 2

        matching = maximum_bipartite_matching(edges, left_size, right_size)

        assert len(matching) == 0

    @pytest.mark.unit
    def test_maximum_bipartite_matching_single_edge(self) -> None:
        edges = [(0, 1)]
        left_size = 1
        right_size = 2

        matching = maximum_bipartite_matching(edges, left_size, right_size)

        assert len(matching) == 1
        assert matching[0] == (0, 1)


class TestStableMarriage:
    @pytest.mark.unit
    def test_stable_marriage_basic(self) -> None:
        men_preferences = [
            [0, 1, 2],  # Man 0 prefers Woman 0, then 1, then 2
            [1, 0, 2],  # Man 1 prefers Woman 1, then 0, then 2
            [0, 1, 2],  # Man 2 prefers Woman 0, then 1, then 2
        ]

        women_preferences = [
            [2, 1, 0],  # Woman 0 prefers Man 2, then 1, then 0
            [0, 1, 2],  # Woman 1 prefers Man 0, then 1, then 2
            [0, 1, 2],  # Woman 2 prefers Man 0, then 1, then 2
        ]

        result = stable_marriage(men_preferences, women_preferences)

        assert len(result) == 3
        assert all(partner != -1 for partner in result)

        assigned_women = set(result)
        assert len(assigned_women) == 3

    @pytest.mark.unit
    def test_stable_marriage_validates_stability(self) -> None:
        men_preferences = [[0, 1], [1, 0]]

        women_preferences = [[0, 1], [1, 0]]

        result = stable_marriage(men_preferences, women_preferences)

        man0_partner = result[0]
        man1_partner = result[1]

        assert man0_partner == 0
        assert man1_partner == 1

    @pytest.mark.unit
    def test_stable_marriage_single_pair(self) -> None:
        men_preferences = [[0]]
        women_preferences = [[0]]

        result = stable_marriage(men_preferences, women_preferences)

        assert result == [0]

    @pytest.mark.unit
    def test_stable_marriage_unequal_sizes_raises_error(self) -> None:
        men_preferences = [[0, 1], [1, 0]]
        women_preferences = [[0]]

        with pytest.raises(ValueError, match="Number of men and women must be equal"):
            stable_marriage(men_preferences, women_preferences)

    @pytest.mark.unit
    def test_stable_marriage_incomplete_preferences_raises_error(self) -> None:
        men_preferences = [[0], [1, 0]]
        women_preferences = [[0, 1], [1, 0]]

        with pytest.raises(ValueError, match="All preference lists must be complete"):
            stable_marriage(men_preferences, women_preferences)

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_stable_marriage_different_sizes(self, n: int) -> None:
        men_preferences = [list(range(n)) for _ in range(n)]
        women_preferences = [list(range(n)) for _ in range(n)]

        result = stable_marriage(men_preferences, women_preferences)

        assert len(result) == n
        assert set(result) == set(range(n))


class TestAssignmentAlgorithmEdgeCases:
    @pytest.mark.unit
    def test_hungarian_non_square_more_rows(self) -> None:
        cost_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

        result = hungarian_algorithm(cost_matrix)

        assert len(result.assignment) == 3

    @pytest.mark.unit
    def test_hungarian_non_square_more_cols(self) -> None:
        cost_matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

        result = hungarian_algorithm(cost_matrix)

        assert len(result.assignment) == 3

    @pytest.mark.unit
    def test_hungarian_with_zero_costs(self) -> None:
        cost_matrix = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]

        result = hungarian_algorithm(cost_matrix)

        assert result.total_cost == 0

    @pytest.mark.unit
    def test_hungarian_with_identical_costs(self) -> None:
        cost_matrix = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]

        result = hungarian_algorithm(cost_matrix)

        assert result.total_cost == 15
        assert len(result.assignment) == 3
