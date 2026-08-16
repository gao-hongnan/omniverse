from __future__ import annotations

from ..core.errors import RectangularityViolation


class HungarianResult:
    def __init__(self, assignment: list[tuple[int, int]], total_cost: float) -> None:
        self.assignment = assignment
        self.total_cost: float = total_cost

    def __repr__(self) -> str:
        return f"HungarianResult(assignment={self.assignment}, total_cost={self.total_cost})"


def hungarian_algorithm[NumericT: (int, float)](cost_matrix: list[list[NumericT]]) -> HungarianResult:
    if not cost_matrix or not cost_matrix[0]:
        return HungarianResult([], 0.0)

    n_rows = len(cost_matrix)
    n_cols = len(cost_matrix[0])

    if not all(len(row) == n_cols for row in cost_matrix):
        raise RectangularityViolation("Cost matrix must be rectangular")

    n = max(n_rows, n_cols)

    matrix: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i < n_rows and j < n_cols:
                row.append(float(cost_matrix[i][j]))
            else:
                row.append(0.0)
        matrix.append(row)

    assignment = _solve_assignment(matrix, n)

    filtered_assignment = [(i, j) for i, j in assignment if i < n_rows and j < n_cols]

    total_cost = sum(float(cost_matrix[i][j]) for i, j in filtered_assignment)

    return HungarianResult(filtered_assignment, total_cost)


def _solve_assignment(matrix: list[list[float]], n: int) -> list[tuple[int, int]]:
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return [(p[j] - 1, j - 1) for j in range(1, n + 1) if p[j] != 0]


def assignment_problem_min[NumericT: (int, float)](cost_matrix: list[list[NumericT]]) -> HungarianResult:
    return hungarian_algorithm(cost_matrix)


def assignment_problem_max[NumericT: (int, float)](profit_matrix: list[list[NumericT]]) -> HungarianResult:
    if not profit_matrix or not profit_matrix[0]:
        return HungarianResult([], 0.0)

    max_val = max(max(row) for row in profit_matrix)
    cost_matrix = [
        [max_val - profit_matrix[i][j] for j in range(len(profit_matrix[i]))] for i in range(len(profit_matrix))
    ]

    result = hungarian_algorithm(cost_matrix)

    original_total = float(sum(profit_matrix[i][j] for i, j in result.assignment))

    return HungarianResult(result.assignment, original_total)


class BipartiteMatching:
    def __init__(self, left_size: int, right_size: int) -> None:
        self.left_size = left_size
        self.right_size = right_size
        self.graph: list[list[int]] = [[] for _ in range(left_size)]
        self.match_left: list[int] = [-1] * left_size
        self.match_right: list[int] = [-1] * right_size

    def add_edge(self, left: int, right: int) -> None:
        if 0 <= left < self.left_size and 0 <= right < self.right_size:
            self.graph[left].append(right)

    def maximum_matching(self) -> list[tuple[int, int]]:
        matching = 0

        for left in range(self.left_size):
            visited = [False] * self.right_size
            if self._dfs(left, visited):
                matching += 1

        result: list[tuple[int, int]] = [
            (left, self.match_left[left]) for left in range(self.left_size) if self.match_left[left] != -1
        ]

        return result

    def _dfs(self, left: int, visited: list[bool]) -> bool:
        for right in self.graph[left]:
            if visited[right]:
                continue

            visited[right] = True

            if self.match_right[right] == -1 or self._dfs(self.match_right[right], visited):
                self.match_left[left] = right
                self.match_right[right] = left
                return True

        return False


def maximum_bipartite_matching(edges: list[tuple[int, int]], left_size: int, right_size: int) -> list[tuple[int, int]]:
    matcher = BipartiteMatching(left_size, right_size)

    for left, right in edges:
        matcher.add_edge(left, right)

    return matcher.maximum_matching()


def stable_marriage(men_preferences: list[list[int]], women_preferences: list[list[int]]) -> list[int]:
    n = len(men_preferences)

    if len(women_preferences) != n:
        raise RectangularityViolation("Number of men and women must be equal")

    if any(len(prefs) != n for prefs in men_preferences + women_preferences):
        raise RectangularityViolation("All preference lists must be complete")

    woman_ranking = [{woman: rank for rank, woman in enumerate(prefs)} for prefs in women_preferences]

    free_men = list(range(n))
    man_next_proposal = [0] * n
    woman_partner = [-1] * n
    man_partner = [-1] * n

    while free_men:
        man = free_men.pop(0)
        woman = men_preferences[man][man_next_proposal[man]]
        man_next_proposal[man] += 1

        if woman_partner[woman] == -1:
            woman_partner[woman] = man
            man_partner[man] = woman
        else:
            current_partner = woman_partner[woman]

            if woman_ranking[woman][man] < woman_ranking[woman][current_partner]:
                woman_partner[woman] = man
                man_partner[man] = woman
                man_partner[current_partner] = -1
                free_men.append(current_partner)
            else:
                free_men.append(man)

    return list(man_partner)
