### The N-Queens Problem (CSP)

Let's consider a classic example of a CSP: the N-Queens problem. In this
problem, you are asked to place $N$ chess queens on an $N \times N$ chessboard
so that no two queens threaten each other. In other words, no two queens can be
placed in the same row, column, or diagonal.

Let's define this as a CSP:

#### Variables

-   $\mathcal{V} = \{V_1, V_2, \ldots, V_N\}$, where each variable $V_i$
    represents the column position of the queen in the $i$-th row.

#### Domains

-   $\mathcal{D} = \{D_1, D_2, \ldots, D_N\}$, where each domain
    $D_i = \{1, 2, \ldots, N\}$ represents the possible column positions for the
    $i$-th queen.

#### Constraints

-   $\mathcal{C}$ contains constraints ensuring that no two queens are in the
    same column or on the same diagonal.
    -   $\forall i, j \in \{1, 2, \ldots, N\}, i \neq j: V_i \neq V_j$
        (Different columns)
    -   $\forall i, j \in \{1, 2, \ldots, N\}, i \neq j: |V_i - V_j| \neq |i - j|$
        (Different diagonals)

#### CSP Definition

The N-Queens problem can be defined as the CSP
$(\mathcal{V}, \mathcal{D}, \mathcal{C})$.

#### Solution

A solution to this CSP is an assignment of values to the variables in
$\mathcal{V}$ such that all constraints in $\mathcal{C}$ are satisfied. Each
solution represents a valid way to place $N$ queens on an $N \times N$
chessboard so that no two queens threaten each other.

#### Example (4-Queens Problem)

-   $\mathcal{V} = \{V_1, V_2, V_3, V_4\}$
-   $\mathcal{D} = \{\{1, 2, 3, 4\}, \{1, 2, 3, 4\}, \{1, 2, 3, 4\}, \{1, 2, 3, 4\}\}$
-   $\mathcal{C}$: As defined above

A solution might be $V_1 = 2, V_2 = 4, V_3 = 1, V_4 = 3$.

This CSP example aligns with the abstract definitions and showcases how the
concepts of variables, domains, and constraints come together to define a
specific problem. The N-Queens problem is also a classic example where
backtracking is commonly used to find solutions.

#### Implementation

TO REVIEW LEETCODE's solution.

Here's a Python code snippet for solving the N-Queens problem using
backtracking, complete with type hints. The function `solve_n_queens` takes the
size of the board, `n`, and returns a list of solutions, where each solution is
represented as a list of strings.

```python
from typing import List, Tuple

def solve_n_queens(n: int) -> List[List[str]]:
    def is_safe(board: List[str], row: int, col: int) -> bool:
        # Check this row on left side
        for i in range(col):
            if board[row][i] == 'Q':
                return False

        # Check upper diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False

        # Check lower diagonal on left side
        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False

        return True

    def solve_queens_util(board: List[str], col: int, solutions: List[List[str]]) -> None:
        if col >= n:
            solutions.append(board.copy())
            return

        for i in range(n):
            if is_safe(board, i, col):
                board[i][col] = 'Q'
                solve_queens_util(board, col + 1, solutions)
                board[i][col] = '.'

    solutions: List[List[str]] = []
    board: List[str] = ['.' * n for _ in range(n)]
    solve_queens_util(board, 0, solutions)
    return solutions

n = 4
solutions = solve_n_queens(n)
for solution in solutions:
    for row in solution:
        print(row)
    print()
```

For `n = 4`, this code will find the two solutions to the 4-Queens problem,
printing them to the console.

This code utilizes backtracking by recursively trying to place a queen in each
row of the current column, then moving on to the next column. If it finds that a
queen placement violates the constraints (i.e., two queens threaten each other),
it "backtracks" by removing the queen and trying the next row in the current
column. It continues this process until it has found all solutions.
