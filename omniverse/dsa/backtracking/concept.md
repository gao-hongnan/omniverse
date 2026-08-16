---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Concept

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gao-hongnan/omniverse/blob/main/omniverse/dsa/backtracking/concept.ipynb)

```{contents}
:local:
```

```{code-cell} ipython3
:tags: [remove-cell]

%config InlineBackend.figure_format = 'svg'

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Optional, TypeVar
from typing import List, Union, TypeVar, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import rich
from rich.jupyter import print



from omnivault.dsa.trees.binary import BinaryTreeNode
from omnivault.dsa.trees.utils import build_binary_tree_from_list_preorder
from omnivault.dsa.trees.utils import print_binary_tree
```

## Introduction

## Definition

**Backtracking** is a systematic method of exploring all possible candidates for
a solution in a search space defined by $\mathcal{X}$ and $\mathcal{C}$. When a
partial solution is determined to be infeasible, the algorithm abandons the
current path and reverts to a previous decision point (backtracks) to explore a
different path. The backtracking process continues until all possible candidates
have been explored, and feasible solutions are processed according to the
problem's requirements.

1. **Decision Space**: A space $\mathcal{X}$ that consists of all possible
   choices that can be made.
2. **Constraints**: A set of conditions, $\mathcal{C}$, which the solutions must
   satisfy.
3. **Solution Space**: A space $\mathcal{S}$ containing all feasible solutions
   that meet the constraints. Formally, $\mathcal{S} \subseteq \mathcal{X}$ and
   $s \in \mathcal{S}$ if and only if $s$ satisfies all constraints in
   $\mathcal{C}$.
4. **Partial Solution**: A partial sequence of choices
   $p = \langle x_1, x_2, \ldots, x_k \rangle$ where $x_i \in \mathcal{X}$ and
   $k \leq n$, where $n$ is the size of a complete solution.
5. **Complete Solution**: A sequence of choices that completely covers the
   solution space, i.e., a partial solution with size $n$.

## Backtracking Algorithm

### Definition

Given a set of choices $\mathcal{X}$, constraints $\mathcal{C}$, and a current
partial solution $p$, the backtracking algorithm can be defined as follows:

1. **Check for Completion**: If $p$ is a complete solution, then process or
   store it.
2. **Extend the Partial Solution**: For each choice $x \in \mathcal{X}$, extend
   the current partial solution $p$ by the choice $x$ and recursively apply the
   backtracking algorithm.

    - **Prune**: If the extended partial solution does not meet the constraints
      in $\mathcal{C}$, it is discarded.

3. **Backtrack**: If no further extension of $p$ leads to a solution, revert to
   the previous state of $p$ and try the next choice.

### Correctness

If the backtracking algorithm is implemented with the correct definition of the
decision space $\mathcal{X}$, constraints $\mathcal{C}$, and proper handling of
partial solutions, then it will enumerate all valid solutions in $\mathcal{S}$.

We want to show that the backtracking algorithm will find every solution in the
solution space $\mathcal{S}$, and it will not find any solutions that are not in
$\mathcal{S}$.

#### Proof of Exhaustiveness

```{prf:proof} **Proof of Exhaustiveness (It will find every solution in $\mathcal{S}$)**:

- **Base Case**: If a partial solution $p$ is a complete solution and
    satisfies all constraints, then by definition, it is in $\mathcal{S}$, and
    the algorithm will process it.

- **Inductive Step**: Suppose the algorithm correctly processes all partial
    solutions at depth $k$, and we want to prove it will do the same for depth
    $k+1$.

    - For every extension of $p$ at depth $k+1$, the algorithm either
        discards it (if it does not meet the constraints), or it continues to
        extend it (if it is not complete), or processes it (if it is complete
        and satisfies the constraints).

    - By the inductive hypothesis, we know that all partial solutions at
        depth $k$ are handled correctly. Thus, all extensions at depth $k+1$
        are also handled correctly.

    - Therefore, every solution in $\mathcal{S}$ will be found by the
        backtracking algorithm.
```

#### Proof of Soundness

```{prf:proof} **Proof of Soundness (It will not find any solutions that are not
in $\mathcal{S}$)**:

- If a sequence of choices $s$ is processed by the algorithm as a solution,
    then it must be a complete solution and satisfy all constraints.

- By the definition of the solution space $\mathcal{S}$, any complete
    solution that satisfies all constraints is in $\mathcal{S}$.

- Therefore, the backtracking algorithm will not find any solutions that are
    not in $\mathcal{S}$.
```

#### Conclusion

By proving both exhaustiveness and soundness, we have shown that the
backtracking algorithm will enumerate all valid solutions in $\mathcal{S}$, and
it will not find any solutions that are not in $\mathcal{S}$. Thus, we have
proved the correctness of the backtracking algorithm.

## Example

Here is our running example:

Suppose you have three variables $A$, $B$, and $C$, each taking values from the
domain $\{0, 1\}$.

There are two constraints:

1. The sum of the three variables must be even: $A + B + C \mod 2 = 0$.
2. The sum of the first two variables must be less than $1$: $A + B < 1$.

_Find all possible solutions to this problem_.

## Common Notations

Backtracking is a rich subject with many related terminologies. Here are some
common terms and concepts:

1. **Backtracking**: The process of retreating (or backtracking) to a previous
   step when the current step does not lead to a solution, allowing for a
   different path to be explored.

2. **Pruning**: The process of discarding branches of the search tree that
   cannot possibly lead to a valid solution. This can significantly reduce the
   search space.

3. **Branching Factor**: The number of choices available at each decision point
   in the search tree.

4. **Search Tree**: A tree structure that represents the various paths that can
   be taken during the execution of the backtracking algorithm. Nodes represent
   partial solutions, and edges represent choices.

5. **Decision Space**: The set of all possible choices or decisions that can be
   made in the problem.

6. **Constraint Satisfaction Problem (CSP)**: A type of problem where the goal
   is to find a solution that meets a series of constraints. Backtracking is
   commonly used to solve CSPs.

7. **Heuristic**: A strategy or technique used to guide the search in a
   particular direction, often to increase efficiency. It may involve ordering
   choices in a certain way or applying specific rules to prune branches
   earlier.

8. **Complete Solution**: A solution that fully satisfies all constraints and
   covers the entire decision space.

9. **Partial Solution**: A solution that represents a partial path through the
   decision space, potentially leading to a complete solution.

10. **Feasibility Check**: The process of determining whether a partial solution
    can possibly lead to a complete solution, often used to decide whether to
    prune a branch.

11. **Dead End**: A point in the search space where it's determined that no
    further extension of the current path can lead to a solution, triggering a
    backtrack.

12. **Recursive Algorithm**: Backtracking is often implemented recursively,
    where a function calls itself with modified arguments to explore different
    parts of the search space.

13. **Chronological Backtracking**: The process of backtracking in the reverse
    order of the decisions made. This is the most common form of backtracking.

14. **Look-Ahead**: A technique where you examine future choices to predict
    whether the current path is likely to lead to a solution. It can be used to
    prune branches early.

15. **Optimization Problem**: A problem where you not only want to find a
    solution but the best solution according to some criteria. Backtracking can
    be adapted to solve optimization problems by keeping track of the best
    solution found so far.

These terms capture the main concepts and techniques often encountered when
dealing with backtracking algorithms, providing a foundation for understanding
and implementing these types of algorithms.

(csp)=

## Constraint Satisfaction Problem (CSP)

> Backtracking is a general algorithm for finding all (or some) solutions to
> some computational problems (notably Constraint satisfaction problems or
> CSPs), which incrementally builds candidates to the solution and abandons a
> candidate ("backtracks") as soon as it determines that the candidate cannot
> lead to a valid solution[^backtracking-wikipedia].

### Definition

**Constraint Satisfaction Problem (CSP)** is defined as a triplet
$(\mathcal{V}, \mathcal{D}, \mathcal{C})$, where:

-   $\mathcal{V} = \{V_1, V_2, \ldots, V_n\}$ is a set of $n$ variables.
-   $\mathcal{D} = \{D_1, D_2, \ldots, D_n\}$ is a set of domains, where
    $D_i = \{d_{i1}, d_{i2}, \ldots, d_{ik}\}$ is the domain of variable $V_i$,
    containing the possible values that $V_i$ can take.
    -   We may also use $v_i$ to refer to a value in $D_i$ if it is clear from
        context. This is the value that $V_i$ takes in a solution.
-   $\mathcal{C} = \{C_1, C_2, \ldots, C_m\}$ is a set of constraints. Each
    constraint $C_j \in \mathcal{C}$ is a pair $\langle T_j, R_j \rangle$, where
    -   $T_j \subseteq \mathcal{V}$ is a subset of $k$ variables and,
    -   $R_j$ is a $k$-ary relation on the corresponding subset of domains
        $D_j$. The relation specifies the allowable combinations of values for
        those variables.

### What is a Relation?

We see that the definition of $\mathcal{C}$ involves a
[relation](<https://en.wikipedia.org/wiki/Relation_(mathematics)>) $R_j$ on a
subset of domains $D_j$. What is a relation?

A $k$-ary relation $R$ on sets $A_1, A_2, \ldots, A_k$ is a subset of the
Cartesian product of these sets:

$$
R \subseteq A_1 \times A_2 \times \ldots \times A_k
$$

In the context of a Constraint Satisfaction Problem (CSP), a $k$-ary relation is
used to define the allowable combinations of values for a subset of $k$
variables.

Let's take the constraint involving $k$ variables and the corresponding domains
$D_1, D_2, \ldots, D_k$:

-   $T = \{V_1, V_2, \ldots, V_k\}$ is the subset of $k$ variables.
-   $D_1, D_2, \ldots, D_k$ are the domains of the variables in $T$, containing
    the possible values that the variables can take.

A $k$-ary relation $R$ on the domains $D_1, D_2, \ldots, D_k$ is then defined
as:

$$
R \subseteq D_1 \times D_2 \times \ldots \times D_k
$$

An element $(d_1, d_2, \ldots, d_k) \in R$ represents an allowable combination
of values for the variables $V_1, V_2, \ldots, V_k$, where $d_i \in D_i$.

Things will get clearer with an example later.

### Solution

A solution to a CSP is a (sequence of) assignment(s) defined by the function
$\mathcal{A}$ that maps the set of variables $\mathcal{V}$ to the corresponding
domains, subject to the condition that for every constraint
$C_j \in \mathcal{C}$, the subset of the assignment restricted to the variables
in $t_j$ satisfies the relation $R_j$:

$$
\mathcal{A}: \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i \quad \text{subject to} \quad \forall C_j \in \mathcal{C}, \, \mathcal{A}(T_j) \in R_j
$$

> Note in particular, that the assignment $\mathcal{A}$ is a function that maps
> from the set of variables $\mathcal{V}$ to the corresponding domains. This
> means that the assignment $\mathcal{A}$ is a complete assignment, and not a
> partial assignment. Consequently, a solution to a CSP needs $n$ assignments,
> one for each variable $V_i \in \mathcal{V}$. So to be pedantic, you can say
> that the solution is a set or sequence of assignments, one for each variable.

Thus, more explicitly, a solution can be defined as a sequence of assignments:

$$
\mathcal{A} = \left(\mathcal{A}(V_1), \mathcal{A}(V_2), \ldots, \mathcal{A}(V_n)\right) \quad \text{subject to} \quad \forall C_j \in \mathcal{C}, \, \mathcal{A}(T_j) \in R_j
$$

Here, $T_j$ is the tuple of variables involved in constraint $C_j$, and
$\mathcal{A}(T_j)$ represents the subset of the assignment restricted to the
variables in $T_j$. The expression $\mathcal{A}(T_j) \in R_j$ ensures that this
subset of the assignment satisfies the corresponding relation $R_j$ for each
constraint in the problem.

One may ask, why are evaluating $\mathcal{A}$ on the tuple of variables $T_j$
and checking if the result is in the relation $R_j$? Why not just check
$\mathcal{A}$ on the whole variable set $V$? This is because some constraints
are **_only imposed on a subset of variables_**. For example, the example we
have, the constraint $C_2$ is imposed on the first two variables and not the
third. So we only need to check if the first two variables satisfy the
constraint, and not the third. This is the motivation of why we need $T_j$.

```{prf:remark} Notation Abuse
:label: prf-notation-abuse

The solution is usually a sequence or set of assignments,
one for each variable.

$$
\left(\mathcal{A}(V_1), \mathcal{A}(V_2), \ldots, \mathcal{A}(V_n)\right)
$$

but we will abuse notation and use $\mathcal{A}$ to refer to the sequence of
assignments where the context is clear.
```

### Evaluation

Now that we have the definition of a solution, we can define the concept of
evaluation.

Evaluation often refers to methods or criteria used to assess potential
solutions, partial assignments, or to guide the search process towards a valid
solution. Evaluation may include concepts such as consistency and completeness,
as previously defined, and it may be part of an algorithmic approach to finding
solutions. Evaluation can be seen as a tool or a process used to explore the
solution space and identify valid solutions.

More formally, an evaluation of the variables is a function $f$ that maps from a
subset of variables to a particular set of values in the corresponding subset of
domains.

$$
f: \mathcal{V}_k \to \bigcup_{V_i \in \mathcal{V}_k} D_i
$$

where

-   $\mathcal{V}_k \subseteq \mathcal{V}$ is a subset of variables, and they
    need not be in any particular order. For example, $\mathcal{V}_k$ could be
    $\{V_1, V_3, V_5\}$.
-   Thus, we should not use $\bigcup_{i=1}^{k} D_i$ to denote the co-domain of
    $f$ because the domains of the variables in the domain $\mathcal{V}_k$ may
    not be contiguous. Instead, we use $\bigcup_{V_i \in \mathcal{V}_k} D_i$ to
    denote the co-domain of $f$, where $D_i$ is the domain of $V_i$. For
    example, if $\mathcal{V}_k = \{V_1, V_3, V_5\}$, then the domain of
    $\mathcal{V}_k$ is
    $\bigcup_{V_i \in \mathcal{V}_k} D_i = D_1 \cup D_3 \cup D_5$.

```{prf:remark} Assignment vs Evaluation
:label: prf-assignment-vs-evaluation

In the context of Constraint Satisfaction Problems (CSP), assignment and
evaluation are related but distinct concepts.

1. **Assignment**: An assignment refers to the process of associating a specific
   value with a particular variable in the problem. In a CSP, a solution
   consists of assignments that satisfy all constraints. An assignment is a
   mapping from variables to values.

2. **Evaluation**: Evaluation, on the other hand, refers to the process of
   assessing whether a given assignment (or set of assignments) satisfies the
   constraints of the problem. In other words, evaluation involves checking if
   the values assigned to the variables meet all the conditions imposed by the
   constraints.

So, while assignment is about setting the variables to specific values,
evaluation is about checking whether those assignments meet the criteria defined
by the problem's constraints. The two concepts are intimately related in the
sense that the evaluation follows the assignment to determine if the solution is
correct, but they represent different aspects of the problem-solving process in
CSP.
```

Now that we have defined an evaluation, we can define the notion of
**consistent**, **complete** and **solution** evaluation.

#### Consistent Evaluation

An evaluation $f: \mathcal{V}_k \to \bigcup_{V_i \in \mathcal{V}_k} D_i$ is said
to be **consistent** if $\forall C_j \in \mathcal{C}$ that involves variables in
$\mathcal{V}_k$, the assignment $f(T_j)$ is in $R_j$. In other words, an
evaluation is **consistent** if it does not violate any of the constraints.

Formally, it means that for every constraint $C_j \in \mathcal{C}$, the subset
of the evaluation restricted to the variables in $T_j$ satisfies the relation
$R_j$:

$$
\forall C_j \in \mathcal{C}, \, f(T_j) \in R_j
$$

An evaluation is a function that maps a subset of variables to their respective
values.

In the solution definition, this consistency is captured in the condition
$\forall C_j \in \mathcal{C}, \, \mathcal{A}(T_j) \in R_j$, which ensures that
the assignment for every constraint's variables satisfies the corresponding
relation.

#### Inconsistent Evaluation

An evaluation $f: \mathcal{V}_k \to \bigcup_{V_i \in \mathcal{V}_k} D_i$ is said
to be **inconsistent** if it violates at least one constraint
$C_j \in
\mathcal{C}$ that involves variables in $\mathcal{V}_k$. In other words,
an evaluation is **inconsistent** if it violates at least one constraint.

Formally, it means that there exists at least one constraint
$C_j \in \mathcal{C}$ such that the subset of the evaluation restricted to the
variables in $T_j$ does not satisfy the relation $R_j$:

$$
\exists C_j \in \mathcal{C} \, \text{such that} \, f(T_j) \notin R_j
$$

#### Complete Evaluation

An evaluation $f: \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i$ is
**complete** if it maps every variable in $\mathcal{V}$ to a value in its
corresponding domain $\mathcal{D}_i$. In other words, an evaluation is
**complete** if it includes all variables in $\mathcal{V}$.

More formally, it means that the domain of the evaluation is the entire set of
variables $\mathcal{V}$:

$$
\bigcup_{V_i \in \mathcal{V}} D_i
$$

This corresponds to the function $\mathcal{A}$ mapping the entire set of
variables $\mathcal{V}$ to their respective domains.

#### Solution Evaluation

An evaluation $f: \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i$ is said to
be a **solution** if it is both **complete** and **consistent**, meaning it
assigns a value from the domain to every variable in $\mathcal{V}$, and it
satisfies all constraints in $\mathcal{C}$. In other words, a **solution
evaluation** is an evaluation that is both **consistent** and **complete**.

Recall that a solution to a CSP is a (sequence of) assignment function(s):

$$
\mathcal{A}: \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i
$$

such that

1. **Consistency**: $\forall C_j \in \mathcal{C}, \mathcal{A}(T_j) \in R_j$.
   Every constraint is satisfied.
2. **Completeness**: $\forall V_i \in \mathcal{V}, \mathcal{A}(V_i) \in D_i$.
   Every variable is assigned a value from its domain.

Here, $\mathcal{A}(T_j)$ refers to the subset of the assignment restricted to
the variables in $T_j$, and the expression ensures that this subset satisfies
the corresponding relation $R_j$ for each constraint.

In the context of solving a CSP, a solution is an assignment of values to
variables that satisfies all the constraints.

1. **Consistency**: Consistency ensures that the assigned values do not violate
   any constraints. If any variable were assigned a value that violated a
   constraint, the entire assignment would be considered invalid. The
   constraints define the rules of the problem, and satisfying them ensures that
   the solution is meaningful and correct.

2. **Completeness**: The completeness aspect ensures that the solution
   encompasses all variables in the problem. If a solution only partially
   assigns values to variables, it may not be usable in the context for which
   the CSP was formulated. A complete solution allows for a full understanding
   of the resolved state of the problem.

So, in summary, a solution to a CSP must involve all variables (completeness)
and must not violate any constraints (consistency). These two aspects together
ensure that the solution fully and accurately resolves the problem, according to
the rules defined by the constraints.

#### Motivation

The evaluation function in a Constraint Satisfaction Problem (CSP) serves
several essential purposes:

1. **Mapping Variables to Values**: The primary role of an evaluation function
   in a CSP is to map a subset of variables (or possibly all variables) to
   specific values within their domains. This mapping represents a partial or
   complete assignment of values to variables.

2. **Determining Consistency**: Through the evaluation, you can determine
   whether a given assignment of values to variables is consistent with the
   constraints of the problem. A consistent evaluation is one where no
   constraints are violated.

3. **Guiding Search**: In solving CSPs, various search strategies might be
   employed to find a solution that satisfies all constraints. The evaluation
   function helps guide this search by allowing you to identify and explore
   consistent assignments, prune inconsistent ones, and gradually build towards
   a complete and consistent solution.

4. **Representing Solutions**: A complete evaluation, where all variables have
   been assigned values that do not violate any constraints, represents a
   solution to the CSP. Thus, the evaluation encapsulates the final answer to
   the problem.

5. **Facilitating Extensions and Constraints Handling**: Having a formal way to
   represent partial or complete assignments enables more complex handling of
   constraints and can facilitate extensions to more complex problems, like
   Max-CSP (where you seek the assignment that satisfies the maximum number of
   constraints) or weighted CSP (where constraints have different importance
   levels).

6. **Analyzing and Understanding the Problem**: The evaluation function provides
   a clear and formal way to analyze and understand the problem's structure. It
   can be used to understand how constraints interact with each other and the
   domains of the variables, which can be crucial in both solving the problem
   and understanding its complexity.

In summary, the concept of an evaluation in CSP is a fundamental building block
that enables consistent and complete solutions to be found, facilitates the
search process, represents the solution, and allows for more complex problem
handling and analysis.

## State Space Tree

A **State Space Tree** represents the possible configurations and choices in a
search space. For a Constraint Satisfaction Problem (CSP) with a set of
variables $\mathcal{V}$ and domains $D_i$ for each variable
$V_i \in \mathcal{V}$, we can define the state space tree as follows:

-   **Nodes**: Each node in the tree corresponds to a partial assignment of
    variables. A node at depth $k$ represents an assignment of $k$ variables,
    denoted by $p_k = \{(V_1, v_1), \ldots, (V_k, v_k)\}$ where
    $V_i \in \mathcal{V}$ and $v_i \in D_i$.
-   **Edges**: An edge between two nodes represents the assignment of a new
    variable. If a node $p_k$ represents the partial assignment of the first $k$
    variables, its child nodes represent all possible extensions of this partial
    assignment by one more variable.
-   **Leaf Nodes**: Leaf nodes in the tree represent complete assignments of all
    variables. A leaf node corresponds to a potential solution to the CSP.

### Visualization

```{figure} ../assets/backtracking/example-0.svg
---
name: state-space-tree-example-0
---
State Space Tree
```

### Sample Template

Here we present a basic template for backtracking on a state space tree with no
pruning or constraints handling.

```python
def backtrack(root, path):
    if is_leaf(root): # base case, leaf=complete assignment
        output(path)
        return

    # iterate all possible candidates.
    for edge in get_edges(root):
        path.add(edge)
        backtrack(root + 1, path)
        path.pop()
```

### Nodes

#### Root Node

Usually this is the initial state of the problem, typically a state where no
decisions have been made.

Formally, we can define the root node as follows:

$$
p_0 = \left( \emptyset \right)
$$

#### Internal Nodes

Each internal node represents a **partial** assignment of variables, meaning
some variables have been assigned values, and others have not yet been
considered. The depth of the node in the tree corresponds to the number of
decisions made so far.

Formally, we can define an internal node at depth $k$ as follows:

$$
p_k = \left(\mathcal{A}(V_1)=v_1, \ldots, \mathcal{A}(V_k)=v_k\right)
$$

where

-   $V_i \in \mathcal{V}$ and $v_i \in D_i$.

> In a sense, the node value $p_k$ is also the path from the root to the node at
> depth $k$. This is an abstract view of the internal node because you can also
> technically view it as just one single variable assignment, but it's useful to
> think of it as a path because it helps us understand the backtracking
> algorithm better. You should also realise that this internal node **does hold
> information about the variables and their values assigned so far**.

#### Leaf Nodes

Each leaf node represents a **complete** assignment of variables, where all
variables have been assigned values. If this assignment satisfies all
constraints, it's a valid solution to the CSP.

Formally, we can define a leaf node as follows:

$$
p_n = \left(\mathcal{A}(V_1)=v_1, \ldots, \mathcal{A}(V_n)=v_n\right)
$$

where

-   $V_i \in \mathcal{V}$ and $v_i \in D_i$.
-   $n$ is the number of variables in the CSP and is also the depth of the leaf
    node, or the height of the tree as usually CSP's state space trees are of
    equal depth.

### Edges

**Edges** in the state space tree represent decisions or transitions between
states (partial assignments). An edge from one node to another means making a
decision about a variable's value based on the problem's constraints.

For example, in our running example, we have $\mathcal{V} = \{A, B, C\}$, if you
are at a node representing a partial assignment where $A = 0$ and $B$ is
unassigned, and the next variable to consider is $B$, then there would be an
edge to a child node for each possible value of $B$ that is consistent with the
constraints.

> So what is the real difference between edge and node in this context? So for
> example in the state tree shown earlier, if you are at the left node $A=0$,
> then the edge to the left child node $B=0$ represents the decision that $B$
> should be assigned the value $0$, similarly the edge to the right child node
> $B=1$ represents the decision that $B$ should be assigned the value $1$. So
> that is why in the code, you do `for edge in get_edges(root):`, then you do
> `path.add(edge)`, so the edge is the decision that you are making, and the
> node is the state that you are in after making that decision. But of course,
> you usually use DFS and thus you will first add the left edge, perform DFS,
> then backtrack and add the right edge, perform DFS and so on.

#### Edges Representing Consistent Transitions

-   These edges lead to nodes (partial assignments) that do not violate any
    constraints so far.
-   They represent decisions that are in line with the constraints and could
    potentially lead to valid solutions.
-   If the algorithm follows such an edge and finds that the subsequent
    decisions lead to a violation, it will backtrack and try a different path.

#### Edges Representing Inconsistent Transitions

-   These edges lead to nodes (partial assignments) that violate one or more
    constraints of the problem.
-   The algorithm will typically prune the subtree rooted at such a node,
    meaning that it will not explore any further down this path, since no valid
    solutions can be found.
-   These edges help visualize the decisions that the algorithm has considered
    and rejected because they do not lead to valid solutions.

#### Using Edges in the Algorithm

The backtracking algorithm doesn't explicitly build the entire state space tree,
including both consistent and inconsistent edges. Instead, it dynamically
explores the tree, constructing nodes and edges as it goes. When it finds an
inconsistent transition, it immediately backtracks without creating the
corresponding subtree.

#### Example

-   Suppose you're working on a CSP with variables $V_1, V_2, \ldots, V_n$ and a
    specific set of constraints.
-   At a node representing a partial assignment $\{V_1 = v_1, V_2 = v_2\}$, you
    have to decide the value for $V_3$.
-   Each potential value for $V_3$ will create an edge from the current node to
    a child node.
-   If assigning $V_3 = v_3$ does not violate any constraints with the existing
    partial assignment, the edge is considered consistent, and the algorithm
    continues to explore.
-   If assigning $V_3 = v_3$ violates a constraint, the edge is considered
    inconsistent, and the algorithm backtracks to try a different value for
    $V_3$ or a previous variable.

So in your state space tree, the edges reflect the decisions made by the
algorithm, and they can be classified as consistent or inconsistent based on
whether they comply with the constraints of the problem.

### Paths

A **path** from the root to any node (internal or leaf) represents a sequence of
decisions made so far. This sequence forms a partial assignment (if it's an
internal node) or a complete assignment (if it's a leaf node).

As mentioned in the internal nodes section, there may be ambiguity in the
definition of a node and that of a path.

> That is why in the code, when you reach leaf node, you "report" the path
> found. That just means return the path, which is a sequence of decisions made
> so far.

### Pruning

**Pruning**: In the backtracking algorithm, if a partial assignment is found to
violate a constraint, the subtree rooted at the corresponding node is "pruned,"
meaning the algorithm does not explore further down that path.

### Summary

In summary:

-   Nodes represent states or assignments (either partial or complete).
-   Edges represent decisions or transitions between these states.
-   Paths represent sequences of decisions that lead to a particular state.

By understanding this structure, you can see how the backtracking algorithm
explores the solution space, extending partial solutions, and backtracking when
necessary to find all valid solutions to the CSP.

### Proof of Equivalence Between Backtracking and DFS Traversal of State Space Tree

Now, let's prove that the backtracking algorithm is equivalent to a DFS
traversal of the state space tree.

-   Let $\mathcal{X}$ be the decision space, the set of all possible choices.
-   Let $\mathcal{C}$ be the constraints that must be satisfied by the
    solutions.
-   Let $\mathcal{S} \subseteq \mathcal{X}$ be the solution space, the set of
    all sequences of choices that form valid solutions according to
    $\mathcal{C}$.
-   Let $\mathcal{T}$ be a tree where each node represents a partial solution,
    and each edge represents an extension of the partial solution by adding a
    new choice from $\mathcal{X}$.

#### Definition 1: State Space Tree (SST)

The state space tree is a rooted tree, where:

-   Root Node:

    $$
    p_0 = \left( \emptyset \right)
    $$

-   Internal Nodes at depth $k$:

    $$
    p_k = \left(\mathcal{A}(V_1)=v_1, \ldots, \mathcal{A}(V_k)=v_k\right)
    $$

-   Leaf Nodes at depth $n$:

    $$
    p_n = \left(\mathcal{A}(V_1)=v_1, \ldots, \mathcal{A}(V_n)=v_n\right)
    $$

#### Definition 2: Depth-First Search (DFS)

We use the same definition as above, where DFS traverses children of a node
before backtracking to visit the siblings, adhering to the structure of the SST.

#### Definition 3: Backtracking Algorithm

The backtracking algorithm involves recursively extending partial solutions at
internal nodes, pruning non-promising branches, and backtracking when a dead-end
is reached.

The relationship between a backtracking algorithm and depth-first search (DFS)
is an interesting one. A backtracking algorithm can be visualized as a traversal
of a DFS tree, where each node represents a decision, and each edge represents a
choice leading to a subsequent decision. The leaf nodes represent potential
solutions, and the paths from the root to the leaves represent different
sequences of choices.

Below is an attempt to formulate this relationship as a theorem and provide a
rigorous proof.

#### Theorem

A backtracking algorithm to find all solutions in $\mathcal{S}$ can be
represented as a traversal of a DFS tree $\mathcal{T}$, where each path from the
root to a leaf of $\mathcal{T}$ corresponds to a sequence of choices in
$\mathcal{X}$, and the traversal explores all such paths that satisfy the
constraints in $\mathcal{C}$.

#### Proof

##### (1) Construction of the DFS Tree $\mathcal{T}$

-   **Nodes**: Each node $v$ in $\mathcal{T}$ represents a partial solution in
    $\mathcal{X}$, where the root represents the empty solution.
-   **Edges**: Each edge $(u, v)$ represents the extension of the partial
    solution represented by $u$ with a new choice from $\mathcal{X}$, leading to
    the partial solution represented by $v$.
-   **Leaves**: The leaf nodes represent either complete solutions (if they
    satisfy $\mathcal{C}$) or sequences of choices that can't be further
    extended to form valid solutions.

##### (2) Correspondence between DFS Traversal and Backtracking

-   **Initialization**: Start at the root of $\mathcal{T}$, representing the
    empty solution.
-   **Exploration**: For each node $v$, representing a partial solution $p$,
    explore each child of $v$ by extending $p$ with a new choice from
    $\mathcal{X}$, recursively applying the same process.
-   **Constraints Handling**: If an extension leads to a partial solution that
    doesn't satisfy $\mathcal{C}$, prune the corresponding subtree of
    $\mathcal{T}$.
-   **Backtracking**: When reaching a leaf or when all children of a node have
    been explored, backtrack to the parent node, effectively reverting to the
    previous partial solution.
-   **Solution Processing**: If a leaf node corresponds to a complete solution
    in $\mathcal{S}$, process or store that solution as required by the problem.

The DFS traversal of $\mathcal{T}$ systematically explores all possible
sequences of choices in $\mathcal{X}$, while respecting the constraints in
$\mathcal{C}$, precisely following the logic of a backtracking algorithm.
Therefore, the backtracking algorithm's exploration of the decision space
$\mathcal{X}$ is equivalent to the DFS traversal of the tree $\mathcal{T}$.

This concludes the proof, establishing that a backtracking algorithm can be
modeled as a traversal of a DFS tree, with a one-to-one correspondence between
the nodes, edges, and paths in the tree and the decision space, constraints, and
solutions in the backtracking problem.

### Conclusion

By defining the state space tree and mapping the steps of the backtracking
algorithm to the operations in a DFS traversal, we have rigorously shown that
the backtracking algorithm is equivalent to a DFS traversal on the state space
tree. This provides a solid understanding of the algorithm's workings and proves
its correctness for solving CSPs.

## Ensuring an Even Sum and a Sum Less Than 1 for the First Two Variables (CSP)

Suppose you have three variables $A$, $B$, and $C$, each taking values from the
domain $\{0, 1\}$.

There are two constraints:

1. The sum of the three variables must be even: $A + B + C \mod 2 = 0$.
2. The sum of the first two variables must be less than $1$: $A + B < 1$.

_Find all possible solutions to this problem_.

Note there exists a hidden/implicit constrant that all variables must be used.

### 1. Variables

In the CSP framework, we start by defining a set of variables. In this example:

$$
\mathcal{V} = \{A, B, C\}
$$

Here, $A$, $B$, and $C$ are the variables, each representing an element that can
take a value from a specific domain.

### 2. Domains

Domains define the possible values that each variable can take. In this problem,
each variable can take on a value of 0 or 1. Thus, we have:

$$
\mathcal{D} = \{D_A, D_B, D_C\}
$$

where

-   $D_A = \{0, 1\}$
-   $D_B = \{0, 1\}$
-   $D_C = \{0, 1\}$

This means that the domain for each variable $A$, $B$, and $C$ consists of two
values, $0$ and $1$. In more complicated problems, the domains of each variable
can be different.

### 3. Constraints

Constraints are the rules that the variables must obey. In this example, the
constraints ensure that the sum of the values of $A$, $B$, and $C$ is even as
well as the sum of the first two variables is less than $1$.

We identified only two constraints in this example. Consequently, the set of
constraints $\mathcal{C}$ in this problem is just a set containing two
constraints:

$$
\mathcal{C} = \{C_1, C_2\}
$$

By our earlier definition, each constraint $C_j \in \mathcal{C}$ is a pair of
the form $\langle T_j, R_j \rangle$, where $T_j$ is a subset of the variables
and $R_j$ is a relation on the variables in $T_j$.

Let's dive deeper.

#### Constraint 1

The first constraint ensures that the sum of the values of $A$, $B$, and $C$ is
even. In all honesty, we can express it simply as:

$$
C_1 = \left\{(A, B, C) \mid A + B + C \mod 2 = 0\right\}
$$

However, we will follow the formal definition of a constraint and define $C_1$
as follows.

Firstly, since the constraint involves all three variables, we have $k = 3$ and
define $T_1 \subseteq \mathcal{V}$ as the subset of variables involved in the
constraint:

$$
T_1 = \{A, B, C\} \subseteq \mathcal{V}
$$

Secondly, we define a 3-ary relation $R_1$. Recall that the relation $R_1$ is a
$k$-ary relation on sub-domains $D_1, D_2, \dots, D_k$. In this case, since
$k = 3$, we are using all 3 domains $D_A$, $D_B$, and $D_C$.

To define the relation $R_1$, we need to consider all possible combinations of
values for the variables $A$, $B$, and $C$ that satisfy the even sum condition.

This set will then be formalized as a 3-ary relation
$R_1 \subseteq D_A \times D_B \times D_C$:

$$
\begin{aligned}
R_1 &= \{(a, b, c) \mid a \in D_A, b \in D_B, c \in D_C \text{ and } a + b + c \text{ is even}\} \\
    &= \{(a, b, c) \mid a, b, c \in \{0, 1\} \text{ and } a + b + c \mod 2 = 0\} \\
    &= \{(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)\}
\end{aligned}
$$

Finally, the constraint $C_1$ is defined as the pair $\langle T_1, R_1 \rangle$:

$$
\begin{aligned}
C_1 &= \langle T_1, R_1 \rangle \\
    &= \left \langle \{A, B, C\}, \{(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)\} \right \rangle
\end{aligned}
$$

#### Constraint 2

The second constraint ensures that the sum of the values of $A$ and $B$ is less
than $1$.

$$
C_2 = \left\{(A, B) \mid A + B < 1\right\}
$$

However, we will follow the formal definition of a constraint and define $C_2$
as follows.

Firstly, since this constraint involves only two variables, we have $k = 2$, and
we define $T_2 \subseteq \mathcal{V}$ as the subset of variables involved in the
constraint:

$$
T_2 = \{A, B\} \subseteq \mathcal{V}
$$

Secondly, we define a 2-ary relation $R_2$. Recall that the relation $R_2$ is a
$k$-ary relation on sub-domains $D_1, D_2$. In this case, since $k = 2$, we are
using 2 domains $D_A$ and $D_B$.

To define the relation $R_2$, we need to consider all possible combinations of
values for the variables $A$ and $B$ that satisfy the condition $A + B < 1$.

This set will then be formalized as a 2-ary relation
$R_2 \subseteq D_A \times D_B$:

$$
\begin{aligned}
R_2 &= \{(a, b) \mid a \in D_A, b \in D_B \text{ and } a + b < 1\} \\
    &= \{(a, b) \mid a, b \in \{0, 1\} \text{ and } a + b = 0\} \\
    &= \{(0, 0)\}
\end{aligned}
$$

Finally, the constraint $C_2$ is defined as the pair $\langle T_2, R_2 \rangle$:

$$
\begin{aligned}
C_2 &= \langle T_2, R_2 \rangle \\
    &= \left \langle \{A, B\}, \{(0, 0)\} \right \rangle
\end{aligned}
$$

Now, we have fully defined both constraints for the given CSP problem. We can
notice that the second constraint imposes a strict condition on the values of
$A$ and $B$, which may affect the feasible solutions for the entire problem.

### Solution

Given the CSP with variables $A$, $B$, and $C$ and the constraints defined, we
can analyze the problem and represent the solution.

First, let's analyze the constraints:

1. $A + B + C \mod 2 = 0$: This requires the sum of the variables to be even,
   which means either all variables are 0, or two out of the three variables
   must be 1.

2. $A + B < 1$: This implies that both $A$ and $B$ must be 0, as even if one of
   them is 1, the constraint would be violated.

Given the second constraint, we can immediately see that $A = B = 0$. Since $A$
and $B$ are both 0, the only way to satisfy the first constraint is to have
$C = 0$ as well.

So the solution to this CSP is:

$$
\begin{align*}
A & = 0 \\
B & = 0 \\
C & = 0
\end{align*}
$$

This assignment satisfies both constraints and is the complete mapping of the
variables to values in their corresponding domain.

Since we found that $A = B = C = 0$, the assignment function $\mathcal{A}$ that
maps each variable to its corresponding value in this solution is:

$$
\mathcal{A} : \{A, B, C\} \to \{0, 1\}
$$

With the specific mapping:

$$
\mathcal{A}(A) = 0, \, \mathcal{A}(B) = 0, \, \mathcal{A}(C) = 0
$$

This assignment function represents the complete solution to the CSP, satisfying
all constraints and assigning a value to each variable from its domain.

### Evaluation

Let's delve into the details of how the evaluation would work within the context
of this particular CSP problem, thereby deriving the solution.

#### 1. Variables and Domains

Firstly, let's restate the variables and domains:

-   Variables: $\mathcal{V} = \{A, B, C\}$
-   Domains:
    $\mathcal{D} = \{D_A, D_B, D_C\} = \{0, 1\} \times \{0, 1\} \times \{0, 1\}$

#### 2. Initial Partial Evaluation

We can initiate a partial evaluation by selecting a variable and assigning a
value from its domain. This could be done systematically (e.g., via a
depth-first search) or heuristically.

However, we know from constraint 2 that $A + B < 1$, so we can immediately
evaluate $A = 0$ and $B = 0$.

Define the initial partial evaluation:

$$
f_1: \{A, B\} \to \{0, 0\}
$$

#### 3. Consistency Check

Now, we check if this partial evaluation is consistent with the constraints.

-   **Constraint 1**: $A + B + C \mod 2 = 0$
-   **Constraint 2**: $A + B < 1$

Since $A = B = 0$, it currently satisfies both constraints as $0$ is even and
$0 + 0 < 1$.

This means that our partial evaluation is consistent, and we can proceed.

#### 4. Extend Evaluation

Next, we want to extend our partial evaluation by assigning a value to $C$.

From the consistent constraint $C \mod 2 = 0$, we know that $C$ must also be
$0$.

Extend the evaluation to:

$$
f_2: \mathcal{V} \to \{0, 0, 0\}
$$

#### 5. Check for Consistency and Completeness (Solution Verification)

We have reached the "leaf" as there are no more variables to assign values to.
Consequently, the next step is to verify that this complete evaluation is a
solution by checking if it is consistent with all constraints.

Since our evaluation $f_2$ includes all variables in $\mathcal{V}$, the
evaluation is complete.

Finally, we verify that this complete evaluation is a solution by checking if it
is consistent with all constraints:

-   For Constraint 1, we have $A + B + C = 0 + 0 + 0 = 0$, so it's satisfied.
-   For Constraint 2, we have $A + B = 0 + 0 < 1$, so it's satisfied as well.

Thus, our complete evaluation $f_2$ is consistent with all constraints, and
hence it is a solution to the CSP problem.

#### Conclusion

In this case, the solution to the problem is the evaluation $f_2$ that maps
$\mathcal{V} \to \{0, 0, 0\}$. The process involved systematically defining
partial evaluations, checking for consistency, extending the evaluations, and
verifying the solution. The notion of evaluation provides a structured way to
navigate through the possible combinations of variable assignments, ensuring
that constraints are respected at each step, leading to the solution of the CSP
problem.

What we are lacking is a systematic way to define the partial evaluations and
extend them. This is where the backtracking algorithm comes in. We will explore
this in the next section.

### Summary

By defining these variables, domains, and constraints, we've framed the given
problem as a CSP. A solution to this CSP would be an assignment of values to
$A$, $B$, and $C$ that satisfies all the defined constraints.

This example serves to illustrate the components of a CSP in a simple yet
rigorous manner. It breaks down the problem into the three key aspects of a CSP
and aligns them with the formal definition of the CSP framework. It also
provides a concrete example that makes the abstract idea of constraints more
tangible, bridging the gap between formal theory and intuitive understanding.

## Constraint Satisfaction Problems and Backtracking

### The Connection between Constraint Satisfaction Problems and Backtracking

The backtracking algorithm can be applied to solve CSPs by systematically
exploring the decision space defined by the variables and domains, guided by the
constraints.

### Notation Mapping Between Backtracking and CSP

Backtracking is a general algorithm for finding all (or some) solutions to
computational problems, particularly constraint satisfaction problems (CSPs).
Here are some common terminologies and notations used in both, and how they can
be mapped:

1. **Variables**:

    - **Backtracking**: Elements to which values are assigned.
    - **CSP**: Variables $V$ with domains $D$.
    - **Notation**: $\mathcal{V} = \{V_1, V_2, \ldots, V_n\}$.

2. **Domain**:

    - **Backtracking**: Set of possible values a variable can take.
    - **CSP**: Domain of each variable $V_i$.
    - **Notation**: $\mathcal{D} = \{D_1, D_2, \ldots, D_n\}$.

3. **Constraints**:

    - **Backtracking**: Conditions that must be met for a solution to be valid.
    - **CSP**: Constraints $C$ that must be satisfied.
    - **Notation**: $\mathcal{C} = \{C_1, C_2, \ldots, C_m\}$.

4. **Decision Space**:

    - **Backtracking**: Set of all possible assignments.
    - **CSP**: Set of all possible assignments of values to variables.
    - **Notation**:
        $$
        \mathcal{X} = \{ \mathcal{A} \mid \mathcal{A} : \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i \, \text{ and } \, \forall V_i \in \mathcal{V}, \mathcal{A}(V_i) \in \mathcal{D}_i \}
        $$

5. **Solution Space**:

    - **Backtracking**: Set of assignments that satisfy the constraints.
    - **CSP**: Subset of the decision space that meets all constraints.
    - **Notation**:
        $$
        \text{{Solution Space}} = \{ x \in \mathcal{X} \mid \text{{Constraints are satisfied}} \}
        $$

6. **State Space Tree**:

    - **Backtracking**: Tree representing all possible assignments, where
      branches are pruned if they cannot lead to a solution.
    - **CSP**: Similar to the state space tree in backtracking.
    - **Notation**: Often visually represented as a tree.

7. **Objective Function** (in case of optimization):
    - **Backtracking**: Function to optimize (minimize or maximize).
    - **CSP**: Function representing the quality of a solution (if there is an
      optimization aspect).
    - **Notation**: $f(\mathcal{A})$, where $f : \mathcal{X} \to \mathbb{R}$.

#### 1. Decision Space

The decision space $\mathcal{X}$ represents all possible assignments that maps
the variables to values, considering their domains. Given a set of variables
$\mathcal{V}$ and a corresponding set of domains $\mathcal{D}$ (where each
domain consists of the values a particular variable can take), the decision
space $\mathcal{X}$ can be defined as:

$$
\begin{aligned}
\mathcal{X} &= \left\{ \mathcal{A} \mid \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i \, \text{ and } \, \forall V_i \in \mathcal{V}, \mathcal{A}(V_i) \in \mathcal{D}_i \right\} \\
            &= \left\{ \left( \mathcal{A}(V_1), \mathcal{A}(V_2), \dots, \mathcal{A}(V_n) \right) \mid \forall V_i \in \mathcal{V}, \mathcal{A}(V_i) \in \mathcal{D}_i \right\}
\end{aligned}
$$

Note very carefully that the set $\mathcal{X}$ contains and enumerates all
possible assignments that map variables to values (solutions), including those
that violate the constraints. The constraints are used to guide the search
process and eliminate invalid assignments.

So in our running example, the decision space $\mathcal{X}$ would be:

$$
\begin{aligned}
\mathcal{X} &= \left\{ \left( \mathcal{A}(V_1), \mathcal{A}(V_2), \mathcal{A}(V_3) \right) \mid \mathcal{A}(V_1) \in \mathcal{D}_1, \mathcal{A}(V_2) \in \mathcal{D}_2, \mathcal{A}(V_3) \in \mathcal{D}_3 \right\} \\
            &= \left\{ \left( \mathcal{A}(A), \mathcal{A}(B), \mathcal{A}(C) \right) \mid \mathcal{A}(A) \in \{0, 1\}, \mathcal{A}(B) \in \{0, 1\}, \mathcal{A}(C) \in \{0, 1\} \right\} \\
            &= \left\{ (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1) \right\}
\end{aligned}
$$

Since the the example consists of binary variables, the decision space is there
are $2^3 = 8$ possible assignments.

#### 2. Constraints

The constraints $\mathcal{C}$ are directly taken from the CSP definition and
guide the search for feasible solutions.

#### 3. Solution Space

The **Solution Space** $\mathcal{S}$ is the set that contains all **feasible**
assignments that satisfy the constraints in $\mathcal{C}$.

Formally, the solution space $\mathcal{S}$ can be defined as:

$$
\begin{aligned}
\mathcal{S} \subseteq \mathcal{X} &= \left\{ \mathcal{A} \mid \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i \, \text{ and } \, \forall C_j \in \mathcal{C}, \, \mathcal{A}(T_j) \in R_j \right\} \\
            &= \left\{ \left( \mathcal{A}(V_1), \mathcal{A}(V_2), \dots, \mathcal{A}(V_n) \right) \mid \forall V_i \in \mathcal{V}, \mathcal{A}(V_i) \in \mathcal{D}_i \, \text{ and } \, \forall C_j \in \mathcal{C}, \, \mathcal{A}(T_j) \in R_j \right\}
\end{aligned}
$$

The definition above coincides with the definition of the solution mentioned in
the CSP section. In simple terms, the solution space $\mathcal{S}$ is the subset
of the decision space $\mathcal{X}$ that contains only those assignments that
satisfy all constraints in $\mathcal{C}$.

> What is the difference between the decision space $\mathcal{X}$ and the
> solution space $\mathcal{S}$? The solution space is a subset of the decision
> space that includes only those assignments or configurations that satisfy all
> constraints of the problem. If there's an optimization goal, the solution
> space may further be narrowed to include only those configurations that
> optimize (maximize or minimize) the objective function.

In our running example, the solution space $\mathcal{S}$ would be:

$$
\begin{aligned}
\mathcal{S} &= \left\{ \left( \mathcal{A}(V_1), \mathcal{A}(V_2), \mathcal{A}(V_3) \right) \mid \mathcal{A}(V_1) \in \mathcal{D}_1, \mathcal{A}(V_2) \in \mathcal{D}_2, \mathcal{A}(V_3) \in \mathcal{D}_3 \, \text{ and } \, \mathcal{A}(A) + \mathcal{A}(B) + \mathcal{A}(C) \mod 2 = 0 \, \text{ and } \, \mathcal{A}(A) + \mathcal{A}(B) < 1 \right\} \\
            &= \left\{ (0, 0, 0)\right\}
\end{aligned}
$$

The solution space $\mathcal{S}$ contains only those assignments that satisfy
the constraints in $\mathcal{C}$.

#### 4. Partial Solution

A **Partial Solution** represents a partially completed assignment that maps
some of the variables in $\mathcal{V}$ to values in their respective domains
$\mathcal{D}$ but leaves other variables **unassigned**. It's a crucial concept
in backtracking, where the algorithm builds up a solution incrementally, making
decisions one at a time, and using constraints to determine if it should
continue in a particular direction or backtrack.

> A partial solution can either be a **consistent** or an **inconsistent**
> assignment. This means the word _solution_ is misleading and doesn't
> necessarily mean a feasible solution. It's just a partial assignment that may
> or may not satisfy the constraints.

You can formally define a partial solution $\mathcal{A}_p$ as a function that
maps a subset of the variables $\mathcal{V}_p \subseteq \mathcal{V}$ to their
respective domains:

$$
\mathcal{A}_p: \mathcal{V}_p \to \bigcup_{V_i \in \mathcal{V}_p} D_i \quad \text{where} \quad \mathcal{V}_p \subset \mathcal{V} \, \text{ and } \, \forall V_i \in \mathcal{V}_p, \mathcal{A}_p(V_i) \in \mathcal{D}_i
$$

Here, $\mathcal{V}_p$ is a subset of $\mathcal{V}$, meaning that only some of
the variables in $\mathcal{V}$ are assigned values in the partial solution. The
rest of the variables are still unassigned.

> A partial solution $\mathcal{A}_p$ can be interpreted as a tuple or sequence
> of $k < n$ assignments:
> $\mathcal{A}_p = \left(\mathcal{A}_p(V_1), \mathcal{A}_p(V_2), \dots, \mathcal{A}_p(V_k) \right)$,
> where $k$ is the number of variables assigned values in the partial solution.

For example, in the context of your running example, a partial solution might
assign values to the variables $A$ and $B$ but leave $C$ unassigned. This
partial solution could be represented as $\mathcal{A}_p = \{(0, 1)\}$, where
$\mathcal{A}_{p}(A) = 0$ and $\mathcal{A}_{p}(B) = 1$. So the the domain of
$\mathcal{A}_{p}$ is $\mathcal{V}_p = \{A, B\}$, a subset of the full variable
set $\mathcal{V} = \{A, B, C\}$.

Backtracking algorithms often work by extending partial solutions, adding one
variable assignment at a time, and checking constraints to see if the partial
solution can be extended into a full solution. If a constraint is violated, the
algorithm will backtrack, undoing some of the variable assignments, and try a
different path through the decision space.

#### 5. Complete Solution

A **Complete Solution** in the context of Constraint Satisfaction Problems
(CSPs) is an assignment that maps all the variables in $\mathcal{V}$ to values
in their respective domains, in a way that satisfies all the given constraints
$\mathcal{C}$. In other words, it is an element of the solution space
$\mathcal{S}$ you've defined earlier.

Formally, a complete solution $\mathcal{A}$ can be defined as:

$$
\mathcal{A} : \mathcal{V} \to \bigcup_{V_i \in \mathcal{V}} D_i \quad \text{subject to} \quad \forall C_j \in \mathcal{C}, \, \mathcal{A}(T_j) \in R_j
$$

Here, $\mathcal{V}$ is the set of all variables, and $D_i$ is the domain of the
values that variable $V_i$ can take. $\mathcal{C}$ is the set of constraints,
with each constraint $C_j$ restricting the tuples $T_j$ of variable values to a
specific relation $R_j$.

The complete solution satisfies all constraints, meaning that for every
constraint $C_j$, the values assigned to the variables in the tuple $T_j$ by the
assignment $\mathcal{A}$ are in the relation $R_j$.

> A complete solution is tuple or sequence of $n$ assignments:
> $\mathcal{A} = \left(\mathcal{A}(V_1), \mathcal{A}(V_2), \dots, \mathcal{A}(V_n) \right)$,
> where $n$ is the number of variables in the CSP.

For example, in the context of your running example with the variables $A, B, C$
and constraints on their sum, the complete solution would be a specific
assignment to these variables that both makes the sum even and ensures that the
sum of the first two variables is less than 1. In this case, the only complete
solution is $\mathcal{A}(A) = 0, \mathcal{A}(B) = 0, \mathcal{A}(C) = 0$.

#### 6. Objective Function

If the CSP includes an optimization goal (like maximizing or minimizing some
value), you might also define an **Objective Function** $\mathcal{J}$:

$$
\mathcal{J} : \mathcal{S} \rightarrow \mathbb{R}
$$

What this means is that the objective function takes a solution $\mathcal{A}$
from the solution space $\mathcal{S}$ and maps it to a real number.

In our running example, let's say that the goal is to maximize the sum of the
variables $A$, $B$, and $C$, subject to the constraints that were previously
defined.

An objective function $\mathcal{J}$ that maximizes the sum of the variables
could be defined as:

$$
\mathcal{J}(\mathcal{A}) = \mathcal{A}(A) + \mathcal{A}(B) + \mathcal{A}(C)
$$

#### 7. Optimal Solution

An **Optimal Solution** is a complete solution that optimizes the objective
function. For a maximization problem:

$$
\mathcal{A}^* = \arg\max_{\mathcal{A} \in \mathcal{S}} \mathcal{J}(\mathcal{A})
$$

And for a minimization problem:

$$
\mathcal{A}^* = \arg\min_{\mathcal{A} \in \mathcal{S}} \mathcal{J}(\mathcal{A})
$$

> Note about notation abuse in {prf:ref}`prf-notation-abuse`.

Given our previous solution space:

$$
\mathcal{S} = \left\{ \left( \mathcal{A}(V_1), \mathcal{A}(V_2), \mathcal{A}(V_3) \right) \mid \mathcal{A}(V_1) \in \mathcal{D}_1, \mathcal{A}(V_2) \in \mathcal{D}_2, \mathcal{A}(V_3) \in \mathcal{D}_3 \, \text{ and } \, \mathcal{A}(A) + \mathcal{A}(B) + \mathcal{A}(C) \mod 2 = 0 \, \text{ and } \, \mathcal{A}(A) + \mathcal{A}(B) < 1 \right\}
$$

we found that the only feasible solution was:

$$
\mathcal{S} = \left\{ (0, 0, 0)\right\}
$$

Although the point shown is moot since there is only one solution, let's just
(for the sake of pedagogy purposes) apply our objective function to this
solution:

$$
\mathcal{J}((0, 0, 0)) = 0 + 0 + 0 = 0
$$

In this specific example, the constraints and the nature of the problem limited
the solution space to one single point, which also defined the value of the
objective function. If the constraints were less restrictive, or the domains
larger, the objective function would help to identify the best solution(s) from
a broader solution space.

### Backtracking Solves Constraint Satisfaction Problems (Theorem)

```{prf:theorem} CSP and Backtracking
:label: csp-backtracking

Given a Constraint Satisfaction Problem (CSP) defined as
$(\mathcal{V}, \mathcal{D}, \mathcal{C})$, the backtracking algorithm can be
used to find all solutions (if they exist) by systematically exploring the
decision space $\mathcal{X}$ defined by $\mathcal{V}$ and $\mathcal{D}$, guided
by the constraints in $\mathcal{C}$.
```

Here's a proof sketch:

```{prf:proof} Here's the proof.

1. **Initialization Phase**:

    - **Partial Solution**: $\mathcal{A} = \{\}$.
    - **Domains**, **Constraints**, **Ordering**, and **Objective Function**.

2. **Exploration Phase**: (Core recursive step that includes variable selection,
   assignment, constraint checking, pruning, backtracking, and completion)
    - **Variable Selection**: Choose an unassigned variable.
    - **Value Assignment**: Iterate through values in the domain, and for each:
        - Assign the value.
        - **Constraint Check** and **Pruning Phase**:
            - **Constraint Violation**: If inconsistent, go to the Backtracking
              Phase.
            - **Domain Reduction**: Optionally prune values.
        - If consistent, recursively call the Exploration Phase.
    - **Backtracking Phase**: (Nested within Exploration)
        - **Failure Handling**: If no valid value can be found, unassign the
          variable and return failure.
        - **Complete Exploration**: If all values have been explored without
          success, continue backtracking.
    - **Completion Phase**: (Part of the recursive exit conditions)
        - **Valid Solution**: If a full assignment satisfies all constraints.
        - **Termination**: Terminate if all branches are explored.
        - **Optimization**: If applicable, evaluate and select the optimal
          solution.
```

The process described in the proof corresponds to the structure of the
backtracking algorithm, adapted to the specific context of a CSP.

By defining the CSP and relating it to the backtracking algorithm, we have
formalized how backtracking can be applied to solve CSPs, with a clear mapping
between the elements of the CSP and the decision space, constraints, and
solutions in the backtracking framework.

> Note that this algorithm coincides with the traversal of the state space tree
> defined earlier in a depth-first search manner.

#### 1. Initialization

-   **Partial Solution**: Start with an empty partial solution, which is a
    mapping of variables to values, denoted as $\mathcal{A}$. Initially,
    $\mathcal{A} = \{\}$, meaning that no variables have been assigned values
    yet.
-   **Domains**: For each variable $V_i$, define its domain $D_i$ of allowable
    values, according to the constraints of the CSP.
-   **Constraints**: Define the set of constraints $\mathcal{C}$ that must be
    satisfied by the solution. These constraints can be equations, inequalities,
    or other relations involving the variables.
-   **Ordering**: Optionally, define a specific ordering in which the variables
    will be assigned values. The ordering can significantly affect the
    efficiency of the backtracking algorithm. For example, if the problem asks
    to find a combination of letters $a$ and $b$ in lexicographic order, the
    ordering of the variables would be $a$ and then $b$.
-   **Objective Function**: If the CSP includes an optimization goal, define the
    objective function $\mathcal{J}$ that maps a solution to a real number, as
    previously discussed.

The initialization sets up the framework within which the backtracking algorithm
operates, providing clear definitions of the problem's variables, domains, and
constraints. The remaining steps of the proof would involve detailing the
recursive assignment of values to variables, consistency checks, and the
backtracking mechanism itself, leading to a formal demonstration that the
algorithm either finds a valid solution or determines that no solution exists.

#### 2. Exploration

This is the core of the backtracking algorithm, where the search through the
decision space occurs. This process will be detailed in the following steps:

-   **Variable Selection**: Choose an unassigned variable $V_i$ from
    $\mathcal{V}$. If order does not matter, select the first unassigned
    variable in the ordering. Else, first sort the unassigned variables in the
    ordering, and then select the first one.
-   **Value Assignment**: For each value $v \in D_i$, where $D_i$ is the domain
    of the variable $V_i$, do the following:
    -   Assign the value $v$ to the variable $V_i$, i.e.,
        $\mathcal{A}(V_i) = v$.
    -   Check for constraint satisfaction: Verify that the current partial
        assignment satisfies all constraints in $\mathcal{C}$ that involve
        $V_i$. In a sense we are doing an evaluation of the partial solution
        here: $f(\cdot)$ on the current partial solution.
    -   If the current partial assignment is **consistent**, recursively call
        the exploration step to assign values to the next variable.
    -   This means if we select the next variable in the ordering, we will
        assign a value to it, and then check for consistency. If the current
        partial assignment is **inconsistent**, backtrack to the previous
        variable and continue exploring its remaining values.
-   **Unassignment**: If no valid value can be found for $V_i$, unassign the
    variable and return failure to trigger backtracking.

The recursion in this step systematically explores all possible value
assignments for each variable, guided by the constraints $\mathcal{C}$.

##### 2.1. Pruning

A sub-phase during value assignment:

-   **Constraint Violation**: If a partial solution violates any constraint,
    prune that path.
-   **Domain Reduction**: Optionally prune values in the domain.

##### 2.2. Backtracking

A nested sub-phase handling failure:

-   **Failure Handling**: If no valid value can be found, backtrack to the
    previous variable.
-   **Complete Exploration**: If all values of a variable have been explored
    without success, continue backtracking.

##### 2.3. Completion

Part of the recursive exit conditions:

-   **Valid Solution**: If a full assignment satisfies all constraints.
-   **Termination**: If all branches are explored, terminate.
-   **Optimization**: If applicable, evaluate and select the optimal solution.

#### 3. Optimization (if applicable)

If the problem includes an optimization goal:

-   **Objective Evaluation**: For each valid solution, evaluate the objective
    function $f(\mathcal{A})$.
-   **Optimal Solution Selection**: Select the solution that optimizes
    (minimizes or maximizes) the objective function.

### Conclusion

The backtracking algorithm is a systematic and efficient method for solving
CSPs. It explores the decision space $\mathcal{X}$, guided by the constraints
$\mathcal{C}$, and finds all valid solutions or determines that no solutions
exist.

By breaking down the process into distinct steps, we have provided a rigorous
and detailed understanding of how backtracking operates within the context of a
CSP. This includes the handling of variables, domains, constraints, the state
space tree, pruning mechanisms, and optional optimization.

The theorem, "Given a Constraint Satisfaction Problem (CSP) defined as
$(\mathcal{V}, \mathcal{D}, \mathcal{C})$, the backtracking algorithm can be
used to find all solutions (if they exist)", has been proved. The outlined proof
highlights the intricate relationship between CSP and backtracking, illustrating
how the latter serves as a powerful tool for solving the former.

## Solving Ensuring an Even Sum and a Sum Less Than 1 for the First Two Variables with Backtracking

so let me details how i approach it with backtracking, it is my first time, so i
may not use words such as pruning, or related words, you will first check if my
logic is sound, then correct my language.

1. Root node is empty **PARTIAL** solution, so we initiate with empty solution
   first then we branch left to start our dfs.
2. Since dfs down the tree, we reach 0,0,0 first and note it is a solution, but
   a nuance I think i s, when i start with a partial empty solution at the root,
   i go left, and reached 0, this is a partial solution (i guess?), then check
   if constraints are fulfilled, yes for now, so i go left and reached 0, check
   both constraints, then continue to last left 0, and all satisfied we got our
   first soln since leaf node is a solution, and before leaf the nodes are a
   state or choices? (i dont know the correct term)
3. We pop the leaf node of 0 and now we have a sequence of 0->0, we branch to
   right 1 and note it is not a soln since it does not satisfy the even
   constraint and the < 1, we pop this leaf 1 and go back to 0->0.
4. Since we explored all children of 0, we pop again to get 0, and branch right,
   to get to 0->1, but immediately we know it is wrong since it does not satisfy
   constraints, (I GUESS we prune here)? If we do prune, then we pop and get
   back to 0. We pop again and get back to root node,
5. We now branch right to go to 1, and 1 is not a solution already, so we know
   we can (prune) again?

```{figure} ../assets/backtracking/example-1.svg
---
name: state-space-tree-example-1
---
State Space Tree
```

```text
graph TD
  root[Start]
  A0[A=0]
  A1[A=1]
  B00[B=0]
  B01[B=1]
  B10[B=0]
  B11[B=1]
  C000[C=0]
  C001[C=1]
  C010[C=0]
  C011[C=1]
  C100[C=0]
  C101[C=1]
  C110[C=0]
  C111[C=1]

  root --> A0
  root --> A1
  A0 --> B00
  A0 --> B01
  A1 --> B10
  A1 --> B11
  B00 --> C000
  B00 --> C001
  B01 --> C010
  B01 --> C011
  B10 --> C100
  B10 --> C101
  B11 --> C110
  B11 --> C111
```

---

Given the specific problem and your approach to solving it using backtracking,
we can now analyze the logic in detail. Here's a step-by-step analysis:

1. **Root Node**: Starting with an empty solution, which means no variables are
   assigned yet, is correct.

2. **DFS to (0,0,0)**:

    - Assigning $A = 0$, $B = 0$, and $C = 0$ results in both constraints being
      satisfied, since $A + B + C \mod 2 = 0$ and $A + B < 1$. This is indeed a
      valid solution. You are correct that leaf nodes represent complete
      solutions.

3. **Backtrack from (0,0,0) to (0,0,1)**:

    - Since $A + B + C \mod 2 = 1$ and $A + B < 1$, the first constraint is
      violated. So you're right that you would prune this branch, backtrack, and
      remove the last assignment.

4. **Backtrack from (0,0,1) to (0,1,?)**:

    - Since $A + B = 1$, the second constraint is violated. You'd prune this
      branch and backtrack again. Here you seem to be doing the right thing.

5. **Backtrack from (0,1,?) to (1,?,?)**:
    - Since $A = 1$, you'd need to assign $B = 0$ to satisfy the second
      constraint. But then you'll find no assignment to $C$ that satisfies the
      first constraint. So you're correct that you would prune this entire
      branch.

Given your detailed description and the specific problem, I can confirm that
your logic is indeed sound. Your approach correctly uses DFS with backtracking,
and you've identified the appropriate times to prune the search.

A minor point of terminology: In your description, "left" and "right" represent
choosing the value 0 or 1 for the current variable, respectively. It's a common
metaphor for binary tree traversal, but in a formal algorithm description, you'd
typically refer to iterating over the domain of the variable (in this case,
$\{0, 1\}$) rather than "branching left" or "branching right."

The complete set of solutions to this CSP is simply $(0,0,0)$, as you've
identified. Your backtracking approach has correctly found this solution and
correctly identified that no other assignments to $A$, $B$, and $C$ satisfy both
constraints.

---

Given the problem with three variables $A$, $B$, and $C$ in the domain
$\{0, 1\}$, and the constraints:

1. $A + B + C \mod 2 = 0$ (even sum constraint)
2. $A + B < 1$ (first two variables sum constraint)

We can outline the backtracking solution as follows:

1. **Initialization**: Start with an empty partial solution, which means no
   variables are assigned.

2. **Exploration**:

    1. **Depth-First Search**: Begin by assigning $A = 0$, and recursively
       explore the decision space by extending the partial solution, following
       the constraints.
    2. **First Solution Found**: Extend to $B = 0$ and $C = 0$. This assignment
       satisfies both constraints, so $(0,0,0)$ is added to the set of
       solutions.
    3. **Backtrack and Prune**: Explore the next child by assigning $C = 1$.
       Since the first constraint is violated, prune this branch and backtrack
       to the previous state.

3. **Further Exploration**: a. **Backtrack and Prune**: Explore the next branch
   by assigning $B = 1$. The second constraint is violated, so prune this branch
   and backtrack to the previous state. b. **Backtrack and Prune**: Explore the
   next branch by assigning $A = 1$. Since there is no valid assignment for $B$
   that satisfies the second constraint, prune this entire branch.

4. **Completion**: The exploration is complete, and the only valid solution to
   the CSP is $(0,0,0)$.

The terminology used here includes:

-   **Initialization**: Starting the process with no variables assigned.
-   **Exploration**: Recursive extension of the solution following the
    constraints.
-   **Pruning**: Removing paths that violate constraints.
-   **Backtrack**: Returning to the previous state if no further extension is
    possible or a constraint is violated.

This formal description retains the essence of your original approach while
adopting a more rigorous and standardized language. It details the process of
systematically exploring the decision space defined by the variables and their
domains, guided by the constraints, to find all valid solutions to the given
CSP.

## Combination of letters (CSP)

The given problem, where you want to find all $n$-letter words composed of 'a'
and 'b', can indeed be formulated as a
[Constraint Satisfaction Problem (CSP)](csp). Here's how the components of a CSP
map to this problem:

### Variables

-   $\mathcal{V} = \{V_1, V_2, \ldots, V_n\}$: There are $n$ variables, where
    each variable $V_i$ represents a letter at the $i$-th position of the word.
-   For example, the word 'abba' can be represented by the variables
    $\{V_1 = a, V_2 = b, V_3 = b, V_4 = a\}$.

### Domains

-   $\mathcal{D} = \{D_1, D_2, \ldots, D_n\}$: Each domain $D_i$ consists of two
    possible values, 'a' and 'b', i.e., $D_i = \{'a', 'b'\}$.

The set of variables $\mathcal{V}$ contains the placeholders that represent the
positions in the word, not the actual values 'a' or 'b'. So, for $n=2$:

-   $\mathcal{V} = \{V_1, V_2\}$

Here, $V_1$ and $V_2$ are variables that can take on the values 'a' or 'b'. They
are not assigned any values yet.

The domains, on the other hand, specify the possible values that each variable
can take:

-   $D_1 = \{a, b\}$
-   $D_2 = \{a, b\}$

So, for each variable $V_i$, the corresponding domain $D_i$ specifies the set of
values that $V_i$ can take, which in this case is $\{a, b\}$.

You can think of the domains $D_1$ and $D_2$ (and more generally, $D_i$ for any
$i$) as the choices available for the corresponding variable at that position in
the word. So it is like a combination of two sets.

We take $a$ from $D_1$ and $a$ from $D_2$ to form the word 'aa', and we take $a$
from $D_1$ and $b$ from $D_2$ to form the word 'ab', etc.

In the context of this problem, each domain $D_i$ represents the two choices,
'a' or 'b', that are available for the $i$-th letter in the word. Since every
position in the word can be either 'a' or 'b', all the domains are identical,
and each one represents the two choices available at that particular position.

So, intuitively, you can think of the domains as the "choice sets" for each
variable, and in the context of this problem, those choices are the possible
letters 'a' or 'b' for each position in the word.

### Constraints

-   $\mathcal{C} = \{\}$: In this case, there are no explicit constraints on the
    values of the variables since any combination of 'a' and 'b' is allowed.

A solution to this CSP is an assignment of values to the variables $\mathcal{V}$
such that no constraints are violated. Since there are no explicit constraints,
any combination of 'a' and 'b' is a valid solution, and there will be $2^n$
possible solutions.

### Solution

A solution to the CSP would be an assignment of values to the variables, such as
$(V_1 = a, V_2 = b)$ or $(V_1 = b, V_2 = a)$, etc. There are $2^n$ possible
solutions in total for this particular problem.

-   Solution 1: $(V_1 = a, V_2 = a)$ where $V_1 = a \in D_1$ and
    $V_2 = a \in D_2$
-   Solution 2: $(V_1 = a, V_2 = b)$ where $V_1 = a \in D_1$ and
    $V_2 = b \in D_2$
-   Solution 3: $(V_1 = b, V_2 = a)$ where $V_1 = b \in D_1$ and
    $V_2 = a \in D_2$
-   Solution 4: $(V_1 = b, V_2 = b)$ where $V_1 = b \in D_1$ and
    $V_2 = b \in D_2$

Each solution is an assignment of values to the variables that satisfies all
constraints in the problem (in this case, there are no explicit constraints
other than the choice of 'a' or 'b' for each variable). You've correctly
enumerated all $2^2 = 4$ possible solutions for the case where $n = 2$.

For a general case with $n$ variables, there would indeed be $2^n$ possible
solutions, as each variable has 2 choices, and there are $n$ variables, giving a
total of $2 \times 2 \times \ldots \times 2 = 2^n$ different ways to assign
values to all the variables.

## Complexity Analysis

### Time Complexity

The time complexity of the backtracking algorithm can be analyzed in terms of
two factors: the branching factor $b$ and the maximum depth of the recursion
$d$.

1. **Branching Factor ($b$)**:

    - At each decision point in the state space tree, there are usually $b$
      different branches that can be followed, where $b$ represents the number
      of choices available for a particular decision.
    - If there are $n$ variables and each variable has a domain of size $m$,
      then in the worst case, the branching factor could be $m$ as each variable
      has $m$ choices.

2. **Maximum Depth of Recursion ($d$)**:

    - The depth of the state space tree corresponds to the size of a complete
      solution, which we denote as $d$. In a typical CSP, $d = n$, where $n$ is
      the number of variables in the problem.
    - The recursion explores each branch until a complete solution is found or a
      branch is pruned, so the maximum depth is reached for each valid solution.

3. **Worst-Case Complexity**:

    - In the worst case, the algorithm explores each branch at every level of
      the tree, leading to $b^d$ possible paths.
    - Checking a complete solution (or pruning an invalid solution) usually
      takes constant time, so the worst-case time complexity is:

        $$
        \mathcal{O}(b^d)
        $$

4. **Pruning Effect**:
    - Intelligent pruning can significantly reduce the actual running time. If
      constraints allow us to prune branches early, fewer branches need to be
      explored, which can bring down the time complexity in practice.

### Space Complexity

The space complexity of the backtracking algorithm is largely determined by the
maximum depth of the recursion:

1. **Recursion Stack**:

    - Every recursive call adds a new frame to the system’s call stack. Since
      the algorithm recurses to a maximum depth of $d$, there will be $d$ frames
      on the call stack at most.

2. **Partial Solutions and Other Overheads**:

    - We must also account for the storage of partial solutions, variables,
      domains, constraints, and any additional data structures used in the
      algorithm.
    - However, these typically don't grow with the depth of the recursion, so
      the dominant factor remains the size of the call stack.

3. **Total Space Complexity**:

    - The total space complexity, considering both the recursion stack and other
      overhead, is:

        $$
        \mathcal{O}(d)
        $$

### Conclusion

The time and space complexity analysis of the backtracking algorithm provides
insight into its efficiency:

-   The time complexity is exponential in the worst case but can be
    significantly improved with intelligent pruning and constraint propagation
    techniques.
-   The space complexity is linear in the depth of the recursion, reflecting the
    need to store the call stack and partial solutions.

These complexities underline the need for careful problem modeling and algorithm
design when using backtracking to solve complex CSPs. By understanding the
factors that contribute to the time and space complexity, one can tailor the
algorithm to exploit problem-specific characteristics and constraints,
potentially leading to more efficient solutions.

## References and Further Readings

-   <http://www.cs.toronto.edu/~torsten/csc384-f11/lectures/csc384f11-Lecture04-BacktrackingSearch.pdf>
-   <https://leetcode.com/explore/featured/card/recursion-ii/472/backtracking/2654/>
-   <https://algo.monster/problems/backtracking>

[^backtracking-wikipedia]:
    [https://en.wikipedia.org/wiki/Backtracking](https://en.wikipedia.org/wiki/Backtracking)
