from __future__ import annotations

import heapq
from collections import defaultdict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic

from .base import AbstractGraph

if TYPE_CHECKING:
    from collections.abc import Iterator

from ..core.errors import InvalidConfiguration, KeyNotFound
from ..core.types import EdgeWeightT, VertexT

type Heuristic[VertexT] = Callable[[VertexT, VertexT], float]


class PathResult(Generic[VertexT]):  # noqa: UP046
    def __init__(self, path: list[VertexT], distance: float) -> None:
        self.path = path
        self.distance: float = distance

    def __repr__(self) -> str:
        return f"PathResult(distance={self.distance}, length={len(self.path)})"


class ShortestPaths(Generic[VertexT]):  # noqa: UP046
    def __init__(self, distances: dict[VertexT, float], predecessors: dict[VertexT, VertexT | None]) -> None:
        self.distances: dict[VertexT, float] = distances
        self.predecessors = predecessors

    def get_path(self, target: VertexT) -> list[VertexT] | None:
        if target not in self.predecessors:
            return None

        path: list[VertexT] = []
        current: VertexT | None = target
        while current is not None:
            path.append(current)
            current = self.predecessors[current]
        return list(reversed(path))

    def get_distance(self, target: VertexT) -> float | None:
        return self.distances.get(target)


def dijkstra[VertexT, WeightT: (int, float)](
    graph: AbstractGraph[VertexT, WeightT], start: VertexT
) -> ShortestPaths[VertexT]:
    if not graph.has_vertex(start):
        raise KeyNotFound(f"Start vertex {start} not found in graph")

    distances: dict[VertexT, float] = {}
    predecessors: dict[VertexT, VertexT | None] = {}
    visited: set[VertexT] = set()
    heap: list[tuple[float, VertexT]] = []

    for vertex in graph.get_vertices():
        distances[vertex] = float("inf")
        predecessors[vertex] = None

    distances[start] = 0.0
    heapq.heappush(heap, (0.0, start))

    while heap:
        current_distance, current = heapq.heappop(heap)

        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph.get_neighbors(current):
            if neighbor in visited:
                continue

            edge_weight = graph.get_edge_weight(current, neighbor)
            distance = current_distance + edge_weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current
                heapq.heappush(heap, (distance, neighbor))

    return ShortestPaths(distances, predecessors)


def a_star[VertexT, WeightT: (int, float)](
    graph: AbstractGraph[VertexT, WeightT],
    start: VertexT,
    goal: VertexT,
    heuristic: Heuristic[VertexT],
) -> PathResult[VertexT] | None:
    if not graph.has_vertex(start) or not graph.has_vertex(goal):
        raise KeyNotFound("Start or goal vertex not found in graph")

    open_set: list[tuple[float, VertexT]] = [(heuristic(start, goal), start)]
    came_from: dict[VertexT, VertexT] = {}
    g_score: dict[VertexT, float] = defaultdict(lambda: float("inf"))
    g_score[start] = 0.0
    f_score: dict[VertexT, float] = defaultdict(lambda: float("inf"))
    f_score[start] = heuristic(start, goal)

    in_open_set: set[VertexT] = {start}

    while open_set:
        current_f, current = heapq.heappop(open_set)
        in_open_set.discard(current)

        if current == goal:
            path: list[VertexT] = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return PathResult(list(reversed(path)), g_score[goal])

        for neighbor in graph.get_neighbors(current):
            tentative_g = g_score[current] + graph.get_edge_weight(current, neighbor)

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in in_open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    in_open_set.add(neighbor)

    return None


def bellman_ford[VertexT, WeightT: (int, float)](
    graph: AbstractGraph[VertexT, WeightT], start: VertexT
) -> ShortestPaths[VertexT] | None:
    if not graph.has_vertex(start):
        raise KeyNotFound(f"Start vertex {start} not found in graph")

    distances: dict[VertexT, float] = {}
    predecessors: dict[VertexT, VertexT | None] = {}

    for vertex in graph.get_vertices():
        distances[vertex] = float("inf")
        predecessors[vertex] = None

    distances[start] = 0.0

    vertices = list(graph.get_vertices())
    edges = list(graph.get_edges())

    for _ in range(len(vertices) - 1):
        for source, target, weight in edges:
            if weight is None:
                continue
            if distances[source] != float("inf") and distances[source] + weight < distances[target]:
                distances[target] = distances[source] + weight
                predecessors[target] = source

    for source, target, weight in edges:
        if weight is None:
            continue
        if distances[source] != float("inf") and distances[source] + weight < distances[target]:
            return None

    return ShortestPaths(distances, predecessors)


def floyd_warshall[VertexT, WeightT: (int, float)](
    graph: AbstractGraph[VertexT, WeightT],
) -> dict[tuple[VertexT, VertexT], float]:
    vertices = list(graph.get_vertices())
    vertex_indices = {vertex: i for i, vertex in enumerate(vertices)}
    n = len(vertices)

    distances: list[list[float]] = [[float("inf")] * n for _ in range(n)]

    for i in range(n):
        distances[i][i] = 0.0

    for source, target, weight in graph.get_edges():
        if weight is not None:
            i, j = vertex_indices[source], vertex_indices[target]
            distances[i][j] = weight

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]

    result: dict[tuple[VertexT, VertexT], float] = {}
    for i, source in enumerate(vertices):
        for j, target in enumerate(vertices):
            result[(source, target)] = distances[i][j]

    return result


def topological_sort(graph: AbstractGraph[VertexT, EdgeWeightT]) -> list[VertexT] | None:
    if not graph.is_directed():
        raise InvalidConfiguration("Topological sort requires a directed graph")

    in_degree: dict[VertexT, int] = {}
    for vertex in graph.get_vertices():
        in_degree[vertex] = graph.get_in_degree(vertex)

    queue: deque[VertexT] = deque([v for v, degree in in_degree.items() if degree == 0])
    result: list[VertexT] = []

    while queue:
        current = queue.popleft()
        result.append(current)

        for neighbor in graph.get_neighbors(current):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != graph.vertex_count():
        return None

    return result


def kruskal_mst[VertexT, WeightT: (int, float)](
    graph: AbstractGraph[VertexT, WeightT],
) -> list[tuple[VertexT, VertexT, float]]:
    if graph.is_directed():
        raise InvalidConfiguration("MST algorithms require an undirected graph")

    class UnionFind:
        def __init__(self, vertices: list[VertexT]) -> None:
            self.parent = {v: v for v in vertices}
            self.rank = dict.fromkeys(vertices, 0)

        def find(self, x: VertexT) -> VertexT:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x: VertexT, y: VertexT) -> bool:
            px, py = self.find(x), self.find(y)
            if px == py:
                return False

            if self.rank[px] < self.rank[py]:
                px, py = py, px

            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True

    vertices = list(graph.get_vertices())
    edges = [(source, target, weight) for source, target, weight in graph.get_edges() if weight is not None]
    edges.sort(key=lambda x: x[2])

    union_find = UnionFind(vertices)
    mst: list[tuple[VertexT, VertexT, float]] = []

    for source, target, weight in edges:
        if union_find.union(source, target):
            mst.append((source, target, weight))
            if len(mst) == len(vertices) - 1:
                break

    return mst


def prim_mst[VertexT, WeightT: (int, float)](
    graph: AbstractGraph[VertexT, WeightT],
) -> list[tuple[VertexT, VertexT, float]]:
    if graph.is_directed():
        raise InvalidConfiguration("MST algorithms require an undirected graph")

    vertices = list(graph.get_vertices())
    if not vertices:
        return []

    mst: list[tuple[VertexT, VertexT, float]] = []
    visited: set[VertexT] = {vertices[0]}
    edges: list[tuple[float, VertexT, VertexT]] = []

    for neighbor in graph.get_neighbors(vertices[0]):
        initial_weight = graph.get_edge_weight(vertices[0], neighbor)
        if initial_weight is not None:
            heapq.heappush(edges, (float(initial_weight), vertices[0], neighbor))

    while edges and len(visited) < len(vertices):
        weight, source, target = heapq.heappop(edges)

        if target in visited:
            continue

        mst.append((source, target, weight))
        visited.add(target)

        for neighbor in graph.get_neighbors(target):
            if neighbor not in visited:
                neighbor_weight = graph.get_edge_weight(target, neighbor)
                if neighbor_weight is not None:
                    heapq.heappush(edges, (float(neighbor_weight), target, neighbor))

    return mst


def strongly_connected_components(graph: AbstractGraph[VertexT, EdgeWeightT]) -> list[list[VertexT]]:
    if not graph.is_directed():
        raise InvalidConfiguration("SCC algorithm requires a directed graph")

    index_counter = [0]
    stack: list[VertexT] = []
    lowlinks: dict[VertexT, int] = {}
    index: dict[VertexT, int] = {}
    on_stack: dict[VertexT, bool] = {}
    components: list[list[VertexT]] = []

    def strongconnect(v: VertexT) -> None:
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in graph.get_neighbors(v):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack.get(w, False):
                lowlinks[v] = min(lowlinks[v], index[w])

        if lowlinks[v] == index[v]:
            component: list[VertexT] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            components.append(component)

    for vertex in graph.get_vertices():
        if vertex not in index:
            strongconnect(vertex)

    return components


def bfs(graph: AbstractGraph[VertexT, EdgeWeightT], start: VertexT) -> Iterator[VertexT]:
    if not graph.has_vertex(start):
        raise KeyNotFound(f"Start vertex {start} not found in graph")

    visited: set[VertexT] = set()
    queue: deque[VertexT] = deque([start])
    visited.add(start)

    while queue:
        current = queue.popleft()
        yield current

        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


def dfs(graph: AbstractGraph[VertexT, EdgeWeightT], start: VertexT) -> Iterator[VertexT]:
    if not graph.has_vertex(start):
        raise KeyNotFound(f"Start vertex {start} not found in graph")

    visited: set[VertexT] = set()
    stack: list[VertexT] = [start]

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            yield current

            stack.extend(neighbor for neighbor in graph.get_neighbors(current) if neighbor not in visited)
