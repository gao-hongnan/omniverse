from __future__ import annotations

from collections import defaultdict, deque
from typing import Generic

from ..core.types import VertexT


class FlowNetwork(Generic[VertexT]):  # noqa: UP046
    def __init__(self) -> None:
        self.graph: dict[VertexT, dict[VertexT, float]] = defaultdict(dict)
        self.vertices: set[VertexT] = set()

    def add_edge(self, source: VertexT, sink: VertexT, capacity: float) -> None:
        self.vertices.add(source)
        self.vertices.add(sink)
        self.graph[source][sink] = capacity
        if sink not in self.graph[source]:
            self.graph[sink][source] = 0.0

    def get_capacity(self, source: VertexT, sink: VertexT) -> float:
        return self.graph[source].get(sink, 0.0)

    def set_capacity(self, source: VertexT, sink: VertexT, capacity: float) -> None:
        self.graph[source][sink] = capacity

    def get_neighbors(self, vertex: VertexT) -> list[VertexT]:
        return list(self.graph[vertex].keys())


class MaxFlowResult(Generic[VertexT]):  # noqa: UP046
    def __init__(self, max_flow_value: float, flow_graph: dict[VertexT, dict[VertexT, float]]) -> None:
        self.max_flow_value: float = max_flow_value
        self.flow_graph: dict[VertexT, dict[VertexT, float]] = flow_graph

    def get_flow(self, source: VertexT, sink: VertexT) -> float:
        return self.flow_graph.get(source, {}).get(sink, 0.0)

    def __repr__(self) -> str:
        return f"MaxFlowResult(max_flow={self.max_flow_value})"


def ford_fulkerson(network: FlowNetwork[VertexT], source: VertexT, sink: VertexT) -> MaxFlowResult[VertexT]:
    residual_graph: dict[VertexT, dict[VertexT, float]] = _create_residual_graph(network)
    flow_graph: dict[VertexT, dict[VertexT, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))
    max_flow_value: float = 0.0

    while True:
        path, bottleneck = _find_augmenting_path_dfs(residual_graph, source, sink)
        if not path:
            break

        max_flow_value += bottleneck

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow_graph[u][v] += bottleneck
            flow_graph[v][u] -= bottleneck
            residual_graph[u][v] -= bottleneck
            residual_graph[v][u] += bottleneck

    return MaxFlowResult(max_flow_value, dict(flow_graph))


def edmonds_karp(network: FlowNetwork[VertexT], source: VertexT, sink: VertexT) -> MaxFlowResult[VertexT]:
    residual_graph: dict[VertexT, dict[VertexT, float]] = _create_residual_graph(network)
    flow_graph: dict[VertexT, dict[VertexT, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))
    max_flow_value: float = 0.0

    while True:
        path, bottleneck = _find_augmenting_path_bfs(residual_graph, source, sink)
        if not path:
            break

        max_flow_value += bottleneck

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            flow_graph[u][v] += bottleneck
            flow_graph[v][u] -= bottleneck
            residual_graph[u][v] -= bottleneck
            residual_graph[v][u] += bottleneck

    return MaxFlowResult(max_flow_value, dict(flow_graph))


def dinic(network: FlowNetwork[VertexT], source: VertexT, sink: VertexT) -> MaxFlowResult[VertexT]:
    residual_graph: dict[VertexT, dict[VertexT, float]] = _create_residual_graph(network)
    flow_graph: dict[VertexT, dict[VertexT, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))
    max_flow_value: float = 0.0

    while True:
        level_graph = _build_level_graph(residual_graph, source, sink)
        if sink not in level_graph:
            break

        while True:
            path, bottleneck = _find_blocking_flow_path(level_graph, residual_graph, source, sink)
            if not path:
                break

            max_flow_value += bottleneck

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                flow_graph[u][v] += bottleneck
                flow_graph[v][u] -= bottleneck
                residual_graph[u][v] -= bottleneck
                residual_graph[v][u] += bottleneck

    return MaxFlowResult(max_flow_value, dict(flow_graph))


def push_relabel(network: FlowNetwork[VertexT], source: VertexT, sink: VertexT) -> MaxFlowResult[VertexT]:
    height: dict[VertexT, int] = dict.fromkeys(network.vertices, 0)
    excess: dict[VertexT, float] = dict.fromkeys(network.vertices, 0.0)
    residual_graph: dict[VertexT, dict[VertexT, float]] = _create_residual_graph(network)
    flow_graph: dict[VertexT, dict[VertexT, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))

    height[source] = len(network.vertices)

    for neighbor in network.get_neighbors(source):
        capacity = network.get_capacity(source, neighbor)
        if capacity > 0:
            flow_graph[source][neighbor] = capacity
            flow_graph[neighbor][source] = -capacity
            excess[neighbor] += capacity
            excess[source] -= capacity
            residual_graph[source][neighbor] -= capacity
            residual_graph[neighbor][source] += capacity

    active_vertices = [v for v in network.vertices if v != source and v != sink and excess[v] > 0]

    while active_vertices:
        vertex = active_vertices.pop()

        if excess[vertex] > 0:
            pushed = _push(vertex, residual_graph, flow_graph, height, excess)
            if not pushed:
                _relabel(vertex, residual_graph, height)
                active_vertices.append(vertex)
            else:
                if excess[vertex] > 0 and vertex not in active_vertices:
                    active_vertices.append(vertex)
                for neighbor in network.get_neighbors(vertex):
                    if (
                        neighbor != source
                        and neighbor != sink
                        and excess[neighbor] > 0
                        and neighbor not in active_vertices
                    ):
                        active_vertices.append(neighbor)

    return MaxFlowResult(excess[sink], dict(flow_graph))


def _create_residual_graph(network: FlowNetwork[VertexT]) -> dict[VertexT, dict[VertexT, float]]:
    residual: dict[VertexT, dict[VertexT, float]] = defaultdict(lambda: defaultdict(lambda: 0.0))

    for source in network.vertices:
        for sink in network.get_neighbors(source):
            capacity = network.get_capacity(source, sink)
            residual[source][sink] = capacity
            if source not in residual[sink]:
                residual[sink][source] = 0.0

    return residual


def _find_augmenting_path_dfs(
    residual_graph: dict[VertexT, dict[VertexT, float]], source: VertexT, sink: VertexT
) -> tuple[list[VertexT], float]:
    visited: set[VertexT] = set()
    path: list[VertexT] = []

    def dfs(current: VertexT) -> float | None:
        if current == sink:
            path.append(current)
            return float("inf")

        visited.add(current)
        path.append(current)

        for neighbor in residual_graph[current]:
            capacity = residual_graph[current][neighbor]
            if neighbor not in visited and capacity > 0:
                bottleneck = dfs(neighbor)
                if bottleneck is not None:
                    return min(capacity, bottleneck)

        path.pop()
        return None

    bottleneck = dfs(source)
    if bottleneck is None:
        return [], 0.0

    return path, bottleneck


def _find_augmenting_path_bfs(
    residual_graph: dict[VertexT, dict[VertexT, float]], source: VertexT, sink: VertexT
) -> tuple[list[VertexT], float]:
    parent: dict[VertexT, VertexT | None] = {source: None}
    visited: set[VertexT] = {source}
    queue: deque[VertexT] = deque([source])

    while queue:
        current = queue.popleft()

        for neighbor in residual_graph[current]:
            capacity = residual_graph[current][neighbor]
            if neighbor not in visited and capacity > 0:
                parent[neighbor] = current
                visited.add(neighbor)
                queue.append(neighbor)

                if neighbor == sink:
                    path = []
                    bottleneck = float("inf")
                    node: VertexT | None = sink

                    while node is not None:
                        path.append(node)
                        parent_node = parent[node]
                        if parent_node is not None:
                            capacity = residual_graph[parent_node][node]
                            bottleneck = min(bottleneck, capacity)
                        node = parent_node

                    return list(reversed(path)), bottleneck

    return [], 0.0


def _build_level_graph(
    residual_graph: dict[VertexT, dict[VertexT, float]], source: VertexT, _sink: VertexT
) -> dict[VertexT, int]:
    level: dict[VertexT, int] = {}
    queue: deque[VertexT] = deque([source])
    level[source] = 0

    while queue:
        current = queue.popleft()

        for neighbor in residual_graph[current]:
            capacity = residual_graph[current][neighbor]
            if neighbor not in level and capacity > 0:
                level[neighbor] = level[current] + 1
                queue.append(neighbor)

    return level


def _find_blocking_flow_path(
    level_graph: dict[VertexT, int], residual_graph: dict[VertexT, dict[VertexT, float]], source: VertexT, sink: VertexT
) -> tuple[list[VertexT], float]:
    path: list[VertexT] = []
    visited: set[VertexT] = set()

    def dfs(current: VertexT) -> float | None:
        if current == sink:
            path.append(current)
            return float("inf")

        visited.add(current)
        path.append(current)

        for neighbor in residual_graph[current]:
            capacity = residual_graph[current][neighbor]
            if (
                neighbor not in visited
                and neighbor in level_graph
                and level_graph[neighbor] == level_graph[current] + 1
                and capacity > 0
            ):
                bottleneck = dfs(neighbor)
                if bottleneck is not None:
                    return min(capacity, bottleneck)

        path.pop()
        return None

    bottleneck = dfs(source)
    if bottleneck is None:
        return [], 0.0

    return path, bottleneck


def _push(
    vertex: VertexT,
    residual_graph: dict[VertexT, dict[VertexT, float]],
    flow_graph: dict[VertexT, dict[VertexT, float]],
    height: dict[VertexT, int],
    excess: dict[VertexT, float],
) -> bool:
    for neighbor in residual_graph[vertex]:
        capacity = residual_graph[vertex][neighbor]
        if capacity > 0 and height[vertex] == height[neighbor] + 1:
            push_amount = min(excess[vertex], capacity)

            flow_graph[vertex][neighbor] += push_amount
            flow_graph[neighbor][vertex] -= push_amount
            residual_graph[vertex][neighbor] -= push_amount
            residual_graph[neighbor][vertex] += push_amount
            excess[vertex] -= push_amount
            excess[neighbor] += push_amount

            return True

    return False


def _relabel(vertex: VertexT, residual_graph: dict[VertexT, dict[VertexT, float]], height: dict[VertexT, int]) -> None:
    min_height = float("inf")

    for neighbor in residual_graph[vertex]:
        capacity = residual_graph[vertex][neighbor]
        if capacity > 0:
            min_height = min(min_height, height[neighbor])

    if min_height != float("inf"):
        height[vertex] = int(min_height) + 1


class MinCostMaxFlow(Generic[VertexT]):  # noqa: UP046
    def __init__(self) -> None:
        self.graph: dict[VertexT, list[tuple[VertexT, float, float]]] = defaultdict(list)
        self.vertices: set[VertexT] = set()

    def add_edge(self, source: VertexT, sink: VertexT, capacity: float, cost: float) -> None:
        self.vertices.add(source)
        self.vertices.add(sink)
        self.graph[source].append((sink, capacity, cost))
        self.graph[sink].append((source, 0.0, -cost))

    def min_cost_max_flow(self, source: VertexT, sink: VertexT) -> tuple[float, float]:
        max_flow: float = 0.0
        min_cost: float = 0.0

        while True:
            distance, parent, edge_id = self._spfa(source, sink)
            if distance[sink] == float("inf"):
                break

            flow: float = float("inf")
            current = sink

            while current != source:
                prev = parent[current]
                edge_idx = edge_id[current]
                capacity = self.graph[prev][edge_idx][1]
                flow = min(flow, capacity)
                current = prev

            max_flow += flow
            min_cost += flow * distance[sink]

            current = sink
            while current != source:
                prev = parent[current]
                edge_idx = edge_id[current]

                dest, capacity, cost = self.graph[prev][edge_idx]
                self.graph[prev][edge_idx] = (dest, capacity - flow, cost)

                reverse_edge_idx = None
                for i, (dest_rev, _cap_rev, _cost_rev) in enumerate(self.graph[current]):
                    if dest_rev == prev:
                        reverse_edge_idx = i
                        break

                if reverse_edge_idx is not None:
                    dest_rev, cap_rev, cost_rev = self.graph[current][reverse_edge_idx]
                    self.graph[current][reverse_edge_idx] = (dest_rev, cap_rev + flow, cost_rev)

                current = prev

        return max_flow, min_cost

    def _spfa(
        self, source: VertexT, _sink: VertexT
    ) -> tuple[dict[VertexT, float], dict[VertexT, VertexT], dict[VertexT, int]]:
        distance: dict[VertexT, float] = {vertex: float("inf") for vertex in self.vertices}
        parent: dict[VertexT, VertexT] = {}
        edge_id: dict[VertexT, int] = {}
        in_queue: dict[VertexT, bool] = dict.fromkeys(self.vertices, False)

        distance[source] = 0.0
        queue: deque[VertexT] = deque([source])
        in_queue[source] = True

        while queue:
            current = queue.popleft()
            in_queue[current] = False

            for i, (neighbor, capacity, cost) in enumerate(self.graph[current]):
                if capacity > 0 and distance[current] + cost < distance[neighbor]:
                    distance[neighbor] = distance[current] + cost
                    parent[neighbor] = current
                    edge_id[neighbor] = i

                    if not in_queue[neighbor]:
                        queue.append(neighbor)
                        in_queue[neighbor] = True

        return distance, parent, edge_id


def bipartite_matching_max_flow(
    left_vertices: list[VertexT], right_vertices: list[VertexT], edges: list[tuple[VertexT, VertexT]]
) -> list[tuple[VertexT, VertexT]]:
    network: FlowNetwork[VertexT | str] = FlowNetwork()
    source: VertexT | str = "source"
    sink: VertexT | str = "sink"

    for left in left_vertices:
        network.add_edge(source, left, 1)

    for right in right_vertices:
        network.add_edge(right, sink, 1)

    for left, right in edges:
        network.add_edge(left, right, 1)

    result = ford_fulkerson(network, source, sink)

    matching: list[tuple[VertexT, VertexT]] = [
        (left, right) for left in left_vertices for right in right_vertices if result.get_flow(left, right) == 1
    ]

    return matching
