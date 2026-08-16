from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from omnivault.dsa.core.errors import KeyNotFound
from omnivault.dsa.graphs import (
    AdjacencyListGraph,
    AdjacencyMatrixGraph,
    a_star,
    bellman_ford,
    bfs,
    dfs,
    dijkstra,
    floyd_warshall,
    kruskal_mst,
    prim_mst,
    strongly_connected_components,
    topological_sort,
)


class TestShortestPathAlgorithms:
    @pytest.fixture
    def weighted_graph(self) -> AdjacencyListGraph[str, int]:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "B", 4)
        graph.add_edge("A", "C", 2)
        graph.add_edge("B", "C", 1)
        graph.add_edge("B", "D", 5)
        graph.add_edge("C", "D", 8)
        graph.add_edge("C", "E", 10)
        graph.add_edge("D", "E", 2)
        return graph

    @pytest.fixture
    def negative_weight_graph(self) -> AdjacencyListGraph[str, int]:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "B", 1)
        graph.add_edge("A", "C", 4)
        graph.add_edge("B", "C", -3)
        graph.add_edge("B", "D", 2)
        graph.add_edge("B", "E", 2)
        graph.add_edge("D", "B", 1)
        graph.add_edge("D", "C", 5)
        graph.add_edge("E", "D", -3)
        return graph

    @pytest.mark.unit
    def test_dijkstra_shortest_paths(self, weighted_graph: AdjacencyListGraph[str, int]) -> None:
        result = dijkstra(weighted_graph, "A")

        assert result.get_distance("A") == 0
        assert result.get_distance("B") == 4
        assert result.get_distance("C") == 2
        assert result.get_distance("D") == 9
        assert result.get_distance("E") == 11

        path_to_e = result.get_path("E")
        assert path_to_e == ["A", "B", "D", "E"]

    @pytest.mark.unit
    def test_bellman_ford_with_negative_weights(self, negative_weight_graph: AdjacencyListGraph[str, int]) -> None:
        result = bellman_ford(negative_weight_graph, "A")
        assert result is not None

        assert result.get_distance("A") == 0
        assert result.get_distance("B") == 1
        assert result.get_distance("C") == -2
        assert result.get_distance("D") == 0
        assert result.get_distance("E") == 3

    @pytest.mark.unit
    def test_bellman_ford_detects_negative_cycle(self) -> None:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", -3)
        graph.add_edge("C", "A", 1)

        result = bellman_ford(graph, "A")
        assert result is None

    @pytest.mark.unit
    def test_floyd_warshall_all_pairs(self, weighted_graph: AdjacencyListGraph[str, int]) -> None:
        distances = floyd_warshall(weighted_graph)

        assert distances[("A", "A")] == 0
        assert distances[("A", "B")] == 4
        assert distances[("A", "C")] == 2
        assert distances[("A", "D")] == 9
        assert distances[("A", "E")] == 11

    @pytest.mark.unit
    def test_a_star_pathfinding(self, weighted_graph: AdjacencyListGraph[str, int]) -> None:
        def manhattan_heuristic(current: str, goal: str) -> int:
            coordinates = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (2, 0), "E": (2, 1)}
            current_pos = coordinates[current]
            goal_pos = coordinates[goal]
            return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])

        result = a_star(weighted_graph, "A", "E", manhattan_heuristic)
        assert result is not None
        assert result.distance == 11
        assert result.path == ["A", "B", "D", "E"]

    @pytest.mark.unit
    def test_a_star_no_path(self) -> None:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "B", 1)
        graph.add_vertex("C")

        def zero_heuristic(_current: str, _goal: str) -> int:
            return 0

        result = a_star(graph, "A", "C", zero_heuristic)
        assert result is None


class TestTopologicalSort:
    @pytest.fixture
    def dag(self) -> AdjacencyListGraph[str, int]:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "C", 1)
        graph.add_edge("B", "C", 1)
        graph.add_edge("B", "D", 1)
        graph.add_edge("C", "E", 1)
        graph.add_edge("D", "F", 1)
        graph.add_edge("E", "F", 1)
        return graph

    @pytest.mark.unit
    def test_topological_sort_valid_dag(self, dag: AdjacencyListGraph[str, int]) -> None:
        result = topological_sort(dag)
        assert result is not None
        assert len(result) == 6

        positions = {vertex: i for i, vertex in enumerate(result)}
        assert positions["A"] < positions["C"]
        assert positions["B"] < positions["C"]
        assert positions["B"] < positions["D"]
        assert positions["C"] < positions["E"]
        assert positions["D"] < positions["F"]
        assert positions["E"] < positions["F"]

    @pytest.mark.unit
    def test_topological_sort_with_cycle(self) -> None:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", 1)
        graph.add_edge("C", "A", 1)

        result = topological_sort(graph)
        assert result is None

    @pytest.mark.unit
    def test_topological_sort_undirected_raises_error(self) -> None:
        graph = AdjacencyListGraph[str, int](directed=False)
        graph.add_edge("A", "B", 1)

        with pytest.raises(ValueError, match="Topological sort requires a directed graph"):
            topological_sort(graph)


class TestMinimumSpanningTree:
    @pytest.fixture
    def weighted_undirected_graph(self) -> AdjacencyListGraph[str, int]:
        graph = AdjacencyListGraph[str, int](directed=False)
        graph.add_edge("A", "B", 4)
        graph.add_edge("A", "H", 8)
        graph.add_edge("B", "C", 8)
        graph.add_edge("B", "H", 11)
        graph.add_edge("C", "D", 7)
        graph.add_edge("C", "F", 4)
        graph.add_edge("C", "I", 2)
        graph.add_edge("D", "E", 9)
        graph.add_edge("D", "F", 14)
        graph.add_edge("E", "F", 10)
        graph.add_edge("F", "G", 2)
        graph.add_edge("G", "H", 1)
        graph.add_edge("G", "I", 6)
        graph.add_edge("H", "I", 7)
        return graph

    @pytest.mark.unit
    def test_kruskal_mst(self, weighted_undirected_graph: AdjacencyListGraph[str, int]) -> None:
        mst = kruskal_mst(weighted_undirected_graph)

        total_weight = sum(weight for _, _, weight in mst)
        assert total_weight == 37
        assert len(mst) == 8

    @pytest.mark.unit
    def test_prim_mst(self, weighted_undirected_graph: AdjacencyListGraph[str, int]) -> None:
        mst = prim_mst(weighted_undirected_graph)

        total_weight = sum(weight for _, _, weight in mst)
        assert total_weight == 37
        assert len(mst) == 8

    @pytest.mark.unit
    def test_mst_with_directed_graph_raises_error(self) -> None:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "B", 1)

        with pytest.raises(ValueError, match="MST algorithms require an undirected graph"):
            kruskal_mst(graph)

        with pytest.raises(ValueError, match="MST algorithms require an undirected graph"):
            prim_mst(graph)


class TestStronglyConnectedComponents:
    @pytest.fixture
    def scc_graph(self) -> AdjacencyListGraph[str, int]:
        graph = AdjacencyListGraph[str, int](directed=True)
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", 1)
        graph.add_edge("C", "A", 1)
        graph.add_edge("B", "D", 1)
        graph.add_edge("D", "E", 1)
        graph.add_edge("E", "F", 1)
        graph.add_edge("F", "D", 1)
        graph.add_edge("G", "F", 1)
        graph.add_edge("G", "H", 1)
        graph.add_edge("H", "I", 1)
        graph.add_edge("I", "G", 1)
        return graph

    @pytest.mark.unit
    def test_strongly_connected_components(self, scc_graph: AdjacencyListGraph[str, int]) -> None:
        components = strongly_connected_components(scc_graph)

        assert len(components) == 3
        component_sets = [set(component) for component in components]

        assert {"A", "B", "C"} in component_sets
        assert {"D", "E", "F"} in component_sets
        assert {"G", "H", "I"} in component_sets

    @pytest.mark.unit
    def test_scc_with_undirected_graph_raises_error(self) -> None:
        graph = AdjacencyListGraph[str, int](directed=False)
        graph.add_edge("A", "B", 1)

        with pytest.raises(ValueError, match="SCC algorithm requires a directed graph"):
            strongly_connected_components(graph)


class TestGraphTraversal:
    @pytest.fixture
    def simple_graph(self) -> AdjacencyListGraph[str, int]:
        graph = AdjacencyListGraph[str, int](directed=False)
        graph.add_edge("A", "B", 1)
        graph.add_edge("A", "C", 1)
        graph.add_edge("B", "D", 1)
        graph.add_edge("C", "E", 1)
        graph.add_edge("D", "F", 1)
        graph.add_edge("E", "F", 1)
        return graph

    @pytest.mark.unit
    def test_bfs_traversal(self, simple_graph: AdjacencyListGraph[str, int]) -> None:
        result = list(bfs(simple_graph, "A"))

        assert result[0] == "A"
        assert "B" in result[:3]
        assert "C" in result[:3]
        assert len(result) == 6

    @pytest.mark.unit
    def test_dfs_traversal(self, simple_graph: AdjacencyListGraph[str, int]) -> None:
        result = list(dfs(simple_graph, "A"))

        assert result[0] == "A"
        assert len(result) == 6

    @pytest.mark.unit
    def test_bfs_invalid_start_vertex(self, simple_graph: AdjacencyListGraph[str, int]) -> None:
        with pytest.raises(KeyNotFound, match="Start vertex Z not found in graph"):
            list(bfs(simple_graph, "Z"))

    @pytest.mark.unit
    def test_dfs_invalid_start_vertex(self, simple_graph: AdjacencyListGraph[str, int]) -> None:
        with pytest.raises(KeyNotFound, match="Start vertex Z not found in graph"):
            list(dfs(simple_graph, "Z"))


class TestGraphAlgorithmsWithDifferentImplementations:
    @pytest.fixture(
        params=[
            lambda: AdjacencyListGraph[int, int](directed=True),
            lambda: AdjacencyMatrixGraph[int, int](directed=True),
        ]
    )
    def graph_factory(self, request: pytest.FixtureRequest) -> Callable[[], AdjacencyListGraph[int, int]]:
        return cast("Callable[[], AdjacencyListGraph[int, int]]", request.param)

    @pytest.mark.unit
    def test_dijkstra_on_different_implementations(
        self, graph_factory: Callable[[], AdjacencyListGraph[int, int]]
    ) -> None:
        graph = graph_factory()
        graph.add_edge(1, 2, 10)
        graph.add_edge(1, 3, 5)
        graph.add_edge(2, 4, 1)
        graph.add_edge(3, 2, 3)
        graph.add_edge(3, 4, 8)

        result = dijkstra(graph, 1)

        assert result.get_distance(1) == 0
        assert result.get_distance(2) == 8
        assert result.get_distance(3) == 5
        assert result.get_distance(4) == 9
