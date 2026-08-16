from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from omnivault.dsa.core.errors import KeyNotFound
from omnivault.dsa.graphs import AbstractGraph, AdjacencyListGraph, AdjacencyMatrixGraph

if TYPE_CHECKING:
    from collections.abc import Callable


class TestGraphImplementations:
    @pytest.fixture(
        params=[
            lambda directed=True: AdjacencyListGraph[str, int](directed=directed),
            lambda directed=True: AdjacencyMatrixGraph[str, int](directed=directed),
        ]
    )
    def graph_factory(self, request: pytest.FixtureRequest) -> Callable[..., AbstractGraph[str, int]]:
        return cast("Callable[..., AbstractGraph[str, int]]", request.param)

    @pytest.fixture
    def empty_graph(self, graph_factory: Callable[..., AbstractGraph[str, int]]) -> AbstractGraph[str, int]:
        return graph_factory()

    @pytest.fixture
    def directed_graph(self, graph_factory: Callable[..., AbstractGraph[str, int]]) -> AbstractGraph[str, int]:
        graph = graph_factory(directed=True)
        graph.add_vertex("A")
        graph.add_vertex("B")
        graph.add_vertex("C")
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", 2)
        graph.add_edge("A", "C", 3)
        return graph

    @pytest.fixture
    def undirected_graph(self, graph_factory: Callable[..., AbstractGraph[str, int]]) -> AbstractGraph[str, int]:
        graph = graph_factory(directed=False)
        graph.add_vertex("A")
        graph.add_vertex("B")
        graph.add_vertex("C")
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", 2)
        return graph

    @pytest.mark.unit
    def test_empty_graph_properties(self, empty_graph: AbstractGraph[str, int]) -> None:
        assert empty_graph.is_empty()
        assert empty_graph.vertex_count() == 0
        assert empty_graph.edge_count() == 0
        assert not empty_graph
        assert list(empty_graph.get_vertices()) == []
        assert list(empty_graph.get_edges()) == []

    @pytest.mark.unit
    def test_add_single_vertex(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_vertex("A")
        assert not empty_graph.is_empty()
        assert empty_graph.vertex_count() == 1
        assert empty_graph.edge_count() == 0
        assert empty_graph.has_vertex("A")
        assert "A" in empty_graph
        assert list(empty_graph.get_vertices()) == ["A"]

    @pytest.mark.unit
    def test_add_duplicate_vertex(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_vertex("A")
        empty_graph.add_vertex("A")
        assert empty_graph.vertex_count() == 1
        assert empty_graph.has_vertex("A")

    @pytest.mark.unit
    def test_remove_vertex(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_vertex("A")
        empty_graph.add_vertex("B")
        empty_graph.add_edge("A", "B", 1)

        empty_graph.remove_vertex("A")
        assert not empty_graph.has_vertex("A")
        assert empty_graph.vertex_count() == 1
        assert empty_graph.edge_count() == 0

    @pytest.mark.unit
    def test_remove_nonexistent_vertex(self, empty_graph: AbstractGraph[str, int]) -> None:
        with pytest.raises(KeyNotFound, match="Vertex .* not found"):
            empty_graph.remove_vertex("nonexistent")

    @pytest.mark.unit
    def test_add_edge_between_existing_vertices(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_vertex("A")
        empty_graph.add_vertex("B")
        empty_graph.add_edge("A", "B", 42)

        assert empty_graph.edge_count() == 1
        assert empty_graph.has_edge("A", "B")
        assert empty_graph.get_edge_weight("A", "B") == 42
        assert list(empty_graph.get_neighbors("A")) == ["B"]

    @pytest.mark.unit
    def test_add_edge_creates_vertices(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_edge("A", "B", 42)

        assert empty_graph.vertex_count() == 2
        assert empty_graph.has_vertex("A")
        assert empty_graph.has_vertex("B")
        assert empty_graph.has_edge("A", "B")

    @pytest.mark.unit
    def test_directed_graph_properties(self, directed_graph: AbstractGraph[str, int]) -> None:
        assert directed_graph.is_directed()
        assert directed_graph.vertex_count() == 3
        assert directed_graph.edge_count() == 3

        assert directed_graph.has_edge("A", "B")
        assert not directed_graph.has_edge("B", "A")

        assert directed_graph.get_degree("A") == 2
        assert directed_graph.get_out_degree("A") == 2
        assert directed_graph.get_in_degree("A") == 0

    @pytest.mark.unit
    def test_undirected_graph_properties(self, undirected_graph: AbstractGraph[str, int]) -> None:
        assert not undirected_graph.is_directed()
        assert undirected_graph.vertex_count() == 3
        assert undirected_graph.edge_count() == 2

        assert undirected_graph.has_edge("A", "B")
        assert undirected_graph.has_edge("B", "A")

        assert undirected_graph.get_degree("B") == 2
        assert undirected_graph.get_in_degree("B") == 2
        assert undirected_graph.get_out_degree("B") == 2

    @pytest.mark.unit
    def test_remove_edge(self, directed_graph: AbstractGraph[str, int]) -> None:
        initial_edge_count = directed_graph.edge_count()
        directed_graph.remove_edge("A", "B")

        assert directed_graph.edge_count() == initial_edge_count - 1
        assert not directed_graph.has_edge("A", "B")
        assert directed_graph.get_degree("A") == 1

    @pytest.mark.unit
    def test_remove_nonexistent_edge(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_vertex("A")
        empty_graph.add_vertex("B")

        with pytest.raises(KeyNotFound, match="Edge .* not found"):
            empty_graph.remove_edge("A", "B")

    @pytest.mark.unit
    def test_get_edge_weight_nonexistent_edge(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_vertex("A")
        empty_graph.add_vertex("B")

        with pytest.raises(KeyNotFound, match="Edge .* not found"):
            empty_graph.get_edge_weight("A", "B")

    @pytest.mark.unit
    def test_get_neighbors_nonexistent_vertex(self, empty_graph: AbstractGraph[str, int]) -> None:
        with pytest.raises(KeyNotFound, match="Vertex .* not found"):
            list(empty_graph.get_neighbors("nonexistent"))

    @pytest.mark.unit
    def test_get_degree_nonexistent_vertex(self, empty_graph: AbstractGraph[str, int]) -> None:
        with pytest.raises(KeyNotFound, match="Vertex .* not found"):
            empty_graph.get_degree("nonexistent")

    @pytest.mark.unit
    def test_edges_iterator(self, directed_graph: AbstractGraph[str, int]) -> None:
        edges = list(directed_graph.get_edges())
        assert len(edges) == 3

        edge_set = {(source, target) for source, target, _ in edges}
        expected_edges = {("A", "B"), ("B", "C"), ("A", "C")}
        assert edge_set == expected_edges

    @pytest.mark.unit
    def test_vertices_iterator(self, directed_graph: AbstractGraph[str, int]) -> None:
        vertices = set(directed_graph.get_vertices())
        assert vertices == {"A", "B", "C"}

    @pytest.mark.unit
    def test_clear_graph(self, directed_graph: AbstractGraph[str, int]) -> None:
        directed_graph.clear()
        assert directed_graph.is_empty()
        assert directed_graph.vertex_count() == 0
        assert directed_graph.edge_count() == 0

    @pytest.mark.unit
    def test_graph_density(self, empty_graph: AbstractGraph[str, int]) -> None:
        assert empty_graph.density() == 0.0

        empty_graph.add_vertex("A")
        assert empty_graph.density() == 0.0

        empty_graph.add_vertex("B")
        empty_graph.add_edge("A", "B")

        if empty_graph.is_directed():
            assert empty_graph.density() == 0.5
        else:
            assert empty_graph.density() == 1.0

    @pytest.mark.unit
    def test_self_loop(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_edge("A", "A", 5)

        assert empty_graph.has_vertex("A")
        assert empty_graph.has_edge("A", "A")
        assert empty_graph.get_edge_weight("A", "A") == 5
        assert empty_graph.vertex_count() == 1
        assert empty_graph.edge_count() == 1

    @pytest.mark.unit
    def test_is_connected(self, directed_graph: AbstractGraph[str, int]) -> None:
        assert directed_graph.is_connected("A", "B")
        assert not directed_graph.is_connected("B", "A")

    @pytest.mark.unit
    def test_repr_output(self, graph_factory: Callable[..., AbstractGraph[str, int]]) -> None:
        graph = graph_factory()
        graph.add_vertex("A")
        graph.add_edge("A", "B", 1)

        repr_str = repr(graph)
        assert "vertices=2" in repr_str
        assert "edges=1" in repr_str

    @pytest.mark.parametrize("size", [10, 50, 100])
    def test_large_graph_operations(self, graph_factory: Callable[..., AbstractGraph[str, int]], size: int) -> None:
        graph = graph_factory()

        for i in range(size):
            graph.add_vertex(f"v{i}")

        for i in range(size - 1):
            graph.add_edge(f"v{i}", f"v{i + 1}", i)

        assert graph.vertex_count() == size
        assert graph.edge_count() == size - 1

        for i in range(size - 1):
            assert graph.has_edge(f"v{i}", f"v{i + 1}")
            assert graph.get_edge_weight(f"v{i}", f"v{i + 1}") == i

    @pytest.mark.edge_case
    def test_update_edge_weight(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_edge("A", "B", 1)
        assert empty_graph.get_edge_weight("A", "B") == 1

        empty_graph.add_edge("A", "B", 2)
        assert empty_graph.get_edge_weight("A", "B") == 2
        assert empty_graph.edge_count() == 1

    @pytest.mark.edge_case
    def test_edge_without_weight(self, empty_graph: AbstractGraph[str, int]) -> None:
        empty_graph.add_edge("A", "B", None)

        assert empty_graph.has_edge("A", "B")

        with pytest.raises(ValueError, match="has no weight"):
            empty_graph.get_edge_weight("A", "B")

    @pytest.mark.benchmark
    def test_performance_with_many_vertices(self, graph_factory: Callable[..., AbstractGraph[str, int]]) -> None:
        graph = graph_factory()

        vertices = [f"vertex_{i}" for i in range(1000)]

        for vertex in vertices:
            graph.add_vertex(vertex)

        for i in range(len(vertices) - 1):
            graph.add_edge(vertices[i], vertices[i + 1], i)

        assert graph.vertex_count() == 1000
        assert graph.edge_count() == 999


class TestAdjacencyListGraphSpecific:
    @pytest.mark.unit
    def test_to_adjacency_matrix_conversion(self) -> None:
        list_graph = AdjacencyListGraph[str, int](directed=True)
        list_graph.add_edge("A", "B", 1)
        list_graph.add_edge("B", "C", 2)

        matrix_graph = list_graph.to_adjacency_matrix()

        assert matrix_graph.vertex_count() == list_graph.vertex_count()
        assert matrix_graph.edge_count() == list_graph.edge_count()
        assert matrix_graph.is_directed() == list_graph.is_directed()

        for source, target, weight in list_graph.get_edges():
            assert matrix_graph.has_edge(source, target)
            assert matrix_graph.get_edge_weight(source, target) == weight

    @pytest.mark.unit
    def test_neighbors_empty_for_isolated_vertex(self) -> None:
        graph = AdjacencyListGraph[str, int]()
        graph.add_vertex("isolated")

        assert list(graph.get_neighbors("isolated")) == []
        assert graph.get_degree("isolated") == 0


class TestAdjacencyMatrixGraphSpecific:
    @pytest.mark.unit
    def test_to_adjacency_list_conversion(self) -> None:
        matrix_graph = AdjacencyMatrixGraph[str, int](directed=False)
        matrix_graph.add_edge("A", "B", 1)
        matrix_graph.add_edge("B", "C", 2)

        list_graph = matrix_graph.to_adjacency_list()

        assert list_graph.vertex_count() == matrix_graph.vertex_count()
        assert list_graph.edge_count() == matrix_graph.edge_count()
        assert list_graph.is_directed() == matrix_graph.is_directed()

        for source, target, weight in matrix_graph.get_edges():
            assert list_graph.has_edge(source, target)
            assert list_graph.get_edge_weight(source, target) == weight

    @pytest.mark.unit
    def test_matrix_resize_on_vertex_addition(self) -> None:
        graph = AdjacencyMatrixGraph[str, int]()

        len(graph._matrix)

        graph.add_vertex("A")
        graph.add_vertex("B")
        graph.add_vertex("C")

        assert len(graph._matrix) == 3
        assert all(len(row) == 3 for row in graph._matrix)

    @pytest.mark.unit
    def test_vertex_removal_updates_indices(self) -> None:
        graph = AdjacencyMatrixGraph[str, int]()
        graph.add_vertex("A")
        graph.add_vertex("B")
        graph.add_vertex("C")
        graph.add_edge("A", "B", 1)
        graph.add_edge("B", "C", 2)

        graph.remove_vertex("B")

        assert graph.vertex_count() == 2
        assert graph.has_vertex("A")
        assert graph.has_vertex("C")
        assert not graph.has_vertex("B")
        assert not graph.has_edge("A", "C")
        assert graph.edge_count() == 0

    @pytest.mark.unit
    def test_neighbors_empty_for_isolated_vertex(self) -> None:
        graph = AdjacencyMatrixGraph[str, int]()
        graph.add_vertex("isolated")

        assert list(graph.get_neighbors("isolated")) == []
        assert graph.get_degree("isolated") == 0
