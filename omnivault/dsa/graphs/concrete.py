from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Generic

from rich.repr import Result

from ..core.errors import InvalidConfiguration, KeyNotFound
from ..core.types import EdgeWeightT, VertexT
from .base import AbstractGraph

if TYPE_CHECKING:
    from collections.abc import Iterator


class AdjacencyListGraph(AbstractGraph[VertexT, EdgeWeightT], Generic[VertexT, EdgeWeightT]):  # noqa: UP046
    def __init__(self, directed: bool = True) -> None:
        self._directed = directed
        self._adjacency_list: dict[VertexT, dict[VertexT, EdgeWeightT | None]] = defaultdict(dict)
        self._vertices: set[VertexT] = set()
        self._edge_count = 0

    def add_vertex(self, vertex: VertexT) -> None:
        if vertex not in self._vertices:
            self._vertices.add(vertex)
            self._adjacency_list[vertex] = {}

    def remove_vertex(self, vertex: VertexT) -> None:
        if vertex not in self._vertices:
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        for neighbor in list(self._adjacency_list[vertex].keys()):
            self.remove_edge(vertex, neighbor)

        for v in list(self._vertices):
            if v != vertex and self.has_edge(v, vertex):
                self.remove_edge(v, vertex)

        del self._adjacency_list[vertex]
        self._vertices.remove(vertex)

    def add_edge(self, source: VertexT, target: VertexT, weight: EdgeWeightT | None = None) -> None:
        self.add_vertex(source)
        self.add_vertex(target)

        if not self.has_edge(source, target):
            self._adjacency_list[source][target] = weight
            self._edge_count += 1

            if not self._directed and source != target:
                self._adjacency_list[target][source] = weight
        else:
            self._adjacency_list[source][target] = weight
            if not self._directed and source != target:
                self._adjacency_list[target][source] = weight

    def remove_edge(self, source: VertexT, target: VertexT) -> None:
        if not self.has_edge(source, target):
            raise KeyNotFound(f"Edge ({source}, {target}) not found in graph")

        del self._adjacency_list[source][target]
        self._edge_count -= 1

        if not self._directed and source != target and self.has_edge(target, source):
            del self._adjacency_list[target][source]

    def has_vertex(self, vertex: VertexT) -> bool:
        return vertex in self._vertices

    def has_edge(self, source: VertexT, target: VertexT) -> bool:
        return source in self._adjacency_list and target in self._adjacency_list[source]

    def get_edge_weight(self, source: VertexT, target: VertexT) -> EdgeWeightT:
        if not self.has_edge(source, target):
            raise KeyNotFound(f"Edge ({source}, {target}) not found in graph")

        weight = self._adjacency_list[source][target]
        if weight is None:
            raise InvalidConfiguration(f"Edge ({source}, {target}) has no weight")
        return weight

    def get_neighbors(self, vertex: VertexT) -> Iterator[VertexT]:
        if vertex not in self._vertices:
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        return iter(self._adjacency_list[vertex].keys())

    def get_vertices(self) -> Iterator[VertexT]:
        return iter(self._vertices)

    def get_edges(self) -> Iterator[tuple[VertexT, VertexT, EdgeWeightT | None]]:
        seen_edges: set[tuple[VertexT, VertexT]] = set()

        for source in self._vertices:
            for target, weight in self._adjacency_list[source].items():
                edge = (source, target)
                reverse_edge = (target, source)

                if self._directed or edge not in seen_edges:
                    yield (source, target, weight)
                    if not self._directed:
                        seen_edges.add(edge)
                        seen_edges.add(reverse_edge)

    def vertex_count(self) -> int:
        return len(self._vertices)

    def edge_count(self) -> int:
        return self._edge_count

    def is_directed(self) -> bool:
        return self._directed

    def get_degree(self, vertex: VertexT) -> int:
        if vertex not in self._vertices:
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        return len(self._adjacency_list[vertex])

    def clear(self) -> None:
        self._adjacency_list.clear()
        self._vertices.clear()
        self._edge_count = 0

    def __rich_repr__(self) -> Result:
        yield "directed", self._directed
        yield "vertices", self.vertex_count()
        yield "edges", self.edge_count()

    def to_adjacency_matrix(self) -> AdjacencyMatrixGraph[VertexT, EdgeWeightT]:
        matrix_graph = AdjacencyMatrixGraph[VertexT, EdgeWeightT](directed=self._directed)

        for vertex in self.get_vertices():
            matrix_graph.add_vertex(vertex)

        for source, target, weight in self.get_edges():
            matrix_graph.add_edge(source, target, weight)

        return matrix_graph


class _NoEdge:
    def __repr__(self) -> str:
        return "NO_EDGE"


NO_EDGE = _NoEdge()


class AdjacencyMatrixGraph(AbstractGraph[VertexT, EdgeWeightT], Generic[VertexT, EdgeWeightT]):  # noqa: UP046
    def __init__(self, directed: bool = True) -> None:
        self._directed = directed
        self._vertices: list[VertexT] = []
        self._vertex_indices: dict[VertexT, int] = {}
        self._matrix: list[list[EdgeWeightT | None | _NoEdge]] = []
        self._edge_count = 0

    def _resize_matrix(self, new_size: int) -> None:
        current_size = len(self._matrix)

        for row in self._matrix:
            row.extend([NO_EDGE] * (new_size - current_size))

        for _ in range(new_size - current_size):
            self._matrix.append([NO_EDGE] * new_size)

    def add_vertex(self, vertex: VertexT) -> None:
        if vertex in self._vertex_indices:
            return

        index = len(self._vertices)
        self._vertices.append(vertex)
        self._vertex_indices[vertex] = index
        self._resize_matrix(len(self._vertices))

    def remove_vertex(self, vertex: VertexT) -> None:
        if vertex not in self._vertex_indices:
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        vertex_index = self._vertex_indices[vertex]

        for i in range(len(self._vertices)):
            if self._matrix[vertex_index][i] is not NO_EDGE:
                self._edge_count -= 1
            if i != vertex_index and self._matrix[i][vertex_index] is not NO_EDGE and self._directed:
                self._edge_count -= 1

        self._matrix.pop(vertex_index)
        for row in self._matrix:
            row.pop(vertex_index)

        self._vertices.pop(vertex_index)

        self._vertex_indices = {v: i for i, v in enumerate(self._vertices)}

    def add_edge(self, source: VertexT, target: VertexT, weight: EdgeWeightT | None = None) -> None:
        self.add_vertex(source)
        self.add_vertex(target)

        source_idx = self._vertex_indices[source]
        target_idx = self._vertex_indices[target]

        if self._matrix[source_idx][target_idx] is NO_EDGE:
            self._edge_count += 1

        self._matrix[source_idx][target_idx] = weight

        if not self._directed and source != target:
            self._matrix[target_idx][source_idx] = weight

    def remove_edge(self, source: VertexT, target: VertexT) -> None:
        if not self.has_edge(source, target):
            raise KeyNotFound(f"Edge ({source}, {target}) not found in graph")

        source_idx = self._vertex_indices[source]
        target_idx = self._vertex_indices[target]

        self._matrix[source_idx][target_idx] = NO_EDGE
        self._edge_count -= 1

        if not self._directed and source != target:
            self._matrix[target_idx][source_idx] = NO_EDGE

    def has_vertex(self, vertex: VertexT) -> bool:
        return vertex in self._vertex_indices

    def has_edge(self, source: VertexT, target: VertexT) -> bool:
        if source not in self._vertex_indices or target not in self._vertex_indices:
            return False

        source_idx = self._vertex_indices[source]
        target_idx = self._vertex_indices[target]

        return self._matrix[source_idx][target_idx] is not NO_EDGE

    def get_edge_weight(self, source: VertexT, target: VertexT) -> EdgeWeightT:
        if not self.has_edge(source, target):
            raise KeyNotFound(f"Edge ({source}, {target}) not found in graph")

        source_idx = self._vertex_indices[source]
        target_idx = self._vertex_indices[target]

        weight = self._matrix[source_idx][target_idx]
        if weight is None:
            raise InvalidConfiguration(f"Edge ({source}, {target}) has no weight")
        if isinstance(weight, _NoEdge):
            raise KeyNotFound(f"Edge ({source}, {target}) not found in graph")
        return weight

    def get_neighbors(self, vertex: VertexT) -> Iterator[VertexT]:
        if vertex not in self._vertex_indices:
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        vertex_idx = self._vertex_indices[vertex]

        for i, neighbor in enumerate(self._vertices):
            if self._matrix[vertex_idx][i] is not NO_EDGE:
                yield neighbor

    def get_vertices(self) -> Iterator[VertexT]:
        return iter(self._vertices)

    def get_edges(self) -> Iterator[tuple[VertexT, VertexT, EdgeWeightT | None]]:
        seen_edges: set[tuple[VertexT, VertexT]] = set()

        for i, source in enumerate(self._vertices):
            for j, target in enumerate(self._vertices):
                if self._matrix[i][j] is not NO_EDGE:
                    edge = (source, target)
                    reverse_edge = (target, source)

                    if self._directed or edge not in seen_edges:
                        matrix_value = self._matrix[i][j]
                        if isinstance(matrix_value, _NoEdge):
                            continue
                        yield (source, target, matrix_value)
                        if not self._directed:
                            seen_edges.add(edge)
                            seen_edges.add(reverse_edge)

    def vertex_count(self) -> int:
        return len(self._vertices)

    def edge_count(self) -> int:
        return self._edge_count

    def is_directed(self) -> bool:
        return self._directed

    def get_degree(self, vertex: VertexT) -> int:
        if vertex not in self._vertex_indices:
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        vertex_idx = self._vertex_indices[vertex]
        degree = 0

        for i in range(len(self._vertices)):
            if self._matrix[vertex_idx][i] is not NO_EDGE:
                degree += 1

        return degree

    def clear(self) -> None:
        self._vertices.clear()
        self._vertex_indices.clear()
        self._matrix.clear()
        self._edge_count = 0

    def __rich_repr__(self) -> Result:
        yield "directed", self._directed
        yield "vertices", self.vertex_count()
        yield "edges", self.edge_count()

    def to_adjacency_list(self) -> AdjacencyListGraph[VertexT, EdgeWeightT]:
        list_graph = AdjacencyListGraph[VertexT, EdgeWeightT](directed=self._directed)

        for vertex in self.get_vertices():
            list_graph.add_vertex(vertex)

        for source, target, weight in self.get_edges():
            list_graph.add_edge(source, target, weight)

        return list_graph
