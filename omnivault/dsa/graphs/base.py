from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..core.errors import KeyNotFound

if TYPE_CHECKING:
    from collections.abc import Iterator


# ABC retained: shared default methods (density, get_in_degree, get_out_degree, copy) per rules/python-typings.md exception clause
class AbstractGraph[VertexT, EdgeWeightT](ABC):
    @abstractmethod
    def add_vertex(self, vertex: VertexT) -> None: ...

    @abstractmethod
    def remove_vertex(self, vertex: VertexT) -> None: ...

    @abstractmethod
    def add_edge(self, source: VertexT, target: VertexT, weight: EdgeWeightT | None = None) -> None: ...

    @abstractmethod
    def remove_edge(self, source: VertexT, target: VertexT) -> None: ...

    @abstractmethod
    def has_vertex(self, vertex: VertexT) -> bool: ...

    @abstractmethod
    def has_edge(self, source: VertexT, target: VertexT) -> bool: ...

    @abstractmethod
    def get_edge_weight(self, source: VertexT, target: VertexT) -> EdgeWeightT: ...

    @abstractmethod
    def get_neighbors(self, vertex: VertexT) -> Iterator[VertexT]: ...

    @abstractmethod
    def get_vertices(self) -> Iterator[VertexT]: ...

    @abstractmethod
    def get_edges(self) -> Iterator[tuple[VertexT, VertexT, EdgeWeightT | None]]: ...

    @abstractmethod
    def vertex_count(self) -> int: ...

    @abstractmethod
    def edge_count(self) -> int: ...

    @abstractmethod
    def is_directed(self) -> bool: ...

    @abstractmethod
    def get_degree(self, vertex: VertexT) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    def __bool__(self) -> bool:
        return self.vertex_count() > 0

    def __repr__(self) -> str:
        vertices = list(self.get_vertices())
        edges = list(self.get_edges())
        return f"{self.__class__.__name__}(vertices={len(vertices)}, edges={len(edges)})"

    def __contains__(self, vertex: VertexT) -> bool:
        return self.has_vertex(vertex)

    def is_empty(self) -> bool:
        return self.vertex_count() == 0

    def density(self) -> float:
        v = self.vertex_count()
        if v <= 1:
            return 0.0

        max_edges = v * (v - 1)
        if not self.is_directed():
            max_edges //= 2

        return self.edge_count() / max_edges if max_edges > 0 else 0.0

    def get_in_degree(self, vertex: VertexT) -> int:
        if not self.has_vertex(vertex):
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        if not self.is_directed():
            return self.get_degree(vertex)

        in_degree = 0
        for v in self.get_vertices():
            if self.has_edge(v, vertex):
                in_degree += 1
        return in_degree

    def get_out_degree(self, vertex: VertexT) -> int:
        if not self.has_vertex(vertex):
            raise KeyNotFound(f"Vertex {vertex} not found in graph")

        return self.get_degree(vertex)

    def is_connected(self, source: VertexT, target: VertexT) -> bool:
        return self.has_edge(source, target)

    def copy(self) -> AbstractGraph[VertexT, EdgeWeightT]:
        new_graph = self.__class__()

        for vertex in self.get_vertices():
            new_graph.add_vertex(vertex)

        for source, target, weight in self.get_edges():
            new_graph.add_edge(source, target, weight)

        return new_graph
