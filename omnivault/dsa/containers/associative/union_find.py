from __future__ import annotations

from collections.abc import Iterator
from typing import Generic

from ...core.errors import InvalidConfiguration, KeyNotFound
from ...core.types import ItemT


class UnionFind(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_parent", "_rank", "_size", "_component_count", "_element_to_id", "_id_to_element")

    def __init__(self, elements: list[ItemT] | None = None) -> None:
        self._parent: list[int] = []
        self._rank: list[int] = []
        self._size: list[int] = []
        self._component_count: int = 0
        self._element_to_id: dict[ItemT, int] = {}
        self._id_to_element: dict[int, ItemT] = {}

        if elements:
            for element in elements:
                self.make_set(element)

    def make_set(self, element: ItemT) -> int:
        if element in self._element_to_id:
            return self._element_to_id[element]

        element_id = len(self._parent)
        self._element_to_id[element] = element_id
        self._id_to_element[element_id] = element

        self._parent.append(element_id)
        self._rank.append(0)
        self._size.append(1)
        self._component_count += 1

        return element_id

    def find(self, element: ItemT) -> ItemT:
        if element not in self._element_to_id:
            raise KeyNotFound(f"Element {element} not found")

        element_id = self._element_to_id[element]
        root_id = self._find_with_compression(element_id)
        return self._id_to_element[root_id]

    def _find_with_compression(self, element_id: int) -> int:
        if self._parent[element_id] != element_id:
            self._parent[element_id] = self._find_with_compression(self._parent[element_id])
        return self._parent[element_id]

    def union(self, element1: ItemT, element2: ItemT) -> bool:
        if element1 not in self._element_to_id:
            self.make_set(element1)
        if element2 not in self._element_to_id:
            self.make_set(element2)

        root1_id = self._find_with_compression(self._element_to_id[element1])
        root2_id = self._find_with_compression(self._element_to_id[element2])

        if root1_id == root2_id:
            return False

        if self._rank[root1_id] < self._rank[root2_id]:
            root1_id, root2_id = root2_id, root1_id

        self._parent[root2_id] = root1_id
        self._size[root1_id] += self._size[root2_id]

        if self._rank[root1_id] == self._rank[root2_id]:
            self._rank[root1_id] += 1

        self._component_count -= 1
        return True

    def connected(self, element1: ItemT, element2: ItemT) -> bool:
        if element1 not in self._element_to_id or element2 not in self._element_to_id:
            return False

        return self.find(element1) == self.find(element2)

    def component_size(self, element: ItemT) -> int:
        if element not in self._element_to_id:
            raise KeyNotFound(f"Element {element} not found")

        element_id = self._element_to_id[element]
        root_id = self._find_with_compression(element_id)
        return self._size[root_id]

    def get_components(self) -> dict[ItemT, list[ItemT]]:
        components: dict[ItemT, list[ItemT]] = {}

        for element in self._element_to_id:
            root = self.find(element)
            if root not in components:
                components[root] = []
            components[root].append(element)

        return components

    def get_component(self, element: ItemT) -> list[ItemT]:
        if element not in self._element_to_id:
            raise KeyNotFound(f"Element {element} not found")

        root = self.find(element)
        component: list[ItemT] = [elem for elem in self._element_to_id if self.find(elem) == root]

        return component

    @property
    def component_count(self) -> int:
        return self._component_count

    @property
    def element_count(self) -> int:
        return len(self._element_to_id)

    def contains(self, element: ItemT) -> bool:
        return element in self._element_to_id

    def clear(self) -> None:
        self._parent.clear()
        self._rank.clear()
        self._size.clear()
        self._element_to_id.clear()
        self._id_to_element.clear()
        self._component_count = 0

    def is_empty(self) -> bool:
        return len(self._element_to_id) == 0

    def __len__(self) -> int:
        return len(self._element_to_id)

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __contains__(self, element: object) -> bool:
        return element in self._element_to_id

    def __iter__(self) -> Iterator[ItemT]:
        yield from self._element_to_id.keys()

    def __repr__(self) -> str:
        components = self.get_components()
        component_list = list(components.values())
        return f"{self.__class__.__name__}({component_list!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnionFind):
            return NotImplemented

        if self.element_count != other.element_count:
            return False

        if self.component_count != other.component_count:
            return False

        return self.get_components() == other.get_components()

    def elements(self) -> Iterator[ItemT]:
        yield from self._element_to_id.keys()

    def roots(self) -> Iterator[ItemT]:
        seen_roots: set[ItemT] = set()
        for element in self._element_to_id:
            root = self.find(element)
            if root not in seen_roots:
                seen_roots.add(root)
                yield root

    def largest_component_size(self) -> int:
        if self.is_empty():
            return 0

        return max(self._size[self._find_with_compression(i)] for i in range(len(self._parent)))

    def smallest_component_size(self) -> int:
        if self.is_empty():
            return 0

        return min(self._size[self._find_with_compression(i)] for i in range(len(self._parent)))


class WeightedUnionFind(UnionFind[ItemT]):
    __slots__ = ("_weight",)

    def __init__(self, elements: list[ItemT] | None = None) -> None:
        self._weight: list[int] = []
        super().__init__(elements)

    def make_set(self, element: ItemT) -> int:
        element_id = super().make_set(element)
        if element_id == len(self._weight):
            self._weight.append(0)
        return element_id

    def union_with_weight(self, element1: ItemT, element2: ItemT, weight: int) -> bool:
        if element1 not in self._element_to_id:
            self.make_set(element1)
        if element2 not in self._element_to_id:
            self.make_set(element2)

        id1 = self._element_to_id[element1]
        id2 = self._element_to_id[element2]

        offset1 = self._get_weight_to_root(id1)
        offset2 = self._get_weight_to_root(id2)

        root1_id = self._find_with_compression(id1)
        root2_id = self._find_with_compression(id2)

        if root1_id == root2_id:
            return False

        root_gap = weight + offset1 - offset2

        if self._rank[root1_id] < self._rank[root2_id]:
            root1_id, root2_id = root2_id, root1_id
            root_gap = -root_gap

        self._parent[root2_id] = root1_id
        self._weight[root2_id] = root_gap
        self._size[root1_id] += self._size[root2_id]

        if self._rank[root1_id] == self._rank[root2_id]:
            self._rank[root1_id] += 1

        self._component_count -= 1
        return True

    def get_weight_difference(self, element1: ItemT, element2: ItemT) -> int:
        if not self.connected(element1, element2):
            raise InvalidConfiguration("Elements are not in the same component")

        id1 = self._element_to_id[element1]
        id2 = self._element_to_id[element2]

        return self._get_weight_to_root(id2) - self._get_weight_to_root(id1)

    def _get_weight_to_root(self, element_id: int) -> int:
        parent_id = self._parent[element_id]

        if parent_id == element_id:
            return 0

        self._weight[element_id] += self._get_weight_to_root(parent_id)
        self._parent[element_id] = self._parent[parent_id]

        return self._weight[element_id]

    def clear(self) -> None:
        super().clear()
        self._weight.clear()
