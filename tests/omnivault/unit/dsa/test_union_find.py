from __future__ import annotations

import pytest

from omnivault.dsa.containers.associative.union_find import UnionFind, WeightedUnionFind


class TestUnionFind:
    @pytest.fixture
    def empty_uf(self) -> UnionFind[str]:
        return UnionFind[str]()

    @pytest.fixture
    def sample_uf(self) -> UnionFind[str]:
        elements = ["A", "B", "C", "D", "E", "F"]
        return UnionFind[str](elements)

    @pytest.fixture
    def connected_uf(self) -> UnionFind[int]:
        uf = UnionFind[int]()
        for i in range(10):
            uf.make_set(i)

        # Create some connections: (0,1), (2,3,4), (5,6), (7,8,9)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(3, 4)
        uf.union(5, 6)
        uf.union(7, 8)
        uf.union(8, 9)

        return uf

    def test_empty_union_find_operations(self, empty_uf: UnionFind[str]) -> None:
        assert empty_uf.is_empty()
        assert len(empty_uf) == 0
        assert empty_uf.element_count == 0
        assert empty_uf.component_count == 0
        assert not empty_uf

    def test_make_set_operation(self, empty_uf: UnionFind[str]) -> None:
        element_id = empty_uf.make_set("A")

        assert not empty_uf.is_empty()
        assert len(empty_uf) == 1
        assert empty_uf.element_count == 1
        assert empty_uf.component_count == 1
        assert bool(empty_uf)
        assert element_id == 0

    def test_make_set_duplicate_element(self, empty_uf: UnionFind[str]) -> None:
        id1 = empty_uf.make_set("A")
        id2 = empty_uf.make_set("A")

        assert id1 == id2
        assert empty_uf.element_count == 1
        assert empty_uf.component_count == 1

    def test_initialization_with_elements(self, sample_uf: UnionFind[str]) -> None:
        assert sample_uf.element_count == 6
        assert sample_uf.component_count == 6

        for element in ["A", "B", "C", "D", "E", "F"]:
            assert sample_uf.contains(element)

    def test_find_operation(self, sample_uf: UnionFind[str]) -> None:
        root_a = sample_uf.find("A")
        assert root_a == "A"

        with pytest.raises(KeyError):
            sample_uf.find("Z")

    def test_union_operation_success(self, sample_uf: UnionFind[str]) -> None:
        result = sample_uf.union("A", "B")
        assert result is True
        assert sample_uf.component_count == 5
        assert sample_uf.connected("A", "B")

    def test_union_operation_already_connected(self, sample_uf: UnionFind[str]) -> None:
        sample_uf.union("A", "B")
        result = sample_uf.union("A", "B")
        assert result is False
        assert sample_uf.component_count == 5

    def test_union_with_new_elements(self, empty_uf: UnionFind[str]) -> None:
        result = empty_uf.union("X", "Y")
        assert result is True
        assert empty_uf.element_count == 2
        assert empty_uf.component_count == 1
        assert empty_uf.connected("X", "Y")

    def test_connected_operation(self, sample_uf: UnionFind[str]) -> None:
        assert not sample_uf.connected("A", "B")

        sample_uf.union("A", "B")
        assert sample_uf.connected("A", "B")

        assert not sample_uf.connected("A", "C")

    def test_connected_nonexistent_elements(self, sample_uf: UnionFind[str]) -> None:
        assert not sample_uf.connected("A", "Z")
        assert not sample_uf.connected("X", "Y")

    def test_component_size(self, sample_uf: UnionFind[str]) -> None:
        assert sample_uf.component_size("A") == 1

        sample_uf.union("A", "B")
        assert sample_uf.component_size("A") == 2
        assert sample_uf.component_size("B") == 2

        sample_uf.union("B", "C")
        assert sample_uf.component_size("A") == 3
        assert sample_uf.component_size("B") == 3
        assert sample_uf.component_size("C") == 3

    def test_component_size_nonexistent_element(self, sample_uf: UnionFind[str]) -> None:
        with pytest.raises(KeyError):
            sample_uf.component_size("Z")

    def test_get_components(self, connected_uf: UnionFind[int]) -> None:
        components = connected_uf.get_components()

        assert len(components) == 4

        # Check component sizes
        component_sizes = [len(comp) for comp in components.values()]
        assert sorted(component_sizes) == [2, 2, 3, 3]

    def test_get_component(self, connected_uf: UnionFind[int]) -> None:
        component_0 = connected_uf.get_component(0)
        assert sorted(component_0) == [0, 1]

        component_2 = connected_uf.get_component(2)
        assert sorted(component_2) == [2, 3, 4]

    def test_get_component_nonexistent_element(self, connected_uf: UnionFind[int]) -> None:
        with pytest.raises(KeyError):
            connected_uf.get_component(100)

    def test_clear_operation(self, sample_uf: UnionFind[str]) -> None:
        sample_uf.clear()
        assert sample_uf.is_empty()
        assert sample_uf.element_count == 0
        assert sample_uf.component_count == 0

    def test_contains_operation(self, sample_uf: UnionFind[str]) -> None:
        assert sample_uf.contains("A")
        assert "A" in sample_uf
        assert not sample_uf.contains("Z")
        assert "Z" not in sample_uf

    def test_iteration(self, sample_uf: UnionFind[str]) -> None:
        elements = list(sample_uf)
        expected_elements = ["A", "B", "C", "D", "E", "F"]
        assert sorted(elements) == sorted(expected_elements)

    def test_elements_method(self, sample_uf: UnionFind[str]) -> None:
        elements = list(sample_uf.elements())
        expected_elements = ["A", "B", "C", "D", "E", "F"]
        assert sorted(elements) == sorted(expected_elements)

    def test_roots_method(self, connected_uf: UnionFind[int]) -> None:
        roots = list(connected_uf.roots())
        assert len(roots) == 4

    def test_largest_component_size(self, connected_uf: UnionFind[int]) -> None:
        assert connected_uf.largest_component_size() == 3

    def test_smallest_component_size(self, connected_uf: UnionFind[int]) -> None:
        assert connected_uf.smallest_component_size() == 2

    def test_largest_smallest_component_empty(self, empty_uf: UnionFind[str]) -> None:
        assert empty_uf.largest_component_size() == 0
        assert empty_uf.smallest_component_size() == 0

    def test_equality(self) -> None:
        uf1 = UnionFind[int]()
        uf2 = UnionFind[int]()

        elements = [1, 2, 3, 4]
        for element in elements:
            uf1.make_set(element)
            uf2.make_set(element)

        uf1.union(1, 2)
        uf1.union(3, 4)

        uf2.union(1, 2)
        uf2.union(3, 4)

        assert uf1 == uf2

        uf2.union(1, 3)
        assert uf1 != uf2

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_performance_path_compression(self, size: int) -> None:
        uf = UnionFind[int]()

        # Create a chain: 0 -> 1 -> 2 -> ... -> size-1
        for i in range(size):
            uf.make_set(i)

        for i in range(size - 1):
            uf.union(i, i + 1)

        # Path compression should make subsequent finds fast
        root = uf.find(0)
        assert uf.find(size - 1) == root
        assert uf.component_size(0) == size

    def test_union_by_rank_optimization(self) -> None:
        uf = UnionFind[int]()

        # Create two trees of different sizes
        for i in range(8):
            uf.make_set(i)

        # Create tree 1: {0, 1, 2, 3}
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 2)

        # Create tree 2: {4, 5}
        uf.union(4, 5)

        # Union by rank should attach smaller tree to larger tree
        uf.union(0, 4)

        assert uf.component_size(0) == 6
        assert uf.component_size(4) == 6

    def test_complex_union_find_operations(self) -> None:
        uf = UnionFind[str]()

        # Create a complex graph structure
        nodes = ["A", "B", "C", "D", "E", "F", "G", "H"]
        for node in nodes:
            uf.make_set(node)

        # Create connections
        connections = [("A", "B"), ("B", "C"), ("D", "E"), ("F", "G"), ("G", "H")]
        for node1, node2 in connections:
            uf.union(node1, node2)

        # Verify components
        assert uf.component_count == 3
        assert uf.connected("A", "C")
        assert uf.connected("D", "E")
        assert uf.connected("F", "H")
        assert not uf.connected("A", "D")

        # Merge two components
        uf.union("C", "D")
        assert uf.component_count == 2
        assert uf.connected("A", "E")

    def test_edge_cases(self) -> None:
        uf = UnionFind[int]()

        # Single element
        uf.make_set(42)
        assert uf.find(42) == 42
        assert uf.component_size(42) == 1

        # Self union
        result = uf.union(42, 42)
        assert result is False
        assert uf.component_size(42) == 1


class TestWeightedUnionFind:
    @pytest.fixture
    def weighted_uf(self) -> WeightedUnionFind[str]:
        elements = ["A", "B", "C", "D"]
        return WeightedUnionFind[str](elements)

    def test_weighted_union_operation(self, weighted_uf: WeightedUnionFind[str]) -> None:
        result = weighted_uf.union_with_weight("A", "B", 5)
        assert result is True
        assert weighted_uf.connected("A", "B")

    def test_weight_difference_calculation(self, weighted_uf: WeightedUnionFind[str]) -> None:
        weighted_uf.union_with_weight("A", "B", 5)
        weighted_uf.union_with_weight("B", "C", 3)

        # A --5--> B --3--> C, so A --8--> C
        weight_diff = weighted_uf.get_weight_difference("A", "C")
        assert weight_diff == 8

        weight_diff_reverse = weighted_uf.get_weight_difference("C", "A")
        assert weight_diff_reverse == -8

    def test_weight_difference_not_connected(self, weighted_uf: WeightedUnionFind[str]) -> None:
        weighted_uf.union_with_weight("A", "B", 5)

        with pytest.raises(ValueError):
            weighted_uf.get_weight_difference("A", "C")

    def test_weighted_union_already_connected(self, weighted_uf: WeightedUnionFind[str]) -> None:
        weighted_uf.union_with_weight("A", "B", 5)
        result = weighted_uf.union_with_weight("A", "B", 10)
        assert result is False

    def test_complex_weighted_structure(self) -> None:
        wuf = WeightedUnionFind[int]()

        # Create weighted graph: 0 --2--> 1 --3--> 2 --1--> 3
        for i in range(4):
            wuf.make_set(i)

        wuf.union_with_weight(0, 1, 2)
        wuf.union_with_weight(1, 2, 3)
        wuf.union_with_weight(2, 3, 1)

        # Check weight differences
        assert wuf.get_weight_difference(0, 1) == 2
        assert wuf.get_weight_difference(0, 2) == 5
        assert wuf.get_weight_difference(0, 3) == 6
        assert wuf.get_weight_difference(1, 3) == 4

    def test_weighted_clear_operation(self, weighted_uf: WeightedUnionFind[str]) -> None:
        weighted_uf.union_with_weight("A", "B", 5)
        weighted_uf.clear()

        assert weighted_uf.is_empty()
        assert weighted_uf.element_count == 0
        assert weighted_uf.component_count == 0

    @pytest.mark.parametrize(
        "operations",
        [
            [("A", "B", 10), ("B", "C", 5), ("C", "D", 3)],
            [("A", "C", 7), ("B", "D", 4), ("A", "B", 2)],
        ],
    )
    def test_weighted_union_find_scenarios(self, operations: list[tuple[str, str, int]]) -> None:
        wuf = WeightedUnionFind[str]()

        elements = set()
        for elem1, elem2, _ in operations:
            elements.update([elem1, elem2])

        for element in elements:
            wuf.make_set(element)

        for elem1, elem2, weight in operations:
            wuf.union_with_weight(elem1, elem2, weight)

        # Verify all elements are connected if we expect them to be
        if len(operations) >= len(elements) - 1:
            first_elem = next(iter(elements))
            for element in elements:
                if element != first_elem:
                    assert wuf.connected(first_elem, element)
