from __future__ import annotations

import pytest

from omnivault.dsa.trees.interval import IntervalTree


class TestIntervalTree:
    @pytest.fixture
    def populated(self) -> IntervalTree:
        tree = IntervalTree()
        tree.insert(1.0, 5.0, "a")
        tree.insert(3.0, 7.0, "b")
        tree.insert(10.0, 15.0, "c")
        return tree

    @pytest.mark.unit
    def test_query_point_in_overlap(self, populated: IntervalTree) -> None:
        hits = {value for _, _, value in populated.query_point(4.0)}
        assert hits == {"a", "b"}

    @pytest.mark.unit
    def test_query_point_outside_all(self, populated: IntervalTree) -> None:
        assert populated.query_point(8.5) == []

    @pytest.mark.unit
    def test_query_point_in_only_c(self, populated: IntervalTree) -> None:
        hits = {value for _, _, value in populated.query_point(12.0)}
        assert hits == {"c"}

    @pytest.mark.unit
    def test_query_range_overlap(self, populated: IntervalTree) -> None:
        hits = {value for _, _, value in populated.query_range(6.0, 12.0)}
        assert hits == {"b", "c"}

    @pytest.mark.unit
    def test_query_range_no_overlap(self, populated: IntervalTree) -> None:
        assert populated.query_range(20.0, 25.0) == []

    @pytest.mark.unit
    def test_empty_tree_queries(self) -> None:
        tree = IntervalTree()
        assert tree.query_point(1.0) == []
        assert tree.query_range(1.0, 2.0) == []
