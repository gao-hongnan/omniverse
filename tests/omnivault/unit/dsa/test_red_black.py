from __future__ import annotations

import math

import pytest

from omnivault.dsa.core.errors import EmptyContainer, KeyNotFound
from omnivault.dsa.trees.red_black import RedBlackTree


class TestRedBlackTree:
    @pytest.fixture
    def empty_tree(self) -> RedBlackTree[int, str]:
        return RedBlackTree[int, str]()

    @pytest.mark.unit
    def test_empty_tree_state(self, empty_tree: RedBlackTree[int, str]) -> None:
        assert empty_tree.is_empty()
        assert len(empty_tree) == 0
        with pytest.raises(EmptyContainer):
            empty_tree.min_key()
        with pytest.raises(EmptyContainer):
            empty_tree.max_key()

    @pytest.mark.unit
    def test_insert_and_search_single(self, empty_tree: RedBlackTree[int, str]) -> None:
        empty_tree.insert(42, "answer")
        assert empty_tree.search(42) == "answer"
        assert 42 in empty_tree

    @pytest.mark.unit
    def test_insert_sorted_inorder_is_sorted(self, empty_tree: RedBlackTree[int, str]) -> None:
        for i in range(1, 101):
            empty_tree.insert(i, str(i))
        assert list(empty_tree.inorder_keys()) == list(range(1, 101))

    @pytest.mark.unit
    def test_red_black_height_bound_after_sorted_inserts(self, empty_tree: RedBlackTree[int, str]) -> None:
        n = 100
        for i in range(1, n + 1):
            empty_tree.insert(i, str(i))
        assert empty_tree.height() <= 2 * math.log2(n + 1)

    @pytest.mark.unit
    def test_search_missing_raises(self, empty_tree: RedBlackTree[int, str]) -> None:
        empty_tree.insert(1, "one")
        with pytest.raises(KeyNotFound):
            empty_tree.search(99)

    @pytest.mark.unit
    def test_delete_keeps_property(self, empty_tree: RedBlackTree[int, str]) -> None:
        for i in [10, 5, 15, 3, 7, 12, 20, 1, 6, 8, 11, 18]:
            empty_tree.insert(i, str(i))
        for victim in [5, 15, 10]:
            empty_tree.delete(victim)
            assert victim not in empty_tree
        remaining = sorted([3, 7, 12, 20, 1, 6, 8, 11, 18])
        assert list(empty_tree.inorder_keys()) == remaining
        assert empty_tree.height() <= 2 * math.log2(len(remaining) + 1)

    @pytest.mark.unit
    def test_delete_missing_raises(self, empty_tree: RedBlackTree[int, str]) -> None:
        empty_tree.insert(1, "a")
        with pytest.raises(KeyNotFound):
            empty_tree.delete(99)

    @pytest.mark.unit
    def test_min_max(self, empty_tree: RedBlackTree[int, str]) -> None:
        for i in [10, 5, 15, 3, 7]:
            empty_tree.insert(i, str(i))
        assert empty_tree.min_key() == 3
        assert empty_tree.max_key() == 15
