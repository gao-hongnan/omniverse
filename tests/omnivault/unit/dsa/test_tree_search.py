from __future__ import annotations

import pytest

from omnivault.dsa.trees.avl import AVLTree
from omnivault.dsa.trees.search import BinarySearchTree


class TestBinarySearchTree:
    @pytest.fixture
    def empty_bst(self) -> BinarySearchTree[int, str]:
        return BinarySearchTree[int, str]()

    @pytest.fixture
    def sample_bst(self) -> BinarySearchTree[int, str]:
        bst = BinarySearchTree[int, str]()
        data = [(50, "fifty"), (30, "thirty"), (70, "seventy"), (20, "twenty"), (40, "forty")]
        for key, value in data:
            bst.insert(key, value)
        return bst

    @pytest.fixture
    def large_bst(self) -> BinarySearchTree[int, int]:
        bst = BinarySearchTree[int, int]()
        for i in range(1000):
            bst.insert(i, i * 2)
        return bst

    def test_empty_tree_operations(self, empty_bst: BinarySearchTree[int, str]) -> None:
        assert empty_bst.is_empty()
        assert len(empty_bst) == 0
        assert empty_bst.size() == 0
        assert not empty_bst
        assert empty_bst.height() == -1

    def test_empty_tree_errors(self, empty_bst: BinarySearchTree[int, str]) -> None:
        with pytest.raises(KeyError):
            empty_bst.search(1)

        with pytest.raises(KeyError):
            empty_bst.delete(1)

        with pytest.raises(ValueError):
            empty_bst.min_key()

        with pytest.raises(ValueError):
            empty_bst.max_key()

    def test_single_element_operations(self, empty_bst: BinarySearchTree[int, str]) -> None:
        empty_bst.insert(42, "answer")

        assert not empty_bst.is_empty()
        assert len(empty_bst) == 1
        assert empty_bst.size() == 1
        assert bool(empty_bst)
        assert empty_bst.height() == 0

        assert empty_bst.search(42) == "answer"
        assert empty_bst.contains(42)
        assert empty_bst.min_key() == 42
        assert empty_bst.max_key() == 42

    def test_insertion_and_search(self, sample_bst: BinarySearchTree[int, str]) -> None:
        assert sample_bst.search(50) == "fifty"
        assert sample_bst.search(30) == "thirty"
        assert sample_bst.search(70) == "seventy"
        assert sample_bst.search(20) == "twenty"
        assert sample_bst.search(40) == "forty"

        assert sample_bst.size() == 5
        assert not sample_bst.is_empty()

    def test_insertion_update_existing(self, empty_bst: BinarySearchTree[int, str]) -> None:
        empty_bst.insert(10, "ten")
        empty_bst.insert(10, "TEN")

        assert empty_bst.search(10) == "TEN"
        assert empty_bst.size() == 1

    def test_deletion_leaf_node(self, sample_bst: BinarySearchTree[int, str]) -> None:
        deleted_value = sample_bst.delete(20)
        assert deleted_value == "twenty"
        assert not sample_bst.contains(20)
        assert sample_bst.size() == 4

    def test_deletion_node_with_one_child(self, sample_bst: BinarySearchTree[int, str]) -> None:
        sample_bst.insert(25, "twenty-five")
        deleted_value = sample_bst.delete(30)
        assert deleted_value == "thirty"
        assert not sample_bst.contains(30)
        assert sample_bst.contains(25)
        assert sample_bst.size() == 5

    def test_deletion_node_with_two_children(self, sample_bst: BinarySearchTree[int, str]) -> None:
        deleted_value = sample_bst.delete(50)
        assert deleted_value == "fifty"
        assert not sample_bst.contains(50)
        assert sample_bst.size() == 4

    def test_deletion_nonexistent_key(self, sample_bst: BinarySearchTree[int, str]) -> None:
        with pytest.raises(KeyError):
            sample_bst.delete(100)

    def test_contains_method(self, sample_bst: BinarySearchTree[int, str]) -> None:
        assert sample_bst.contains(50)
        assert sample_bst.contains(30)
        assert not sample_bst.contains(100)
        assert not sample_bst.contains(-10)

    def test_min_max_keys(self, sample_bst: BinarySearchTree[int, str]) -> None:
        assert sample_bst.min_key() == 20
        assert sample_bst.max_key() == 70

    def test_traversal_orders(self, sample_bst: BinarySearchTree[int, str]) -> None:
        inorder = list(sample_bst.inorder_keys())
        assert inorder == [20, 30, 40, 50, 70]

        preorder = list(sample_bst.preorder_keys())
        assert preorder == [50, 30, 20, 40, 70]

        postorder = list(sample_bst.postorder_keys())
        assert postorder == [20, 40, 30, 70, 50]

        level_order = list(sample_bst.level_order_keys())
        assert level_order == [50, 30, 70, 20, 40]

    def test_clear_operation(self, sample_bst: BinarySearchTree[int, str]) -> None:
        sample_bst.clear()
        assert sample_bst.is_empty()
        assert sample_bst.size() == 0

    def test_keys_values_items_iteration(self, sample_bst: BinarySearchTree[int, str]) -> None:
        keys = list(sample_bst.keys())
        assert keys == [20, 30, 40, 50, 70]

        values = list(sample_bst.values())
        assert values == ["twenty", "thirty", "forty", "fifty", "seventy"]

        items = list(sample_bst.items())
        expected_items = [(20, "twenty"), (30, "thirty"), (40, "forty"), (50, "fifty"), (70, "seventy")]
        assert items == expected_items

    def test_magic_methods(self, sample_bst: BinarySearchTree[int, str]) -> None:
        assert 50 in sample_bst
        assert 100 not in sample_bst

        assert sample_bst[50] == "fifty"

        sample_bst[80] = "eighty"
        assert sample_bst.search(80) == "eighty"

        del sample_bst[80]
        assert not sample_bst.contains(80)

    def test_iteration(self, sample_bst: BinarySearchTree[int, str]) -> None:
        keys = list(sample_bst)
        assert keys == [20, 30, 40, 50, 70]

    def test_equality(self) -> None:
        bst1 = BinarySearchTree[int, str]()
        bst2 = BinarySearchTree[int, str]()

        data = [(5, "five"), (3, "three"), (7, "seven")]
        for key, value in data:
            bst1.insert(key, value)
            bst2.insert(key, value)

        assert bst1 == bst2

        bst2.insert(9, "nine")
        assert bst1 != bst2

    def test_get_default(self, sample_bst: BinarySearchTree[int, str]) -> None:
        assert sample_bst.get_default(50, "default") == "fifty"
        assert sample_bst.get_default(100, "default") == "default"

    def test_pop_operations(self, sample_bst: BinarySearchTree[int, str]) -> None:
        value = sample_bst.pop(50)
        assert value == "fifty"
        assert not sample_bst.contains(50)

        with pytest.raises(KeyError):
            sample_bst.pop(100)

        default_value = sample_bst.pop(100, "default")
        assert default_value == "default"

    def test_popitem(self, sample_bst: BinarySearchTree[int, str]) -> None:
        key, value = sample_bst.popitem()
        assert key == 70
        assert value == "seventy"
        assert not sample_bst.contains(70)

    def test_setdefault(self, sample_bst: BinarySearchTree[int, str]) -> None:
        value = sample_bst.setdefault(50, "default")
        assert value == "fifty"

        value = sample_bst.setdefault(100, "hundred")
        assert value == "hundred"
        assert sample_bst.search(100) == "hundred"

    def test_update_operations(self, empty_bst: BinarySearchTree[int, str]) -> None:
        other_bst = BinarySearchTree[int, str]()
        other_bst.insert(1, "one")
        other_bst.insert(2, "two")

        empty_bst.update(other_bst)
        assert empty_bst.size() == 2
        assert empty_bst.search(1) == "one"
        assert empty_bst.search(2) == "two"

        dict_data = {3: "three", 4: "four"}
        empty_bst.update(dict_data)
        assert empty_bst.size() == 4

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_performance_scalability(self, size: int) -> None:
        bst = BinarySearchTree[int, int]()

        for i in range(size):
            bst.insert(i, i * 2)

        assert bst.size() == size

        for i in range(0, size, 10):
            assert bst.search(i) == i * 2

        for i in range(0, size, 10):
            bst.delete(i)

        remaining_size = size - len(range(0, size, 10))
        assert bst.size() == remaining_size

    def test_large_dataset_operations(self, large_bst: BinarySearchTree[int, int]) -> None:
        assert large_bst.size() == 1000
        assert large_bst.min_key() == 0
        assert large_bst.max_key() == 999

        middle_key = 500
        assert large_bst.search(middle_key) == middle_key * 2

    @pytest.mark.unit
    def test_sorted_insert_survives_default_recursion_limit(self) -> None:
        bst: BinarySearchTree[int, int] = BinarySearchTree()
        for i in range(2000):
            bst.insert(i, i)

        assert bst.size() == 2000
        assert bst.search(0) == 0
        assert bst.search(1999) == 1999
        assert bst.contains(1500) is True
        bst.delete(1000)
        assert bst.contains(1000) is False
        assert bst.size() == 1999


class TestAVLTree:
    @pytest.fixture
    def empty_avl(self) -> AVLTree[int, str]:
        return AVLTree[int, str]()

    @pytest.fixture
    def sample_avl(self) -> AVLTree[int, str]:
        avl = AVLTree[int, str]()
        data = [(50, "fifty"), (30, "thirty"), (70, "seventy"), (20, "twenty"), (40, "forty")]
        for key, value in data:
            avl.insert(key, value)
        return avl

    def test_empty_avl_operations(self, empty_avl: AVLTree[int, str]) -> None:
        assert empty_avl.is_empty()
        assert len(empty_avl) == 0
        assert empty_avl.size() == 0
        assert not empty_avl
        assert empty_avl.height() == -1

    def test_avl_balancing_right_rotation(self) -> None:
        avl = AVLTree[int, int]()
        avl.insert(30, 30)
        avl.insert(20, 20)
        avl.insert(10, 10)

        assert avl.root is not None
        assert avl.root.key == 20
        assert avl.height() == 1

    def test_avl_balancing_left_rotation(self) -> None:
        avl = AVLTree[int, int]()
        avl.insert(10, 10)
        avl.insert(20, 20)
        avl.insert(30, 30)

        assert avl.root is not None
        assert avl.root.key == 20
        assert avl.height() == 1

    def test_avl_balancing_left_right_rotation(self) -> None:
        avl = AVLTree[int, int]()
        avl.insert(30, 30)
        avl.insert(10, 10)
        avl.insert(20, 20)

        assert avl.root is not None
        assert avl.root.key == 20

    def test_avl_balancing_right_left_rotation(self) -> None:
        avl = AVLTree[int, int]()
        avl.insert(10, 10)
        avl.insert(30, 30)
        avl.insert(20, 20)

        assert avl.root is not None
        assert avl.root.key == 20

    def test_avl_height_property(self) -> None:
        avl = AVLTree[int, int]()

        for i in range(1, 16):
            avl.insert(i, i)

        assert avl.height() <= 5

    def test_avl_maintains_bst_property(self, sample_avl: AVLTree[int, str]) -> None:
        inorder = list(sample_avl.inorder_keys())
        assert inorder == sorted(inorder)

    @pytest.mark.parametrize(
        "sequence",
        [
            list(range(10)),
            list(range(10, 0, -1)),
            [5, 3, 7, 1, 9, 2, 8, 4, 6],
        ],
    )
    def test_avl_various_insertion_sequences(self, sequence: list[int]) -> None:
        avl = AVLTree[int, int]()

        for value in sequence:
            avl.insert(value, value)

        assert avl.size() == len(sequence)

        inorder = list(avl.inorder_keys())
        assert inorder == sorted(sequence)

    def test_avl_deletion_maintains_balance(self) -> None:
        avl = AVLTree[int, int]()

        for i in range(1, 8):
            avl.insert(i, i)

        avl.delete(1)
        avl.delete(2)
        avl.delete(3)

        inorder = list(avl.inorder_keys())
        assert inorder == [4, 5, 6, 7]

    def test_avl_complex_operations(self) -> None:
        avl = AVLTree[int, str]()

        operations = [
            ("insert", 10, "ten"),
            ("insert", 5, "five"),
            ("insert", 15, "fifteen"),
            ("insert", 3, "three"),
            ("insert", 7, "seven"),
            ("insert", 12, "twelve"),
            ("insert", 18, "eighteen"),
            ("delete", 5, None),
            ("insert", 6, "six"),
            ("delete", 15, None),
        ]

        for op, key, value in operations:
            if op == "insert":
                assert value is not None
                avl.insert(key, value)
            elif op == "delete":
                avl.delete(key)

        expected_keys = [3, 6, 7, 10, 12, 18]
        assert list(avl.inorder_keys()) == expected_keys
