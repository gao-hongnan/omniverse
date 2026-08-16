from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import cast

from ..core.errors import EmptyContainer
from ..core.types import Comparable, ItemT
from .binary import BinaryTree, BinaryTreeNode


class HeapType(Enum):
    """Enum for the type of heap."""

    MIN = auto()
    MAX = auto()


class HeapNode(BinaryTreeNode[ItemT]):
    """Implementation of a heap node."""

    def __init__(
        self,
        value: ItemT,
        parent: HeapNode[ItemT] | None = None,
        left: HeapNode[ItemT] | None = None,
        right: HeapNode[ItemT] | None = None,
    ) -> None:
        """
        Initialize a new heap node.

        Parameters
        ----------
        value : ItemT
            The value to store in this node.
        parent : HeapNode[ItemT] | None, optional
            The parent of this node, by default None
        left : HeapNode[ItemT] | None, optional
            The left child of this node, by default None
        right : HeapNode[ItemT] | None, optional
            The right child of this node, by default None
        """
        super().__init__(value, left, right)
        self._parent = parent

    @property
    def parent(self) -> HeapNode[ItemT] | None:
        """The parent of this node."""
        return self._parent

    @parent.setter
    def parent(self, node: HeapNode[ItemT] | None) -> None:
        """
        Set the parent of this node.

        Parameters
        ----------
        node : HeapNode[ItemT] | None
            The new parent
        """
        self._parent = node

    @property
    def left(self) -> BinaryTreeNode[ItemT] | None:
        """The left child of this node."""
        return self._left

    @left.setter
    def left(self, node: BinaryTreeNode[ItemT] | None) -> None:
        """
        Set the left child of this node.

        Parameters
        ----------
        node : HeapNode[ItemT] | None
            The new left child
        """
        if self._left is not None and self._left in self._children:
            self._children.remove(self._left)

        self._left = node

        if node is not None:
            if isinstance(node, HeapNode):
                node.parent = self
            self._children.append(node)

    @property
    def right(self) -> BinaryTreeNode[ItemT] | None:
        """The right child of this node."""
        return self._right

    @right.setter
    def right(self, node: BinaryTreeNode[ItemT] | None) -> None:
        """
        Set the right child of this node.

        Parameters
        ----------
        node : HeapNode[ItemT] | None
            The new right child
        """
        if self._right is not None and self._right in self._children:
            self._children.remove(self._right)

        self._right = node

        if node is not None:
            if isinstance(node, HeapNode):
                node.parent = self
            self._children.append(node)


class Heap[ItemT, KeyT: Comparable]:
    """
    Implementation of a heap data structure.

    A heap is a complete binary tree where each node's value is ordered with respect to
    its children according to a specified comparison function.

    Type parameters
    --------------
    ItemT:
        The type of elements stored in the heap
    KeyT:
        The type of keys used for comparison (must be comparable)
    """

    def __init__(
        self,
        heap_type: HeapType = HeapType.MIN,
        key_func: Callable[[ItemT], KeyT] | None = None,
        items: list[ItemT] | None = None,
    ) -> None:
        """
        Initialize a new heap.

        Parameters
        ----------
        heap_type : HeapType, optional
            The type of heap (MIN or MAX), by default HeapType.MIN
        key_func : Callable[[ItemT], KeyT] | None, optional
            A function to extract a comparison key from each element, by default None
            If None, the elements must be directly comparable.
        items : list[ItemT] | None, optional
            Initial items to add to the heap, by default None
        """
        self._heap_type = heap_type
        self._key_func = key_func
        self._data: list[ItemT] = []

        if items:
            for item in items:
                self.push(item)

    def _compare(self, a: ItemT, b: ItemT) -> bool:
        """
        Compare two items according to the heap type and key function.

        Parameters
        ----------
        a : ItemT
            The first item
        b : ItemT
            The second item

        Returns
        -------
        bool
            True if a should be higher in the heap than b, False otherwise
        """
        if self._key_func:
            key_a = self._key_func(a)
            key_b = self._key_func(b)
            return key_a < key_b if self._heap_type == HeapType.MIN else key_a > key_b
        comp_a = cast(Comparable, a)
        comp_b = cast(Comparable, b)
        return comp_a < comp_b if self._heap_type == HeapType.MIN else comp_a > comp_b

    def push(self, item: ItemT) -> None:
        """
        Add an item to the heap.

        Parameters
        ----------
        item : ItemT
            The item to add

        Time Complexity
        --------------
        O(log n) where n is the number of items in the heap.
        """
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> ItemT:
        """
        Remove and return the top item from the heap.

        Returns
        -------
        ItemT
            The top item from the heap

        Raises
        ------
        IndexError
            If the heap is empty

        Time Complexity
        --------------
        O(log n) where n is the number of items in the heap.
        """
        if not self._data:
            raise EmptyContainer("pop from an empty heap")

        result = self._data[0]

        last_item = self._data.pop()
        if self._data:
            self._data[0] = last_item
            self._sift_down(0)

        return result

    def peek(self) -> ItemT:
        """
        Return the top item from the heap without removing it.

        Returns
        -------
        ItemT
            The top item from the heap

        Raises
        ------
        IndexError
            If the heap is empty

        Time Complexity
        --------------
        O(1)
        """
        if not self._data:
            raise EmptyContainer("peek from an empty heap")

        return self._data[0]

    def _sift_up(self, index: int) -> None:
        """
        Restore the heap property by moving the item at the given index up the heap.

        Parameters
        ----------
        index : int
            The index of the item to sift up

        Time Complexity
        --------------
        O(log n) where n is the number of items in the heap.
        """
        parent_idx = (index - 1) // 2

        if index <= 0 or not self._compare(self._data[index], self._data[parent_idx]):
            return

        self._data[index], self._data[parent_idx] = (
            self._data[parent_idx],
            self._data[index],
        )

        self._sift_up(parent_idx)

    def _sift_down(self, index: int) -> None:
        """
        Restore the heap property by moving the item at the given index down the heap.

        Parameters
        ----------
        index : int
            The index of the item to sift down

        Time Complexity
        --------------
        O(log n) where n is the number of items in the heap.
        """
        left_idx = 2 * index + 1
        right_idx = 2 * index + 2

        extreme_idx = index

        if left_idx < len(self._data) and self._compare(self._data[left_idx], self._data[extreme_idx]):
            extreme_idx = left_idx

        if right_idx < len(self._data) and self._compare(self._data[right_idx], self._data[extreme_idx]):
            extreme_idx = right_idx

        if extreme_idx != index:
            self._data[index], self._data[extreme_idx] = (
                self._data[extreme_idx],
                self._data[index],
            )
            self._sift_down(extreme_idx)

    def heapify(self, items: list[ItemT]) -> None:
        """
        Build a heap from a list of items.

        This is more efficient than pushing items one by one.

        Parameters
        ----------
        items : list[ItemT]
            The items to add to the heap

        Time Complexity
        --------------
        O(n) where n is the number of items in the list.
        """
        self._data = items.copy()
        for i in range(len(self._data) // 2 - 1, -1, -1):
            self._sift_down(i)

    @property
    def size(self) -> int:
        """
        The number of items in the heap.

        Returns
        -------
        int
            The number of items in the heap

        Time Complexity
        --------------
        O(1)
        """
        return len(self._data)

    def is_empty(self) -> bool:
        """
        Check if the heap is empty.

        Returns
        -------
        bool
            True if the heap is empty, False otherwise

        Time Complexity
        --------------
        O(1)
        """
        return len(self._data) == 0

    def clear(self) -> None:
        """
        Remove all items from the heap.

        Time Complexity
        --------------
        O(1)
        """
        self._data = []

    def to_binary_tree(self) -> BinaryTree[ItemT]:
        """
        Convert the heap to a binary tree for visualization purposes.

        Returns
        -------
        BinaryTree[ItemT]
            A binary tree representation of the heap

        Time Complexity
        --------------
        O(n) where n is the number of items in the heap.
        """
        if not self._data:
            return BinaryTree[ItemT]()

        nodes: dict[int, HeapNode[ItemT]] = {}

        for i, item in enumerate(self._data):
            nodes[i] = HeapNode[ItemT](item)

        for i in range(len(self._data)):
            left_idx = 2 * i + 1
            right_idx = 2 * i + 2

            if left_idx < len(self._data):
                nodes[i].left = nodes[left_idx]

            if right_idx < len(self._data):
                nodes[i].right = nodes[right_idx]

        return BinaryTree[ItemT](cast(BinaryTreeNode[ItemT], nodes[0]))
