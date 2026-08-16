from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Generic, Protocol, runtime_checkable

from ..core.types import ItemT, ItemT_co


@runtime_checkable
class TreeNode(Protocol[ItemT_co]):  # noqa: UP046
    """Protocol defining the interface for a tree node."""

    @property
    def value(self) -> ItemT_co:
        """The value stored in this node."""
        ...

    @property
    def children(self) -> Sequence[TreeNode[ItemT_co]]:
        """The children of this node."""
        ...

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a leaf (has no children)."""
        ...

    def __iter__(self) -> Iterator[TreeNode[ItemT_co]]:
        """Iterate over this node and all its descendants."""
        ...


class TreeNodeImpl(Generic[ItemT]):  # noqa: UP046
    """Base implementation of a tree node."""

    def __init__(self, value: ItemT, children: list[TreeNodeImpl[ItemT]] | None = None) -> None:
        """
        Initialize a new tree node.

        Parameters
        ----------
        value : ItemT
            The value to store in this node.
        children : list[TreeNodeImpl[ItemT]] | None, optional
            The children of this node, by default None
        """
        self._value = value
        self._children: list[TreeNodeImpl[ItemT]] = children or []

    @property
    def value(self) -> ItemT:
        """The value stored in this node."""
        return self._value

    @property
    def children(self) -> list[TreeNodeImpl[ItemT]]:
        """The children of this node."""
        return self._children

    @property
    def is_leaf(self) -> bool:
        """Whether this node is a leaf (has no children)."""
        return len(self._children) == 0

    def add_child(self, child: TreeNodeImpl[ItemT]) -> None:
        """
        Add a child to this node.

        Parameters
        ----------
        child : TreeNodeImpl[ItemT]
            The child to add
        """
        self._children.append(child)

    def __iter__(self) -> Iterator[TreeNodeImpl[ItemT]]:
        """
        Iterate over this node and all its descendants in pre-order traversal.

        Yields
        ------
        TreeNodeImpl[ItemT]
            The nodes in pre-order traversal.

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        yield self
        for child in self._children:
            yield from child


@runtime_checkable
class Tree(Protocol[ItemT_co]):  # noqa: UP046
    """Protocol defining the interface for a tree."""

    @property
    def root(self) -> TreeNode[ItemT_co] | None:
        """The root node of the tree."""
        ...

    @property
    def height(self) -> int:
        """The height of the tree."""
        ...

    @property
    def size(self) -> int:
        """The number of nodes in the tree."""
        ...

    def __iter__(self) -> Iterator[TreeNode[ItemT_co]]:
        """Iterate over all nodes in the tree."""
        ...


class TreeImpl(Generic[ItemT]):  # noqa: UP046
    """Base implementation of a tree."""

    def __init__(self, root: TreeNodeImpl[ItemT] | None = None) -> None:
        """
        Initialize a new tree.

        Parameters
        ----------
        root : TreeNodeImpl[ItemT] | None, optional
            The root node of the tree, by default None
        """
        self._root = root
        self._size = self._calculate_size() if root else 0
        self._height = self._calculate_height() if root else 0

    @property
    def root(self) -> TreeNodeImpl[ItemT] | None:
        """The root node of the tree."""
        return self._root

    @property
    def height(self) -> int:
        """
        The height of the tree.

        Returns
        -------
        int
            The height of the tree, or -1 if the tree is empty.

        Time Complexity
        --------------
        O(1) - Cached value
        """
        return self._height

    @property
    def size(self) -> int:
        """
        The number of nodes in the tree.

        Returns
        -------
        int
            The number of nodes in the tree.

        Time Complexity
        --------------
        O(1) - Cached value
        """
        return self._size

    def _calculate_size(self) -> int:
        """
        Calculate the size of the tree.

        Returns
        -------
        int
            The number of nodes in the tree.

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self._root is None:
            return 0
        return sum(1 for _ in self._root)

    def _calculate_height(self) -> int:
        """
        Calculate the height of the tree.

        Returns
        -------
        int
            The height of the tree, or -1 if the tree is empty.

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self._root is None:
            return -1

        def height_recursive(node: TreeNodeImpl[ItemT]) -> int:
            if node.is_leaf:
                return 0
            return 1 + max(height_recursive(child) for child in node.children)

        return height_recursive(self._root)

    def __iter__(self) -> Iterator[TreeNodeImpl[ItemT]]:
        """
        Iterate over all nodes in the tree in pre-order traversal.

        Yields
        ------
        TreeNodeImpl[ItemT]
            The nodes in pre-order traversal.

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self._root is None:
            return
        yield from self._root
