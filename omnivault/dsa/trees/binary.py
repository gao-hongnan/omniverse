from __future__ import annotations

from collections.abc import Iterator
from typing import cast

from ..core.types import ItemT
from .base import TreeImpl, TreeNodeImpl


class BinaryTreeNode(TreeNodeImpl[ItemT]):
    """Implementation of a binary tree node."""

    def __init__(
        self,
        value: ItemT,
        left: BinaryTreeNode[ItemT] | None = None,
        right: BinaryTreeNode[ItemT] | None = None,
    ) -> None:
        """
        Initialize a new binary tree node.

        Parameters
        ----------
        value : ItemT
            The value to store in this node.
        left : BinaryTreeNode[ItemT] | None, optional
            The left child of this node, by default None
        right : BinaryTreeNode[ItemT] | None, optional
            The right child of this node, by default None
        """
        super().__init__(value, [])
        self._left = left
        self._right = right
        if left:
            self._children.append(left)
        if right:
            self._children.append(right)

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
        node : BinaryTreeNode[ItemT] | None
            The new left child
        """
        if self._left is not None and self._left in self._children:
            self._children.remove(self._left)

        self._left = node

        if node is not None:
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
        node : BinaryTreeNode[ItemT] | None
            The new right child
        """
        if self._right is not None and self._right in self._children:
            self._children.remove(self._right)

        self._right = node

        if node is not None:
            self._children.append(node)

    def inorder_traversal(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Perform an inorder traversal of the binary tree rooted at this node.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in inorder traversal (left, root, right).

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self.left:
            yield from self.left.inorder_traversal()
        yield self
        if self.right:
            yield from self.right.inorder_traversal()

    def postorder_traversal(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Perform a postorder traversal of the binary tree rooted at this node.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in postorder traversal (left, right, root).

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self.left:
            yield from self.left.postorder_traversal()
        if self.right:
            yield from self.right.postorder_traversal()
        yield self

    def levelorder_traversal(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Perform a level-order traversal of the binary tree rooted at this node.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in level-order traversal (breadth-first).

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        queue: list[BinaryTreeNode[ItemT]] = [self]
        while queue:
            node = queue.pop(0)
            yield node
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    def __iter__(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Iterate over this node and all its descendants in pre-order traversal.

        This is an override of the base implementation to ensure correct typing.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in pre-order traversal (root, left, right).

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        yield self
        if self.left:
            yield from self.left
        if self.right:
            yield from self.right


class BinaryTree(TreeImpl[ItemT]):
    """Implementation of a binary tree."""

    def __init__(self, root: BinaryTreeNode[ItemT] | None = None) -> None:
        """
        Initialize a new binary tree.

        Parameters
        ----------
        root : BinaryTreeNode[ItemT] | None, optional
            The root node of the tree, by default None
        """
        super().__init__(root)

    @property
    def root(self) -> BinaryTreeNode[ItemT] | None:
        """The root node of the tree."""
        return cast(BinaryTreeNode[ItemT] | None, self._root)

    def inorder_traversal(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Perform an inorder traversal of the binary tree.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in inorder traversal (left, root, right).

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self.root is None:
            return
        yield from self.root.inorder_traversal()

    def postorder_traversal(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Perform a postorder traversal of the binary tree.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in postorder traversal (left, right, root).

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self.root is None:
            return
        yield from self.root.postorder_traversal()

    def levelorder_traversal(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Perform a level-order traversal of the binary tree.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in level-order traversal (breadth-first).

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self.root is None:
            return
        yield from self.root.levelorder_traversal()

    def __iter__(self) -> Iterator[BinaryTreeNode[ItemT]]:
        """
        Iterate over all nodes in the tree in pre-order traversal (root, left, right).

        This is an override of the base implementation to ensure correct typing.

        Yields
        ------
        BinaryTreeNode[ItemT]
            The nodes in pre-order traversal.

        Time Complexity
        --------------
        O(n) where n is the number of nodes in the tree.
        """
        if self.root is None:
            return
        yield from self.root
