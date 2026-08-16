from __future__ import annotations

from collections.abc import Iterator
from enum import Enum

from pydantic import BaseModel, ConfigDict

from ..core.errors import EmptyContainer, KeyNotFound
from ..core.types import Comparable


class Color(Enum):
    RED = "red"
    BLACK = "black"


class RedBlackNode[KeyT: Comparable, ValueT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: KeyT
    value: ValueT
    color: Color = Color.RED
    left: RedBlackNode[KeyT, ValueT] | None = None
    right: RedBlackNode[KeyT, ValueT] | None = None
    parent: RedBlackNode[KeyT, ValueT] | None = None


class RedBlackTree[KeyT: Comparable, ValueT]:
    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        self._root: RedBlackNode[KeyT, ValueT] | None = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: object) -> bool:
        try:
            self._lookup(key)  # type: ignore[arg-type]
        except KeyNotFound:
            return False
        return True

    def is_empty(self) -> bool:
        return self._root is None

    def insert(self, key: KeyT, value: ValueT) -> None:
        new_node = RedBlackNode[KeyT, ValueT](key=key, value=value, color=Color.RED)
        parent: RedBlackNode[KeyT, ValueT] | None = None
        cursor = self._root
        while cursor is not None:
            parent = cursor
            if key < cursor.key:
                cursor = cursor.left
            elif key > cursor.key:
                cursor = cursor.right
            else:
                cursor.value = value
                return
        new_node.parent = parent
        if parent is None:
            self._root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        self._size += 1
        self._fix_insert(new_node)

    def search(self, key: KeyT) -> ValueT:
        return self._lookup(key).value

    def delete(self, key: KeyT) -> ValueT:
        node = self._lookup(key)
        deleted_value = node.value
        self._delete_node(node)
        self._size -= 1
        return deleted_value

    def min_key(self) -> KeyT:
        if self._root is None:
            raise EmptyContainer("min_key on empty RedBlackTree")
        return self._min_node(self._root).key

    def max_key(self) -> KeyT:
        if self._root is None:
            raise EmptyContainer("max_key on empty RedBlackTree")
        node = self._root
        while node.right is not None:
            node = node.right
        return node.key

    def inorder_keys(self) -> Iterator[KeyT]:
        yield from self._inorder(self._root)

    def height(self) -> int:
        return self._height(self._root)

    def _height(self, node: RedBlackNode[KeyT, ValueT] | None) -> int:
        if node is None:
            return -1
        return 1 + max(self._height(node.left), self._height(node.right))

    def _inorder(self, node: RedBlackNode[KeyT, ValueT] | None) -> Iterator[KeyT]:
        if node is None:
            return
        yield from self._inorder(node.left)
        yield node.key
        yield from self._inorder(node.right)

    def _lookup(self, key: KeyT) -> RedBlackNode[KeyT, ValueT]:
        cursor = self._root
        while cursor is not None:
            if key == cursor.key:
                return cursor
            cursor = cursor.left if key < cursor.key else cursor.right
        raise KeyNotFound(f"Key {key!r} not found in RedBlackTree")

    def _min_node(self, node: RedBlackNode[KeyT, ValueT]) -> RedBlackNode[KeyT, ValueT]:
        while node.left is not None:
            node = node.left
        return node

    def _rotate_left(self, x: RedBlackNode[KeyT, ValueT]) -> None:
        y = x.right
        if y is None:
            return
        x.right = y.left
        if y.left is not None:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self._root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, x: RedBlackNode[KeyT, ValueT]) -> None:
        y = x.left
        if y is None:
            return
        x.left = y.right
        if y.right is not None:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self._root = y
        elif x is x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def _fix_insert(self, node: RedBlackNode[KeyT, ValueT]) -> None:
        while node.parent is not None and node.parent.color is Color.RED:
            parent = node.parent
            grandparent = parent.parent
            if grandparent is None:
                break
            if parent is grandparent.left:
                uncle = grandparent.right
                if uncle is not None and uncle.color is Color.RED:
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    node = grandparent
                else:
                    if node is parent.right:
                        node = parent
                        self._rotate_left(node)
                        rotated_parent = node.parent
                        assert rotated_parent is not None
                        parent = rotated_parent
                        rotated_grandparent = parent.parent
                        assert rotated_grandparent is not None
                        grandparent = rotated_grandparent
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    self._rotate_right(grandparent)
            else:
                uncle = grandparent.left
                if uncle is not None and uncle.color is Color.RED:
                    parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    grandparent.color = Color.RED
                    node = grandparent
                else:
                    if node is parent.left:
                        node = parent
                        self._rotate_right(node)
                        rotated_parent = node.parent
                        assert rotated_parent is not None
                        parent = rotated_parent
                        rotated_grandparent = parent.parent
                        assert rotated_grandparent is not None
                        grandparent = rotated_grandparent
                    parent.color = Color.BLACK
                    grandparent.color = Color.RED
                    self._rotate_left(grandparent)
        assert self._root is not None
        self._root.color = Color.BLACK

    def _transplant(
        self,
        old: RedBlackNode[KeyT, ValueT],
        new: RedBlackNode[KeyT, ValueT] | None,
    ) -> None:
        if old.parent is None:
            self._root = new
        elif old is old.parent.left:
            old.parent.left = new
        else:
            old.parent.right = new
        if new is not None:
            new.parent = old.parent

    def _delete_node(self, node: RedBlackNode[KeyT, ValueT]) -> None:
        successor = node
        successor_original_color = successor.color
        replacement: RedBlackNode[KeyT, ValueT] | None
        replacement_parent: RedBlackNode[KeyT, ValueT] | None
        if node.left is None:
            replacement = node.right
            replacement_parent = node.parent
            self._transplant(node, node.right)
        elif node.right is None:
            replacement = node.left
            replacement_parent = node.parent
            self._transplant(node, node.left)
        else:
            successor = self._min_node(node.right)
            successor_original_color = successor.color
            replacement = successor.right
            if successor.parent is node:
                replacement_parent = successor
            else:
                replacement_parent = successor.parent
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor
            self._transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor
            successor.color = node.color
        if successor_original_color is Color.BLACK:
            self._fix_delete(replacement, replacement_parent)

    def _is_black(self, node: RedBlackNode[KeyT, ValueT] | None) -> bool:
        return node is None or node.color is Color.BLACK

    def _fix_delete(
        self,
        node: RedBlackNode[KeyT, ValueT] | None,
        parent: RedBlackNode[KeyT, ValueT] | None,
    ) -> None:
        while node is not self._root and self._is_black(node):
            if parent is None:
                break
            if node is parent.left:
                sibling = parent.right
                if sibling is None:
                    break
                if sibling.color is Color.RED:
                    sibling.color = Color.BLACK
                    parent.color = Color.RED
                    self._rotate_left(parent)
                    sibling = parent.right
                    if sibling is None:
                        break
                if self._is_black(sibling.left) and self._is_black(sibling.right):
                    sibling.color = Color.RED
                    node = parent
                    parent = node.parent
                else:
                    if self._is_black(sibling.right):
                        if sibling.left is not None:
                            sibling.left.color = Color.BLACK
                        sibling.color = Color.RED
                        self._rotate_right(sibling)
                        sibling = parent.right
                        if sibling is None:
                            break
                    sibling.color = parent.color
                    parent.color = Color.BLACK
                    if sibling.right is not None:
                        sibling.right.color = Color.BLACK
                    self._rotate_left(parent)
                    node = self._root
                    parent = None
            else:
                sibling = parent.left
                if sibling is None:
                    break
                if sibling.color is Color.RED:
                    sibling.color = Color.BLACK
                    parent.color = Color.RED
                    self._rotate_right(parent)
                    sibling = parent.left
                    if sibling is None:
                        break
                if self._is_black(sibling.right) and self._is_black(sibling.left):
                    sibling.color = Color.RED
                    node = parent
                    parent = node.parent
                else:
                    if self._is_black(sibling.left):
                        if sibling.right is not None:
                            sibling.right.color = Color.BLACK
                        sibling.color = Color.RED
                        self._rotate_left(sibling)
                        sibling = parent.left
                        if sibling is None:
                            break
                    sibling.color = parent.color
                    parent.color = Color.BLACK
                    if sibling.left is not None:
                        sibling.left.color = Color.BLACK
                    self._rotate_right(parent)
                    node = self._root
                    parent = None
        if node is not None:
            node.color = Color.BLACK
