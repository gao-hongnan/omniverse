from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, ConfigDict

from ..core.errors import EmptyContainer, KeyNotFound
from ..core.types import Comparable
from .search import AbstractSearchTree


class AVLNode[KeyT: Comparable, ValueT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: KeyT
    value: ValueT
    left: AVLNode[KeyT, ValueT] | None = None
    right: AVLNode[KeyT, ValueT] | None = None
    height: int = 0
    size: int = 1

    def model_post_init(self, __context: dict[str, object] | None) -> None:  # noqa: PYI063
        self.update_height()
        self.update_size()

    def update_height(self) -> None:
        left_height = self.left.height if self.left else -1
        right_height = self.right.height if self.right else -1
        self.height = 1 + max(left_height, right_height)

    def update_size(self) -> None:
        left_size = self.left.size if self.left else 0
        right_size = self.right.size if self.right else 0
        self.size = 1 + left_size + right_size

    def balance_factor(self) -> int:
        left_height = self.left.height if self.left else -1
        right_height = self.right.height if self.right else -1
        return left_height - right_height


class AVLTree[K: Comparable, V](AbstractSearchTree[K, V]):
    __slots__ = ("_root",)

    def __init__(self) -> None:
        self._root: AVLNode[K, V] | None = None

    @property
    def root(self) -> AVLNode[K, V] | None:
        return self._root

    def insert(self, key: K, value: V) -> None:
        self._root = self._insert_recursive(self._root, key, value)

    def _insert_recursive(self, node: AVLNode[K, V] | None, key: K, value: V) -> AVLNode[K, V]:
        if node is None:
            return AVLNode(key=key, value=value)

        if key < node.key:
            node.left = self._insert_recursive(node.left, key, value)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key, value)
        else:
            node.value = value
            return node

        node.update_height()
        node.update_size()

        return self._rebalance(node)

    def search(self, key: K) -> V:
        node = self._search_recursive(self._root, key)
        if node is None:
            raise KeyNotFound(f"Key {key} not found")
        return node.value

    def _search_recursive(self, node: AVLNode[K, V] | None, key: K) -> AVLNode[K, V] | None:
        if node is None:
            return None

        if key == node.key:
            return node
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)

    def delete(self, key: K) -> V:
        if self._root is None:
            raise KeyNotFound(f"Key {key} not found")

        value_holder: list[V] = []
        self._root = self._delete_recursive(self._root, key, value_holder)

        if not value_holder:
            raise KeyNotFound(f"Key {key} not found")

        return value_holder[0]

    def _delete_recursive(self, node: AVLNode[K, V] | None, key: K, value_holder: list[V]) -> AVLNode[K, V] | None:
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete_recursive(node.left, key, value_holder)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key, value_holder)
        else:
            value_holder.append(node.value)

            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                successor = self._find_min_node(node.right)
                node.key = successor.key
                node.value = successor.value
                node.right = self._delete_recursive(node.right, successor.key, [])

        node.update_height()
        node.update_size()

        return self._rebalance(node)

    def _find_min_node(self, node: AVLNode[K, V]) -> AVLNode[K, V]:
        while node.left is not None:
            node = node.left
        return node

    def _find_max_node(self, node: AVLNode[K, V]) -> AVLNode[K, V]:
        while node.right is not None:
            node = node.right
        return node

    def _rebalance(self, node: AVLNode[K, V]) -> AVLNode[K, V]:
        balance = node.balance_factor()

        if balance > 1:
            if node.left and node.left.balance_factor() < 0:
                node.left = self._rotate_left(node.left)
            node = self._rotate_right(node)
        elif balance < -1:
            if node.right and node.right.balance_factor() > 0:
                node.right = self._rotate_right(node.right)
            node = self._rotate_left(node)

        return node

    def _rotate_left(self, node: AVLNode[K, V]) -> AVLNode[K, V]:
        assert node.right is not None
        new_root = node.right
        node.right = new_root.left
        new_root.left = node

        node.update_height()
        node.update_size()
        new_root.update_height()
        new_root.update_size()

        return new_root

    def _rotate_right(self, node: AVLNode[K, V]) -> AVLNode[K, V]:
        assert node.left is not None
        new_root = node.left
        node.left = new_root.right
        new_root.right = node

        node.update_height()
        node.update_size()
        new_root.update_height()
        new_root.update_size()

        return new_root

    def min_key(self) -> K:
        if self._root is None:
            raise EmptyContainer("Tree is empty")
        return self._find_min_node(self._root).key

    def max_key(self) -> K:
        if self._root is None:
            raise EmptyContainer("Tree is empty")
        return self._find_max_node(self._root).key

    def contains(self, key: K) -> bool:
        return self._search_recursive(self._root, key) is not None

    def is_empty(self) -> bool:
        return self._root is None

    def size(self) -> int:
        return self._root.size if self._root else 0

    def height(self) -> int:
        return self._root.height if self._root else -1

    def inorder_keys(self) -> Iterator[K]:
        yield from self._inorder_recursive(self._root)

    def _inorder_recursive(self, node: AVLNode[K, V] | None) -> Iterator[K]:
        if node is not None:
            yield from self._inorder_recursive(node.left)
            yield node.key
            yield from self._inorder_recursive(node.right)

    def preorder_keys(self) -> Iterator[K]:
        yield from self._preorder_recursive(self._root)

    def _preorder_recursive(self, node: AVLNode[K, V] | None) -> Iterator[K]:
        if node is not None:
            yield node.key
            yield from self._preorder_recursive(node.left)
            yield from self._preorder_recursive(node.right)

    def postorder_keys(self) -> Iterator[K]:
        yield from self._postorder_recursive(self._root)

    def _postorder_recursive(self, node: AVLNode[K, V] | None) -> Iterator[K]:
        if node is not None:
            yield from self._postorder_recursive(node.left)
            yield from self._postorder_recursive(node.right)
            yield node.key

    def level_order_keys(self) -> Iterator[K]:
        if self._root is None:
            return

        queue: list[AVLNode[K, V]] = [self._root]

        while queue:
            node = queue.pop(0)
            yield node.key

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    def clear(self) -> None:
        self._root = None

    def keys(self) -> Iterator[K]:
        yield from self.inorder_keys()

    def values(self) -> Iterator[V]:
        yield from self._values_inorder(self._root)

    def _values_inorder(self, node: AVLNode[K, V] | None) -> Iterator[V]:
        if node is not None:
            yield from self._values_inorder(node.left)
            yield node.value
            yield from self._values_inorder(node.right)

    def items(self) -> Iterator[tuple[K, V]]:
        yield from self._items_inorder(self._root)

    def _items_inorder(self, node: AVLNode[K, V] | None) -> Iterator[tuple[K, V]]:
        if node is not None:
            yield from self._items_inorder(node.left)
            yield (node.key, node.value)
            yield from self._items_inorder(node.right)

    def __iter__(self) -> Iterator[K]:
        yield from self.keys()

    def __repr__(self) -> str:
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{self.__class__.__name__}({{{items}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AVLTree):
            return NotImplemented

        if self.size() != other.size():
            return False

        return all(self_item == other_item for self_item, other_item in zip(self.items(), other.items(), strict=True))

    def get_default(self, key: K, default: V) -> V:
        try:
            return self.search(key)
        except KeyError:
            return default

    def update(self, other: AbstractSearchTree[K, V] | dict[K, V]) -> None:
        if isinstance(other, AbstractSearchTree):
            for key, value in other.items():
                self.insert(key, value)
        else:
            for key, value in other.items():
                self.insert(key, value)

    def pop(self, key: K, default: V | None = None) -> V:
        try:
            return self.delete(key)
        except KeyError:
            if default is not None:
                return default
            raise

    def popitem(self) -> tuple[K, V]:
        if self.is_empty():
            raise EmptyContainer("popitem(): tree is empty")

        max_key = self.max_key()
        value = self.delete(max_key)
        return (max_key, value)

    def setdefault(self, key: K, default: V) -> V:
        try:
            return self.search(key)
        except KeyError:
            self.insert(key, default)
            return default
