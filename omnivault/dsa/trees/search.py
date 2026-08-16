from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ..core.errors import EmptyContainer, InvalidConfiguration, KeyNotFound
from ..core.types import Comparable, KeyT_co, ValueT_co


@runtime_checkable
class SearchTreeNode(Protocol[KeyT_co, ValueT_co]):
    @property
    def key(self) -> KeyT_co: ...

    @property
    def value(self) -> ValueT_co: ...

    @property
    def left(self) -> SearchTreeNode[KeyT_co, ValueT_co] | None: ...

    @property
    def right(self) -> SearchTreeNode[KeyT_co, ValueT_co] | None: ...


class AbstractSearchTree[KeyT: Comparable, ValueT](ABC):
    @abstractmethod
    def insert(self, key: KeyT, value: ValueT) -> None: ...

    @abstractmethod
    def search(self, key: KeyT) -> ValueT: ...

    @abstractmethod
    def delete(self, key: KeyT) -> ValueT: ...

    @abstractmethod
    def min_key(self) -> KeyT: ...

    @abstractmethod
    def max_key(self) -> KeyT: ...

    @abstractmethod
    def contains(self, key: KeyT) -> bool: ...

    @abstractmethod
    def is_empty(self) -> bool: ...

    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def height(self) -> int: ...

    @abstractmethod
    def inorder_keys(self) -> Iterator[KeyT]: ...

    @abstractmethod
    def preorder_keys(self) -> Iterator[KeyT]: ...

    @abstractmethod
    def postorder_keys(self) -> Iterator[KeyT]: ...

    @abstractmethod
    def level_order_keys(self) -> Iterator[KeyT]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def items(self) -> Iterator[tuple[KeyT, ValueT]]: ...

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, key: object) -> bool:
        try:
            return self.contains(cast(KeyT, key))
        except TypeError, AttributeError, KeyError:
            return False

    def __getitem__(self, key: KeyT) -> ValueT:
        return self.search(key)

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        self.insert(key, value)

    def __delitem__(self, key: KeyT) -> None:
        self.delete(key)


class BSTNode[KeyT: Comparable, ValueT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: KeyT
    value: ValueT
    left: BSTNode[KeyT, ValueT] | None = None
    right: BSTNode[KeyT, ValueT] | None = None
    size: int = 1

    def model_post_init(self, __context: dict[str, object] | None) -> None:  # noqa: PYI063
        self._update_size()

    def _update_size(self) -> None:
        left_size = self.left.size if self.left else 0
        right_size = self.right.size if self.right else 0
        self.size = 1 + left_size + right_size

    def update_sizes_up(self) -> None:
        self._update_size()


class BinarySearchTree[KeyT: Comparable, ValueT](AbstractSearchTree[KeyT, ValueT]):
    __slots__ = ("_root",)

    def __init__(self) -> None:
        self._root: BSTNode[KeyT, ValueT] | None = None

    @property
    def root(self) -> BSTNode[KeyT, ValueT] | None:
        return self._root

    def insert(self, key: KeyT, value: ValueT) -> None:
        if self._root is None:
            self._root = BSTNode(key=key, value=value)
            return

        path: list[BSTNode[KeyT, ValueT]] = []
        current: BSTNode[KeyT, ValueT] = self._root

        while True:
            path.append(current)
            if key < current.key:
                if current.left is None:
                    current.left = BSTNode(key=key, value=value)
                    break
                current = current.left
            elif key > current.key:
                if current.right is None:
                    current.right = BSTNode(key=key, value=value)
                    break
                current = current.right
            else:
                current.value = value
                return

        for ancestor in reversed(path):
            ancestor.update_sizes_up()

    def search(self, key: KeyT) -> ValueT:
        node = self._find_node(key)
        if node is None:
            raise KeyNotFound(f"Key {key} not found")
        return node.value

    def _find_node(self, key: KeyT) -> BSTNode[KeyT, ValueT] | None:
        current: BSTNode[KeyT, ValueT] | None = self._root
        while current is not None:
            if key == current.key:
                return current
            current = current.left if key < current.key else current.right
        return None

    def delete(self, key: KeyT) -> ValueT:
        parent: BSTNode[KeyT, ValueT] | None = None
        current: BSTNode[KeyT, ValueT] | None = self._root
        path: list[BSTNode[KeyT, ValueT]] = []

        while current is not None and current.key != key:
            path.append(current)
            parent = current
            current = current.left if key < current.key else current.right

        if current is None:
            raise KeyNotFound(f"Key {key} not found")

        removed_value: ValueT = current.value

        if current.left is not None and current.right is not None:
            successor_parent: BSTNode[KeyT, ValueT] = current
            successor: BSTNode[KeyT, ValueT] = current.right
            successor_path: list[BSTNode[KeyT, ValueT]] = [current]
            while successor.left is not None:
                successor_path.append(successor)
                successor_parent = successor
                successor = successor.left

            current.key = successor.key
            current.value = successor.value

            replacement: BSTNode[KeyT, ValueT] | None = successor.right
            if successor_parent is current:
                successor_parent.right = replacement
            else:
                successor_parent.left = replacement

            for ancestor in reversed(successor_path):
                ancestor.update_sizes_up()
        else:
            child: BSTNode[KeyT, ValueT] | None = current.left if current.left is not None else current.right
            if parent is None:
                self._root = child
            elif parent.left is current:
                parent.left = child
            else:
                parent.right = child

        for ancestor in reversed(path):
            ancestor.update_sizes_up()

        return removed_value

    def _find_min_node(self, node: BSTNode[KeyT, ValueT]) -> BSTNode[KeyT, ValueT]:
        while node.left is not None:
            node = node.left
        return node

    def _find_max_node(self, node: BSTNode[KeyT, ValueT]) -> BSTNode[KeyT, ValueT]:
        while node.right is not None:
            node = node.right
        return node

    def min_key(self) -> KeyT:
        if self._root is None:
            raise InvalidConfiguration("Tree is empty")
        return self._find_min_node(self._root).key

    def max_key(self) -> KeyT:
        if self._root is None:
            raise InvalidConfiguration("Tree is empty")
        return self._find_max_node(self._root).key

    def contains(self, key: KeyT) -> bool:
        return self._find_node(key) is not None

    def is_empty(self) -> bool:
        return self._root is None

    def size(self) -> int:
        return self._root.size if self._root else 0

    def height(self) -> int:
        if self._root is None:
            return -1

        stack: list[tuple[BSTNode[KeyT, ValueT], int]] = [(self._root, 0)]
        max_depth: int = 0

        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                max_depth = depth
            if node.left is not None:
                stack.append((node.left, depth + 1))
            if node.right is not None:
                stack.append((node.right, depth + 1))

        return max_depth

    def inorder_keys(self) -> Iterator[KeyT]:
        yield from self._inorder_recursive(self._root)

    def _inorder_recursive(self, node: BSTNode[KeyT, ValueT] | None) -> Iterator[KeyT]:
        if node is not None:
            yield from self._inorder_recursive(node.left)
            yield node.key
            yield from self._inorder_recursive(node.right)

    def preorder_keys(self) -> Iterator[KeyT]:
        yield from self._preorder_recursive(self._root)

    def _preorder_recursive(self, node: BSTNode[KeyT, ValueT] | None) -> Iterator[KeyT]:
        if node is not None:
            yield node.key
            yield from self._preorder_recursive(node.left)
            yield from self._preorder_recursive(node.right)

    def postorder_keys(self) -> Iterator[KeyT]:
        yield from self._postorder_recursive(self._root)

    def _postorder_recursive(self, node: BSTNode[KeyT, ValueT] | None) -> Iterator[KeyT]:
        if node is not None:
            yield from self._postorder_recursive(node.left)
            yield from self._postorder_recursive(node.right)
            yield node.key

    def level_order_keys(self) -> Iterator[KeyT]:
        if self._root is None:
            return

        queue: list[BSTNode[KeyT, ValueT]] = [self._root]

        while queue:
            node = queue.pop(0)
            yield node.key

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    def clear(self) -> None:
        self._root = None

    def keys(self) -> Iterator[KeyT]:
        yield from self.inorder_keys()

    def values(self) -> Iterator[ValueT]:
        yield from self._values_inorder(self._root)

    def _values_inorder(self, node: BSTNode[KeyT, ValueT] | None) -> Iterator[ValueT]:
        if node is not None:
            yield from self._values_inorder(node.left)
            yield node.value
            yield from self._values_inorder(node.right)

    def items(self) -> Iterator[tuple[KeyT, ValueT]]:
        yield from self._items_inorder(self._root)

    def _items_inorder(self, node: BSTNode[KeyT, ValueT] | None) -> Iterator[tuple[KeyT, ValueT]]:
        if node is not None:
            yield from self._items_inorder(node.left)
            yield (node.key, node.value)
            yield from self._items_inorder(node.right)

    def __iter__(self) -> Iterator[KeyT]:
        yield from self.keys()

    def __repr__(self) -> str:
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{self.__class__.__name__}({{{items}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinarySearchTree):
            return NotImplemented

        if self.size() != other.size():
            return False

        return all(self_item == other_item for self_item, other_item in zip(self.items(), other.items(), strict=True))

    def get_default(self, key: KeyT, default: ValueT) -> ValueT:
        try:
            return self.search(key)
        except KeyError:
            return default

    def update(self, other: AbstractSearchTree[KeyT, ValueT] | dict[KeyT, ValueT]) -> None:
        if isinstance(other, AbstractSearchTree):
            for key, value in other.items():
                self.insert(key, value)
        else:
            for key, value in other.items():
                self.insert(key, value)

    def pop(self, key: KeyT, default: ValueT | None = None) -> ValueT:
        try:
            return self.delete(key)
        except KeyError:
            if default is not None:
                return default
            raise

    def popitem(self) -> tuple[KeyT, ValueT]:
        if self.is_empty():
            raise EmptyContainer("popitem(): tree is empty")

        max_key = self.max_key()
        value = self.delete(max_key)
        return (max_key, value)

    def setdefault(self, key: KeyT, default: ValueT) -> ValueT:
        try:
            return self.search(key)
        except KeyError:
            self.insert(key, default)
            return default
