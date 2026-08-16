from __future__ import annotations

from collections.abc import Iterator
from typing import Generic

from pydantic import BaseModel, Field

from ..core.errors import IncompatibleSketch, KeyNotFound
from ..core.types import ItemT


class TrieNode(BaseModel, Generic[ItemT]):  # noqa: UP046
    children: dict[str, TrieNode[ItemT]] = Field(default_factory=dict)
    value: ItemT | None = None
    is_end_of_word: bool = False

    def has_child(self, char: str) -> bool:
        return char in self.children

    def get_child(self, char: str) -> TrieNode[ItemT] | None:
        return self.children.get(char)

    def add_child(self, char: str) -> TrieNode[ItemT]:
        if char not in self.children:
            self.children[char] = TrieNode[ItemT]()
        return self.children[char]

    def remove_child(self, char: str) -> None:
        if char in self.children:
            del self.children[char]

    def has_children(self) -> bool:
        return len(self.children) > 0

    def child_count(self) -> int:
        return len(self.children)


class Trie(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        self._root: TrieNode[ItemT] = TrieNode[ItemT]()
        self._size: int = 0

    @property
    def root(self) -> TrieNode[ItemT]:
        return self._root

    def insert(self, word: str, value: ItemT | None = None) -> None:
        if not word:
            raise IncompatibleSketch("Word cannot be empty")

        current = self._root

        for char in word:
            current = current.add_child(char)

        if not current.is_end_of_word:
            self._size += 1

        current.is_end_of_word = True
        current.value = value

    def search(self, word: str) -> ItemT | None:
        node = self._find_node(word)
        if node and node.is_end_of_word:
            return node.value
        raise KeyNotFound(f"Word '{word}' not found")

    def contains(self, word: str) -> bool:
        node = self._find_node(word)
        return node is not None and node.is_end_of_word

    def starts_with(self, prefix: str) -> bool:
        return self._find_node(prefix) is not None

    def _find_node(self, word: str) -> TrieNode[ItemT] | None:
        if not word:
            return self._root

        current: TrieNode[ItemT] | None = self._root

        for char in word:
            if current is None:
                return None
            current = current.get_child(char)

        return current

    def delete(self, word: str) -> ItemT | None:
        if not word:
            raise IncompatibleSketch("Word cannot be empty")

        def _delete_recursive(node: TrieNode[ItemT], word: str, index: int) -> bool:
            if index == len(word):
                if not node.is_end_of_word:
                    return False

                node.is_end_of_word = False
                node.value = None
                return not node.has_children()

            char = word[index]
            child = node.get_child(char)

            if child is None:
                return False

            should_delete_child = _delete_recursive(child, word, index + 1)

            if should_delete_child:
                node.remove_child(char)
                return not node.is_end_of_word and not node.has_children()

            return False

        node = self._find_node(word)
        if node is None or not node.is_end_of_word:
            raise KeyNotFound(f"Word '{word}' not found")

        value = node.value
        _delete_recursive(self._root, word, 0)
        self._size -= 1
        return value

    def get_words_with_prefix(self, prefix: str) -> Iterator[str]:
        prefix_node = self._find_node(prefix)
        if prefix_node is None:
            return

        yield from self._collect_words(prefix_node, prefix)

    def _collect_words(self, node: TrieNode[ItemT], current_word: str) -> Iterator[str]:
        stack: list[tuple[TrieNode[ItemT], str]] = [(node, current_word)]

        while stack:
            current_node, current_path = stack.pop()

            if current_node.is_end_of_word:
                yield current_path

            for char, child in sorted(current_node.children.items()):
                stack.append((child, current_path + char))

    def get_words_with_values(self) -> Iterator[tuple[str, ItemT]]:
        yield from self._collect_words_with_values(self._root, "")

    def _collect_words_with_values(self, node: TrieNode[ItemT], current_word: str) -> Iterator[tuple[str, ItemT]]:
        stack: list[tuple[TrieNode[ItemT], str]] = [(node, current_word)]

        while stack:
            current_node, current_path = stack.pop()

            if current_node.is_end_of_word and current_node.value is not None:
                yield (current_path, current_node.value)

            for char, child in sorted(current_node.children.items()):
                stack.append((child, current_path + char))

    def get_all_words(self) -> Iterator[str]:
        yield from self._collect_words(self._root, "")

    def autocomplete(self, prefix: str, limit: int = 10) -> list[str]:
        results: list[str] = []

        for word in self.get_words_with_prefix(prefix):
            if len(results) >= limit:
                break
            results.append(word)

        return results

    def longest_common_prefix(self) -> str:
        if self.is_empty():
            return ""

        current = self._root
        prefix = ""

        while current.child_count() == 1 and not current.is_end_of_word:
            char = next(iter(current.children.keys()))
            prefix += char
            current = current.children[char]

        return prefix

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._root = TrieNode[ItemT]()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        return self.contains(word)

    def __getitem__(self, word: str) -> ItemT | None:
        return self.search(word)

    def __setitem__(self, word: str, value: ItemT) -> None:
        self.insert(word, value)

    def __delitem__(self, word: str) -> None:
        self.delete(word)

    def __iter__(self) -> Iterator[str]:
        yield from self.get_all_words()

    def __repr__(self) -> str:
        words = list(self.get_all_words())
        return f"{self.__class__.__name__}({words!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Trie):
            return NotImplemented

        if self.size() != other.size():
            return False

        self_words = set(self.get_all_words())
        other_words = set(other.get_all_words())

        return self_words == other_words

    def keys(self) -> Iterator[str]:
        yield from self.get_all_words()

    def values(self) -> Iterator[ItemT]:
        for _, value in self.get_words_with_values():
            yield value

    def items(self) -> Iterator[tuple[str, ItemT]]:
        yield from self.get_words_with_values()

    def get_default(self, word: str, default: ItemT) -> ItemT:
        try:
            result = self.search(word)
            return result if result is not None else default
        except KeyError:
            return default

    def update(self, other: Trie[ItemT] | dict[str, ItemT]) -> None:
        if isinstance(other, Trie):
            for word, value in other.items():
                self.insert(word, value)
        else:
            for word, value in other.items():
                self.insert(word, value)

    def pop(self, word: str, default: ItemT | None = None) -> ItemT | None:
        try:
            return self.delete(word)
        except KeyError:
            if default is not None:
                return default
            raise

    def setdefault(self, word: str, default: ItemT) -> ItemT | None:
        try:
            return self.search(word)
        except KeyError:
            self.insert(word, default)
            return default
