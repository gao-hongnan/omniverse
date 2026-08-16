from __future__ import annotations

import random
from collections.abc import Iterator
from typing import cast

from pydantic import BaseModel, ConfigDict

from ...core.errors import EmptyContainer, InvalidConfiguration, KeyNotFound
from ...core.types import Comparable


class SkipListNode[KeyT: Comparable, ValueT](BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: KeyT | None = None
    value: ValueT | None = None
    forward: list[SkipListNode[KeyT, ValueT] | None] = []

    def __init__(self, level: int, key: KeyT | None = None, value: ValueT | None = None, **data: object) -> None:
        super().__init__(key=key, value=value, **data)
        self.forward = [None] * (level + 1)


class SkipList[KeyT: Comparable, ValueT]:
    __slots__ = ("_header", "_level", "_size", "_max_level", "_probability")

    def __init__(self, max_level: int = 16, probability: float = 0.5) -> None:
        if max_level <= 0:
            raise InvalidConfiguration("Max level must be positive")
        if not 0 < probability < 1:
            raise InvalidConfiguration("Probability must be between 0 and 1")

        self._max_level = max_level
        self._probability = probability
        self._level = 0
        self._size = 0
        self._header = SkipListNode[KeyT, ValueT](max_level)

    def _random_level(self) -> int:
        level = 0
        while random.random() < self._probability and level < self._max_level:
            level += 1
        return level

    def search(self, key: KeyT) -> ValueT:
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                node = current.forward[i]
                assert node is not None
                if node.key is not None and node.key < key:
                    current = node
                else:
                    break

        next_node = current.forward[0]

        if next_node is not None and next_node.key == key:
            assert next_node.value is not None
            return next_node.value

        raise KeyNotFound(f"Key {key} not found")

    def insert(self, key: KeyT, value: ValueT) -> None:
        update: list[SkipListNode[KeyT, ValueT]] = []
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                node = current.forward[i]
                assert node is not None
                if node.key is not None and node.key < key:
                    current = node
                else:
                    break
            update.insert(0, current)

        next_node = current.forward[0]

        if next_node is not None and next_node.key == key:
            next_node.value = value
            return

        new_level = self._random_level()

        if new_level > self._level:
            update.extend([self._header] * (new_level - self._level))
            self._level = new_level

        new_node = SkipListNode[KeyT, ValueT](new_level, key, value)

        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self._size += 1

    def delete(self, key: KeyT) -> ValueT:
        update: list[SkipListNode[KeyT, ValueT]] = []
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                node = current.forward[i]
                assert node is not None
                if node.key is not None and node.key < key:
                    current = node
                else:
                    break
            update.insert(0, current)

        target_node = current.forward[0]

        if target_node is None or target_node.key != key:
            raise KeyNotFound(f"Key {key} not found")

        assert target_node.value is not None
        deleted_value = target_node.value

        for i in range(len(target_node.forward)):
            if update[i].forward[i] is not target_node:
                break
            update[i].forward[i] = target_node.forward[i]

        while self._level > 0 and self._header.forward[self._level] is None:
            self._level -= 1

        self._size -= 1
        return deleted_value

    def contains(self, key: KeyT) -> bool:
        try:
            self.search(key)
            return True
        except KeyError:
            return False

    def min_key(self) -> KeyT:
        if self.is_empty():
            raise EmptyContainer("Skip list is empty")

        first_node = self._header.forward[0]
        assert first_node is not None and first_node.key is not None
        return first_node.key

    def max_key(self) -> KeyT:
        if self.is_empty():
            raise EmptyContainer("Skip list is empty")

        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                node = current.forward[i]
                assert node is not None
                current = node

        assert current.key is not None
        return current.key

    def predecessor(self, key: KeyT) -> KeyT | None:
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                node = current.forward[i]
                assert node is not None
                if node.key is not None and node.key < key:
                    current = node
                else:
                    break

        return current.key if current != self._header else None

    def successor(self, key: KeyT) -> KeyT | None:
        current = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                node = current.forward[i]
                assert node is not None
                if node.key is not None and node.key <= key:
                    current = node
                else:
                    break

        next_node = current.forward[0]
        return next_node.key if next_node is not None else None

    def range_search(self, start_key: KeyT, end_key: KeyT) -> Iterator[tuple[KeyT, ValueT]]:
        if start_key > end_key:
            return

        current: SkipListNode[KeyT, ValueT] = self._header

        for i in range(self._level, -1, -1):
            while current.forward[i] is not None:
                node = current.forward[i]
                assert node is not None
                if node.key is not None and node.key < start_key:
                    current = node
                else:
                    break

        cursor: SkipListNode[KeyT, ValueT] | None = current.forward[0]

        while cursor is not None:
            if cursor.key is not None and cursor.key <= end_key:
                assert cursor.value is not None
                yield (cursor.key, cursor.value)
                cursor = cursor.forward[0]
            else:
                break

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._header = SkipListNode[KeyT, ValueT](self._max_level)
        self._level = 0
        self._size = 0

    def keys(self) -> Iterator[KeyT]:
        current = self._header.forward[0]

        while current is not None:
            assert current.key is not None
            yield current.key
            current = current.forward[0]

    def values(self) -> Iterator[ValueT]:
        current = self._header.forward[0]

        while current is not None:
            assert current.value is not None
            yield current.value
            current = current.forward[0]

    def items(self) -> Iterator[tuple[KeyT, ValueT]]:
        current = self._header.forward[0]

        while current is not None:
            assert current.key is not None and current.value is not None
            yield (current.key, current.value)
            current = current.forward[0]

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __contains__(self, key: object) -> bool:
        try:
            if isinstance(key, Comparable):
                return self.contains(cast(KeyT, key))
            return False
        except TypeError, AttributeError:
            return False

    def __getitem__(self, key: KeyT) -> ValueT:
        return self.search(key)

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        self.insert(key, value)

    def __delitem__(self, key: KeyT) -> None:
        self.delete(key)

    def __iter__(self) -> Iterator[KeyT]:
        yield from self.keys()

    def __repr__(self) -> str:
        items = ", ".join(f"{k!r}: {v!r}" for k, v in self.items())
        return f"{self.__class__.__name__}({{{items}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SkipList):
            return NotImplemented

        if self.size() != other.size():
            return False

        return all(self_item == other_item for self_item, other_item in zip(self.items(), other.items(), strict=True))

    def get_default(self, key: KeyT, default: ValueT) -> ValueT:
        try:
            return self.search(key)
        except KeyError:
            return default

    def update(self, other: SkipList[KeyT, ValueT] | dict[KeyT, ValueT]) -> None:
        if isinstance(other, SkipList):
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
            raise EmptyContainer("popitem(): skip list is empty")

        max_key = self.max_key()
        value = self.delete(max_key)
        return (max_key, value)

    def setdefault(self, key: KeyT, default: ValueT) -> ValueT:
        try:
            return self.search(key)
        except KeyError:
            self.insert(key, default)
            return default

    @property
    def height(self) -> int:
        return self._level + 1

    @property
    def max_height(self) -> int:
        return self._max_level + 1

    def display_structure(self) -> str:
        lines: list[str] = []

        for level in range(self._level, -1, -1):
            line = f"Level {level}: "
            current = self._header

            while current.forward[level] is not None:
                node = current.forward[level]
                assert node is not None
                current = node
                line += f"{current.key} -> "

            line += "None"
            lines.append(line)

        return "\n".join(lines)
