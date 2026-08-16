from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic

from ..core.errors import EmptyContainer, IncompatibleSketch, InvalidConfiguration
from ..core.types import ItemT


class SegmentTree(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_tree", "_size", "_combine", "_identity", "_lazy", "_has_lazy")

    def __init__(self, data: Sequence[ItemT], combine: Callable[[ItemT, ItemT], ItemT], identity: ItemT) -> None:
        if not data:
            raise EmptyContainer("Data cannot be empty")

        self._size = len(data)
        self._combine = combine
        self._identity = identity
        self._tree = [identity] * (4 * self._size)
        self._lazy = [identity] * (4 * self._size)
        self._has_lazy = [False] * (4 * self._size)

        self._build(data, 0, 0, self._size - 1)

    def _build(self, data: Sequence[ItemT], node: int, start: int, end: int) -> None:
        if start == end:
            self._tree[node] = data[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            self._build(data, left_child, start, mid)
            self._build(data, right_child, mid + 1, end)

            self._tree[node] = self._combine(self._tree[left_child], self._tree[right_child])

    def _push(self, node: int, start: int, end: int) -> None:
        if self._has_lazy[node]:
            self._tree[node] = self._combine(self._tree[node], self._lazy[node])

            if start != end:
                left_child = 2 * node + 1
                right_child = 2 * node + 2

                self._lazy[left_child] = self._combine(self._lazy[left_child], self._lazy[node])
                self._lazy[right_child] = self._combine(self._lazy[right_child], self._lazy[node])
                self._has_lazy[left_child] = True
                self._has_lazy[right_child] = True

            self._lazy[node] = self._identity
            self._has_lazy[node] = False

    def query(self, left: int, right: int) -> ItemT:
        if left < 0 or right >= self._size or left > right:
            raise IncompatibleSketch("Invalid range")

        return self._query_recursive(0, 0, self._size - 1, left, right)

    def _query_recursive(self, node: int, start: int, end: int, left: int, right: int) -> ItemT:
        if right < start or end < left:
            return self._identity

        self._push(node, start, end)

        if left <= start and end <= right:
            return self._tree[node]

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        left_result = self._query_recursive(left_child, start, mid, left, right)
        right_result = self._query_recursive(right_child, mid + 1, end, left, right)

        return self._combine(left_result, right_result)

    def update_point(self, index: int, value: ItemT) -> None:
        if index < 0 or index >= self._size:
            raise IncompatibleSketch("Index out of bounds")

        self._update_point_recursive(0, 0, self._size - 1, index, value)

    def _update_point_recursive(self, node: int, start: int, end: int, index: int, value: ItemT) -> None:
        self._push(node, start, end)

        if start == end:
            self._tree[node] = value
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            if index <= mid:
                self._update_point_recursive(left_child, start, mid, index, value)
            else:
                self._update_point_recursive(right_child, mid + 1, end, index, value)

            self._push(left_child, start, mid)
            self._push(right_child, mid + 1, end)

            self._tree[node] = self._combine(self._tree[left_child], self._tree[right_child])

    def update_range(self, left: int, right: int, value: ItemT) -> None:
        if left < 0 or right >= self._size or left > right:
            raise IncompatibleSketch("Invalid range")

        self._update_range_recursive(0, 0, self._size - 1, left, right, value)

    def _update_range_recursive(self, node: int, start: int, end: int, left: int, right: int, value: ItemT) -> None:
        self._push(node, start, end)

        if right < start or end < left:
            return

        if left <= start and end <= right:
            self._lazy[node] = self._combine(self._lazy[node], value)
            self._has_lazy[node] = True
            self._push(node, start, end)
            return

        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2

        self._update_range_recursive(left_child, start, mid, left, right, value)
        self._update_range_recursive(right_child, mid + 1, end, left, right, value)

        self._push(left_child, start, mid)
        self._push(right_child, mid + 1, end)

        self._tree[node] = self._combine(self._tree[left_child], self._tree[right_child])

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> ItemT:
        return self.query(index, index)

    def __setitem__(self, index: int, value: ItemT) -> None:
        self.update_point(index, value)

    def __repr__(self) -> str:
        values = [self[i] for i in range(self._size)]
        return f"{self.__class__.__name__}({values!r})"


class FenwickTree(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_tree", "_size", "_combine", "_identity", "_inverse")

    def __init__(
        self,
        size: int,
        combine: Callable[[ItemT, ItemT], ItemT],
        identity: ItemT,
        inverse: Callable[[ItemT], ItemT] | None = None,
    ) -> None:
        if size <= 0:
            raise InvalidConfiguration("Size must be positive")

        self._size = size
        self._combine = combine
        self._identity = identity
        self._inverse = inverse
        self._tree = [identity] * (size + 1)

    @classmethod
    def from_data(
        cls,
        data: Sequence[ItemT],
        combine: Callable[[ItemT, ItemT], ItemT],
        identity: ItemT,
        inverse: Callable[[ItemT], ItemT] | None = None,
    ) -> FenwickTree[ItemT]:
        tree = cls(len(data), combine, identity, inverse)

        for i, value in enumerate(data):
            tree.update(i, value)

        return tree

    def update(self, index: int, delta: ItemT) -> None:
        if index < 0 or index >= self._size:
            raise IncompatibleSketch("Index out of bounds")

        index += 1

        while index <= self._size:
            self._tree[index] = self._combine(self._tree[index], delta)
            index += index & (-index)

    def query(self, index: int) -> ItemT:
        if index < 0 or index >= self._size:
            raise IncompatibleSketch("Index out of bounds")

        index += 1
        result = self._identity

        while index > 0:
            result = self._combine(result, self._tree[index])
            index -= index & (-index)

        return result

    def range_query(self, left: int, right: int) -> ItemT:
        if left < 0 or right >= self._size or left > right:
            raise IncompatibleSketch("Invalid range")

        if left == 0:
            return self.query(right)

        if self._inverse is None:
            raise IncompatibleSketch("Inverse function required for range queries")

        right_sum = self.query(right)
        left_sum = self.query(left - 1)

        return self._combine(right_sum, self._inverse(left_sum))

    def set_value(self, index: int, value: ItemT) -> None:
        if self._inverse is None:
            raise IncompatibleSketch("Inverse function required for setting values")

        current = self.query(index)
        if index > 0:
            current = self._combine(current, self._inverse(self.query(index - 1)))

        delta = self._combine(value, self._inverse(current))
        self.update(index, delta)

    def lower_bound(self, target: ItemT, compare: Callable[[ItemT, ItemT], bool]) -> int:
        if self._size == 0:
            return 0

        index = 0
        bit_mask = 1

        while bit_mask <= self._size:
            bit_mask <<= 1
        bit_mask >>= 1

        current_sum = self._identity

        while bit_mask > 0:
            next_index = index + bit_mask

            if next_index <= self._size:
                next_sum = self._combine(current_sum, self._tree[next_index])

                if compare(next_sum, target):
                    index = next_index
                    current_sum = next_sum

            bit_mask >>= 1

        return index

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> ItemT:
        return self.query(index)

    def __repr__(self) -> str:
        values = [self.query(i) for i in range(self._size)]
        return f"{self.__class__.__name__}({values!r})"


class BinaryIndexedTree2D(Generic[ItemT]):  # noqa: UP046
    __slots__ = ("_tree", "_rows", "_cols", "_combine", "_identity")

    def __init__(self, rows: int, cols: int, combine: Callable[[ItemT, ItemT], ItemT], identity: ItemT) -> None:
        if rows <= 0 or cols <= 0:
            raise InvalidConfiguration("Dimensions must be positive")

        self._rows = rows
        self._cols = cols
        self._combine = combine
        self._identity = identity
        self._tree = [[identity] * (cols + 1) for _ in range(rows + 1)]

    def update(self, row: int, col: int, delta: ItemT) -> None:
        if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
            raise IncompatibleSketch("Coordinates out of bounds")

        row += 1
        original_col = col + 1

        while row <= self._rows:
            col = original_col
            while col <= self._cols:
                self._tree[row][col] = self._combine(self._tree[row][col], delta)
                col += col & (-col)
            row += row & (-row)

    def query(self, row: int, col: int) -> ItemT:
        if row < 0 or row >= self._rows or col < 0 or col >= self._cols:
            raise IncompatibleSketch("Coordinates out of bounds")

        row += 1
        original_col = col + 1
        result = self._identity

        while row > 0:
            col = original_col
            while col > 0:
                result = self._combine(result, self._tree[row][col])
                col -= col & (-col)
            row -= row & (-row)

        return result

    def range_query(self, row1: int, col1: int, row2: int, col2: int) -> ItemT:
        if (
            row1 < 0
            or row1 >= self._rows
            or col1 < 0
            or col1 >= self._cols
            or row2 < 0
            or row2 >= self._rows
            or col2 < 0
            or col2 >= self._cols
            or row1 > row2
            or col1 > col2
        ):
            raise IncompatibleSketch("Invalid coordinates")

        total = self.query(row2, col2)

        if row1 > 0:
            total = self._subtract(total, self.query(row1 - 1, col2))

        if col1 > 0:
            total = self._subtract(total, self.query(row2, col1 - 1))

        if row1 > 0 and col1 > 0:
            total = self._combine(total, self.query(row1 - 1, col1 - 1))

        return total

    def _subtract(self, a: ItemT, _b: ItemT) -> ItemT:
        return a

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self._rows, self._cols)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rows={self._rows}, cols={self._cols})"
