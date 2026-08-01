from collections.abc import Iterator
from typing import override

from omnivault.dsa.stack.base import EmptyStackError, Stack


class StackList[ItemT](Stack[ItemT]):
    """Creates a stack that uses python's default list as the underlying
    data structure.

    Note
    ----
    Methods are ordered with
    dunder/magic/property -> public -> private -> static/class.

    Attributes
    ----------
    _stack_items : List[ItemT]
        The list that stores the items in the stack. We treat the end of the
        list as the top of the stack.
    """

    @override
    def __len__(self) -> int:
        """Return the size of the stack."""
        return len(self.stack_items)

    @override
    def __iter__(self) -> Iterator[ItemT]:
        """Iterate over the stack items.

        Note
        ----
        If we return self, then we need to define `__next__`
        to make it an iterator. Else, python handles the
        `__next__` method for us if `__iter__` returns an
        iterator.

        ```python
        def __next__(self) -> StackList[ItemT]:
            if self.is_empty():
                raise StopIteration
            return self.pop()
        ```

        Returns
        -------
        StackList[ItemT]
            The stack.
        """

        while not self.is_empty():
            yield self.pop()

    def __repr__(self) -> str:
        """Return the official string representation of the StackList."""
        return f"StackList(stack_items={self.stack_items})"

    @property
    def stack_items(self) -> list[ItemT]:
        """Read only property for the stack items."""
        return self._stack_items

    @property
    def size(self) -> int:
        """Return the size of the stack.

        Note
        ----
        When you call `len(self)` from within the class, it will call internally
        `self.__len__()` (`StackList.__len__()`) which will return the size of
        the stack.

        Returns
        -------
        int
            The size of the stack.
        """
        return len(self)

    @override
    def is_empty(self) -> bool:
        """Check if stack is empty.

        Returns
        -------
        bool
            True if stack is empty, False otherwise.
        """
        return not self.stack_items

    @override
    def peek(self) -> ItemT:
        """Return the top most item in the stack without modifying the stack.

        This is different from pop in that it does not remove the item from the
        stack.

        Returns
        -------
        ItemT
            The top most item in the stack.
        """
        return self.stack_items[-1]

    @override
    def pop(self) -> ItemT:
        """Pop an item from the top of the stack.

        In this implementation, the item at the end of the list is returned
        and removed. We are using the list's pop method to do this.

        Raises
        ------
        (Exception): If stack is empty.

        Returns
        -------
        ItemT
            The top most item in the stack.
        """
        if self.is_empty():
            raise EmptyStackError("Stack is empty")
        return self.stack_items.pop()

    @override
    def push(self, item: ItemT) -> None:
        """Push an item on top of the stack.

        In this implementation, the item is appended to the end of the list.

        Parameters
        ----------
        item : ItemT
            The current item pushed into the stack.
        """
        self.stack_items.append(item)
