# from __future__ import annotations

# from typing import Iterator, Optional, Sequence

# from omnivault._types._generic import T
# from omnivault.dsa.linked_list.base import AbstractLinkedList, DoublyNode, SinglyNode


# class SinglyLinkedList(AbstractLinkedList[T, SinglyNode[T]]):
#     """Concrete implementation of a singly linked list.

#     The `SinglyLinkedList` object is initialized with a head node.

#     The `head` node (the first node) of a **Linked List** is of a `Node` object.
#     The `head` **entirely determines** the entirety of the whole **Linked List**.
#     Because knowing the head node of the **Linked List**, we will be able to
#     know every single node that comes after it sequentially (if exists).
#     """

#     def add_multiple_nodes(self) -> None:
#         assert self._values is not None

#         first = SinglyNode(value=self._values[0])
#         self._head = first

#         for value in self._values[1:]:
#             current = SinglyNode(value=value)
#             current.next = self._head
#             self._head = current

#     def traverse(self) -> str:
#         temp_node = self._head
#         string = ""
#         while temp_node is not None:
#             string += f"{temp_node.value} -> "
#             temp_node = temp_node.next
#             if temp_node is None:
#                 string += "None"
#         return string

#     def append(self, value: T) -> None:
#         """Append a node to the end of the linked list.

#         Traverses the linked list until reaching the last node, then appends
#         the new node with the given value.

#         Parameters
#         ----------
#         value: T
#             The value to be stored in the new node that will be appended.

#         Notes
#         -----
#         Given a linked list with structure `head -> 1 -> 2 -> 3 -> None`,
#         appending value 4 involves:

#         1. Traversing until reaching the last node (node with value 3)
#         2. Setting its next pointer to the new node
#         3. Resulting in: `head -> 1 -> 2 -> 3 -> 4 -> None`

#         In more details, we would have `ll.head` to be `1 -> 2 -> 3 -> None` and
#         to have `4` appended to the end of the linked list done via the code
#         `ll.append(4)`. Then we would have the below pointer changes:

#         1. `new = SinglyNode(value=4)`;
#         2. Since `ll.head` is not None, we set `current = ll.head`;
#         3. We then traverse the linked list using `while current.next:` until
#         `current.next` is None;
#             - loop 1: `current = ll.head.next` which points to Node(2)
#             - loop 2: `current = ll.head.next.next` which points to Node(3)
#             - loop 3: `current.next` is None so we break out of the loop
#         4. We then set `current.next = new` which points the last node to the
#         new node. This is equivalent to saying `ll.head.next.next.next = Node(4)`.

#         Recall currently temp_node is Node(3) and temp_node.next_node is None
#         so we set temp_node.next_node to be the new_node
#         so now Node(3).next_node is Node(4)
#         this is equivalent to saying ll.head.next_node.next_node.next_node = Node(4)
#         so now the linked list is 1 -> 2 -> 3 -> 4 -> None

#         Time Complexity
#         --------------
#         O(n)
#             Where n is the number of nodes in the linked list.
#             This is due to the required traversal to reach the end.
#             Could be optimized to O(1) with a tail pointer.

#         Examples
#         --------
#         >>> ll = SinglyLinkedList()
#         >>> ll.append(1)
#         >>> ll.append(2)
#         >>> ll.append(3)
#         >>> print(ll)
#         1 -> 2 -> 3 -> None
#         """
#         new = SinglyNode(value=value)
#         if not self._head:  # same as self.is_empty()
#             self._head = new
#             return

#         # the head defines the whole linked list 1->2->3->None
#         current = self._head  # current is now pointing to the first node Node(1)
#         # traverse to the last node
#         while current.next:
#             current = current.next
#         current.next = new
#         self._size += 1

#     def prepend(self, value: T) -> None:
#         """Prepend a node to the beginning of the linked list.

#         Sets the new node as the head of the list and points its next pointer
#         to the current head.

#         Time Complexity
#         --------------
#         O(1)
#             Since we are only changing the pointers.

#         Parameters
#         ----------
#         value: T
#             The value to be stored in the new node that will be prepended.
#         """
#         new = SinglyNode(value=value)
#         new.next = self._head
#         self._head = new
#         self._size += 1

#     def remove(self, value: T) -> bool:
#         if not self._head:
#             return False

#         # if the head node is the value to remove
#         # for example if we have 1->2->3->None and we want to remove value=1
#         # we set the head to be the next node which is Node(2)
#         if self._head.value == value:
#             self._head = self._head.next
#             self._size -= 1
#             return True

#         # else we traverse the linked list until we find the value to remove
#         current = self._head
#         while current.next:
#             if current.next.value == value:
#                 current.next = current.next.next
#                 self._size -= 1
#                 return True
#             current = current.next
#         return False

#     def __iter__(self) -> Iterator[T]:
#         """Iterate through the linked list.

#         >>> ll = SinglyLinkedList()
#         >>> ll.append(1)
#         >>> ll.append(2)
#         >>> ll.append(3)
#         >>> for value in ll:
#         ...     print(value)
#         1
#         2
#         3
#         """
#         current = self._head
#         while current:
#             yield current.value
#             current = current.next


# class DoublyLinkedList(AbstractLinkedList[T, DoublyNode[T]]):
#     """Concrete implementation of a doubly linked list.

#     The DoublyLinkedList maintains both head and tail pointers for efficient
#     operations at both ends of the list. Each node contains references to both
#     its next and previous nodes.
#     """

#     def __init__(self, values: Sequence[T] | None = None) -> None:
#         super().__init__(values)
#         self._tail: Optional[DoublyNode[T]] = None

#     def add_multiple_nodes(self) -> None:
#         """Add multiple nodes to the linked list from initialization values."""
#         assert self._values is not None

#         first = DoublyNode(value=self._values[0])
#         self._head = self._tail = first
#         self._size += 1

#         # Append remaining values to maintain proper order
#         for value in self._values[1:]:
#             self.append(value)

#     def traverse(self) -> str:
#         """Return a string representation of the list in forward direction."""
#         temp_node = self._head
#         string = ""
#         while temp_node is not None:
#             string += f"{temp_node.value} <-> "
#             temp_node = temp_node.next
#             if temp_node is None:
#                 string += "None"
#         return string

#     def append(self, value: T) -> None:
#         """Append a node to the end of the linked list.

#         Time Complexity: O(1) - Using tail pointer for constant time append
#         """
#         new_node = DoublyNode(value=value)
#         self._size += 1

#         if not self._head:
#             self._head = self._tail = new_node
#             return

#         assert self._tail is not None  # for type checker
#         new_node.prev = self._tail
#         self._tail.next = new_node
#         self._tail = new_node

#     def prepend(self, value: T) -> None:
#         """Prepend a node to the beginning of the linked list.

#         Time Complexity: O(1)
#         """
#         new_node = DoublyNode(value=value)
#         self._size += 1

#         if not self._head:
#             self._head = self._tail = new_node
#             return

#         new_node.next = self._head
#         self._head.prev = new_node
#         self._head = new_node

#     def remove(self, value: T) -> bool:
#         """Remove the first occurrence of value from the list.

#         Time Complexity: O(n) where n is the number of nodes
#         """
#         current = self._head
#         while current:
#             if current.value == value:
#                 # Handle removal at head
#                 if current.prev is None:
#                     self._head = current.next
#                 else:
#                     current.prev.next = current.next

#                 # Handle removal at tail
#                 if current.next is None:
#                     self._tail = current.prev
#                 else:
#                     current.next.prev = current.prev

#                 self._size -= 1
#                 return True
#             current = current.next
#         return False

#     def __iter__(self) -> Iterator[T]:
#         """Forward iterator through the linked list."""
#         current = self._head
#         while current:
#             yield current.value
#             current = current.next

#     def __reversed__(self) -> Iterator[T]:
#         """Reverse iterator through the linked list."""
#         current = self._tail
#         while current:
#             yield current.value
#             current = current.prev


# def test_singly_linked_list():
#     print("\n=== Testing Singly Linked List ===")

#     # Create a new list
#     sll = SinglyLinkedList()
#     print("Created empty list:", list(sll))  # []

#     # Add elements
#     sll.append(1)
#     sll.append(2)
#     sll.append(3)
#     print("After appending 1, 2, 3:", list(sll))  # [1, 2, 3]

#     sll.prepend(0)
#     print("After prepending 0:", list(sll))  # [0, 1, 2, 3]

#     # Remove elements
#     sll.remove(2)
#     print("After removing 2:", list(sll))  # [0, 1, 3]

#     # Check length
#     print("Length of list:", len(sll))  # 3

#     # Iterate through elements
#     print("Iterating through list:")
#     for value in sll:
#         print(value)

#     print(sll)


# def test_doubly_linked_list():
#     print("\n=== Testing Doubly Linked List ===")

#     # Create a new list
#     dll = DoublyLinkedList()
#     print("Created empty list:", list(dll))  # []

#     # Add elements
#     dll.append(1)
#     dll.append(2)
#     dll.append(3)
#     print("After appending 1, 2, 3:", list(dll))  # [1, 2, 3]

#     dll.prepend(0)
#     print("After prepending 0:", list(dll))  # [0, 1, 2, 3]

#     # Remove elements
#     dll.remove(2)
#     print("After removing 2:", list(dll))  # [0, 1, 3]

#     # Check length
#     print("Length of list:", len(dll))  # 3

#     # Forward iteration
#     print("Forward iteration:")
#     for value in dll:
#         print(value)

#     # Backward iteration
#     print("Backward iteration:")
#     for value in reversed(dll):
#         print(value)

#     print(dll)


# if __name__ == "__main__":
#     test_singly_linked_list()
#     test_doubly_linked_list()


# from dsa.complexity.time import time_complexity_iterable
# from dsa.linked_list.linked_list import SinglyLinkedList


# @time_complexity_iterable("singly_linked_list", repeat=3, plot=True)
# def singly_linked_list_insert(n, ll: SinglyLinkedList):
#     ll.insert(n)  # O(1)


# @time_complexity_iterable("singly_linked_list", repeat=3, plot=True)
# def singly_linked_list_append(n, ll: SinglyLinkedList):
#     ll.append(n)  # O(n)


# if __name__ == "__main__":
#     n_values = list(range(1000, 10001, 1000))
#     for func in [singly_linked_list_append, singly_linked_list_insert]:
#         _ = func(n_values)
