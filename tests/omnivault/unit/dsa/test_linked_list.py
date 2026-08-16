from omnivault.dsa.containers.linear.linked_list import DoublyLinkedList, SinglyLinkedList


class TestSinglyLinkedList:
    def test_empty_list(self) -> None:
        sll = SinglyLinkedList[int]()
        assert len(sll) == 0
        assert sll.is_empty()
        assert list(sll) == []
        assert str(sll) == "None"

    def test_append(self) -> None:
        sll = SinglyLinkedList[int]()
        sll.append(1)
        sll.append(2)
        sll.append(3)
        assert len(sll) == 3
        assert list(sll) == [1, 2, 3]
        assert str(sll) == "1 -> 2 -> 3 -> None"

    def test_prepend(self) -> None:
        sll = SinglyLinkedList[int]()
        sll.prepend(3)
        sll.prepend(2)
        sll.prepend(1)
        assert len(sll) == 3
        assert list(sll) == [1, 2, 3]
        assert str(sll) == "1 -> 2 -> 3 -> None"

    def test_remove_head(self) -> None:
        sll = SinglyLinkedList[int]()
        sll.append(1)
        sll.append(2)
        sll.append(3)
        assert sll.remove(1)
        assert len(sll) == 2
        assert list(sll) == [2, 3]

    def test_remove_middle(self) -> None:
        sll = SinglyLinkedList[int]()
        sll.append(1)
        sll.append(2)
        sll.append(3)
        assert sll.remove(2)
        assert len(sll) == 2
        assert list(sll) == [1, 3]

    def test_remove_tail(self) -> None:
        sll = SinglyLinkedList[int]()
        sll.append(1)
        sll.append(2)
        sll.append(3)
        assert sll.remove(3)
        assert len(sll) == 2
        assert list(sll) == [1, 2]

    def test_remove_not_found(self) -> None:
        sll = SinglyLinkedList[int]()
        sll.append(1)
        sll.append(2)
        assert not sll.remove(3)
        assert len(sll) == 2

    def test_clear(self) -> None:
        sll = SinglyLinkedList[int]()
        sll.append(1)
        sll.append(2)
        sll.clear()
        assert len(sll) == 0
        assert sll.is_empty()

    def test_initialize_with_values(self) -> None:
        sll = SinglyLinkedList[int]([1, 2, 3])
        assert len(sll) == 3
        assert list(sll) == [1, 2, 3]

    def test_iteration(self) -> None:
        sll = SinglyLinkedList[int]([1, 2, 3])
        values = list(sll)
        assert values == [1, 2, 3]

    def test_bool(self) -> None:
        sll = SinglyLinkedList[int]()
        assert not sll
        sll.append(1)
        assert sll


class TestDoublyLinkedList:
    def test_empty_list(self) -> None:
        dll = DoublyLinkedList[int]()
        assert len(dll) == 0
        assert dll.is_empty()
        assert list(dll) == []
        assert str(dll) == "None"

    def test_append(self) -> None:
        dll = DoublyLinkedList[int]()
        dll.append(1)
        dll.append(2)
        dll.append(3)
        assert len(dll) == 3
        assert list(dll) == [1, 2, 3]
        assert str(dll) == "1 <-> 2 <-> 3 <-> None"

    def test_prepend(self) -> None:
        dll = DoublyLinkedList[int]()
        dll.prepend(3)
        dll.prepend(2)
        dll.prepend(1)
        assert len(dll) == 3
        assert list(dll) == [1, 2, 3]
        assert str(dll) == "1 <-> 2 <-> 3 <-> None"

    def test_remove_head(self) -> None:
        dll = DoublyLinkedList[int]()
        dll.append(1)
        dll.append(2)
        dll.append(3)
        assert dll.remove(1)
        assert len(dll) == 2
        assert list(dll) == [2, 3]

    def test_remove_middle(self) -> None:
        dll = DoublyLinkedList[int]()
        dll.append(1)
        dll.append(2)
        dll.append(3)
        assert dll.remove(2)
        assert len(dll) == 2
        assert list(dll) == [1, 3]

    def test_remove_tail(self) -> None:
        dll = DoublyLinkedList[int]()
        dll.append(1)
        dll.append(2)
        dll.append(3)
        assert dll.remove(3)
        assert len(dll) == 2
        assert list(dll) == [1, 2]

    def test_remove_not_found(self) -> None:
        dll = DoublyLinkedList[int]()
        dll.append(1)
        dll.append(2)
        assert not dll.remove(3)
        assert len(dll) == 2

    def test_clear(self) -> None:
        dll = DoublyLinkedList[int]()
        dll.append(1)
        dll.append(2)
        dll.clear()
        assert len(dll) == 0
        assert dll.is_empty()
        assert dll.tail is None

    def test_initialize_with_values(self) -> None:
        dll = DoublyLinkedList[int]([1, 2, 3])
        assert len(dll) == 3
        assert list(dll) == [1, 2, 3]

    def test_forward_iteration(self) -> None:
        dll = DoublyLinkedList[int]([1, 2, 3])
        values = list(dll)
        assert values == [1, 2, 3]

    def test_reverse_iteration(self) -> None:
        dll = DoublyLinkedList[int]([1, 2, 3])
        values = list(reversed(dll))
        assert values == [3, 2, 1]

    def test_bool(self) -> None:
        dll = DoublyLinkedList[int]()
        assert not dll
        dll.append(1)
        assert dll

    def test_tail_property(self) -> None:
        dll = DoublyLinkedList[int]()
        assert (dll.tail,) == (None,)
        dll.append(1)
        tail = dll.tail
        assert tail is not None
        assert tail.value == 1
        dll.append(2)
        tail = dll.tail
        assert tail is not None
        assert tail.value == 2
