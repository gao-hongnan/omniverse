# pytest tests/omnivault/unit/_types/test_sentinel.py -v
import threading
from threading import Thread
from typing import Generic

from omnivault._types._generic import T
from omnivault._types._sentinel import Singleton


class SingletonExample(metaclass=Singleton):
    """Basic singleton class for testing."""

    def __init__(self) -> None:
        self.value: int = 0


class GenericSingleton(Generic[T], metaclass=Singleton):
    """Generic singleton class for testing type parameters."""

    def __init__(self, value: T) -> None:
        self.value: T = value


class MutableSingleton(metaclass=Singleton):
    """Singleton class with mutable state for testing thread safety."""

    def __init__(self) -> None:
        self.counter: int = 0

    def increment(self) -> None:
        self.counter += 1


def test_singleton_identity() -> None:
    """Test that multiple instantiations return the same instance."""
    first = SingletonExample()
    second = SingletonExample()
    assert first is second
    assert id(first) == id(second)


def test_singleton_state() -> None:
    """Test that singleton maintains state across instances."""
    first = SingletonExample()
    first.value = 42
    second = SingletonExample()
    assert second.value == 42


# def test_generic_singleton_type_safety() -> None:
#     """Test generic singleton with different type parameters."""
#     int_singleton = GenericSingleton[int](42)
#     str_singleton = GenericSingleton[str]("test")

#     assert isinstance(int_singleton.value, int)
#     assert isinstance(str_singleton.value, str)

#     # Verify type consistency
#     same_int_singleton = GenericSingleton[int](100)
#     assert same_int_singleton is int_singleton
#     assert same_int_singleton.value == 42  # Original value preserved


def test_singleton_thread_safety() -> None:
    """Test thread safety of singleton creation."""
    singleton_instances: list[MutableSingleton] = []

    def create_singleton() -> None:
        singleton_instances.append(MutableSingleton())

    threads = [Thread(target=create_singleton) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Verify all instances are the same
    first_instance = singleton_instances[0]
    for instance in singleton_instances[1:]:
        assert instance is first_instance


def test_singleton_concurrent_state_modification() -> None:
    """Test thread safety of singleton state modifications."""
    singleton = MutableSingleton()
    thread_count = 100

    def modify_singleton() -> None:
        singleton.increment()

    threads = [Thread(target=modify_singleton) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert singleton.counter == thread_count


def test_singleton_inheritance() -> None:
    """Test that singleton behavior works with inheritance."""

    class ChildSingleton(SingletonExample):
        pass

    parent1 = SingletonExample()
    parent2 = SingletonExample()
    child1 = ChildSingleton()
    child2 = ChildSingleton()

    assert parent1 is parent2
    assert child1 is child2
    assert parent1 is not child1


def test_singleton_args_ignored() -> None:
    """Test that subsequent instantiations ignore constructor arguments."""
    first = GenericSingleton[int](42)
    second = GenericSingleton[int](99)  # Different argument

    assert first is second
    assert first.value == 42  # Original value preserved
    assert second.value == 42


# def test_singleton_type_errors() -> None:
#     """Test type-related errors with generic singleton."""
#     with pytest.raises(TypeError):
#         # This should fail static type checking, but we test runtime behavior
#         GenericSingleton[int]("wrong type")  # type: ignore


def test_singleton_metaclass_instances() -> None:
    """Test the singleton metaclass instance storage."""
    singleton_instance = SingletonExample()
    metaclass_instance = type(singleton_instance)
    assert isinstance(metaclass_instance._instances, dict)
    assert isinstance(metaclass_instance._lock, type(threading.Lock()))
