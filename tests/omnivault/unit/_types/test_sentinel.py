# pytest tests/omnivault/unit/_types/test_sentinel.py -v
import threading
from collections.abc import Iterator
from threading import Thread

import pytest

from omnivault._types._sentinel import Singleton


class SingletonExample(metaclass=Singleton):
    """Basic singleton class for testing."""

    def __init__(self) -> None:
        self.value: int = 0


class GenericSingleton[T](metaclass=Singleton):
    """Generic singleton class for testing type parameters."""

    def __init__(self, value: T) -> None:
        self.value: T = value


class MutableSingleton(metaclass=Singleton):
    """Singleton class with mutable state for testing thread safety."""

    def __init__(self) -> None:
        self.counter: int = 0

    def increment(self) -> None:
        self.counter += 1


@pytest.fixture(autouse=True)
def _isolated_singleton_registry() -> Iterator[None]:
    """Run every test against an empty singleton registry.

    ``Singleton._instances`` is module-global state: without this reset, a test
    observes instances (and mutated state) created by whichever test ran first,
    making the file order-dependent. Pre-existing registrations from other test
    modules are restored on teardown.
    """
    registry = Singleton._instances  # type: ignore[misc]  # deliberate white-box access to the metaclass registry for isolation
    saved = dict(registry)
    registry.clear()
    yield
    registry.clear()
    registry.update(saved)


def test_singleton_identity() -> None:
    """Test that multiple instantiations return the same instance."""
    first = SingletonExample()
    second = SingletonExample()

    assert first is second


def test_singleton_state() -> None:
    """Test that singleton maintains state across instances."""
    first = SingletonExample()
    first.value = 42

    second = SingletonExample()

    assert second.value == 42


def test_singleton_thread_safety() -> None:
    """Test thread safety of singleton creation."""
    thread_count = 10
    singleton_instances: list[MutableSingleton] = []

    def create_singleton() -> None:
        singleton_instances.append(MutableSingleton())

    threads = [Thread(target=create_singleton) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(singleton_instances) == thread_count
    assert all(instance is singleton_instances[0] for instance in singleton_instances)


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


def test_singleton_metaclass_instances() -> None:
    """White-box test of the singleton metaclass instance storage."""
    singleton_instance = SingletonExample()

    metaclass_instance = type(singleton_instance)

    # `_instances` is declared on the generic metaclass `Singleton[T]`; reading it off
    # the class object is what this test asserts, which pyright flags as ambiguous.
    assert isinstance(metaclass_instance._instances, dict)  # pyright: ignore[reportGeneralTypeIssues]
    assert isinstance(metaclass_instance._lock, type(threading.Lock()))
