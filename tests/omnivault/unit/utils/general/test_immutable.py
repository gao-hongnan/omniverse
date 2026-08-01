from typing import List

import pytest

from omnivault.utils.general.immutable import ImmutableProxy


def test_immutable_proxy_attribute_access() -> None:
    """Test that we can access attributes of the wrapped object."""
    original = [1, 2, 3]
    proxy = ImmutableProxy(original)

    # We can only access non-callable attributes
    assert proxy._obj == original

    # Read-only operations like len() are allowed on the proxy.
    assert len(proxy) == 3


def test_immutable_proxy_prevent_modification() -> None:
    """Test that modification attempts raise AttributeError."""
    original: List[int] = [1, 2, 3]
    proxy = ImmutableProxy(original)

    with pytest.raises(AttributeError) as exc_info:
        proxy.append(4)
    assert "Attempting to modify object with method `append`" in str(exc_info.value)
    assert "`list` object is immutable" in str(exc_info.value)


def test_immutable_proxy_prevent_attribute_setting() -> None:
    """Test that setting attributes raises AttributeError."""
    proxy = ImmutableProxy([1, 2, 3])

    with pytest.raises(AttributeError) as exc_info:
        proxy.new_attr = 42
    assert "Attempting to set attribute" in str(exc_info.value)
    assert "`list` object is immutable" in str(exc_info.value)


def test_immutable_proxy_with_custom_object() -> None:
    """Test ImmutableProxy with a custom class."""

    class CustomClass:
        def __init__(self) -> None:
            self.value = 42

        def modify(self) -> None:
            self.value += 1

    proxy = ImmutableProxy(CustomClass())

    # Can access attributes
    assert proxy.value == 42

    # Cannot modify
    with pytest.raises(AttributeError):
        proxy.modify()

    with pytest.raises(AttributeError):
        proxy.value = 43


def test_immutable_proxy_callable_attributes() -> None:
    """Test that callable attributes are properly handled."""
    original = [1, 2, 3]
    proxy = ImmutableProxy(original)

    # Read-only methods like count() are allowed on the proxy.
    assert proxy.count(1) == 1


def test_immutable_proxy_original_unchanged() -> None:
    """Test that the original object remains unchanged and mutable."""
    original = [1, 2, 3]
    proxy = ImmutableProxy(original)

    # Proxy operations shouldn't affect original
    with pytest.raises(AttributeError):
        proxy.append(4)

    # Original should still be mutable
    original.append(4)
    assert original == [1, 2, 3, 4]
