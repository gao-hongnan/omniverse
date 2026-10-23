"""Guard ``omnivault.__version__`` — release.yaml's test-install job asserts it."""

import re
from importlib.metadata import PackageNotFoundError, version

import pytest

import omnivault


def test_version_attribute_exists_and_is_formatted() -> None:
    assert isinstance(omnivault.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+\S*", omnivault.__version__), omnivault.__version__


def test_version_matches_installed_distribution_metadata() -> None:
    try:
        installed = version("omniverse")
    except PackageNotFoundError:  # pragma: no cover
        pytest.skip("omniverse distribution not installed (bare source checkout)")
    assert omnivault.__version__ == installed
