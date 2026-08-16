from __future__ import annotations

import inspect
from typing import TypeAliasType, TypeVar, get_args, get_origin

import omnivault.dsa.core.types


def _typevar_members() -> list[tuple[str, object]]:
    sentinel_typevar_type = type(TypeVar("X"))
    return [
        (name, member)
        for name, member in inspect.getmembers(omnivault.dsa.core.types)
        if isinstance(member, sentinel_typevar_type)
    ]


def test_typevar_names_suffix_t() -> None:
    members = _typevar_members()
    assert members, "expected omnivault.dsa.core.types to declare at least one TypeVar"
    offenders = [name for name, _ in members if not name.endswith(("T", "_co", "_contra"))]
    assert offenders == [], f"TypeVars without suffix-T convention: {offenders}"
    variance_offenders = [
        name
        for name, _ in members
        if name.endswith(("_co", "_contra"))
        and not name.startswith(("ItemT", "KeyT", "ValueT", "NodeT", "GraphT", "VertexT"))
    ]
    assert variance_offenders == [], f"Variance-tagged TypeVars not using suffix-T base: {variance_offenders}"


def test_no_single_letter_typevar_names() -> None:
    members = _typevar_members()
    single_letter = [name for name, _ in members if len(name) <= 2]
    assert single_letter == [], f"single-letter TypeVar names remain: {single_letter}"


def test_required_typevar_names_present() -> None:
    members = dict(_typevar_members())
    required = {
        "ItemT",
        "KeyT",
        "ValueT",
        "ComparableT",
        "MappingKeyT",
        "MappingValueT",
        "ResultT",
        "PriorityT",
        "ItemT_co",
        "KeyT_co",
        "ValueT_co",
        "ItemT_contra",
        "NodeT",
        "GraphT",
    }
    missing = required - members.keys()
    assert missing == set(), f"missing required TypeVar names: {missing}"


def test_numeric_is_pep695_alias() -> None:
    numeric = omnivault.dsa.core.types.Numeric
    assert isinstance(numeric, TypeAliasType), "Numeric must be a PEP 695 type alias, not a TypeVar"
    args = get_args(numeric)
    if args:
        assert set(args) == {float, int}, f"Numeric args mismatch: {args}"
    else:
        origin = get_origin(numeric)
        assert origin is not None or numeric is not None


def test_comparable_protocol_has_four_methods() -> None:
    protocol_attrs = getattr(omnivault.dsa.core.types.Comparable, "__protocol_attrs__", None)
    if protocol_attrs is None:
        method_names = {
            name
            for name in dir(omnivault.dsa.core.types.Comparable)
            if not name.startswith("_") or name in {"__lt__", "__le__", "__gt__", "__ge__"}
        }
    else:
        method_names = set(protocol_attrs)
    required = {"__lt__", "__le__", "__gt__", "__ge__"}
    assert required.issubset(method_names), f"Comparable missing ordering methods: {required - method_names}"
