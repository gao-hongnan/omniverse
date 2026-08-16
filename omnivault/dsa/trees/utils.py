from __future__ import annotations

import functools as fn
from typing import TYPE_CHECKING, cast

from .binary import BinaryTreeNode

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

type NodeInfo[NodeT] = Callable[[NodeT], tuple[str, NodeT | None, NodeT | None]]


def build_binary_tree_from_list_preorder[ItemT](values: Iterable[ItemT | None]) -> BinaryTreeNode[ItemT] | None:
    cursor = iter(values)

    def build() -> BinaryTreeNode[ItemT] | None:
        value = next(cursor, None)
        if value is None:
            return None

        node = BinaryTreeNode(value)
        node.left = build()
        node.right = build()
        return node

    return build()


def _binary_tree_node_info[ItemT](
    node: BinaryTreeNode[ItemT],
) -> tuple[str, BinaryTreeNode[ItemT] | None, BinaryTreeNode[ItemT] | None]:
    return str(node.value), node.left, node.right


def print_binary_tree[NodeT](
    node: NodeT,
    node_info: NodeInfo[NodeT] | None = None,
    *,
    inverted: bool = False,
    is_top: bool = True,
) -> list[str] | None:
    describe: NodeInfo[NodeT] = node_info if node_info is not None else cast("NodeInfo[NodeT]", _binary_tree_node_info)
    string_value, left_child, right_child = describe(node)
    string_width = len(string_value)

    left_block = (
        [] if not left_child else print_binary_tree(left_child, describe, inverted=inverted, is_top=False) or []
    )
    right_block = (
        [] if not right_child else print_binary_tree(right_child, describe, inverted=inverted, is_top=False) or []
    )

    common_lines = min(len(left_block), len(right_block))
    sub_level_lines = max(len(left_block), len(right_block))

    left_sub_lines = left_block + [""] * (sub_level_lines - len(left_block))
    right_sub_lines = right_block + [""] * (sub_level_lines - len(right_block))

    left_line_widths = [len(line) for line in left_sub_lines]
    right_line_indents = [len(line) - len(line.lstrip(" ")) for line in right_sub_lines]

    first_left_width = (left_line_widths + [0])[0]
    first_right_indent = (right_line_indents + [0])[0]

    link_spacing = min(string_width, 2 - string_width % 2)
    left_link_bar = 1 if left_child else 0
    right_link_bar = 1 if right_child else 0
    min_link_width = left_link_bar + link_spacing + right_link_bar
    value_offset = (string_width - link_spacing) // 2

    min_spacing = 2
    right_node_position = fn.reduce(
        lambda r, pair: max(r, pair[0] + min_spacing + first_right_indent - pair[1]),
        zip(left_line_widths, right_line_indents[:common_lines], strict=False),
        first_left_width + min_link_width,
    )

    link_extra_width = max(0, right_node_position - first_left_width - min_link_width)
    right_link_extra = link_extra_width // 2
    left_link_extra = link_extra_width - right_link_extra

    value_indent = max(0, first_left_width + left_link_extra + left_link_bar - value_offset)
    value_line = " " * max(0, value_indent) + string_value
    slash = "\\" if inverted else "/"
    backslash = "/" if inverted else "\\"
    u_line = "¯" if inverted else "_"

    left_link = "" if not left_child else (" " * first_left_width + u_line * left_link_extra + slash)
    right_link_offset = link_spacing + value_offset * (1 - left_link_bar)
    right_link = "" if not right_child else (" " * right_link_offset + backslash + u_line * right_link_extra)
    link_line = left_link + right_link

    left_indent_width = max(0, first_right_indent - right_node_position)
    left_indent = " " * left_indent_width
    indented_left_lines = [(left_indent if line else "") + line for line in left_sub_lines]

    merge_offsets_raw = [len(line) for line in indented_left_lines]
    merge_offsets = [left_indent_width + right_node_position - first_right_indent - w for w in merge_offsets_raw]
    merge_offsets = [p if right_sub_lines[i] else 0 for i, p in enumerate(merge_offsets)]

    merged_sub_lines: list[str] = []
    for i, p in enumerate(merge_offsets):
        line = indented_left_lines[i] + (" " * max(0, p))
        line = line + right_sub_lines[i][max(0, -p) :]
        merged_sub_lines.append(line)

    tree_lines = [left_indent + value_line]
    if link_line:
        tree_lines.append(left_indent + link_line)
    tree_lines.extend(merged_sub_lines)

    if inverted and is_top:
        tree_lines = list(reversed(tree_lines))

    if is_top:
        print("\n".join(tree_lines))
        return None
    return tree_lines
