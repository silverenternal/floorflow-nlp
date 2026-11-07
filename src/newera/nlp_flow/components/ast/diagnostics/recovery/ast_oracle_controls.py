"""Offline-only topology/lexical oracle controls for diagnostic fixtures.

These controls consume target information and must never be used by inference.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import render_tree_structure


def _leaves(structure: dict[str, Any]) -> dict[int, dict[str, Any]]:
    nodes = structure.get("nodes")
    if not isinstance(nodes, list) or not nodes or structure.get("root_index") != 0:
        raise ValueError("structure requires root index zero and nonempty nodes")
    seen: set[int] = set()
    leaves = {}

    def visit(index: int, parent: int) -> None:
        if index in seen or index < 0 or index >= len(nodes):
            raise ValueError("tree has a cycle, shared child, or invalid index")
        seen.add(index)
        node = nodes[index]
        if node.get("index") != index or node.get("parent_index") != parent:
            raise ValueError("node index or parent does not match topology")
        children = node.get("children")
        if not isinstance(children, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in children):
            raise ValueError("invalid children")
        if not children:
            leaves[index] = node
        for child in children:
            visit(child, index)

    visit(0, -1)
    if len(seen) != len(nodes):
        raise ValueError("tree contains unreachable nodes")
    return leaves


def oracle_projection(
    topology: dict[str, Any], lexical_source: dict[str, Any], leaf_alignment: dict[int, int]
) -> dict[str, Any]:
    """Project lexical leaves onto topology using an explicit bijective typed map.

    Missing or nonmatching leaves are rejected rather than filled from either
    source. This bounds the diagnostic to aligned, complete trees.
    """
    if topology.get("language") != lexical_source.get("language"):
        raise ValueError("cross-language projection is unsupported")
    target_leaves = _leaves(topology)
    source_leaves = _leaves(lexical_source)
    if set(leaf_alignment) != set(target_leaves) or set(leaf_alignment.values()) != set(source_leaves):
        raise ValueError("leaf alignment must cover both trees")
    if len(set(leaf_alignment.values())) != len(leaf_alignment):
        raise ValueError("leaf alignment must be one-to-one")
    result = deepcopy(topology)
    for target_id, source_id in leaf_alignment.items():
        target, source = target_leaves[target_id], source_leaves[source_id]
        if target["kind"] != source["kind"] or target.get("leaf_class") != source.get("leaf_class"):
            raise ValueError("aligned leaves have incompatible kinds or classes")
        if not isinstance(source.get("lexical"), str) or not isinstance(source.get("gap_before"), str):
            raise TypeError("source leaf lacks lexical value or gap")
        result["nodes"][target_id]["lexical"] = source["lexical"]
        result["nodes"][target_id]["gap_before"] = source["gap_before"]
    return result


def render_oracle_projection(
    topology: dict[str, Any], lexical_source: dict[str, Any], leaf_alignment: dict[int, int]
) -> str:
    """Render one offline diagnostic arm; no parser repair or retries."""
    return render_tree_structure(oracle_projection(topology, lexical_source, leaf_alignment))
