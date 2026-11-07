"""Explicit grammar-checked Seq2Tree actions for AST construction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TypeAlias

from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTGrammarState, ASTTransitionGrammar
from newera.nlp_flow.components.ast.representation.structure.ast_leaf_classifier import classify_leaf


@dataclass(frozen=True)
class AttachEdgeAction:
    parent_index: int
    edge_field: str


@dataclass(frozen=True)
class OpenNodeAction:
    node_kind: str


@dataclass(frozen=True)
class OpenScopeAction:
    scope_id: int
    node_index: int
    scope_kind: str


@dataclass(frozen=True)
class DeclareLeafAction:
    node_index: int
    leaf_class: str
    lexical: str | None = None
    gap_before: str = ""


@dataclass(frozen=True)
class CloseScopeAction:
    scope_id: int


@dataclass(frozen=True)
class TerminateAction:
    pass


TreeAction: TypeAlias = (
    AttachEdgeAction
    | OpenNodeAction
    | OpenScopeAction
    | DeclareLeafAction
    | CloseScopeAction
    | TerminateAction
)


@dataclass
class TreeActionNode:
    kind: str
    parent_index: int
    edge_field: str
    scope_id: int
    children: list[int] = field(default_factory=list)
    leaf_class: str | None = None
    lexical: str | None = None
    gap_before: str = ""


class TreeActionDecoder:
    """Apply explicit typed actions while enforcing the frozen AST grammar."""

    def __init__(self, grammar: ASTTransitionGrammar) -> None:
        self.grammar = grammar
        self.state = ASTGrammarState.initial(grammar)
        self.nodes: list[TreeActionNode] = []
        self.actions: list[TreeAction] = []
        self.pending_edge: AttachEdgeAction | None = None
        self.expected_scope: tuple[int, int, str] | None = None
        self.terminated = False

    def _require_active(self) -> None:
        if self.terminated:
            raise ValueError("tree decoder has already terminated")

    def apply(self, action: TreeAction) -> TreeActionDecoder:
        self._require_active()
        if isinstance(action, AttachEdgeAction):
            self._attach_edge(action)
        elif isinstance(action, OpenNodeAction):
            self._open_node(action)
        elif isinstance(action, OpenScopeAction):
            self._open_scope(action)
        elif isinstance(action, DeclareLeafAction):
            self._declare_leaf(action)
        elif isinstance(action, CloseScopeAction):
            self._close_scope(action)
        elif isinstance(action, TerminateAction):
            self._terminate()
        else:
            raise TypeError(f"unsupported tree action: {type(action).__name__}")
        self.actions.append(action)
        return self

    def _attach_edge(self, action: AttachEdgeAction) -> None:
        if self.expected_scope is not None:
            raise ValueError("scope-open action must follow node opening")
        if self.pending_edge is not None:
            raise ValueError("previous edge attachment has no opened child")
        if action.parent_index < 0 or action.parent_index >= len(self.nodes):
            raise ValueError("edge attachment parent is outside the tree")
        if not action.edge_field:
            raise ValueError("edge field must be non-empty")
        self.pending_edge = action

    def _open_node(self, action: OpenNodeAction) -> None:
        if self.expected_scope is not None:
            raise ValueError("scope-open action must follow node opening")
        if not self.nodes:
            if self.pending_edge is not None:
                raise ValueError("root node cannot have a parent edge")
            next_state = self.state.start_root(action.node_kind)
            parent_index = -1
            edge_field = "__root__"
        else:
            if self.pending_edge is None:
                raise ValueError("non-root node requires an edge attachment")
            parent_index = self.pending_edge.parent_index
            edge_field = self.pending_edge.edge_field
            next_state = self.state.add_child(
                parent_index=parent_index,
                edge_field=edge_field,
                child_kind=action.node_kind,
            )
        node_index = len(next_state.nodes) - 1
        grammar_node = next_state.nodes[node_index]
        if parent_index >= 0:
            self.nodes[parent_index].children.append(node_index)
        self.nodes.append(
            TreeActionNode(
                kind=action.node_kind,
                parent_index=parent_index,
                edge_field=edge_field,
                scope_id=grammar_node.scope_id,
            )
        )
        previous_scopes = set(self.state.open_scopes)
        self.state = next_state
        opened = set(self.state.open_scopes) - previous_scopes
        if opened:
            scope_id = next(iter(opened))
            self.expected_scope = (scope_id, node_index, action.node_kind)
        self.pending_edge = None

    def _open_scope(self, action: OpenScopeAction) -> None:
        if self.expected_scope is None:
            raise ValueError("no scope opening is pending")
        expected_id, expected_node, expected_kind = self.expected_scope
        if (
            action.scope_id != expected_id
            or action.node_index != expected_node
            or action.scope_kind != expected_kind
        ):
            raise ValueError("scope-open action does not match the grammar transition")
        self.expected_scope = None

    def _declare_leaf(self, action: DeclareLeafAction) -> None:
        if self.expected_scope is not None:
            raise ValueError("scope-open action must follow node opening")
        if action.node_index < 0 or action.node_index >= len(self.nodes):
            raise ValueError("leaf node index is outside the tree")
        node = self.nodes[action.node_index]
        if node.children:
            raise ValueError("node with children cannot be declared as a leaf")
        allowed = self.state.allowed_leaf_classes(node_kind=node.kind)
        if allowed and action.leaf_class not in allowed:
            raise ValueError(
                f"leaf class {action.leaf_class} is illegal for {node.kind}"
            )
        node.leaf_class = action.leaf_class
        node.lexical = action.lexical
        node.gap_before = action.gap_before

    def _close_scope(self, action: CloseScopeAction) -> None:
        if self.expected_scope is not None or self.pending_edge is not None:
            raise ValueError("cannot close a scope during a pending action")
        self.state = self.state.close_scope(action.scope_id)

    def _terminate(self) -> None:
        if self.expected_scope is not None or self.pending_edge is not None:
            raise ValueError("cannot terminate with a pending action")
        undeclared = [
            index
            for index, node in enumerate(self.nodes)
            if not node.children and node.leaf_class is None
        ]
        if undeclared:
            raise ValueError(f"leaf nodes lack declarations: {undeclared}")
        self.state = self.state.finish()
        self.terminated = True

    def structure(self) -> dict[str, object]:
        if not self.terminated:
            raise ValueError("tree action sequence is incomplete")
        return {
            "language": self.grammar.language,
            "root_index": 0,
            "nodes": [
                {
                    "index": index,
                    "kind": node.kind,
                    "parent_index": node.parent_index,
                    "edge_field": node.edge_field,
                    "scope_id": node.scope_id,
                    "children": list(node.children),
                    "leaf_class": node.leaf_class,
                    "lexical": node.lexical,
                    "gap_before": node.gap_before,
                }
                for index, node in enumerate(self.nodes)
            ],
            "scope_ids": list(self.state.scope_kind_by_id),
        }


def trace_teacher_actions(
    grammar: ASTTransitionGrammar,
    ast_payload: dict[str, object],
) -> list[TreeAction]:
    """Convert a parsed AST payload into explicit actions for lossless replay."""
    decoder = TreeActionDecoder(grammar)
    actions: list[TreeAction] = []
    root = ast_payload["root"]

    def emit(action: TreeAction) -> None:
        decoder.apply(action)
        actions.append(action)

    def visit(node: dict[str, object], node_index: int, is_root: bool) -> None:
        for child in node["children"]:
            child_node = child["node"]
            emit(
                AttachEdgeAction(
                    parent_index=node_index,
                    edge_field=str(child["field"]),
                )
            )
            emit(OpenNodeAction(node_kind=str(child_node["type"])))
            child_index = len(decoder.nodes) - 1
            if decoder.expected_scope is not None:
                scope_id, expected_node, scope_kind = decoder.expected_scope
                if expected_node != child_index:
                    raise ValueError("scope owner does not match opened node")
                emit(
                    OpenScopeAction(
                        scope_id=scope_id,
                        node_index=child_index,
                        scope_kind=scope_kind,
                    )
                )
            visit(child_node, child_index, False)
            if not child_node["children"]:
                lexical = child_node["lexical"]
                if not isinstance(lexical, str):
                    raise TypeError("teacher AST leaf has no lexical value")
                emit(
                    DeclareLeafAction(
                        node_index=child_index,
                        leaf_class=classify_leaf(
                            str(child_node["type"]),
                            lexical,
                        ),
                        lexical=lexical,
                        gap_before=str(child_node.get("gap_before", "")),
                    )
                )
            if (
                not is_root
                and str(child_node["type"]) in decoder.state.scope_owner_kinds
            ):
                emit(CloseScopeAction(scope_id=decoder.state.open_scopes[-1]))

    emit(OpenNodeAction(node_kind=str(root["type"])))
    scope_id, owner_index, scope_kind = decoder.expected_scope
    emit(
        OpenScopeAction(
            scope_id=scope_id,
            node_index=owner_index,
            scope_kind=scope_kind,
        )
    )
    visit(root, 0, True)
    while decoder.state.open_scopes:
        emit(CloseScopeAction(scope_id=decoder.state.open_scopes[-1]))
    emit(TerminateAction())
    return actions


def decode_actions(
    grammar: ASTTransitionGrammar,
    actions: list[TreeAction],
) -> dict[str, object]:
    decoder = TreeActionDecoder(grammar)
    for action in actions:
        decoder.apply(action)
    return decoder.structure()


def render_tree_structure(structure: dict[str, object]) -> str:
    """Render the unique source projection carried by leaf actions."""
    nodes = structure["nodes"]
    root_index = int(structure["root_index"])

    def render(index: int) -> str:
        node = nodes[index]
        children = node["children"]
        if not children:
            lexical = node["lexical"]
            if not isinstance(lexical, str):
                raise ValueError("leaf action has no lexical value")
            return f"{node['gap_before']}{lexical}"
        return "".join(render(int(child)) for child in children)

    return render(root_index)


def load_ast_payload(path: str) -> dict[str, object]:
    payload = json.loads(path)
    if not isinstance(payload, dict) or "root" not in payload:
        raise ValueError("AST payload must contain a root")
    return payload


__all__ = [
    "AttachEdgeAction",
    "CloseScopeAction",
    "DeclareLeafAction",
    "OpenNodeAction",
    "OpenScopeAction",
    "TerminateAction",
    "TreeAction",
    "TreeActionDecoder",
    "decode_actions",
    "render_tree_structure",
    "trace_teacher_actions",
]
