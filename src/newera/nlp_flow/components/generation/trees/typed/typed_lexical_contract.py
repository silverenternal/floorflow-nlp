"""Typed lexical decisions with evidence, scope and Unicode-span contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EvidenceOrigin = Literal["user_available", "target_derived", "training_only", "evaluation_only"]
ValueOrigin = Literal["user_available", "generated"]
LexicalMode = Literal["copy", "generate"]


@dataclass(frozen=True, order=True)
class LexicalNode:
    node_id: int
    node_kind: str
    scope_id: int
    leaf_classes: tuple[str, ...]
    minimum_length: int = 0
    maximum_length: int = 4096


@dataclass(frozen=True, order=True)
class LexicalEvidence:
    evidence_id: str
    origin: EvidenceOrigin
    content: str
    visible_to_model: bool
    scope_id: int | None = None


@dataclass(frozen=True)
class LexicalDecision:
    node_id: int
    node_kind: str
    leaf_class: str
    mode: LexicalMode
    minimum_length: int
    maximum_length: int
    value: str | None = None
    evidence_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None


@dataclass(frozen=True)
class TypedLexicalValue:
    node_id: int
    node_kind: str
    leaf_class: str
    scope_id: int
    value: str
    origin: ValueOrigin
    minimum_length: int
    maximum_length: int
    evidence_id: str | None = None
    span_start: int | None = None
    span_end: int | None = None


@dataclass(frozen=True)
class TypedLexicalContract:
    nodes: tuple[LexicalNode, ...]
    scope_parents: tuple[tuple[int, int | None], ...]
    evidence: tuple[LexicalEvidence, ...]
    coordinate_system: str = "unicode_code_points_half_open"

    def __post_init__(self) -> None:
        if tuple(sorted(self.nodes)) != self.nodes or len({node.node_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("lexical node bindings must be sorted and uniquely identified")
        if tuple(sorted(self.scope_parents)) != self.scope_parents:
            raise ValueError("scope parent bindings must be sorted")
        if len({scope for scope, _parent in self.scope_parents}) != len(self.scope_parents):
            raise ValueError("scope IDs must be unique")
        if tuple(sorted(self.evidence)) != self.evidence:
            raise ValueError("lexical evidence must be sorted by identity")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        if self.coordinate_system != "unicode_code_points_half_open":
            raise ValueError("unsupported lexical span coordinate system")
        scopes = dict(self.scope_parents)
        for node in self.nodes:
            if (
                node.node_id < 0
                or not node.node_kind
                or node.scope_id < 0
                or node.minimum_length < 0
                or node.maximum_length < node.minimum_length
                or not node.leaf_classes
                or any(not value for value in node.leaf_classes)
                or len(set(node.leaf_classes)) != len(node.leaf_classes)
            ):
                raise ValueError("lexical node bounds and leaf classes are invalid")
            if node.scope_id not in scopes:
                raise ValueError(f"lexical node has unknown scope ID: {node.scope_id}")
        for item in self.evidence:
            if item.origin not in {"user_available", "target_derived", "training_only", "evaluation_only"}:
                raise ValueError(f"unknown lexical evidence origin: {item.origin}")
            if not item.evidence_id:
                raise ValueError("lexical evidence ID cannot be empty")
            if item.scope_id is not None and item.scope_id not in scopes:
                raise ValueError(f"lexical evidence has unknown scope ID: {item.scope_id}")
        for scope_id in scopes:
            self._ancestors(scope_id, scopes)

    def resolve(self, decision: LexicalDecision) -> TypedLexicalValue:
        nodes = {node.node_id: node for node in self.nodes}
        node = nodes.get(decision.node_id)
        if node is None:
            raise ValueError(f"unknown lexical node ID: {decision.node_id}")
        if decision.node_kind != node.node_kind or decision.leaf_class not in node.leaf_classes:
            raise ValueError("lexical node type or class differs from its registered AST node")
        if (
            decision.minimum_length < node.minimum_length
            or decision.maximum_length > node.maximum_length
            or decision.minimum_length < 0
            or decision.maximum_length < decision.minimum_length
        ):
            raise ValueError("lexical length bounds exceed the node contract")

        if decision.mode == "copy":
            value, origin, evidence_id, span_start, span_end = self._copy(decision, node)
        elif decision.mode == "generate":
            if decision.evidence_id is not None or decision.span_start is not None or decision.span_end is not None:
                raise ValueError("generated lexical values cannot reference source evidence")
            if decision.value is None:
                raise ValueError("generated lexical value is missing")
            value, origin = decision.value, "generated"
            evidence_id = span_start = span_end = None
        else:
            raise ValueError(f"unsupported lexical mode: {decision.mode}")

        length = len(value)
        if not decision.minimum_length <= length <= decision.maximum_length:
            raise ValueError("lexical value length is outside its declared variable-length bounds")
        return TypedLexicalValue(
            node_id=node.node_id,
            node_kind=node.node_kind,
            leaf_class=decision.leaf_class,
            scope_id=node.scope_id,
            value=value,
            origin=origin,
            minimum_length=decision.minimum_length,
            maximum_length=decision.maximum_length,
            evidence_id=evidence_id,
            span_start=span_start,
            span_end=span_end,
        )

    def _copy(
        self,
        decision: LexicalDecision,
        node: LexicalNode,
    ) -> tuple[str, ValueOrigin, str, int, int]:
        if decision.value is not None or decision.evidence_id is None:
            raise ValueError("copy decisions require evidence identity and derive value from its span")
        if decision.span_start is None or decision.span_end is None:
            raise ValueError("copy decisions require a half-open evidence span")
        evidence = next((item for item in self.evidence if item.evidence_id == decision.evidence_id), None)
        if evidence is None:
            raise ValueError("copy decision references unknown evidence")
        if evidence.origin != "user_available" or not evidence.visible_to_model:
            raise ValueError("copy source is not user-visible evidence")
        if evidence.scope_id is not None:
            scopes = dict(self.scope_parents)
            if node.scope_id not in scopes or evidence.scope_id not in self._ancestors(node.scope_id, scopes):
                raise ValueError("copy source is outside the node's visible lexical scope")
        start, end = decision.span_start, decision.span_end
        if start < 0 or end <= start or end > len(evidence.content):
            raise ValueError("copy span is outside the evidence text")
        return evidence.content[start:end], "user_available", evidence.evidence_id, start, end

    @staticmethod
    def _ancestors(scope_id: int, parents: dict[int, int | None]) -> frozenset[int]:
        output = set()
        current: int | None = scope_id
        while current is not None:
            if current in output:
                raise ValueError("scope parent graph contains a cycle")
            output.add(current)
            if current not in parents:
                raise ValueError(f"unknown scope ID: {current}")
            current = parents[current]
        return frozenset(output)


__all__ = [
    "EvidenceOrigin",
    "LexicalDecision",
    "LexicalEvidence",
    "LexicalMode",
    "LexicalNode",
    "TypedLexicalContract",
    "TypedLexicalValue",
    "ValueOrigin",
]
