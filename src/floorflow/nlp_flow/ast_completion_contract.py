"""Immutable schema and closure state shared by AST training and inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from floorflow.nlp_flow.ast_grammar import ASTTransition, ASTTransitionGrammar


@dataclass(frozen=True, order=True)
class FieldAlternative:
    kind: str
    named: bool


@dataclass(frozen=True, order=True)
class FieldRule:
    parent_kind: str
    parent_named: bool
    field: str
    alternatives: tuple[FieldAlternative, ...]
    required: bool
    repeated: bool


@dataclass(frozen=True, order=True)
class ChildEdge:
    field: str
    child_kind: str
    child_named: bool


@dataclass(frozen=True)
class ASTCompletionContract:
    """Pinned parser schema plus train-derived transitions and closure rules."""

    language: str
    schema_sha256: str
    transitions: tuple[ASTTransition, ...]
    legal_root_kinds: tuple[str, ...]
    fields: tuple[FieldRule, ...]
    extra_child_kinds: tuple[tuple[str, str, bool], ...]
    lexical_classes: tuple[tuple[str, tuple[str, ...]], ...]
    scope_owner_kinds: tuple[str, ...]
    root_closure_required_before_eos: bool = True
    node_namedness: tuple[tuple[str, bool], ...] = ()

    def __post_init__(self) -> None:
        if not self.language or len(self.schema_sha256) != 64:
            raise ValueError("completion contract requires a language and SHA-256 schema identity")
        if not self.legal_root_kinds:
            raise ValueError("completion contract requires at least one root kind")
        if len(set(self.legal_root_kinds)) != len(self.legal_root_kinds):
            raise ValueError("completion contract root kinds must be unique")
        if tuple(sorted(self.fields)) != self.fields:
            raise ValueError("completion contract fields must be deterministically sorted")
        if tuple(sorted(set(self.extra_child_kinds))) != self.extra_child_kinds:
            raise ValueError("extra child kinds must be sorted and unique")
        if tuple(sorted(set(self.scope_owner_kinds))) != self.scope_owner_kinds:
            raise ValueError("scope owner kinds must be sorted and unique")

    @cached_property
    def transition_keys(self) -> frozenset[tuple[str, str, str]]:
        return frozenset(
            (item.parent_kind, item.edge_field, item.child_kind)
            for item in self.transitions
        )

    @cached_property
    def field_rules_by_relation(self) -> dict[tuple[str, bool, str], FieldRule]:
        return {
            (rule.parent_kind, rule.parent_named, rule.field): rule
            for rule in self.fields
        }

    @cached_property
    def alternatives_by_relation(
        self,
    ) -> dict[tuple[str, bool, str], frozenset[tuple[str, bool]]]:
        return {
            key: frozenset((alternative.kind, alternative.named) for alternative in rule.alternatives)
            for key, rule in self.field_rules_by_relation.items()
        }

    @cached_property
    def extra_namedness_by_relation(self) -> dict[tuple[str, str], frozenset[bool]]:
        forms: dict[tuple[str, str], set[bool]] = {}
        for parent, child, named in self.extra_child_kinds:
            forms.setdefault((parent, child), set()).add(named)
        return {key: frozenset(values) for key, values in forms.items()}

    @cached_property
    def namedness_forms_by_kind(self) -> dict[str, frozenset[bool]]:
        forms: dict[str, set[bool]] = {}
        for kind, named in self.node_namedness:
            forms.setdefault(kind, set()).add(named)
        return {key: frozenset(values) for key, values in forms.items()}

    @cached_property
    def scope_flags_by_relation(self) -> dict[tuple[str, str, str], frozenset[bool]]:
        flags: dict[tuple[str, str, str], set[bool]] = {}
        for transition in self.transitions:
            key = (transition.parent_kind, transition.edge_field, transition.child_kind)
            flags.setdefault(key, set()).add(transition.opens_scope)
        return {key: frozenset(values) for key, values in flags.items()}

    @classmethod
    def from_files(
        cls,
        transition_grammar: ASTTransitionGrammar,
        parser_schema_path: str | Path,
        *,
        expected_schema_sha256: str,
    ) -> ASTCompletionContract:
        payload = Path(parser_schema_path).read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_schema_sha256:
            raise ValueError("parser schema hash differs from its pinned identity")
        return cls.from_payload(
            transition_grammar,
            json.loads(payload),
            schema_sha256=actual_hash,
        )

    @classmethod
    def from_payload(
        cls,
        transition_grammar: ASTTransitionGrammar,
        schema: Any,
        *,
        schema_sha256: str,
    ) -> ASTCompletionContract:
        if not isinstance(schema, list):
            raise TypeError("parser node schema must be a list")
        entries = {
            (str(row["type"]), bool(row["named"])): row
            for row in schema
            if isinstance(row, dict) and isinstance(row.get("type"), str)
        }

        def expand(kind: str, named: bool, seen: frozenset[tuple[str, bool]] = frozenset()) -> set[FieldAlternative]:
            key = (kind, named)
            if key in seen:
                raise ValueError(f"recursive parser schema subtype: {kind}")
            entry = entries.get(key)
            subtypes = () if entry is None else entry.get("subtypes", ())
            if not subtypes:
                return {FieldAlternative(kind, named)}
            return {
                alternative
                for subtype in subtypes
                for alternative in expand(
                    str(subtype["type"]),
                    bool(subtype["named"]),
                    seen | {key},
                )
            }

        rules: list[FieldRule] = []
        for (parent_kind, parent_named), row in entries.items():
            relations = dict(row.get("fields", {}))
            if row.get("children") is not None:
                relations["child"] = row["children"]
            for field, relation in relations.items():
                if not isinstance(relation, dict) or not isinstance(relation.get("types"), list):
                    raise TypeError(f"invalid parser field schema: {parent_kind}.{field}")
                alternatives = tuple(
                    sorted(
                        {
                            alternative
                            for item in relation["types"]
                            for alternative in expand(str(item["type"]), bool(item["named"]))
                        }
                    )
                )
                rules.append(
                    FieldRule(
                        parent_kind=parent_kind,
                        parent_named=parent_named,
                        field=str(field),
                        alternatives=alternatives,
                        required=bool(relation.get("required", False)),
                        repeated=bool(relation.get("multiple", False)),
                    )
                )
        lexical_classes = tuple(
            sorted(
                (str(kind), tuple(sorted(set(values))))
                for kind, values in transition_grammar.leaf_classes.items()
            )
        )
        transitions = tuple(sorted(set(transition_grammar.transitions), key=lambda item: (
            item.parent_kind,
            item.edge_field,
            item.child_kind,
            item.parent_scope_kind,
            item.child_scope_kind,
            item.opens_scope,
        )))
        schema_children = {
            (rule.parent_kind, alternative.kind, alternative.named)
            for rule in rules
            if rule.field == "child"
            for alternative in rule.alternatives
        }
        schema_child_pairs = {(parent, child) for parent, child, _named in schema_children}
        named_forms_by_kind: dict[str, set[bool]] = {}
        for kind, named in entries:
            named_forms_by_kind.setdefault(kind, set()).add(named)
        extra_child_kinds = tuple(
            sorted(
                {
                    (transition.parent_kind, transition.child_kind, named)
                    for transition in transitions
                    if transition.edge_field == "child"
                    and (transition.parent_kind, transition.child_kind) not in schema_child_pairs
                    for named in (
                        named_forms_by_kind.get(transition.child_kind) or {True}
                    )
                }
            )
        )
        return cls(
            language=transition_grammar.language,
            schema_sha256=schema_sha256,
            transitions=transitions,
            legal_root_kinds=tuple(sorted(set(transition_grammar.legal_root_kinds))),
            fields=tuple(sorted(rules)),
            extra_child_kinds=extra_child_kinds,
            lexical_classes=lexical_classes,
            scope_owner_kinds=tuple(sorted({item.child_kind for item in transitions if item.opens_scope})),
            node_namedness=tuple(sorted(entries)),
        )

    def namedness_for_kind(self, kind: str) -> bool:
        """Resolve parser namedness when the node kind has one schema form."""
        forms = self.namedness_forms_by_kind.get(kind, frozenset())
        if len(forms) != 1:
            raise ValueError(f"node namedness is ambiguous for {kind}: {sorted(forms)}")
        return next(iter(forms))

    def child_namedness(
        self,
        parent_kind: str,
        parent_named: bool,
        field: str,
        child_kind: str,
    ) -> bool:
        """Resolve a child form from its exact parser parent/field relation."""
        relation = (parent_kind, parent_named, field)
        forms = {
            named
            for kind, named in self.alternatives_by_relation.get(relation, frozenset())
            if kind == child_kind
        }
        if field == "child":
            forms.update(self.extra_namedness_by_relation.get((parent_kind, child_kind), ()))
        if not forms:
            forms.update(self.namedness_forms_by_kind.get(child_kind, ()))
            if not forms:
                # Corpus-only terminal types (for example trivia) are modeled as named leaves.
                forms.add(True)
        if len(forms) > 1 and child_kind in self.scope_owner_kinds:
            scope_flags = self.scope_flags_by_relation.get(
                (parent_kind, field, child_kind),
            )
            scope_flags = set() if scope_flags is None else set(scope_flags)
            if len(scope_flags) == 1:
                forms = scope_flags
        if len(forms) != 1:
            raise ValueError(
                "parser schema does not uniquely resolve child namedness: "
                f"{parent_kind}[named={parent_named}].{field}->{child_kind}: {sorted(forms)}"
            )
        return next(iter(forms))

    def initial_state(self) -> ASTCompletionState:
        return ASTCompletionState(contract=self)

    def training_state(self) -> ASTCompletionState:
        """Return the same immutable transition state used by inference."""
        return self.initial_state()

    def inference_state(self) -> ASTCompletionState:
        return self.initial_state()


@dataclass(frozen=True)
class _NodeFrame:
    kind: str
    named: bool
    child_edges: tuple[ChildEdge, ...] = ()
    field_counts: tuple[tuple[str, int], ...] = ()
    scope_id: int | None = None


@dataclass(frozen=True)
class ASTCompletionState:
    """Persistent pre-order builder; every transition returns a fresh state."""

    contract: ASTCompletionContract
    frames: tuple[_NodeFrame, ...] = ()
    closed_nodes: tuple[tuple[str, tuple[ChildEdge, ...]], ...] = ()
    open_scopes: tuple[int, ...] = ()
    next_scope_id: int = 0
    root_closed: bool = False
    finished: bool = False

    @property
    def child_order(self) -> tuple[tuple[str, tuple[ChildEdge, ...]], ...]:
        return self.closed_nodes + tuple((frame.kind, frame.child_edges) for frame in self.frames)

    @property
    def eos_ready(self) -> bool:
        return self.root_closed and not self.frames and not self.open_scopes and not self.finished

    @classmethod
    def open_parent_context(
        cls,
        contract: ASTCompletionContract,
        kind: str,
        *,
        named: bool,
        child_edges: tuple[ChildEdge, ...],
    ) -> ASTCompletionState:
        """Build a completion context for a decoder node and its current children."""
        counts: dict[str, int] = {}
        for edge in child_edges:
            transition_key = (kind, edge.field, edge.child_kind)
            if transition_key not in contract.transition_keys:
                raise ValueError(f"existing child is outside the completion grammar: {kind}.{edge.field}")
            relation = (kind, named, edge.field)
            rule = contract.field_rules_by_relation.get(relation)
            if rule is not None:
                schema_allows = (edge.child_kind, edge.child_named) in (
                    contract.alternatives_by_relation[relation]
                )
                parser_extra_allows = (
                    edge.field == "child"
                    and edge.child_named
                    in contract.extra_namedness_by_relation.get((kind, edge.child_kind), ())
                )
                if not schema_allows and not parser_extra_allows:
                    raise ValueError(f"existing child is outside parser alternatives: {kind}.{edge.field}")
                if parser_extra_allows and not schema_allows:
                    continue
                if counts.get(edge.field, 0) and not rule.repeated:
                    raise ValueError(f"existing non-repeated field appears more than once: {kind}.{edge.field}")
            counts[edge.field] = counts.get(edge.field, 0) + 1
        frame = _NodeFrame(
            kind=kind,
            named=named,
            child_edges=child_edges,
            field_counts=tuple(sorted(counts.items())),
        )
        return cls(contract=contract, frames=(frame,))

    def start_root(self, kind: str, *, named: bool = True) -> ASTCompletionState:
        if self.finished or self.frames or self.root_closed:
            raise ValueError("AST root can only be opened once")
        if kind not in self.contract.legal_root_kinds:
            raise ValueError(f"illegal AST root kind: {kind}")
        scope_id = self.next_scope_id
        return ASTCompletionState(
            contract=self.contract,
            frames=(_NodeFrame(kind=kind, named=named, scope_id=scope_id),),
            closed_nodes=self.closed_nodes,
            open_scopes=() if scope_id is None else (scope_id,),
            next_scope_id=self.next_scope_id + (scope_id is not None),
        )

    def append_child(self, field: str, kind: str, *, named: bool = True) -> ASTCompletionState:
        if self.finished or not self.frames:
            raise ValueError("child requires an open parent node")
        parent = self.frames[-1]
        matching = [
            item for item in self.contract.transitions
            if item.parent_kind == parent.kind and item.edge_field == field and item.child_kind == kind
        ]
        if not matching:
            raise ValueError(f"illegal AST child: {parent.kind}.{field}->{kind}")
        rule = self._field_rule(parent.kind, parent.named, field)
        if rule is not None:
            schema_allows = any(item.kind == kind and item.named == named for item in rule.alternatives)
            parser_extra_allows = field == "child" and (parent.kind, kind, named) in self.contract.extra_child_kinds
            if not schema_allows and not parser_extra_allows:
                raise ValueError(f"child kind {kind} is outside parser alternatives for {parent.kind}.{field}")
            counts = dict(parent.field_counts)
            if parser_extra_allows and not schema_allows:
                pass
            elif counts.get(field, 0) and not rule.repeated:
                raise ValueError(f"non-repeated parser field appears more than once: {parent.kind}.{field}")
            else:
                counts[field] = counts.get(field, 0) + 1
        else:
            counts = dict(parent.field_counts)
            counts[field] = counts.get(field, 0) + 1
        updated_parent = _NodeFrame(
            kind=parent.kind,
            named=parent.named,
            child_edges=parent.child_edges + (ChildEdge(field, kind, named),),
            field_counts=tuple(sorted(counts.items())),
            scope_id=parent.scope_id,
        )
        opens_scope = named and any(item.opens_scope for item in matching)
        scope_id = self.next_scope_id if opens_scope else None
        child = _NodeFrame(kind=kind, named=named, scope_id=scope_id)
        return ASTCompletionState(
            contract=self.contract,
            frames=self.frames[:-1] + (updated_parent, child),
            closed_nodes=self.closed_nodes,
            open_scopes=self.open_scopes + ((scope_id,) if scope_id is not None else ()),
            next_scope_id=self.next_scope_id + (scope_id is not None),
            root_closed=self.root_closed,
        )

    def allowed_lexical_classes(self, kind: str) -> tuple[str, ...]:
        return dict(self.contract.lexical_classes).get(kind, ())

    def validate_lexical_class(self, kind: str, lexical_class: str) -> None:
        allowed = self.allowed_lexical_classes(kind)
        if allowed and lexical_class not in allowed:
            raise ValueError(f"lexical class {lexical_class} is illegal for {kind}")

    def close_node(self) -> ASTCompletionState:
        if self.finished or not self.frames:
            raise ValueError("no open node to complete")
        frame = self.frames[-1]
        counts = dict(frame.field_counts)
        missing = [
            rule.field for rule in self.contract.fields
            if (
                rule.parent_kind == frame.kind
                and rule.parent_named == frame.named
                and rule.required
                and counts.get(rule.field, 0) == 0
            )
        ]
        if missing:
            raise ValueError(f"required parser fields are missing from {frame.kind}: {sorted(missing)}")
        scopes = self.open_scopes
        if frame.scope_id is not None:
            if not scopes or scopes[-1] != frame.scope_id:
                raise ValueError("scope obligations must close in LIFO order")
            scopes = scopes[:-1]
        remaining = self.frames[:-1]
        return ASTCompletionState(
            contract=self.contract,
            frames=remaining,
            closed_nodes=self.closed_nodes + ((frame.kind, frame.child_edges),),
            open_scopes=scopes,
            next_scope_id=self.next_scope_id,
            root_closed=not remaining,
        )

    def finish(self) -> ASTCompletionState:
        if not self.contract.root_closure_required_before_eos or not self.eos_ready:
            raise ValueError("EOS requires every node and scope obligation to be closed")
        return ASTCompletionState(
            contract=self.contract,
            frames=self.frames,
            closed_nodes=self.closed_nodes,
            open_scopes=self.open_scopes,
            next_scope_id=self.next_scope_id,
            root_closed=self.root_closed,
            finished=True,
        )

    def _field_rule(self, parent_kind: str, parent_named: bool, field: str) -> FieldRule | None:
        return next(
            (
                rule for rule in self.contract.fields
                if rule.parent_kind == parent_kind and rule.parent_named == parent_named and rule.field == field
            ),
            None,
        )


__all__ = [
    "ASTCompletionContract",
    "ASTCompletionState",
    "ChildEdge",
    "FieldAlternative",
    "FieldRule",
]
