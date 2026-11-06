"""Typed AST grammar state with native structural action masks."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class ASTTransition:
    parent_kind: str
    edge_field: str
    child_kind: str
    parent_scope_kind: str
    child_scope_kind: str
    opens_scope: bool


@dataclass(frozen=True)
class ASTGrammarNode:
    kind: str
    parent_index: int
    edge_field: str
    scope_id: int


@dataclass(frozen=True)
class ASTTransitionGrammar:
    """Indexed train-only transition grammar for one language."""

    language: str
    legal_root_kinds: tuple[str, ...]
    transitions: tuple[ASTTransition, ...]
    leaf_classes: dict[str, tuple[str, ...]]
    corpus_sha256: str

    def __post_init__(self) -> None:
        by_relation: dict[tuple[str, str, str], list[ASTTransition]] = {}
        scope_owners: set[str] = set()
        for transition in self.transitions:
            by_relation.setdefault(
                (
                    transition.parent_kind,
                    transition.edge_field,
                    transition.child_kind,
                ),
                [],
            ).append(transition)
            if transition.opens_scope:
                scope_owners.add(transition.child_kind)
        object.__setattr__(
            self,
            "_by_relation",
            {key: tuple(value) for key, value in by_relation.items()},
        )
        object.__setattr__(self, "_scope_owner_kinds", frozenset(scope_owners))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        language: str,
    ) -> ASTTransitionGrammar:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        language_payload = payload["languages"][language]
        leaf_classes: dict[str, set[str]] = {}
        for value in language_payload["leaf_classes"]:
            leaf_classes.setdefault(str(value["node_kind"]), set()).add(
                str(value["leaf_class"])
            )
        return cls(
            language=language,
            legal_root_kinds=tuple(language_payload["legal_root_kinds"]),
            transitions=tuple(
                ASTTransition(
                    parent_kind=str(value["parent_kind"]),
                    edge_field=str(value["edge_field"]),
                    child_kind=str(value["child_kind"]),
                    parent_scope_kind=str(value["parent_scope_kind"]),
                    child_scope_kind=str(value["child_scope_kind"]),
                    opens_scope=bool(value["opens_scope"]),
                )
                for value in language_payload["transitions"]
            ),
            leaf_classes={
                node_kind: tuple(sorted(values))
                for node_kind, values in leaf_classes.items()
            },
            corpus_sha256=str(payload["corpus_sha256"]),
        )


@dataclass(frozen=True)
class ASTGrammarState:
    grammar: ASTTransitionGrammar
    nodes: tuple[ASTGrammarNode, ...] = ()
    scope_kinds: tuple[tuple[int, str], ...] = ()
    open_scopes: tuple[int, ...] = ()
    next_scope_id: int = 0
    finished: bool = False

    @classmethod
    def initial(cls, grammar: ASTTransitionGrammar) -> ASTGrammarState:
        return cls(grammar=grammar)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        language: str,
    ) -> ASTGrammarState:
        return cls.initial(ASTTransitionGrammar.from_file(path, language))

    @property
    def root_kind(self) -> str | None:
        return None if not self.nodes else self.nodes[0].kind

    @property
    def scope_kind_by_id(self) -> dict[int, str]:
        return dict(self.scope_kinds)

    @property
    def scope_owner_kinds(self) -> frozenset[str]:
        return self.grammar._scope_owner_kinds

    @property
    def eos_ready(self) -> bool:
        return bool(self.nodes) and not self.open_scopes and not self.finished

    def start_root(self, root_kind: str) -> ASTGrammarState:
        if self.nodes:
            raise ValueError("AST root has already been selected")
        if root_kind not in self.grammar.legal_root_kinds:
            raise ValueError(f"illegal AST root: {root_kind}")
        node = ASTGrammarNode(
            kind=root_kind,
            parent_index=-1,
            edge_field="__root__",
            scope_id=0,
        )
        return replace(
            self,
            nodes=(node,),
            scope_kinds=((0, root_kind),),
            open_scopes=(0,),
            next_scope_id=1,
        )

    def _parent(self, parent_index: int) -> ASTGrammarNode:
        if parent_index < 0 or parent_index >= len(self.nodes):
            raise ValueError("parent index is outside the grammar state")
        parent = self.nodes[parent_index]
        if parent.scope_id not in self.open_scopes:
            raise ValueError("parent belongs to a closed scope")
        return parent

    def _matching_transitions(
        self,
        parent: ASTGrammarNode,
        edge_field: str,
        child_kind: str,
    ) -> tuple[ASTTransition, ...]:
        return self.grammar._by_relation.get(
            (parent.kind, edge_field, child_kind),
            (),
        )

    def add_child(
        self,
        *,
        parent_index: int,
        edge_field: str,
        child_kind: str,
    ) -> ASTGrammarState:
        if self.finished:
            raise ValueError("cannot add a child after EOS")
        parent = self._parent(parent_index)
        transitions = self._matching_transitions(parent, edge_field, child_kind)
        if not transitions:
            raise ValueError(
                f"illegal AST transition {parent.kind}.{edge_field}->{child_kind}"
            )
        opens_scope = any(item.opens_scope for item in transitions)
        if opens_scope:
            scope_id = self.next_scope_id
            scope_kinds = self.scope_kinds + ((scope_id, child_kind),)
            open_scopes = self.open_scopes + (scope_id,)
            next_scope_id = scope_id + 1
        else:
            scope_id = parent.scope_id
            scope_kinds = self.scope_kinds
            open_scopes = self.open_scopes
            next_scope_id = self.next_scope_id
        node = ASTGrammarNode(
            kind=child_kind,
            parent_index=parent_index,
            edge_field=edge_field,
            scope_id=scope_id,
        )
        return replace(
            self,
            nodes=self.nodes + (node,),
            scope_kinds=scope_kinds,
            open_scopes=open_scopes,
            next_scope_id=next_scope_id,
        )

    def close_scope(self, scope_id: int | None = None) -> ASTGrammarState:
        if self.finished:
            raise ValueError("cannot close a scope after EOS")
        if not self.open_scopes:
            raise ValueError("no open AST scope")
        target = self.open_scopes[-1] if scope_id is None else int(scope_id)
        if target != self.open_scopes[-1]:
            raise ValueError("AST scopes must close in LIFO order")
        return replace(self, open_scopes=self.open_scopes[:-1])

    def finish(self) -> ASTGrammarState:
        if not self.eos_ready:
            raise ValueError("EOS is illegal before root closure")
        return replace(self, finished=True)

    def allowed_child_kinds(
        self,
        *,
        parent_index: int,
        edge_field: str | None = None,
    ) -> tuple[str, ...]:
        parent = self._parent(parent_index)
        allowed = {
            transition.child_kind
            for transition in self.grammar.transitions
            if transition.parent_kind == parent.kind
            and (edge_field is None or transition.edge_field == edge_field)
        }
        return tuple(sorted(allowed))

    def allowed_edge_fields(
        self,
        *,
        parent_index: int,
        child_kind: str,
    ) -> tuple[str, ...]:
        parent = self._parent(parent_index)
        allowed = {
            transition.edge_field
            for transition in self.grammar.transitions
            if transition.parent_kind == parent.kind
            and transition.child_kind == child_kind
        }
        return tuple(sorted(allowed))

    def allowed_parent_indices(self, *, child_kind: str) -> tuple[int, ...]:
        allowed = []
        for index in range(len(self.nodes)):
            parent = self.nodes[index]
            if parent.scope_id not in self.open_scopes:
                continue
            if self.allowed_edge_fields(
                parent_index=index,
                child_kind=child_kind,
            ):
                allowed.append(index)
        return tuple(allowed)

    def allowed_leaf_classes(self, *, node_kind: str) -> tuple[str, ...]:
        return self.grammar.leaf_classes.get(node_kind, ())

    def mask_root_logits(
        self,
        logits: Tensor,
        node_kinds: tuple[str, ...] | list[str],
    ) -> Tensor:
        allowed = set(self.grammar.legal_root_kinds) if not self.nodes else set()
        mask = torch.tensor(
            [value in allowed for value in node_kinds],
            dtype=torch.bool,
            device=logits.device,
        )
        return _mask(logits, mask)

    def mask_child_logits(
        self,
        logits: Tensor,
        node_kinds: tuple[str, ...] | list[str],
        *,
        parent_index: int,
        edge_field: str | None = None,
    ) -> Tensor:
        allowed = set(
            self.allowed_child_kinds(
                parent_index=parent_index,
                edge_field=edge_field,
            )
        )
        mask = torch.tensor(
            [value in allowed for value in node_kinds],
            dtype=torch.bool,
            device=logits.device,
        )
        return _mask(logits, mask)

    def mask_edge_logits(
        self,
        logits: Tensor,
        edge_fields: tuple[str, ...] | list[str],
        *,
        parent_index: int,
        child_kind: str,
    ) -> Tensor:
        allowed = set(
            self.allowed_edge_fields(
                parent_index=parent_index,
                child_kind=child_kind,
            )
        )
        mask = torch.tensor(
            [value in allowed for value in edge_fields],
            dtype=torch.bool,
            device=logits.device,
        )
        return _mask(logits, mask)

    def mask_parent_logits(
        self,
        logits: Tensor,
        *,
        child_kind: str,
    ) -> Tensor:
        allowed = set(self.allowed_parent_indices(child_kind=child_kind))
        mask = torch.tensor(
            [value in allowed for value in range(logits.numel())],
            dtype=torch.bool,
            device=logits.device,
        )
        return _mask(logits, mask)

    def mask_scope_close_logits(
        self,
        logits: Tensor,
        scope_ids: tuple[int, ...] | list[int],
    ) -> Tensor:
        allowed = {self.open_scopes[-1]} if self.open_scopes else set()
        mask = torch.tensor(
            [value in allowed for value in scope_ids],
            dtype=torch.bool,
            device=logits.device,
        )
        return _mask(logits, mask)

    def mask_leaf_logits(
        self,
        logits: Tensor,
        leaf_classes: tuple[str, ...] | list[str],
        *,
        node_kind: str,
    ) -> Tensor:
        allowed = set(self.allowed_leaf_classes(node_kind=node_kind))
        mask = torch.tensor(
            [value in allowed for value in leaf_classes],
            dtype=torch.bool,
            device=logits.device,
        )
        return _mask(logits, mask)

    def mask_eos_logits(self, logits: Tensor) -> Tensor:
        mask = torch.full(
            logits.shape,
            self.eos_ready,
            dtype=torch.bool,
            device=logits.device,
        )
        return _mask(logits, mask)

    @staticmethod
    def topology_is_acyclic(parent_indices: tuple[int, ...] | list[int]) -> bool:
        if not parent_indices:
            return False
        if sum(index < 0 for index in parent_indices) != 1:
            return False
        for start in range(len(parent_indices)):
            seen: set[int] = set()
            current = start
            while current >= 0:
                if current in seen or current >= len(parent_indices):
                    return False
                seen.add(current)
                current = parent_indices[current]
        return True

    @staticmethod
    def mask_cycle_edges(
        logits: Tensor,
        parent_indices: tuple[int, ...] | list[int],
    ) -> Tensor:
        if logits.ndim != 2 or logits.shape != (
            len(parent_indices),
            len(parent_indices),
        ):
            raise ValueError("cycle logits must be [nodes, nodes]")
        allowed = torch.ones_like(logits, dtype=torch.bool)
        for child in range(len(parent_indices)):
            for candidate_parent in range(len(parent_indices)):
                current = candidate_parent
                seen: set[int] = set()
                while current >= 0:
                    if current == child or current in seen:
                        allowed[child, candidate_parent] = False
                        break
                    if current >= len(parent_indices):
                        break
                    seen.add(current)
                    current = parent_indices[current]
        return _mask(logits, allowed)


def _mask(logits: Tensor, allowed: Tensor) -> Tensor:
    if logits.shape != allowed.shape:
        raise ValueError("grammar mask must match the logit shape")
    return logits.masked_fill(~allowed, -torch.inf)


def load_ast_grammar(
    path: str | Path,
    language: Literal["python", "rust", "cpp"],
) -> ASTTransitionGrammar:
    return ASTTransitionGrammar.from_file(path, language)


__all__ = [
    "ASTGrammarNode",
    "ASTGrammarState",
    "ASTTransition",
    "ASTTransitionGrammar",
    "load_ast_grammar",
]
