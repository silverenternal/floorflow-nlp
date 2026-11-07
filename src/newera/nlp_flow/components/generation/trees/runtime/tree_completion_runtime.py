"""Certified parser-schema completion budgets for the action-tree runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from newera.nlp_flow.components.ast.diagnostics.validation.ast_completion_contract import (
    ASTCompletionContract,
    ASTCompletionState,
    ChildEdge,
)
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import ASTTransitionGrammar
from newera.nlp_flow.components.generation.trees.actions.terminalizable_action_mask import (
    TerminalizableActionMask,
    UnsupportedCompletionGrammar,
)
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import TreeActionDecoder


@dataclass(frozen=True)
class _CompletionSnapshot:
    decoder: TreeActionDecoder
    action_count: int
    namedness: dict[int, bool]
    node_costs: tuple[int, ...]

    @property
    def total_required_nodes(self) -> int:
        return self.node_costs[0] if self.node_costs else 0


class TreeCompletionBudget:
    """Resolve parser namedness and reserve enough nodes to complete a partial tree."""

    def __init__(self, contract: ASTCompletionContract) -> None:
        self.contract = contract
        self.mask = TerminalizableActionMask(contract)
        self._snapshot: _CompletionSnapshot | None = None

    def minimum_root_nodes(self) -> int:
        """Return the schema-certified minimum size of any legal root completion."""
        costs = []
        for kind in self.contract.legal_root_kinds:
            try:
                costs.append(
                    self.mask.certificate(
                        kind,
                        named=self.contract.namedness_for_kind(kind),
                    ).minimum_nodes
                )
            except (ValueError, UnsupportedCompletionGrammar):
                continue
        if not costs:
            raise UnsupportedCompletionGrammar("grammar has no certified root completion")
        return min(costs)

    def root_candidates(self, kinds: Sequence[str], *, remaining_nodes: int) -> set[str]:
        allowed = set()
        for kind in kinds:
            try:
                cost = self.mask.certificate(
                    kind,
                    named=self.contract.namedness_for_kind(kind),
                ).minimum_nodes
            except (ValueError, UnsupportedCompletionGrammar):
                continue
            if cost <= remaining_nodes:
                allowed.add(kind)
        return allowed

    def child_candidates(
        self,
        decoder: TreeActionDecoder,
        *,
        parent_index: int,
        field: str,
        kinds: Sequence[str],
        remaining_nodes: int,
    ) -> set[str]:
        if remaining_nodes < 0:
            return set()
        parent = decoder.nodes[parent_index]
        snapshot = self._completion_snapshot(decoder)
        named_by_index = snapshot.namedness
        existing_edges = tuple(
            ChildEdge(
                field=decoder.nodes[index].edge_field,
                child_kind=decoder.nodes[index].kind,
                child_named=named_by_index[index],
            )
            for index in parent.children
        )
        context = ASTCompletionState.open_parent_context(
            self.contract,
            parent.kind,
            named=named_by_index[parent_index],
            child_edges=existing_edges,
        )
        parent_missing = self.mask.minimum_additional_nodes(context)
        existing_children = sum(snapshot.node_costs[index] for index in parent.children)
        outstanding = snapshot.total_required_nodes
        unrelated_obligations = outstanding - parent_missing - existing_children
        if unrelated_obligations < 0:
            raise ValueError("completion accounting underflowed for the current action tree")
        budget_for_parent = remaining_nodes - unrelated_obligations - existing_children
        if budget_for_parent < 0:
            return set()
        named_kinds = tuple(
            (kind, self.contract.child_namedness(parent.kind, named_by_index[parent_index], field, kind))
            for kind in kinds
        )
        result = self.mask.candidate_mask(
            context,
            field=field,
            candidates=named_kinds,
            remaining_node_budget=budget_for_parent,
        )
        return {kind for kind, _named in result.allowed_candidates}

    def minimum_required_nodes(
        self,
        decoder: TreeActionDecoder,
        named_by_index: dict[int, bool] | None = None,
    ) -> int:
        if not decoder.nodes:
            return 0
        if named_by_index is not None:
            return self._build_node_costs(decoder, named_by_index)[0]
        return self._completion_snapshot(decoder).total_required_nodes

    def can_close_active_scope(self, decoder: TreeActionDecoder) -> bool:
        if not decoder.state.open_scopes:
            return False
        scope_id = decoder.state.open_scopes[-1]
        scope_kind = dict(decoder.state.scope_kind_by_id)[scope_id]
        owners = [
            index
            for index, node in enumerate(decoder.nodes)
            if node.scope_id == scope_id and node.kind == scope_kind
        ]
        if len(owners) != 1:
            raise ValueError(f"active parser scope has {len(owners)} owner nodes")
        snapshot = self._completion_snapshot(decoder)
        return snapshot.node_costs[owners[0]] == 0

    def _named_nodes(self, decoder: TreeActionDecoder) -> dict[int, bool]:
        if not decoder.nodes:
            return {}
        root = decoder.nodes[0]
        namedness = {0: self.contract.namedness_for_kind(root.kind)}
        for index, node in enumerate(decoder.nodes[1:], start=1):
            if node.parent_index not in namedness:
                raise ValueError(f"tree parent is not in preorder for node {index}")
            parent = decoder.nodes[node.parent_index]
            namedness[index] = self.contract.child_namedness(
                parent.kind,
                namedness[node.parent_index],
                node.edge_field,
                node.kind,
            )
        return namedness

    def _completion_snapshot(self, decoder: TreeActionDecoder) -> _CompletionSnapshot:
        snapshot = self._snapshot
        if snapshot is not None and snapshot.decoder is decoder and snapshot.action_count == len(decoder.actions):
            return snapshot
        namedness = self._named_nodes(decoder)
        costs = self._build_node_costs(decoder, namedness)
        snapshot = _CompletionSnapshot(
            decoder=decoder,
            action_count=len(decoder.actions),
            namedness=namedness,
            node_costs=tuple(costs),
        )
        self._snapshot = snapshot
        return snapshot

    def _build_node_costs(
        self,
        decoder: TreeActionDecoder,
        namedness: dict[int, bool],
    ) -> list[int]:
        costs = [0] * len(decoder.nodes)
        for index in range(len(decoder.nodes) - 1, -1, -1):
            node = decoder.nodes[index]
            children = tuple(
                ChildEdge(
                    field=decoder.nodes[child].edge_field,
                    child_kind=decoder.nodes[child].kind,
                    child_named=namedness[child],
                )
                for child in node.children
            )
            context = ASTCompletionState.open_parent_context(
                self.contract,
                node.kind,
                named=namedness[index],
                child_edges=children,
            )
            costs[index] = self.mask.minimum_additional_nodes(context) + sum(
                costs[child] for child in node.children
            )
        return costs

def load_tree_completion_budget(
    grammar: ASTTransitionGrammar,
    schema_path: str | Path,
    *,
    expected_schema_sha256: str,
) -> TreeCompletionBudget:
    contract = ASTCompletionContract.from_files(
        grammar,
        schema_path,
        expected_schema_sha256=expected_schema_sha256,
    )
    return TreeCompletionBudget(contract)


__all__ = ["TreeCompletionBudget", "load_tree_completion_budget"]
