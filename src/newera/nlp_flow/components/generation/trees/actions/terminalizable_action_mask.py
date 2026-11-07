"""Exact finite-grammar completion costs and node-candidate masks."""

from __future__ import annotations

from dataclasses import dataclass

from newera.nlp_flow.components.ast.diagnostics.validation.ast_completion_contract import ASTCompletionContract, ASTCompletionState


class UnsupportedCompletionGrammar(ValueError):
    """Raised when a required production has no finite derivation."""


@dataclass(frozen=True, order=True)
class CompletionChoice:
    field: str
    child_kind: str
    child_named: bool
    minimum_nodes: int


@dataclass(frozen=True)
class CompletionCertificate:
    node_kind: str
    node_named: bool
    minimum_nodes: int
    required_choices: tuple[CompletionChoice, ...]


@dataclass(frozen=True)
class CandidateMask:
    field: str
    candidates: tuple[tuple[str, bool], ...]
    allowed: tuple[bool, ...]
    added_node_costs: tuple[int | None, ...]
    remaining_node_budget: int

    @property
    def allowed_candidates(self) -> tuple[tuple[str, bool], ...]:
        return tuple(candidate for candidate, allowed in zip(self.candidates, self.allowed, strict=True) if allowed)


class TerminalizableActionMask:
    """Solve required-child conjunctions and mask choices that cannot finish in budget."""

    def __init__(self, contract: ASTCompletionContract) -> None:
        self.contract = contract
        self._rules = contract.fields
        self._minimum_costs = self._solve_minimum_costs()

    def certificate(self, node_kind: str, *, named: bool = True) -> CompletionCertificate:
        cost = self._minimum_costs.get((node_kind, named))
        if cost is None:
            raise UnsupportedCompletionGrammar(
                f"no finite completion derivation for {node_kind} (named={named})"
            )
        choices = []
        for rule in self._required_rules(node_kind, named):
            choice = self._best_alternative(rule.alternatives)
            if choice is None:
                raise UnsupportedCompletionGrammar(
                    f"required field has no finite alternative: {node_kind}.{rule.field}"
                )
            child_kind, child_named, child_cost = choice
            choices.append(CompletionChoice(rule.field, child_kind, child_named, child_cost))
        return CompletionCertificate(node_kind, named, cost, tuple(choices))

    def minimum_additional_nodes(self, state: ASTCompletionState) -> int:
        if state.contract != self.contract:
            raise ValueError("completion state belongs to a different grammar contract")
        required_cost = 0
        for frame in state.frames:
            counts = dict(frame.field_counts)
            for rule in self._required_rules(frame.kind, frame.named):
                if counts.get(rule.field, 0):
                    continue
                choice = self._best_alternative(rule.alternatives)
                if choice is None:
                    raise UnsupportedCompletionGrammar(
                        f"required field has no finite alternative: {frame.kind}.{rule.field}"
                    )
                required_cost += choice[2]
        return required_cost

    def candidate_mask(
        self,
        state: ASTCompletionState,
        *,
        field: str,
        candidates: tuple[tuple[str, bool], ...],
        remaining_node_budget: int,
    ) -> CandidateMask:
        if remaining_node_budget < 0:
            raise ValueError("remaining node budget cannot be negative")
        if state.contract != self.contract:
            raise ValueError("completion state belongs to a different grammar contract")
        if not state.frames or state.finished:
            raise ValueError("candidate masking requires an active parent node")
        allowed = []
        costs: list[int | None] = []
        for kind, named in candidates:
            try:
                next_state = state.append_child(field, kind, named=named)
            except ValueError:
                allowed.append(False)
                costs.append(None)
                continue
            certificate = self.certificate(kind, named=named)
            added_cost = 1 + self.minimum_additional_nodes(next_state)
            allowed.append(added_cost <= remaining_node_budget)
            costs.append(added_cost)
            if certificate.minimum_nodes < 1:
                raise AssertionError("subtree completion cost must include its root")
        return CandidateMask(field, candidates, tuple(allowed), tuple(costs), remaining_node_budget)

    def teacher_action_reachable(
        self,
        state: ASTCompletionState,
        *,
        field: str,
        child_kind: str,
        child_named: bool,
        remaining_node_budget: int,
    ) -> bool:
        mask = self.candidate_mask(
            state,
            field=field,
            candidates=((child_kind, child_named),),
            remaining_node_budget=remaining_node_budget,
        )
        return mask.allowed[0]

    def can_close_node(self, state: ASTCompletionState) -> bool:
        if not state.frames:
            return False
        frame = state.frames[-1]
        counts = dict(frame.field_counts)
        return all(counts.get(rule.field, 0) > 0 for rule in self._required_rules(frame.kind, frame.named))

    def _required_rules(self, kind: str, named: bool):
        return tuple(rule for rule in self._rules if rule.parent_kind == kind and rule.parent_named == named and rule.required)

    def _best_alternative(self, alternatives):
        choices = [
            (item.kind, item.named, self._minimum_costs[(item.kind, item.named)])
            for item in alternatives
            if (item.kind, item.named) in self._minimum_costs
        ]
        return min(choices, key=lambda item: (item[2], item[0], item[1])) if choices else None

    def _solve_minimum_costs(self) -> dict[tuple[str, bool], int]:
        node_kinds = {
            (rule.parent_kind, rule.parent_named)
            for rule in self._rules
        }
        node_kinds.update((alternative.kind, alternative.named) for rule in self._rules for alternative in rule.alternatives)
        node_kinds.update((kind, named) for _parent, kind, named in self.contract.extra_child_kinds)
        node_kinds.update((kind, True) for kind in self.contract.legal_root_kinds)
        for kind, _classes in self.contract.lexical_classes:
            node_kinds.update(((kind, True), (kind, False)))
        costs: dict[tuple[str, bool], int] = {}
        for kind, named in node_kinds:
            if not self._required_rules(kind, named):
                costs[(kind, named)] = 1
        changed = True
        while changed:
            changed = False
            for kind, named in node_kinds:
                rules = self._required_rules(kind, named)
                child_costs = [
                    min(
                        (
                            costs[(alternative.kind, alternative.named)]
                            for alternative in rule.alternatives
                            if (alternative.kind, alternative.named) in costs
                        ),
                        default=None,
                    )
                    for rule in rules
                ]
                if any(value is None for value in child_costs):
                    continue
                candidate = 1 + sum(int(value) for value in child_costs)
                if candidate < costs.get((kind, named), candidate + 1):
                    costs[(kind, named)] = candidate
                    changed = True
        return costs


__all__ = [
    "CandidateMask",
    "CompletionCertificate",
    "CompletionChoice",
    "TerminalizableActionMask",
    "UnsupportedCompletionGrammar",
]
