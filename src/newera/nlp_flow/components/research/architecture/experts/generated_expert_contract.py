"""Bounded generated-E contracts over the existing factory and runtime store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.model.components.experts.banks.expert_factory import (
    ContextConditionedExpertFactory,
    MaterializedExpert,
)
from newera.nlp_flow.model.components.experts.competence.runtime_competence import (
    RuntimeCompetenceManager,
    RuntimeCompetenceRecord,
)
from newera.nlp_flow.components.research.evaluation.evidence.verifier_result_contract import VerificationEvidence


@dataclass(frozen=True)
class GeneratedExpertEnvelope:
    max_rank: int
    max_width: int
    max_basis_parameters: int
    max_materialized_parameters: int
    max_expert_norm: float

    def __post_init__(self) -> None:
        if min(
            self.max_rank,
            self.max_width,
            self.max_basis_parameters,
            self.max_materialized_parameters,
        ) <= 0 or self.max_expert_norm <= 0:
            raise ValueError("generated expert envelope values must be positive")


def validate_generated_expert(
    factory: ContextConditionedExpertFactory,
    expert: MaterializedExpert,
    envelope: GeneratedExpertEnvelope,
) -> dict[str, int | float]:
    if factory.max_rank > envelope.max_rank or factory.hidden_size > envelope.max_width:
        raise ValueError("factory exceeds the fixed generated expert envelope")
    basis_parameters = factory.left_basis.numel() + factory.right_basis.numel()
    if basis_parameters > envelope.max_basis_parameters:
        raise ValueError("generated expert basis exceeds the resident parameter envelope")
    if expert.left.ndim != 3 or expert.right.ndim != 3:
        raise ValueError("materialized generated expert tensors have invalid rank")
    if expert.left.shape[1] > envelope.max_rank or expert.right.shape[2] > envelope.max_rank:
        raise ValueError("materialized rank exceeds the active envelope")
    if expert.left.shape[2] > envelope.max_width or expert.right.shape[1] > envelope.max_width:
        raise ValueError("materialized width exceeds the active envelope")
    if expert.left.shape[0] * expert.left.shape[1] * expert.left.shape[2] + (
        expert.right.shape[0] * expert.right.shape[1] * expert.right.shape[2]
    ) > envelope.max_materialized_parameters:
        raise ValueError("materialized expert exceeds the active parameter envelope")
    if expert.rank.numel() == 0 or expert.width.numel() == 0:
        raise ValueError("blueprint rank and width must be explicit")
    if torch.any(expert.rank <= 0) or torch.any(expert.rank > envelope.max_rank) or torch.any(expert.rank > factory.max_rank):
        raise ValueError("blueprint rank is outside the envelope")
    if torch.any(expert.width <= 0) or torch.any(expert.width > envelope.max_width):
        raise ValueError("blueprint width is outside the envelope")
    for scale in (expert.input_scale, expert.output_scale):
        if not torch.isfinite(scale).all() or torch.any(scale < 0) or torch.any(scale > 1):
            raise ValueError("blueprint scales must be finite and in [0, 1]")
    norm = float((expert.left.square().sum() + expert.right.square().sum()).sqrt().detach())
    if not torch.isfinite(torch.tensor(norm)) or norm > envelope.max_expert_norm:
        raise ValueError("generated expert norm is non-finite or exceeds its envelope")
    return {
        "basis_parameters": basis_parameters,
        "materialized_parameters": expert.left.numel() + expert.right.numel(),
        "rank": int(expert.rank.max()),
        "width": int(expert.width.max()),
        "norm": norm,
    }


class GeneratedExpertLifecycle:
    """Candidate creation, gated application and verifier-bound promotion."""

    def __init__(
        self,
        factory: ContextConditionedExpertFactory,
        runtime: RuntimeCompetenceManager,
        envelope: GeneratedExpertEnvelope,
    ) -> None:
        if runtime.factory is not factory:
            raise ValueError("runtime manager must reuse the configured expert factory")
        self.factory = factory
        self.runtime = runtime
        self.envelope = envelope

    @torch.no_grad()
    def create_candidate(
        self,
        pooled_hidden: torch.Tensor,
        evidence_features: torch.Tensor,
        *,
        gap_key: str,
    ) -> RuntimeCompetenceRecord | None:
        materialized = self.factory.materialize(pooled_hidden[:1])
        validate_generated_expert(self.factory, materialized, self.envelope)
        record = self.runtime.synthesize(
            pooled_hidden,
            None,
            evidence_features,
            gap_key=gap_key,
        )
        if record is not None:
            validate_generated_expert(self.factory, record.expert, self.envelope)
        return record

    def apply_promoted(
        self,
        hidden: torch.Tensor,
        record: RuntimeCompetenceRecord,
    ) -> torch.Tensor:
        if record.status != "promoted":
            return torch.zeros_like(hidden)
        validate_generated_expert(self.factory, record.expert, self.envelope)
        return self.runtime.apply(hidden, record)

    def observe(
        self,
        record: RuntimeCompetenceRecord,
        evidence: VerificationEvidence,
        *,
        utility: float,
    ) -> str:
        if evidence.identity.candidate_sha256 != record.signature:
            raise ValueError("verifier evidence belongs to another generated expert")
        if evidence.identity.case_id != record.gap_key:
            raise ValueError("verifier evidence belongs to another runtime gap")
        if not evidence.passed and utility != 0:
            raise ValueError("failed verifier evidence cannot receive positive utility")
        return self.runtime.observe(record, valid=evidence.passed, utility=utility)

    def retire(self, record: RuntimeCompetenceRecord) -> None:
        self.runtime.retire(record.signature)

    def save(self, path: str) -> None:
        self.runtime.save(path)

    def export_state(self) -> dict[str, object]:
        return self.runtime.export_state()

    def restore(self, payload: dict[str, Any]) -> None:
        self.runtime.load_state(payload)

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location=next(self.factory.parameters()).device, weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("generated expert store checkpoint must be a mapping")
        self.runtime.load_state(payload)


__all__ = [
    "GeneratedExpertEnvelope",
    "GeneratedExpertLifecycle",
    "validate_generated_expert",
]
