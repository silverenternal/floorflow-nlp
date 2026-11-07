"""Runtime synthesis and CFM inference APIs for the compact model."""

from __future__ import annotations

import torch

from newera.nlp_flow.components.ast.diagnostics.validation.ast_quality_gate import ASTHandoffEvidence
from newera.nlp_flow.model.components.training.objectives.cfm import TwoStageFlowOutput
from newera.nlp_flow.model.components.experts.competence.runtime_competence import (
    GAP_EVIDENCE_FIELDS,
    RuntimeCompetenceRecord,
)


class RuntimeModelMixin:
    def runtime_competence_evidence(
        self,
        hidden: torch.Tensor,
        *,
        novelty: torch.Tensor | None = None,
        router_entropy: torch.Tensor | None = None,
        validity_margin: torch.Tensor | None = None,
        verifier_failure: torch.Tensor | None = None,
        repetition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [batch, sequence, hidden]")
        batch = hidden.shape[0]

        def default(value, fill):
            return (
                hidden.new_full((batch,), fill)
                if value is None
                else value.to(hidden.device, hidden.dtype)
            )

        return torch.stack(
            (
                default(novelty, 0.0),
                default(router_entropy, 0.5),
                default(validity_margin, 0.5),
                default(verifier_failure, 0.0),
                default(repetition, 0.0),
            ),
            dim=-1,
        )

    def synthesize_runtime_competence(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        evidence: torch.Tensor,
        *,
        gap_key: str,
    ) -> RuntimeCompetenceRecord | None:
        if self.runtime_competence is None:
            raise RuntimeError("runtime competence manager is disabled")
        if evidence.shape != (hidden.shape[0], len(GAP_EVIDENCE_FIELDS)):
            raise ValueError("runtime evidence shape is invalid")
        return self.runtime_competence.synthesize(
            hidden.mean(dim=1),
            context,
            evidence,
            gap_key=gap_key,
        )

    def observe_runtime_competence(
        self,
        record: RuntimeCompetenceRecord,
        *,
        valid: bool,
        utility: float,
    ) -> str:
        if self.runtime_competence is None:
            raise RuntimeError("runtime competence manager is disabled")
        return self.runtime_competence.observe(
            record,
            valid=valid,
            utility=utility,
        )

    def apply_runtime_competence(
        self,
        hidden: torch.Tensor,
        record: RuntimeCompetenceRecord,
    ) -> torch.Tensor:
        if self.runtime_competence is None:
            raise RuntimeError("runtime competence manager is disabled")
        return self.runtime_competence.apply(hidden, record)

    def predict_cfm(
        self,
        ast_state: torch.Tensor,
        result_state: torch.Tensor,
        time_a: torch.Tensor,
        time_b: torch.Tensor,
        context: torch.Tensor,
    ) -> TwoStageFlowOutput:
        if self.cfm is None:
            raise RuntimeError("two-stage CFM is disabled")
        return self.cfm.predict(
            ast_state,
            result_state,
            time_a,
            time_b,
            context,
        )

    @torch.no_grad()
    def sample_cfm(
        self,
        context: torch.Tensor,
        *,
        steps_a: int = 4,
        steps_b: int = 4,
        ast_noise: torch.Tensor | None = None,
        result_noise: torch.Tensor | None = None,
    ) -> TwoStageFlowOutput:
        if self.cfm is None:
            raise RuntimeError("two-stage CFM is disabled")
        return self.cfm.sample(
            context,
            steps_a=steps_a,
            steps_b=steps_b,
            ast_noise=ast_noise,
            result_noise=result_noise,
        )

    @torch.no_grad()
    def sample_verified_cfm(
        self,
        context: torch.Tensor,
        ast_handoff: ASTHandoffEvidence,
        *,
        steps_a: int = 4,
        steps_b: int = 4,
        ast_noise: torch.Tensor | None = None,
        result_noise: torch.Tensor | None = None,
    ) -> TwoStageFlowOutput:
        """Final compact-model CFM path; unverified Stage A cannot reach Stage B."""

        ast_handoff.require_stage_b_ready(require_reference_metrics=True)
        return self.sample_cfm(
            context,
            steps_a=steps_a,
            steps_b=steps_b,
            ast_noise=ast_noise,
            result_noise=result_noise,
        )

    def encode_cfm_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self(input_ids, attention_mask=attention_mask)
        mask = (
            torch.ones(
                input_ids.shape,
                dtype=torch.bool,
                device=input_ids.device,
            )
            if attention_mask is None
            else attention_mask.to(device=input_ids.device, dtype=torch.bool)
        )
        if mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        weights = mask.to(output.hidden_states.dtype)[..., None]
        return (output.hidden_states * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    def ast_result_token_logits(self, result_state: torch.Tensor) -> torch.Tensor:
        if self.cfm is None:
            raise RuntimeError("two-stage CFM is disabled")
        if result_state.ndim != 3 or result_state.shape[-1] != self.config.hidden_size:
            raise ValueError("result_state must be [batch, length, hidden]")
        weight = self.embedding.weight if self.output is None else self.output.weight
        return self.final_norm(result_state) @ weight.transpose(0, 1)


__all__ = ["RuntimeModelMixin"]
