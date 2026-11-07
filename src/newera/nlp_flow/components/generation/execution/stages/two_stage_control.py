"""Two-stage AST-first pipeline and matched one-stage result-CFM control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from newera.nlp_flow.components.ast.flow.core.ast_cfm import ASTFlowOutput, ASTStageAFlow
from newera.nlp_flow.components.ast.diagnostics.tracing.ast_generation import (
    StructuredASTDecoding,
    predict_generated_topology_batch,
)
from newera.nlp_flow.components.ast.diagnostics.validation.ast_quality_gate import ASTHandoffEvidence
from newera.nlp_flow.components.ast.representation.structure.ast_token_decoder import decode_source_tokens
from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.model.components.architecture.representations.prompt_encoding import SemanticPromptEncoder
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_encoder import ParsedASTEncoder
from newera.nlp_flow.components.ast.representation.parsed.parsed_ast_handoff import (
    ParsedASTBatch,
    ParsedASTRuntimeRequest,
    sample_stage_b_from_parsed_sources,
)
from newera.nlp_flow.components.ast.flow.runtime.result_cfm import (
    GeneratedASTResultCFM,
    ResultCFMConfig,
    ResultCFMOutput,
)


@dataclass(frozen=True)
class TwoStagePipelineOutput:
    ast_output: ASTFlowOutput
    result_output: ResultCFMOutput
    nfe_a: int
    nfe_b: int
    ast_handoff: ASTHandoffEvidence | None = None
    decoded_ast: StructuredASTDecoding | None = None
    ast_source: str | None = None
    parsed_ast: ParsedASTBatch | None = None

    @property
    def total_nfe(self) -> int:
        return self.nfe_a + self.nfe_b


class TwoStageASTFirstPipeline(nn.Module):
    """Stage A generated AST feeds Stage B; no target AST is accepted."""

    def __init__(
        self,
        stage_a: ASTStageAFlow,
        stage_b: GeneratedASTResultCFM,
        prompt_encoder: SemanticPromptEncoder | None = None,
    ) -> None:
        super().__init__()
        if stage_a.hidden_size != stage_b.config.hidden_size:
            raise ValueError("Stage A and Stage B hidden sizes must match")
        if (
            prompt_encoder is not None
            and prompt_encoder.hidden_size != stage_a.hidden_size
        ):
            raise ValueError("prompt encoder and AST stage hidden sizes must match")
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.prompt_encoder = prompt_encoder

    @torch.no_grad()
    def sample_fixture_latent(
        self,
        prompt_context: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        level_masks: dict[str, torch.Tensor] | None = None,
        parent_indices: torch.Tensor | None = None,
        node_levels: torch.Tensor | None = None,
        result_length: int,
        steps_a: int = 4,
        steps_b: int = 4,
        ast_noise: torch.Tensor | None = None,
        result_noise: torch.Tensor | None = None,
        ast_quality: torch.Tensor | None = None,
    ) -> TwoStagePipelineOutput:
        """Test-only latent baseline; final sampling uses receipt-checked parsed graphs."""
        ast_output = self.stage_a.sample(
            prompt_context,
            node_mask,
            level_masks=level_masks,
            steps=steps_a,
            noise=ast_noise,
        )
        if parent_indices is None or node_levels is None:
            parent_indices, node_levels = predict_generated_topology_batch(
                ast_output,
                node_mask,
            )
        result_output = self.stage_b.sample(
            prompt_context,
            ast_output.endpoint,
            node_mask,
            (
                ast_quality
                if ast_quality is not None
                else ast_output.quality_score(node_mask)
            ),
            result_length=result_length,
            steps=steps_b,
            noise=result_noise,
            parent_indices=parent_indices,
            node_levels=node_levels,
        )
        return TwoStagePipelineOutput(
            ast_output=ast_output,
            result_output=result_output,
            nfe_a=steps_a,
            nfe_b=steps_b,
        )

    @torch.no_grad()
    def sample_target_free(
        self,
        prompt_context: torch.Tensor,
        *,
        language: Literal["python", "rust", "cpp"],
        result_length: int,
        steps_a: int | None = None,
        steps_b: int | None = None,
        ast_noise: torch.Tensor | None = None,
        result_noise: torch.Tensor | None = None,
        request: ParsedASTRuntimeRequest,
        parsed_ast_encoder: ParsedASTEncoder,
    ) -> TwoStagePipelineOutput:
        """Generate source, parse once, and route the exact receipt-bound graph to Stage B."""

        ast_output = self.stage_a.sample_target_free(
            prompt_context,
            steps=steps_a,
            noise=ast_noise,
        )
        node_mask = ast_output.predicted_node_mask()
        if ast_output.source_token_ids is None:
            raise ValueError("Stage A did not emit token-endpoint ids")
        sources = [
            decode_source_tokens(
                ast_output.source_token_ids[index],
                self.stage_a.vocabulary,
            )
            for index in range(node_mask.shape[0])
        ]
        for source in sources:
            if not source:
                raise ValueError("token-endpoint AST source is empty")
        handoff = sample_stage_b_from_parsed_sources(
            self.stage_b,
            prompt_context,
            sources,
            language=language,
            request=request,
            encoder=parsed_ast_encoder,
            result_length=result_length,
            steps=steps_b,
            noise=result_noise,
        )
        return TwoStagePipelineOutput(
            ast_output=ast_output,
            result_output=handoff.result,
            nfe_a=ast_output.executed_nfe,
            nfe_b=handoff.result.executed_nfe,
            ast_source="\n".join(sources) if len(sources) > 1 else sources[0],
            parsed_ast=handoff.parsed_ast,
        )

    @torch.no_grad()
    def sample_target_free_from_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        language: Literal["python", "rust", "cpp"],
        language_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        result_length: int,
        steps_a: int | None = None,
        steps_b: int | None = None,
        ast_noise: torch.Tensor | None = None,
        result_noise: torch.Tensor | None = None,
        request: ParsedASTRuntimeRequest,
        parsed_ast_encoder: ParsedASTEncoder,
    ) -> TwoStagePipelineOutput:
        if self.prompt_encoder is None:
            raise RuntimeError("token prompt generation requires a prompt encoder")
        prompt = self.prompt_encoder(
            input_ids,
            attention_mask,
            language_ids,
        )
        return self.sample_target_free(
            prompt.context,
            language=language,
            result_length=result_length,
            steps_a=steps_a,
            steps_b=steps_b,
            ast_noise=ast_noise,
            result_noise=result_noise,
            request=request,
            parsed_ast_encoder=parsed_ast_encoder,
        )

    @torch.no_grad()
    def sample_verified(
        self,
        prompt_context: torch.Tensor,
        *,
        language: Literal["python", "rust", "cpp"],
        result_length: int,
        request: ParsedASTRuntimeRequest,
        parsed_ast_encoder: ParsedASTEncoder,
        steps_a: int | None = None,
        steps_b: int | None = None,
        ast_noise: torch.Tensor | None = None,
        result_noise: torch.Tensor | None = None,
    ) -> TwoStagePipelineOutput:
        """Verified production alias; raw latent, topology, and quality metrics are unavailable."""
        return self.sample_target_free(
            prompt_context,
            language=language,
            result_length=result_length,
            steps_a=steps_a,
            steps_b=steps_b,
            ast_noise=ast_noise,
            result_noise=result_noise,
            request=request,
            parsed_ast_encoder=parsed_ast_encoder,
        )


class PromptConditionProjector(nn.Module):
    """Expand prompt state into fixed slots without receiving an AST."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        slots: int,
    ) -> None:
        super().__init__()
        self.slots = int(slots)
        self.project = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.RMSNorm(width),
            nn.Linear(width, hidden_size),
        )

    def forward(self, prompt_context: torch.Tensor) -> torch.Tensor:
        if prompt_context.ndim != 2:
            raise ValueError("prompt_context must be [batch, hidden]")
        return self.project(prompt_context)[:, None].expand(
            -1,
            self.slots,
            -1,
        )


class OneStageResultCFMControl(nn.Module):
    """Direct result CFM control over prompt/repository state, never AST."""

    def __init__(
        self,
        config: ResultCFMConfig,
        *,
        condition_width: int = 64,
        condition_slots: int = 8,
    ) -> None:
        super().__init__()
        self.config = config
        self.condition = PromptConditionProjector(
            config.hidden_size,
            condition_width,
            condition_slots,
        )
        self.result_cfm = GeneratedASTResultCFM(config)

    def _condition(
        self,
        prompt_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = self.condition(prompt_context)
        return states, torch.ones(states.shape[:2], device=states.device, dtype=torch.bool)

    @torch.no_grad()
    def sample(
        self,
        prompt_context: torch.Tensor,
        *,
        result_length: int,
        steps: int = 8,
        noise: torch.Tensor | None = None,
    ) -> ResultCFMOutput:
        condition, mask = self._condition(prompt_context)
        return self.result_cfm.sample(
            prompt_context,
            condition,
            mask,
            result_length=result_length,
            steps=steps,
            noise=noise,
        )


@dataclass(frozen=True)
class ParameterMatch:
    width: int
    target_parameters: int
    control_parameters: int
    relative_difference: float

    @property
    def passed(self) -> bool:
        return self.relative_difference <= 0.02


def match_control_width(
    pipeline: TwoStageASTFirstPipeline,
    *,
    candidate_widths: tuple[int, ...] = tuple(range(16, 513, 8)),
    condition_slots: int = 8,
) -> tuple[OneStageResultCFMControl, ParameterMatch]:
    target = count_unique_trainable_parameters(pipeline)
    best: tuple[OneStageResultCFMControl, ParameterMatch] | None = None
    for width in candidate_widths:
        control = OneStageResultCFMControl(
            pipeline.stage_b.config,
            condition_width=width,
            condition_slots=condition_slots,
        )
        parameters = count_unique_trainable_parameters(control)
        match = ParameterMatch(
            width=width,
            target_parameters=target,
            control_parameters=parameters,
            relative_difference=abs(parameters - target) / max(target, 1),
        )
        if best is None or match.relative_difference < best[1].relative_difference:
            best = (control, match)
    assert best is not None
    return best


__all__ = [
    "OneStageResultCFMControl",
    "ParameterMatch",
    "PromptConditionProjector",
    "TwoStageASTFirstPipeline",
    "TwoStagePipelineOutput",
    "match_control_width",
]
