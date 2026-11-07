"""Stage-B result flow conditioned only on generated Stage-A AST state."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from newera.nlp_flow.components.research.training.objectives.cfm_training import (
    CFMTrainingConfig,
    sample_flow_times,
)


@dataclass(frozen=True)
class ResultCFMConfig:
    hidden_size: int
    rank: int = 16
    primary_nfe: int = 4
    dropout: float = 0.0
    early_fraction: float = 0.2
    generated_only_fraction: float = 0.7
    timestep_sampler: Literal["uniform", "logit_normal"] = "logit_normal"
    logit_normal_std: float = 0.8
    minimum_ast_quality: float = 0.8
    enforce_ast_quality: bool = False

    def __post_init__(self) -> None:
        for name in ("hidden_size", "rank", "primary_nfe"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.early_fraction < self.generated_only_fraction <= 1.0:
            raise ValueError("curriculum fractions must be ordered in (0, 1]")
        if self.timestep_sampler not in {"uniform", "logit_normal"}:
            raise ValueError("unsupported timestep sampler")
        if self.logit_normal_std <= 0.0:
            raise ValueError("logit-normal standard deviation must be positive")
        if not 0.0 <= self.minimum_ast_quality <= 1.0:
            raise ValueError("minimum AST quality must be in [0, 1]")
        if not isinstance(self.enforce_ast_quality, bool):
            raise TypeError("enforce_ast_quality must be bool")


@dataclass(frozen=True)
class ResultCFMOutput:
    velocity: torch.Tensor
    endpoint: torch.Tensor
    condition: torch.Tensor
    verifier_logits: torch.Tensor
    ast_consistency: torch.Tensor
    uncertainty_log_std: torch.Tensor
    executed_nfe: int = 0


@dataclass(frozen=True)
class ResultCFMLoss:
    total: torch.Tensor
    velocity: torch.Tensor
    endpoint: torch.Tensor
    reconstruction: torch.Tensor
    verifier: torch.Tensor
    ast_consistency: torch.Tensor
    uncertainty: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "velocity": self.velocity,
            "endpoint": self.endpoint,
            "reconstruction": self.reconstruction,
            "verifier": self.verifier,
            "ast_consistency": self.ast_consistency,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class CurriculumStep:
    phase: str
    teacher_weight: float
    generated_weight: float


class GeneratedASTCurriculum:
    """Explicit teacher -> mixed -> generated-only schedule."""

    def __init__(
        self,
        total_steps: int,
        *,
        early_fraction: float = 0.2,
        generated_only_fraction: float = 0.7,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 < early_fraction < generated_only_fraction <= 1.0:
            raise ValueError("curriculum fractions must be ordered in (0, 1]")
        self.total_steps = int(total_steps)
        self.early_fraction = float(early_fraction)
        self.generated_only_fraction = float(generated_only_fraction)

    def at(self, step: int) -> CurriculumStep:
        if not 0 <= step < self.total_steps:
            raise ValueError("step is outside the curriculum")
        progress = step / max(1, self.total_steps - 1)
        if progress < self.early_fraction:
            teacher_weight = 1.0
            phase = "teacher"
        elif progress < self.generated_only_fraction:
            remaining = 1.0 - (
                progress - self.early_fraction
            ) / (self.generated_only_fraction - self.early_fraction)
            teacher_weight = remaining
            phase = "mixed"
        else:
            teacher_weight = 0.0
            phase = "generated_only"
        return CurriculumStep(
            phase=phase,
            teacher_weight=teacher_weight,
            generated_weight=1.0 - teacher_weight,
        )

    def blend(
        self,
        step: int,
        teacher_ast: torch.Tensor,
        generated_ast: torch.Tensor,
    ) -> tuple[torch.Tensor, CurriculumStep]:
        if teacher_ast.shape != generated_ast.shape:
            raise ValueError("teacher and generated AST states must match")
        schedule = self.at(step)
        condition = (
            schedule.teacher_weight * teacher_ast
            + schedule.generated_weight * generated_ast
        )
        return condition, schedule


class _TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(1, half - 1)
        )
        phases = time[:, None] * frequencies[None]
        values = torch.cat((phases.sin(), phases.cos()), dim=-1)
        if self.dimension % 2:
            values = F.pad(values, (0, 1))
        return values


class GeneratedASTConditioner(nn.Module):
    """Encode generated AST content and topology without target-AST input."""

    def __init__(self, hidden_size: int, heads: int = 4) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("conditioner hidden_size must be divisible by heads")
        self.norm = nn.RMSNorm(hidden_size)
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.position_embedding = nn.Embedding(512, hidden_size)
        self.level_embedding = nn.Embedding(4, hidden_size)
        self.parent_distance_embedding = nn.Embedding(65, hidden_size)
        self.attention_norm = nn.RMSNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            batch_first=True,
        )
        self.ffn_norm = nn.RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        generated_ast: torch.Tensor,
        ast_mask: torch.Tensor,
        prompt_context: torch.Tensor,
        ast_quality: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
        node_levels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if generated_ast.ndim != 3:
            raise ValueError("generated_ast must be [batch, nodes, hidden]")
        if ast_mask.shape != generated_ast.shape[:2]:
            raise ValueError("ast_mask must match generated AST nodes")
        if prompt_context.shape != generated_ast.shape[:1] + generated_ast.shape[-1:]:
            raise ValueError("prompt_context must be [batch, hidden]")
        batch, sequence, _hidden = generated_ast.shape
        if sequence > self.position_embedding.num_embeddings:
            raise ValueError("generated AST exceeds conditioner position capacity")
        parents = (
            torch.full(
                (batch, sequence),
                -1,
                dtype=torch.long,
                device=generated_ast.device,
            )
            if parent_indices is None
            else parent_indices.to(generated_ast.device, torch.long)
        )
        levels = (
            torch.zeros(
                (batch, sequence),
                dtype=torch.long,
                device=generated_ast.device,
            )
            if node_levels is None
            else node_levels.to(generated_ast.device, torch.long)
        )
        if parents.shape != generated_ast.shape[:2] or levels.shape != parents.shape:
            raise ValueError("AST topology tensors must match node slots")
        if levels.lt(0).any() or levels.ge(4).any():
            raise ValueError("AST node levels must be in [0, 4)")
        positions = torch.arange(sequence, device=generated_ast.device)
        distance = torch.where(
            parents.ge(0),
            (positions[None] - parents).abs(),
            torch.zeros_like(parents),
        ).clamp_max(64)
        states = (
            generated_ast
            + self.position_embedding(positions)[None]
            + self.level_embedding(levels)
            + self.parent_distance_embedding(distance)
        )
        attended, _weights = self.attention(
            self.attention_norm(states),
            self.attention_norm(states),
            self.attention_norm(states),
            key_padding_mask=~ast_mask,
            need_weights=False,
        )
        states = states + attended
        states = states + self.ffn(self.ffn_norm(states))
        if ast_quality is not None:
            quality = ast_quality.to(
                device=generated_ast.device,
                dtype=generated_ast.dtype,
            )
            if quality.shape != (generated_ast.shape[0],):
                raise ValueError("ast_quality must be [batch]")
            if any(
                value < 0.0 or value > 1.0
                for value in quality.detach().cpu().tolist()
            ):
                raise ValueError("ast_quality must be in [0, 1]")
        else:
            quality = torch.ones(
                generated_ast.shape[0],
                device=generated_ast.device,
                dtype=generated_ast.dtype,
            )
        weights = ast_mask.to(generated_ast.dtype)[..., None]
        weighted = states * weights * quality[:, None, None]
        pooled = weighted.sum(dim=1)
        pooled = pooled / (
            weights.sum(dim=1) * quality[:, None]
        ).clamp_min(1.0)
        return self.project(self.norm(pooled + prompt_context))


class ResultVectorField(nn.Module):
    def __init__(self, config: ResultCFMConfig) -> None:
        super().__init__()
        self.time = nn.Sequential(
            _TimeEmbedding(config.rank),
            nn.Linear(config.rank, config.rank),
            nn.SiLU(),
            nn.Linear(config.rank, config.rank),
        )
        self.state = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.context = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.up = nn.Linear(config.rank, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.RMSNorm(config.hidden_size)
        self.self_attention = nn.MultiheadAttention(
            config.hidden_size,
            4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.RMSNorm(config.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
        )

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        attended, _weights = self.self_attention(
            self.attention_norm(state),
            self.attention_norm(state),
            self.attention_norm(state),
            need_weights=False,
        )
        state = state + self.dropout(attended)
        state = state + self.dropout(self.ffn(self.ffn_norm(state)))
        hidden = (
            self.state(state)
            + self.context(condition)[:, None]
            + self.time(time)[:, None]
        )
        velocity = self.up(self.dropout(F.silu(hidden)))
        return self.norm(velocity + state)


class GeneratedASTResultCFM(nn.Module):
    """Stage B consumes Stage-A generated AST and never a target AST."""

    def __init__(self, config: ResultCFMConfig) -> None:
        super().__init__()
        self.config = config
        self.conditioner = GeneratedASTConditioner(config.hidden_size)
        self.vector_field = ResultVectorField(config)
        self.verifier = nn.Linear(config.hidden_size, 1)
        self.consistency = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.uncertainty = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        result_state: torch.Tensor,
        time: torch.Tensor,
        prompt_context: torch.Tensor,
        generated_ast: torch.Tensor,
        ast_mask: torch.Tensor,
        ast_quality: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
        node_levels: torch.Tensor | None = None,
    ) -> ResultCFMOutput:
        if result_state.ndim != 3:
            raise ValueError("result_state must be [batch, length, hidden]")
        if time.shape != (result_state.shape[0],):
            raise ValueError("time must be [batch]")
        condition = self.conditioner(
            generated_ast,
            ast_mask,
            prompt_context,
            ast_quality,
            parent_indices,
            node_levels,
        )
        velocity = self.vector_field(result_state, time, condition)
        endpoint = result_state + (1.0 - time[:, None, None]) * velocity
        pooled_endpoint = endpoint.mean(dim=1)
        consistency_target = self.consistency(condition)
        ast_consistency = F.cosine_similarity(
            pooled_endpoint,
            consistency_target,
            dim=-1,
        )
        return ResultCFMOutput(
            velocity=velocity,
            endpoint=endpoint,
            condition=condition,
            verifier_logits=self.verifier(pooled_endpoint).squeeze(-1),
            ast_consistency=ast_consistency,
            uncertainty_log_std=self.uncertainty(pooled_endpoint)
            .squeeze(-1)
            .clamp(-10.0, 5.0),
        )

    def training_loss(
        self,
        result_target: torch.Tensor,
        prompt_context: torch.Tensor,
        generated_ast: torch.Tensor,
        ast_mask: torch.Tensor,
        ast_quality: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
        node_levels: torch.Tensor | None = None,
        *,
        verifier_target: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[ResultCFMLoss, ResultCFMOutput]:
        """Training accepts generated AST state; teacher mixing happens upstream."""

        if result_target.ndim != 3:
            raise ValueError("result_target must be [batch, length, hidden]")
        if noise is None:
            noise = torch.randn_like(result_target)
        if noise.shape != result_target.shape:
            raise ValueError("noise must match result_target")
        if time is None:
            time = sample_flow_times(
                result_target.shape[0],
                CFMTrainingConfig(
                    timestep_sampler=self.config.timestep_sampler,
                    logit_normal_std=self.config.logit_normal_std,
                ),
                device=result_target.device,
                dtype=result_target.dtype,
            )
        state = (
            (1.0 - time[:, None, None]) * noise
            + time[:, None, None] * result_target
        )
        output = self.forward(
            state,
            time,
            prompt_context,
            generated_ast,
            ast_mask,
            ast_quality,
            parent_indices,
            node_levels,
        )
        velocity = (output.velocity - (result_target - noise)).square().mean()
        endpoint = (output.endpoint - result_target).square().mean()
        reconstruction = F.smooth_l1_loss(output.endpoint, result_target)
        if verifier_target is None:
            verifier_target = torch.ones_like(output.verifier_logits)
        verifier = F.binary_cross_entropy_with_logits(
            output.verifier_logits,
            verifier_target.to(output.verifier_logits),
        )
        ast_consistency = (1.0 - output.ast_consistency).mean()
        remaining = (output.endpoint - result_target).square().mean(dim=(1, 2))
        variance = (2.0 * output.uncertainty_log_std).exp()
        uncertainty = (
            remaining / variance.clamp_min(1.0e-8)
            + 2.0 * output.uncertainty_log_std
        ).mean()
        total = (
            velocity
            + endpoint
            + reconstruction
            + 0.25 * verifier
            + 0.5 * ast_consistency
            + 0.1 * uncertainty
        )
        return (
            ResultCFMLoss(
                total=total,
                velocity=velocity,
                endpoint=endpoint,
                reconstruction=reconstruction,
                verifier=verifier,
                ast_consistency=ast_consistency,
                uncertainty=uncertainty,
            ),
            output,
        )

    @torch.no_grad()
    def sample(
        self,
        prompt_context: torch.Tensor,
        generated_ast: torch.Tensor,
        ast_mask: torch.Tensor,
        ast_quality: torch.Tensor | None = None,
        parent_indices: torch.Tensor | None = None,
        node_levels: torch.Tensor | None = None,
        *,
        result_length: int,
        steps: int | None = None,
        max_steps: int = 32,
        noise: torch.Tensor | None = None,
    ) -> ResultCFMOutput:
        """Primary inference path: prompt plus Stage-A generated AST only."""

        if result_length <= 0:
            raise ValueError("result_length must be positive")
        if ast_quality is not None:
            quality = ast_quality.to(
                device=prompt_context.device,
                dtype=prompt_context.dtype,
            )
            if quality.shape != (prompt_context.shape[0],):
                raise ValueError("ast_quality must be [batch]")
            if (
                self.config.enforce_ast_quality
                and quality.min().item() < self.config.minimum_ast_quality
            ):
                raise ValueError("generated AST does not meet Stage-B quality gate")
        loop_steps = self.config.primary_nfe if steps is None else steps
        if steps is not None and steps <= 0:
            raise ValueError("steps must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        batch = prompt_context.shape[0]
        if noise is None:
            state = torch.randn(
                batch,
                result_length,
                self.config.hidden_size,
                device=prompt_context.device,
                dtype=prompt_context.dtype,
            )
        else:
            state = noise.to(prompt_context)
        if state.shape != (batch, result_length, self.config.hidden_size):
            raise ValueError("noise must be [batch, result_length, hidden]")
        executed_steps = loop_steps
        for step in range(max_steps if steps is None else loop_steps):
            time = torch.full(
                (batch,),
                step / max(1, max_steps if steps is None else loop_steps),
                device=prompt_context.device,
                dtype=prompt_context.dtype,
            )
            step_output = self.forward(
                state,
                time,
                prompt_context,
                generated_ast,
                ast_mask,
                ast_quality,
                parent_indices,
                node_levels,
            )
            delta = step_output.velocity / max(
                1,
                max_steps if steps is None else loop_steps,
            )
            state = state + delta
            if steps is None:
                delta_rms = delta.square().mean(dim=(1, 2)).sqrt()
                if torch.all(
                    delta_rms <= step_output.uncertainty_log_std.exp()
                ):
                    executed_steps = step + 1
                    break
        final = self.forward(
            state,
            torch.ones(
                batch,
                device=prompt_context.device,
                dtype=prompt_context.dtype,
            ),
            prompt_context,
            generated_ast,
            ast_mask,
            ast_quality,
            parent_indices,
            node_levels,
        )
        return replace(final, executed_nfe=executed_steps)


__all__ = [
    "CurriculumStep",
    "GeneratedASTConditioner",
    "GeneratedASTCurriculum",
    "GeneratedASTResultCFM",
    "ResultCFMConfig",
    "ResultCFMLoss",
    "ResultCFMOutput",
    "ResultVectorField",
]
