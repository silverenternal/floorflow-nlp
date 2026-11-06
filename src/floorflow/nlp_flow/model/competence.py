"""Unified competence cells and planner for depth, lanes, gaps, and E birth."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from floorflow.nlp_flow.model.expert_factory import (
    ContextConditionedExpertFactory,
    MaterializedExpert,
)
from floorflow.nlp_flow.model.gap_ensemble import GapCrossAttentionAggregator


@dataclass(frozen=True)
class UnifiedCompetenceConfig:
    hidden_size: int
    competence_names: tuple[str, ...]
    cell_rank: int = 8
    top_k: int = 3
    planner_dim: int = 64
    generated_rank: int = 16
    generated_basis_count: int = 4
    max_depth_steps: int = 8
    parallel_lanes: int = 4
    generated_expert_mode: str = "online"

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "cell_rank",
            "top_k",
            "planner_dim",
            "generated_rank",
            "generated_basis_count",
            "max_depth_steps",
            "parallel_lanes",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.competence_names or len(set(self.competence_names)) != len(
            self.competence_names
        ):
            raise ValueError("competence names must be non-empty and unique")
        if self.top_k > len(self.competence_names):
            raise ValueError("top_k cannot exceed competence count")
        if self.generated_expert_mode not in {"online", "disabled"}:
            raise ValueError("unsupported generated expert mode")


@dataclass(frozen=True)
class UnifiedPlannerOutput:
    competence_logits: torch.Tensor
    depth_logits: torch.Tensor
    lane_logits: torch.Tensor
    birth_logits: torch.Tensor
    mode_logits: torch.Tensor
    budget_logits: torch.Tensor


@dataclass(frozen=True)
class UnifiedCompetenceOutput:
    hidden_states: torch.Tensor
    planner: UnifiedPlannerOutput
    competence_states: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor | str]


class CompetenceCell(nn.Module):
    """One stateful/residual competence primitive used by all capability gaps."""

    def __init__(self, hidden_size: int, rank: int, *, stateful: bool = False) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.norm = nn.RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.stateful = bool(stateful)
        self.state_gate = nn.GRUCell(rank, rank) if self.stateful else None
        self.state_scale = nn.Linear(rank, rank) if self.stateful else None
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim not in {2, 3} or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must be [batch, (sequence,) hidden]")
        if self.stateful and hidden.ndim != 3:
            raise ValueError("stateful competence cells require a sequence")
        local = self.down(self.norm(hidden))
        if self.stateful:
            assert self.state_gate is not None
            assert self.state_scale is not None
            state = torch.zeros(
                hidden.shape[0],
                self.rank,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            values = []
            for index in range(hidden.shape[1]):
                state = self.state_gate(local[:, index], state)
                values.append(state * torch.sigmoid(self.state_scale(state)))
            local = torch.stack(values, dim=1)
        residual = self.up(F.silu(local))
        return self.gate.to(hidden.dtype) * residual


class UnifiedPlanner(nn.Module):
    """One context router for depth bias, lanes, gap cells, and generated E."""

    def __init__(self, config: UnifiedCompetenceConfig) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size)
        self.latent = nn.Sequential(
            nn.Linear(config.hidden_size, config.planner_dim),
            nn.SiLU(),
        )
        self.head_adapters = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(config.planner_dim, config.planner_dim),
                    nn.SiLU(),
                )
                for name in (
                    "competence",
                    "depth",
                    "lane",
                    "birth",
                    "mode",
                    "budget",
                )
            }
        )
        self.competence = nn.Linear(
            config.planner_dim,
            len(config.competence_names),
        )
        self.depth = nn.Linear(config.planner_dim, 1)
        self.lane = nn.Linear(config.planner_dim, config.parallel_lanes)
        self.birth = nn.Linear(config.planner_dim, 1)
        self.mode = nn.Linear(config.planner_dim, 4)
        self.budget = nn.Linear(config.planner_dim, 1)

    def forward(self, hidden: torch.Tensor) -> UnifiedPlannerOutput:
        state = self.latent(self.norm(hidden))
        adapted = {
            name: adapter(state)
            for name, adapter in self.head_adapters.items()
        }
        return UnifiedPlannerOutput(
            competence_logits=self.competence(adapted["competence"]),
            depth_logits=self.depth(adapted["depth"]).squeeze(-1),
            lane_logits=self.lane(adapted["lane"]),
            birth_logits=self.birth(adapted["birth"]).squeeze(-1),
            mode_logits=self.mode(adapted["mode"]),
            budget_logits=self.budget(adapted["budget"]).squeeze(-1),
        )


class UnifiedCompetenceBank(nn.Module):
    """Shared cell bank plus one context-generated E slot."""

    def __init__(self, config: UnifiedCompetenceConfig) -> None:
        super().__init__()
        self.config = config
        self.planner = UnifiedPlanner(config)
        self.cells = nn.ModuleDict(
            {
                name: CompetenceCell(
                    config.hidden_size,
                    config.cell_rank,
                    stateful=False,
                )
                for name in config.competence_names
            }
        )
        self.generated_expert = (
            ContextConditionedExpertFactory(
                config.hidden_size,
                max_rank=config.generated_rank,
                basis_count=config.generated_basis_count,
                context_dim=config.planner_dim,
            )
            if config.generated_expert_mode == "online"
            else None
        )
        self.aggregate_scale = nn.Parameter(torch.zeros(1))
        self.aggregator = GapCrossAttentionAggregator(
            config.hidden_size,
            config.planner_dim,
            heads=4,
        )

    def depth_bias(self, hidden: torch.Tensor) -> torch.Tensor:
        planner = self.planner(hidden)
        return torch.tanh(planner.depth_logits) * torch.sigmoid(
            planner.budget_logits
        )

    def lane_bias(self, hidden: torch.Tensor) -> torch.Tensor:
        planner = self.planner(hidden)
        return planner.lane_logits * torch.sigmoid(
            planner.budget_logits
        )[..., None]

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> UnifiedCompetenceOutput:
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden must be [batch, sequence, hidden_size]")
        planner = self.planner(hidden)
        budget = torch.sigmoid(planner.budget_logits)
        mask = (
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            if attention_mask is None
            else attention_mask.to(device=hidden.device, dtype=torch.bool)
        )
        if mask.shape != hidden.shape[:2]:
            raise ValueError("attention_mask must match hidden")
        weights = torch.softmax(planner.competence_logits, dim=-1)
        top_values, top_indices = torch.topk(
            weights,
            self.config.top_k,
            dim=-1,
        )
        sparse = torch.zeros_like(weights).scatter(
            -1,
            top_indices,
            top_values,
        )
        residual = torch.zeros_like(hidden)
        competence_states = {
            name: torch.zeros_like(hidden)
            for name in self.cells
        }
        selected_cell_counts = torch.zeros(
            len(self.cells),
            dtype=torch.long,
            device=hidden.device,
        )
        competence_token_calls = 0
        for index, (name, cell) in enumerate(self.cells.items()):
            selected = (top_indices == index).any(dim=-1) & mask
            selected_cell_counts[index] = selected.sum()
            if not selected.any():
                continue
            token_indices = selected.nonzero(as_tuple=False)
            flat_indices = token_indices[:, 0] * hidden.shape[1] + token_indices[:, 1]
            selected_hidden = hidden[selected]
            cell_output = cell(selected_hidden)
            competence_token_calls += selected_hidden.shape[0]
            gate = (top_values * (top_indices == index)).sum(dim=-1)[selected]
            residual_flat = residual.reshape(-1, hidden.shape[-1])
            residual_flat = torch.index_add(
                residual_flat,
                0,
                flat_indices,
                cell_output * gate[:, None],
            )
            residual = residual_flat.reshape_as(hidden)
            state_flat = competence_states[name].reshape(-1, hidden.shape[-1])
            state_flat = torch.index_add(
                state_flat,
                0,
                flat_indices,
                cell_output,
            )
            competence_states[name] = state_flat.reshape_as(hidden)

        pooled_weights = mask.to(hidden.dtype)[..., None]
        pooled = (hidden * pooled_weights).sum(dim=1) / pooled_weights.sum(
            dim=1
        ).clamp_min(1.0)
        birth_probability = torch.sigmoid(planner.birth_logits)
        if self.config.generated_expert_mode == "online":
            if self.generated_expert is None:
                raise RuntimeError("online generated-expert mode has no factory")
            generated = self.generated_expert(hidden, pooled)
            generated_output = generated.output
            generated_gate = generated.blueprint.birth_gate
            generated_rank = generated.blueprint.rank
            generated_width = generated.blueprint.width
        else:
            generated_output = torch.zeros_like(hidden)
            generated_gate = hidden.new_zeros(())
            generated_rank = 0
            generated_width = 0
        residual = residual + generated_output * birth_probability[..., None]
        cross_residual, cross_weights = self.aggregator(
            hidden,
            [
                competence_states[name]
                for name in self.cells
            ],
        )
        residual = (residual + cross_residual) * budget[..., None]
        updated = hidden + residual
        usage = sparse.mean(dim=(0, 1))
        balance = usage.square().sum() * len(self.cells) - 1.0
        lane_usage = torch.softmax(planner.lane_logits, dim=-1).mean(
            dim=(0, 1)
        )
        diagnostics = {
            "route_mode": "unified",
            "inference_generated_weights": torch.tensor(
                1.0,
                device=hidden.device,
            ),
            "competence_balance_loss": balance,
            "competence_token_calls": torch.tensor(
                competence_token_calls,
                device=hidden.device,
            ),
            "dense_competence_token_calls": torch.tensor(
                hidden.shape[0] * hidden.shape[1] * len(self.cells),
                device=hidden.device,
            ),
            "selected_cell_counts": selected_cell_counts,
            "competence_entropy": -(
                usage.clamp_min(1.0e-8).log() * usage
            ).sum(),
            "mean_birth_probability": birth_probability[mask].mean(),
            "mean_compute_budget": budget[mask].mean(),
            "generated_expert_mode": self.config.generated_expert_mode,
            "generated_expert_gate": generated_gate,
            "generated_rank": hidden.new_tensor(float(generated_rank)),
            "generated_width": hidden.new_tensor(float(generated_width)),
            "generated_weight_elements": hidden.new_tensor(
                float(
                    2
                    * hidden.shape[0]
                    * generated_rank
                    * self.config.hidden_size
                )
            ),
            "generated_basis_resident_parameters": hidden.new_tensor(
                float(
                    (
                        self.generated_expert.left_basis.numel()
                        + self.generated_expert.right_basis.numel()
                    )
                    if self.generated_expert is not None
                    else 0
                )
            ),
            "generated_apply_flops_proxy": hidden.new_tensor(
                float(
                    2
                    * hidden.shape[0]
                    * hidden.shape[1]
                    * generated_rank
                    * self.config.hidden_size
                )
            ),
            "cross_attention_residual_norm": cross_residual.square().mean(),
            "cross_attention_entropy": -(
                cross_weights.mean(dim=(0, 1)).clamp_min(1.0e-8).log()
                * cross_weights.mean(dim=(0, 1))
            ).sum(),
            "mean_depth_bias": planner.depth_logits[mask].mean(),
            "lane_entropy": -(
                lane_usage.clamp_min(1.0e-8).log() * lane_usage
            ).sum(),
            "generated_expert_residual_norm": generated_output.square().mean(),
        }
        return UnifiedCompetenceOutput(
            hidden_states=updated,
            planner=planner,
            competence_states=competence_states,
            diagnostics=diagnostics,
        )

    def materialize_generated_expert(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MaterializedExpert:
        if self.generated_expert is None:
            raise RuntimeError("generated expert is disabled")
        mask = (
            torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
            if attention_mask is None
            else attention_mask.to(device=hidden.device, dtype=torch.bool)
        )
        weights = mask.to(hidden.dtype)[..., None]
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.generated_expert.materialize(pooled)

    def apply_materialized_generated_expert(
        self,
        hidden: torch.Tensor,
        materialized: MaterializedExpert,
    ) -> torch.Tensor:
        if self.generated_expert is None:
            raise RuntimeError("generated expert is disabled")
        return self.generated_expert.apply_materialized(hidden, materialized)

    @torch.no_grad()
    def freeze_generated_expert(self) -> None:
        if self.generated_expert is None:
            return
        for parameter in self.generated_expert.parameters():
            parameter.requires_grad_(False)


__all__ = [
    "CompetenceCell",
    "UnifiedCompetenceBank",
    "UnifiedCompetenceConfig",
    "UnifiedCompetenceOutput",
    "UnifiedPlanner",
    "UnifiedPlannerOutput",
]
