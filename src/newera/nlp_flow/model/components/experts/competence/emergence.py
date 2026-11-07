"""Compositional emergence objective over sparse semantic basis vectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class EmergenceConfig:
    hidden_size: int
    basis_count: int = 32
    active_basis: int = 4
    novelty_margin: float = 0.15
    reconstruction_weight: float = 10.0
    sparsity_weight: float = 0.01
    diversity_weight: float = 0.05
    composition_weight: float = 1.0
    novelty_weight: float = 0.05
    balance_weight: float = 0.01
    validity_weight: float = 5.0
    support_capacity: int = 256
    reward_weight: float = 0.1
    norm_envelope_ratio: float = 1.5

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "basis_count",
            "active_basis",
            "support_capacity",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.active_basis > self.basis_count:
            raise ValueError("active_basis cannot exceed basis_count")
        if self.norm_envelope_ratio <= 0.0:
            raise ValueError("norm_envelope_ratio must be positive")
        for name in (
            "novelty_margin",
            "reconstruction_weight",
            "sparsity_weight",
            "diversity_weight",
            "composition_weight",
            "novelty_weight",
            "balance_weight",
            "validity_weight",
            "support_capacity",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class EmergenceLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    sparsity: torch.Tensor
    diversity: torch.Tensor
    composition_cycle: torch.Tensor
    novelty: torch.Tensor
    balance: torch.Tensor
    validity: torch.Tensor
    novel_distance: torch.Tensor
    support_recall: torch.Tensor
    surface_positive_rate: torch.Tensor
    execution_coverage: torch.Tensor
    novelty_component: torch.Tensor
    reconstruction_gate: torch.Tensor
    cycle_gate: torch.Tensor
    support_gate: torch.Tensor
    validity_gate: torch.Tensor
    norm_gate: torch.Tensor
    emergence_score: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "reconstruction": self.reconstruction,
            "sparsity": self.sparsity,
            "diversity": self.diversity,
            "composition_cycle": self.composition_cycle,
            "novelty": self.novelty,
            "balance": self.balance,
            "validity": self.validity,
            "novel_distance": self.novel_distance,
            "support_recall": self.support_recall,
            "surface_positive_rate": self.surface_positive_rate,
            "execution_coverage": self.execution_coverage,
            "novelty_component": self.novelty_component,
            "reconstruction_gate": self.reconstruction_gate,
            "cycle_gate": self.cycle_gate,
            "support_gate": self.support_gate,
            "validity_gate": self.validity_gate,
            "norm_gate": self.norm_gate,
            "emergence_score": self.emergence_score,
        }


def _topk_unit(coefficients: torch.Tensor, active: int) -> torch.Tensor:
    values, indices = torch.topk(coefficients, active, dim=-1)
    sparse = torch.zeros_like(coefficients)
    sparse.scatter_(-1, indices, values)
    return F.normalize(sparse, dim=-1)


class EmergentSemanticObjective(nn.Module):
    """Learn sparse basis atoms and a composition operator without target samples."""

    def __init__(self, config: EmergenceConfig) -> None:
        super().__init__()
        if config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.config = config
        self.encoder = nn.Linear(config.hidden_size, config.basis_count)
        self.composer = nn.Bilinear(
            config.basis_count,
            config.basis_count,
            config.basis_count,
            bias=False,
        )
        self.decoder = nn.Linear(config.basis_count, config.hidden_size, bias=False)
        self.validity_head = nn.Linear(config.hidden_size, 2)
        self.register_buffer(
            "support_vectors",
            torch.zeros(
                config.support_capacity,
                config.hidden_size,
            ),
            persistent=False,
        )
        self.register_buffer(
            "support_valid",
            torch.zeros(config.support_capacity, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "support_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )

    @torch.no_grad()
    def _update_support(self, hidden: torch.Tensor) -> None:
        values = F.normalize(hidden.detach().float(), dim=-1)
        available = int((~self.support_valid).sum())
        start = int(self.support_count.item())
        if available >= values.shape[0] and start < self.support_vectors.shape[0]:
            count = min(values.shape[0], self.support_vectors.shape[0] - start)
            self.support_vectors[start : start + count] = values[:count]
            self.support_valid[start : start + count] = True
            self.support_count.fill_(start + count)
            return
        compact = self.support_vectors[self.support_valid]
        if compact.numel() == 0:
            count = min(self.support_vectors.shape[0], values.shape[0])
            self.support_vectors[:count] = values[-count:]
            self.support_valid[:count] = True
            self.support_count.fill_(count)
            return
        self.support_vectors.zero_()
        self.support_valid.zero_()
        count = min(
            self.support_vectors.shape[0],
            compact.shape[0] + values.shape[0],
        )
        self.support_vectors[:count] = torch.cat((compact, values), dim=0)[-count:]
        self.support_valid[:count] = True
        self.support_count.fill_(count)

    def _support_distance(self, composed_state: torch.Tensor) -> torch.Tensor:
        seen = self.support_vectors[self.support_valid]
        if seen.numel() == 0:
            return composed_state.new_zeros(composed_state.shape[0])
        novel = F.normalize(composed_state.float(), dim=-1)
        maximum_similarity = (
            novel @ F.normalize(seen, dim=-1).transpose(0, 1)
        ).amax(dim=-1)
        return (1.0 - maximum_similarity).to(composed_state.dtype)

    def compose(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError("basis coefficient tensors must match")
        return _topk_unit(
            F.relu(first + second + self.composer(first, second)),
            self.config.active_basis,
        )

    def _basis_diversity(self) -> torch.Tensor:
        basis = self.decoder.weight.transpose(0, 1)
        basis = F.normalize(basis, dim=-1)
        gram = basis @ basis.transpose(0, 1)
        off_diagonal = gram - torch.diag_embed(torch.diagonal(gram))
        return off_diagonal.square().mean()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        surface_targets: torch.Tensor | None = None,
        execution_targets: torch.Tensor | None = None,
    ) -> EmergenceLoss:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [batch, sequence, hidden]")
        batch, sequence, _hidden = hidden_states.shape
        mask = (
            torch.ones(batch, sequence, dtype=torch.bool, device=hidden_states.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        if mask.shape != (batch, sequence):
            raise ValueError("attention_mask must match hidden states")
        active = hidden_states[mask]
        negative_active = active.float().flip(0)
        if active.shape[0] < 2:
            raise ValueError("emergence objective needs at least two active states")

        coefficients = _topk_unit(
            F.relu(self.encoder(active.float())),
            self.config.active_basis,
        )
        reconstruction = self.decoder(coefficients)
        reconstruction_loss = F.mse_loss(
            reconstruction,
            active.float(),
        )
        sparsity = coefficients.abs().mean()
        diversity = self._basis_diversity()

        positive_logits = self.validity_head(active.float())
        negative_states = negative_active
        negative_logits = self.validity_head(negative_states)
        surface_loss = active.new_zeros(())
        execution_loss = active.new_zeros(())
        surface_positive_rate = active.new_zeros(())
        execution_coverage = active.new_zeros(())
        active_surface = None
        active_execution = None
        if surface_targets is not None:
            if surface_targets.shape != mask.shape:
                raise ValueError("surface_targets must match attention_mask")
            active_surface = surface_targets.to(mask.device)[mask].float()
            surface_loss = F.binary_cross_entropy_with_logits(
                positive_logits[:, 0],
                active_surface,
            ) + F.binary_cross_entropy_with_logits(
                negative_logits[:, 0],
                torch.zeros_like(active_surface),
            )
            surface_positive_rate = active_surface.mean()
        if execution_targets is not None:
            if execution_targets.shape != mask.shape:
                raise ValueError("execution_targets must match attention_mask")
            active_execution = execution_targets.to(mask.device)[mask]
            known_execution = active_execution.ge(0)
            execution_coverage = known_execution.float().mean()
            if known_execution.any():
                execution_loss = F.binary_cross_entropy_with_logits(
                    positive_logits[known_execution, 1],
                    active_execution[known_execution].float(),
                ) + F.binary_cross_entropy_with_logits(
                    negative_logits[known_execution, 1],
                    torch.zeros(
                        int(known_execution.sum()),
                        device=active.device,
                    ),
                )

        pair_count = active.shape[0] // 2
        first = coefficients[:pair_count]
        second = coefficients[pair_count : 2 * pair_count]
        composed = self.compose(first, second)
        composed_state = self.decoder(composed)
        recovered = _topk_unit(
            F.relu(self.encoder(composed_state)),
            self.config.active_basis,
        )
        composition_cycle = F.mse_loss(
            F.normalize(recovered, dim=-1),
            F.normalize(composed, dim=-1),
        )
        composition_validity = active.new_zeros(())
        if active_surface is not None:
            first_surface = active_surface[:pair_count]
            second_surface = active_surface[pair_count : 2 * pair_count]
            composed_surface_target = (
                first_surface.gt(0.5)
                & second_surface.gt(0.5)
            )
            composed_logits = self.validity_head(composed_state.float())
            composition_validity = F.binary_cross_entropy_with_logits(
                composed_logits[:, 0],
                composed_surface_target.float(),
            )
            if active_execution is not None:
                first_execution = active_execution[:pair_count]
                second_execution = active_execution[
                    pair_count : 2 * pair_count
                ]
                known_pair = (
                    first_execution.ge(0)
                    & second_execution.ge(0)
                )
                if known_pair.any():
                    execution_pair_target = (
                        first_execution[known_pair].gt(0)
                        & second_execution[known_pair].gt(0)
                    )
                    composition_validity = composition_validity + (
                        F.binary_cross_entropy_with_logits(
                            composed_logits[known_pair, 1],
                            execution_pair_target.float(),
                        )
                    )

        novel_distance = self._support_distance(composed_state).mean()
        novelty_loss = F.relu(self.config.novelty_margin - novel_distance)
        if self.training:
            self._update_support(active)

        usage = coefficients.mean(dim=0).clamp_min(1.0e-8)
        usage = usage / usage.sum()
        balance = -(usage * usage.log()).sum() / torch.log(
            usage.new_tensor(float(self.config.basis_count))
        )

        union = (first + second).gt(0.0)
        support_recall = (
            ((composed.gt(0.0) & union).sum(dim=-1).float()
             / union.sum(dim=-1).clamp_min(1.0))
            .mean()
        )
        active_norm = active.float().norm(dim=-1)
        composed_norm = composed_state.float().norm(dim=-1)
        envelope = (
            active_norm.quantile(0.95) * self.config.norm_envelope_ratio
        ).clamp_min(1.0e-6)
        norm_excess = (composed_norm / envelope - 1.0).clamp_min(0.0)
        reconstruction_gate = torch.exp(-reconstruction_loss).clamp(0.0, 1.0)
        cycle_gate = torch.exp(-composition_cycle).clamp(0.0, 1.0)
        support_gate = support_recall.clamp(0.0, 1.0)
        validity_total = surface_loss + execution_loss + composition_validity
        validity_gate = torch.exp(-validity_total).clamp(0.0, 1.0)
        norm_gate = torch.exp(-norm_excess.square()).mean().clamp(0.0, 1.0)
        novelty_span = max(1.0e-6, 1.0 - self.config.novelty_margin)
        novelty_component = (
            (novel_distance - self.config.novelty_margin) / novelty_span
        ).clamp(0.0, 1.0)
        emergence_score = (
            novelty_component
            * reconstruction_gate
            * cycle_gate
            * support_gate
            * validity_gate
            * norm_gate
        )
        total = (
            self.config.reconstruction_weight * reconstruction_loss
            + self.config.sparsity_weight * sparsity
            + self.config.diversity_weight * diversity
            + self.config.composition_weight
            * (composition_cycle + 0.1 * (1.0 - support_recall))
            + self.config.novelty_weight * novelty_loss * cycle_gate
            + self.config.balance_weight * balance
            + self.config.validity_weight
            * validity_total
            - self.config.reward_weight * emergence_score
        )
        return EmergenceLoss(
            total=total,
            reconstruction=reconstruction_loss,
            sparsity=sparsity,
            diversity=diversity,
            composition_cycle=composition_cycle,
            novelty=novelty_loss,
            balance=balance,
            validity=validity_total,
            novel_distance=novel_distance,
            support_recall=support_recall,
            surface_positive_rate=surface_positive_rate,
            execution_coverage=execution_coverage,
            novelty_component=novelty_component,
            reconstruction_gate=reconstruction_gate,
            cycle_gate=cycle_gate,
            support_gate=support_gate,
            validity_gate=validity_gate,
            norm_gate=norm_gate,
            emergence_score=emergence_score,
        )


__all__ = [
    "EmergenceConfig",
    "EmergenceLoss",
    "EmergentSemanticObjective",
]
