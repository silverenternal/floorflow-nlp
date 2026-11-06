"""Verifier-grounded lifecycle for inference-time generated competences."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from floorflow.nlp_flow.engineering.storage import save_torch_atomic
from floorflow.nlp_flow.model.expert_factory import (
    ContextConditionedExpertFactory,
    MaterializedExpert,
)

GAP_EVIDENCE_FIELDS = (
    "novelty",
    "router_entropy",
    "validity_margin",
    "verifier_failure",
    "repetition",
)
RuntimeStatus = Literal["candidate", "promoted", "retired"]
RUNTIME_STORE_VERSION = "nlp-runtime-competence-store-v1"


@dataclass
class RuntimeCompetenceRecord:
    gap_key: str
    signature: str
    expert: MaterializedExpert
    trigger_score: float
    parameter_elements: int
    expert_norm: float
    novelty_distance: float
    status: RuntimeStatus = "candidate"
    trials: int = 0
    successes: int = 0
    failures: int = 0
    utility: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.trials)

    @property
    def utility_per_parameter(self) -> float:
        return self.utility / max(1, self.parameter_elements)


@dataclass(frozen=True)
class RuntimeCompetenceConfig:
    hidden_size: int
    capacity: int = 8
    trigger_threshold: float = 0.5
    promotion_trials: int = 2
    promotion_success_rate: float = 0.75
    minimum_expert_norm: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.capacity <= 0:
            raise ValueError("runtime competence dimensions must be positive")
        if not 0.0 <= self.trigger_threshold <= 1.0:
            raise ValueError("trigger threshold must be in [0, 1]")
        if self.promotion_trials <= 0:
            raise ValueError("promotion trials must be positive")
        if not 0.0 <= self.promotion_success_rate <= 1.0:
            raise ValueError("promotion success rate must be in [0, 1]")
        if self.minimum_expert_norm < 0.0:
            raise ValueError("minimum expert norm must be non-negative")


class RuntimeGapDetector(nn.Module):
    """Learn when a verifier-visible gap warrants generating a new competence."""

    def __init__(self, hidden_size: int, evidence_size: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.evidence = nn.Linear(evidence_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(
        self,
        pooled_hidden: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        if pooled_hidden.ndim != 2:
            raise ValueError("pooled_hidden must be [batch, hidden]")
        if evidence.shape != (pooled_hidden.shape[0], len(GAP_EVIDENCE_FIELDS)):
            raise ValueError("evidence must match batch and evidence fields")
        features = torch.tanh(self.hidden(pooled_hidden) + self.evidence(evidence))
        return torch.sigmoid(self.score(features).squeeze(-1))


def _detached_expert(expert: MaterializedExpert) -> MaterializedExpert:
    return replace(
        expert,
        left=expert.left.detach().clone(),
        right=expert.right.detach().clone(),
        rank_mask=expert.rank_mask.detach().clone(),
        width_mask=expert.width_mask.detach().clone(),
        input_scale=expert.input_scale.detach().clone(),
        output_scale=expert.output_scale.detach().clone(),
        birth_gate=expert.birth_gate.detach().clone(),
        rank=expert.rank.detach().clone(),
        width=expert.width.detach().clone(),
    )


def _expert_signature(expert: MaterializedExpert) -> str:
    digest = hashlib.sha256()
    for value in (expert.left, expert.right, expert.rank, expert.width):
        digest.update(
            value.detach().float().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


class RuntimeCompetenceManager(nn.Module):
    """Manage synthesize -> verify -> promote/retain/retire without hidden mutation."""

    def __init__(
        self,
        factory: ContextConditionedExpertFactory,
        config: RuntimeCompetenceConfig,
    ) -> None:
        super().__init__()
        if factory.hidden_size != config.hidden_size:
            raise ValueError("factory and runtime config hidden sizes differ")
        self.config = config
        object.__setattr__(self, "_factory", factory)
        self.detector = RuntimeGapDetector(
            config.hidden_size,
            len(GAP_EVIDENCE_FIELDS),
        )
        self.records: dict[str, RuntimeCompetenceRecord] = {}

    @property
    def factory(self) -> ContextConditionedExpertFactory:
        return self._factory

    def detect(
        self,
        pooled_hidden: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        return self.detector(pooled_hidden, evidence)

    def reusable(self, gap_key: str) -> RuntimeCompetenceRecord | None:
        candidates = [
            record
            for record in self.records.values()
            if record.gap_key == gap_key and record.status == "promoted"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda record: record.utility_per_parameter)

    @torch.no_grad()
    def synthesize(
        self,
        pooled_hidden: torch.Tensor,
        context: torch.Tensor | None,
        evidence: torch.Tensor,
        *,
        gap_key: str,
        require_trigger: bool = True,
    ) -> RuntimeCompetenceRecord | None:
        if pooled_hidden.ndim != 2:
            raise ValueError("pooled_hidden must be [batch, hidden]")
        context = pooled_hidden if context is None else context
        if context.shape != pooled_hidden.shape:
            raise ValueError("context must match pooled hidden")
        reusable = self.reusable(gap_key)
        if reusable is not None:
            return reusable
        score = self.detect(pooled_hidden, evidence)
        selected = score.argmax()
        if require_trigger and score[selected] < self.config.trigger_threshold:
            return None
        expert = _detached_expert(self.factory.materialize(context[selected : selected + 1]))
        expert_norm = float(
            expert.left.square().sum() + expert.right.square().sum()
        ) ** 0.5
        if expert_norm < self.config.minimum_expert_norm:
            return None
        novelty_distance = self._novelty_distance(expert)
        parameter_elements = sum(
            value.numel()
            for value in (expert.left, expert.right, expert.input_scale, expert.output_scale)
        )
        signature = _expert_signature(expert)
        record = RuntimeCompetenceRecord(
            gap_key=str(gap_key),
            signature=signature,
            expert=expert,
            trigger_score=float(score[selected].detach()),
            parameter_elements=parameter_elements,
            expert_norm=expert_norm,
            novelty_distance=novelty_distance,
        )
        self.records[signature] = record
        self._evict_if_needed()
        return record

    def apply(
        self,
        hidden: torch.Tensor,
        record: RuntimeCompetenceRecord,
    ) -> torch.Tensor:
        if hidden.shape[0] != record.expert.left.shape[0]:
            raise ValueError("runtime expert batch does not match hidden")
        return self.factory.apply_materialized(hidden, record.expert)

    def observe(
        self,
        record: RuntimeCompetenceRecord,
        *,
        valid: bool,
        utility: float,
    ) -> RuntimeStatus:
        if utility < 0.0:
            raise ValueError("utility cannot be negative")
        record.trials += 1
        record.successes += int(valid)
        record.failures += int(not valid)
        record.utility += float(utility)
        if (
            record.trials >= self.config.promotion_trials
            and record.success_rate >= self.config.promotion_success_rate
            and record.successes > 0
        ):
            record.status = "promoted"
        elif (
            record.failures >= self.config.promotion_trials
            or record.trials >= self.config.promotion_trials
        ):
            record.status = "retired"
        return record.status

    def retire(self, signature: str) -> None:
        if signature in self.records:
            self.records[signature].status = "retired"

    def active_records(self) -> tuple[RuntimeCompetenceRecord, ...]:
        return tuple(
            record
            for record in self.records.values()
            if record.status != "retired"
        )

    def metrics(self) -> dict[str, int]:
        return {
            "candidate": sum(
                record.status == "candidate" for record in self.records.values()
            ),
            "promoted": sum(
                record.status == "promoted" for record in self.records.values()
            ),
            "retired": sum(
                record.status == "retired" for record in self.records.values()
            ),
        }

    def export_state(self) -> dict[str, object]:
        records = []
        for record in self.records.values():
            expert = record.expert
            records.append(
                {
                    "gap_key": record.gap_key,
                    "signature": record.signature,
                    "trigger_score": record.trigger_score,
                    "parameter_elements": record.parameter_elements,
                    "expert_norm": record.expert_norm,
                    "novelty_distance": record.novelty_distance,
                    "status": record.status,
                    "trials": record.trials,
                    "successes": record.successes,
                    "failures": record.failures,
                    "utility": record.utility,
                    "expert": {
                        "left": expert.left.detach().cpu(),
                        "right": expert.right.detach().cpu(),
                        "rank_mask": expert.rank_mask.detach().cpu(),
                        "width_mask": expert.width_mask.detach().cpu(),
                        "input_scale": expert.input_scale.detach().cpu(),
                        "output_scale": expert.output_scale.detach().cpu(),
                        "birth_gate": expert.birth_gate.detach().cpu(),
                        "rank": expert.rank.detach().cpu(),
                        "width": expert.width.detach().cpu(),
                    },
                }
            )
        return {
            "version": RUNTIME_STORE_VERSION,
            "config": {
                "hidden_size": self.config.hidden_size,
                "capacity": self.config.capacity,
                "trigger_threshold": self.config.trigger_threshold,
                "promotion_trials": self.config.promotion_trials,
                "promotion_success_rate": self.config.promotion_success_rate,
                "minimum_expert_norm": self.config.minimum_expert_norm,
            },
            "records": records,
        }

    def save(self, path: str | Path) -> None:
        save_torch_atomic(path, self.export_state())

    def load_state(self, payload: dict[str, object]) -> None:
        if payload.get("version") != RUNTIME_STORE_VERSION:
            raise ValueError("unsupported runtime competence store")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise TypeError("runtime competence records must be a list")
        device = next(self.parameters()).device
        loaded: dict[str, RuntimeCompetenceRecord] = {}
        for item in raw_records:
            if not isinstance(item, dict) or not isinstance(item.get("expert"), dict):
                raise TypeError("runtime competence record is malformed")
            tensors = {
                name: value.to(device)
                for name, value in item["expert"].items()
            }
            expert = MaterializedExpert(
                left=tensors["left"],
                right=tensors["right"],
                rank_mask=tensors["rank_mask"],
                width_mask=tensors["width_mask"],
                input_scale=tensors["input_scale"],
                output_scale=tensors["output_scale"],
                birth_gate=tensors["birth_gate"],
                rank=tensors["rank"],
                width=tensors["width"],
            )
            record = RuntimeCompetenceRecord(
                gap_key=str(item["gap_key"]),
                signature=str(item["signature"]),
                expert=expert,
                trigger_score=float(item["trigger_score"]),
                parameter_elements=int(item["parameter_elements"]),
                expert_norm=float(item["expert_norm"]),
                novelty_distance=float(item["novelty_distance"]),
                status=str(item["status"]),
                trials=int(item["trials"]),
                successes=int(item["successes"]),
                failures=int(item["failures"]),
                utility=float(item["utility"]),
            )
            loaded[record.signature] = record
        self.records = loaded

    def load(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state(payload)

    def _novelty_distance(self, expert: MaterializedExpert) -> float:
        if not self.records:
            return float(expert.left.square().sum() + expert.right.square().sum()) ** 0.5
        current = torch.cat(
            (expert.left.flatten(), expert.right.flatten())
        )
        distances = []
        for record in self.records.values():
            existing = torch.cat(
                (record.expert.left.flatten(), record.expert.right.flatten())
            )
            distances.append(float((current - existing).square().sum()) ** 0.5)
        return min(distances)

    def _evict_if_needed(self) -> None:
        if len(self.records) <= self.config.capacity:
            return
        candidate = min(
            self.records.values(),
            key=lambda record: (record.status != "candidate", record.utility),
        )
        self.records.pop(candidate.signature, None)


__all__ = [
    "GAP_EVIDENCE_FIELDS",
    "RUNTIME_STORE_VERSION",
    "RuntimeCompetenceConfig",
    "RuntimeCompetenceManager",
    "RuntimeCompetenceRecord",
    "RuntimeGapDetector",
    "RuntimeStatus",
]
