"""Canonical experiment specifications for nlp_flow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS = ("data", "model", "optimizer", "runtime", "evaluation")
VALID_ARMS = frozenset({"control", "candidate"})
VALID_TIERS = frozenset({"P2M", "P5M", "P12M", "P23M", "P46M", "OPEN"})


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class ExperimentSpec:
    """Validated run spec with a stable content hash."""

    payload: Mapping[str, Any]
    path: Path | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, path: Path | None = None) -> ExperimentSpec:
        spec = cls(payload=dict(payload), path=path)
        errors = spec.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return spec

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentSpec:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("experiment spec must be a JSON object")
        return cls.from_mapping(payload, path=source)

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    @property
    def arm(self) -> str:
        return str(self.payload["arm"])

    @property
    def tier(self) -> str:
        return str(self.payload["tier"])

    @property
    def seed(self) -> int:
        return int(self.payload["seed"])

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.payload).encode("utf-8")).hexdigest()

    def validate(self) -> tuple[str, ...]:
        payload = self.payload
        errors: list[str] = []
        if not str(payload.get("version", "")).startswith("nlp-"):
            errors.append("version must identify an nlp_flow experiment")
        if not payload.get("name"):
            errors.append("name is required")
        if payload.get("arm") not in VALID_ARMS:
            errors.append(f"arm must be one of {sorted(VALID_ARMS)}")
        if payload.get("tier") not in VALID_TIERS:
            errors.append(f"tier must be one of {sorted(VALID_TIERS)}")
        seed = payload.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            errors.append("seed must be a non-negative integer")
        for section in REQUIRED_SECTIONS:
            if not isinstance(payload.get(section), Mapping):
                errors.append(f"{section} must be an object")
        data = payload.get("data", {})
        if isinstance(data, Mapping):
            if data.get("target_used_at_inference") is not False:
                errors.append("data.target_used_at_inference must be false")
            for split in ("train", "validation", "test"):
                value = data.get(split)
                valid_string = isinstance(value, str) and bool(value)
                valid_shards = (
                    isinstance(value, list)
                    and bool(value)
                    and all(isinstance(item, str) and item for item in value)
                )
                if not valid_string and not valid_shards:
                    errors.append(f"data.{split} path is required")
        model = payload.get("model", {})
        if isinstance(model, Mapping):
            open_capability_track = (
                payload.get("track") == "open_capability_exploratory"
                or model.get("parameter_budget_mode")
                == "open_capability_exploratory"
            )
            parameters = model.get("target_parameters")
            if not isinstance(parameters, int) or isinstance(parameters, bool) or parameters <= 0:
                errors.append("model.target_parameters must be positive")
            elif parameters > 23_000_000 and not open_capability_track:
                errors.append("primary experiment target exceeds the 23M claim cap")
            growth_flag = model.get("net_parameter_growth_allowed")
            if open_capability_track:
                if growth_flag not in {False, "exploratory_only"}:
                    errors.append(
                        "open-capability growth flag must be false or exploratory_only"
                    )
            elif growth_flag is not False:
                errors.append("model.net_parameter_growth_allowed must be false")
            corpus_tokens = model.get("planned_corpus_tokens")
            if not isinstance(corpus_tokens, int) or isinstance(corpus_tokens, bool):
                errors.append("model.planned_corpus_tokens must be an integer")
            elif not open_capability_track and parameters != corpus_tokens // 20:
                errors.append("model target parameters must equal planned corpus tokens / 20")
        runtime = payload.get("runtime", {})
        if isinstance(runtime, Mapping):
            if runtime.get("precision") not in {"fp32", "fp16", "bf16"}:
                errors.append("runtime.precision must be fp32, fp16, or bf16")
            if not isinstance(runtime.get("deterministic"), bool):
                errors.append("runtime.deterministic must be bool")
        evaluation = payload.get("evaluation", {})
        if (
            isinstance(evaluation, Mapping)
            and evaluation.get("best_of_n_in_primary") is not False
        ):
            errors.append("best-of-N is forbidden in the primary endpoint")
        return tuple(errors)

    def manifest(self) -> dict[str, Any]:
        return {
            "version": "nlp-experiment-spec-v1",
            "name": self.name,
            "arm": self.arm,
            "tier": self.tier,
            "seed": self.seed,
            "spec_sha256": self.sha256,
            "path": None if self.path is None else str(self.path),
        }

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["REQUIRED_SECTIONS", "VALID_ARMS", "VALID_TIERS", "ExperimentSpec", "canonical_json"]
