"""Strict identity joins for model, corpus, split, task, case and verifier labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

JOIN_FIELDS = (
    "model_hash",
    "config_hash",
    "corpus_hash",
    "split",
    "task_id",
    "case_id",
    "seed",
    "parser_hash",
    "evaluator_hash",
)
HASH_FIELDS = (
    "model_hash",
    "config_hash",
    "corpus_hash",
    "parser_hash",
    "evaluator_hash",
)
ALLOWED_SPLITS = frozenset({"train", "validation", "test", "fixture"})


def _digest(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be lowercase SHA-256")


@dataclass(frozen=True)
class EvaluationIdentity:
    model_hash: str
    config_hash: str
    corpus_hash: str
    split: str
    task_id: str
    case_id: str
    seed: int
    parser_hash: str
    evaluator_hash: str

    def __post_init__(self) -> None:
        for name in HASH_FIELDS:
            _digest(getattr(self, name), name)
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"unsupported evaluation split: {self.split}")
        if not self.task_id or not self.case_id:
            raise ValueError("task_id and case_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> EvaluationIdentity:
        missing = set(JOIN_FIELDS) - set(row)
        if missing:
            raise ValueError(f"evaluation row is missing identity fields: {sorted(missing)}")
        return cls(**{field: row[field] for field in JOIN_FIELDS})

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in JOIN_FIELDS}


def strict_join_labels(
    predictions: Iterable[dict[str, Any]],
    labels: Iterable[dict[str, Any]],
    *,
    label_field: str,
    allowed_label_splits: frozenset[str] = frozenset({"validation"}),
) -> list[dict[str, Any]]:
    """Join one-to-one labels; reject stale, test, fixture, duplicate and absent rows."""
    label_index: dict[EvaluationIdentity, dict[str, Any]] = {}
    for row in labels:
        identity = EvaluationIdentity.from_mapping(row)
        if identity.split not in allowed_label_splits:
            raise ValueError(f"label split is not selectable for proxy alignment: {identity.split}")
        if row.get("provenance") not in {"verifier", "external_metric"}:
            raise ValueError("label provenance must name a verified runner or external metric")
        if row.get("stale") is True or not row.get("evidence_sha256"):
            raise ValueError("stale or unproven label cannot be joined")
        if identity in label_index:
            raise ValueError("duplicate label identity")
        if label_field not in row:
            raise ValueError(f"label row is missing {label_field}")
        label_index[identity] = row

    output = []
    seen: set[EvaluationIdentity] = set()
    for prediction in predictions:
        identity = EvaluationIdentity.from_mapping(prediction)
        if identity in seen:
            raise ValueError("duplicate prediction identity")
        seen.add(identity)
        label = label_index.get(identity)
        if label is None:
            raise ValueError(f"missing exact identity label for {identity.case_id}")
        output.append(
            {
                **identity.as_dict(),
                "prediction": prediction.get("prediction"),
                "label": label[label_field],
                "label_provenance": label["provenance"],
                "evidence_sha256": label["evidence_sha256"],
                "identity_sha256": identity.digest,
            }
        )
    if seen != set(label_index):
        raise ValueError("label rows contain unmatched identities")
    return output


__all__ = ["JOIN_FIELDS", "EvaluationIdentity", "strict_join_labels"]
