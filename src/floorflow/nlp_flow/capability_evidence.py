"""Evidence-origin guard separating internal recovery from external capabilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvidenceOrigin(StrEnum):
    IMPLEMENTATION = "implementation"
    FIXTURE = "fixture"
    TEACHER_REPLAY = "teacher_replay"
    ORACLE = "oracle"
    INTERNAL_RECOVERY = "internal_recovery"
    USER_TASK_VERIFIED = "user_task_verified"


class ClaimScope(StrEnum):
    IMPLEMENTATION = "implementation"
    INTERNAL_RECONSTRUCTION = "internal_reconstruction"
    EXTERNAL_TASK_SUCCESS = "external_task_success"
    SOTA = "sota"


@dataclass(frozen=True)
class CapabilityEvidence:
    origin: EvidenceOrigin
    scope: ClaimScope
    split: str
    case_count: int
    evidence_sha256: str
    verifier_sha256: str | None
    independent_audit_sha256: str | None = None
    baseline_audit_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test"}:
            raise ValueError("capability evidence must identify validation or test split")
        if self.case_count < 0:
            raise ValueError("case_count cannot be negative")
        _require_digest(self.evidence_sha256, "evidence_sha256")
        if self.verifier_sha256 is not None:
            _require_digest(self.verifier_sha256, "verifier_sha256")
        for name in ("independent_audit_sha256", "baseline_audit_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_digest(value, name)

    @property
    def claimable(self) -> bool:
        if self.scope in {ClaimScope.IMPLEMENTATION, ClaimScope.INTERNAL_RECONSTRUCTION}:
            return self.origin in {
                EvidenceOrigin.IMPLEMENTATION,
                EvidenceOrigin.INTERNAL_RECOVERY,
            }
        if self.origin != EvidenceOrigin.USER_TASK_VERIFIED or not self.verifier_sha256:
            return False
        if self.scope == ClaimScope.SOTA:
            return bool(self.independent_audit_sha256 and self.baseline_audit_sha256)
        return self.split == "test" and bool(self.independent_audit_sha256)

    def require_claimable(self) -> CapabilityEvidence:
        if not self.claimable:
            raise ValueError(f"{self.origin.value} evidence cannot support {self.scope.value}")
        return self


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def evidence_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = [
    "CapabilityEvidence",
    "ClaimScope",
    "EvidenceOrigin",
    "evidence_digest",
]
