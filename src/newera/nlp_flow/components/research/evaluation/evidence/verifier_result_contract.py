"""Case- and hash-bound outcomes for patch and engineering verification."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class VerificationStatus(StrEnum):
    PATCH_APPLIED = "patch_applied"
    PATCH_REJECTED = "patch_rejected"
    COMPILE_PASSED = "compile_passed"
    TESTS_PASSED = "tests_passed"
    NUMERICAL_PASSED = "numerical_passed"
    PERFORMANCE_PASSED = "performance_passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CRASHED = "crashed"
    WRONG_OUTPUT = "wrong_output"
    ENVIRONMENT_MISSING = "environment_missing"
    UNSUPPORTED = "unsupported"


SUCCESS_STATUSES = frozenset(
    {
        VerificationStatus.PATCH_APPLIED,
        VerificationStatus.COMPILE_PASSED,
        VerificationStatus.TESTS_PASSED,
        VerificationStatus.NUMERICAL_PASSED,
        VerificationStatus.PERFORMANCE_PASSED,
    }
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_bytes(encoded.encode("utf-8"))


@dataclass(frozen=True)
class VerificationIdentity:
    case_id: str
    case_sha256: str
    candidate_sha256: str
    verifier_sha256: str
    environment_sha256: str

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        for name in ("case_sha256", "candidate_sha256", "verifier_sha256", "environment_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class VerificationEvidence:
    identity: VerificationIdentity
    stage: str
    status: VerificationStatus
    exit_code: int | None = None
    timed_out: bool = False
    environment_available: bool = True
    expected_output_sha256: str | None = None
    observed_output_sha256: str | None = None
    measurement: float | None = None
    limit: float | None = None

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("verification stage must be non-empty")
        if self.timed_out and self.status != VerificationStatus.TIMED_OUT:
            raise ValueError("timeout cannot be represented as a success or generic failure")
        if not self.environment_available and self.status != VerificationStatus.ENVIRONMENT_MISSING:
            raise ValueError("missing environment cannot be represented as a pass")
        if self.status == VerificationStatus.TIMED_OUT and not self.timed_out:
            raise ValueError("timed_out status requires a timeout observation")
        if self.status == VerificationStatus.ENVIRONMENT_MISSING and self.environment_available:
            raise ValueError("environment_missing status requires an unavailable environment")
        if self.status in SUCCESS_STATUSES and self.exit_code not in (None, 0):
            raise ValueError("nonzero process exit cannot produce a successful verification status")
        if self.status == VerificationStatus.WRONG_OUTPUT:
            if not self.expected_output_sha256 or not self.observed_output_sha256:
                raise ValueError("wrong_output requires expected and observed output digests")
            if self.expected_output_sha256 == self.observed_output_sha256:
                raise ValueError("wrong_output requires different output digests")
        if (
            self.status == VerificationStatus.NUMERICAL_PASSED
            and self.expected_output_sha256 != self.observed_output_sha256
        ):
            raise ValueError("numerical success requires exact expected output digest parity")
        if self.status == VerificationStatus.PERFORMANCE_PASSED:
            if self.measurement is None or self.limit is None:
                raise ValueError("performance success requires measurement and limit")
            if not math.isfinite(self.measurement) or not math.isfinite(self.limit) or self.measurement > self.limit:
                raise ValueError("performance success exceeds its declared limit")

    @property
    def passed(self) -> bool:
        return self.status in SUCCESS_STATUSES

    @property
    def evidence_sha256(self) -> str:
        return canonical_sha256(
            {
                "identity": self.identity.__dict__,
                "stage": self.stage,
                "status": self.status.value,
                "exit_code": self.exit_code,
                "timed_out": self.timed_out,
                "environment_available": self.environment_available,
                "expected_output_sha256": self.expected_output_sha256,
                "observed_output_sha256": self.observed_output_sha256,
                "measurement": self.measurement,
                "limit": self.limit,
            }
        )

    def require_current(self, expected: VerificationIdentity) -> VerificationEvidence:
        if self.identity != expected:
            raise ValueError("verification evidence is stale or belongs to another case/candidate/verifier/environment")
        return self


@dataclass(frozen=True)
class PatchApplicationReceipt:
    identity: VerificationIdentity
    base_revision: str
    status: VerificationStatus
    patch_sha256: str
    applied_tree_sha256: str | None

    def __post_init__(self) -> None:
        if not self.base_revision.strip():
            raise ValueError("patch base revision must be explicit")
        if len(self.patch_sha256) != 64:
            raise ValueError("patch digest must be SHA-256")
        if self.status == VerificationStatus.PATCH_APPLIED and not self.applied_tree_sha256:
            raise ValueError("applied patch requires the resulting tree digest")
        if self.status != VerificationStatus.PATCH_APPLIED and self.applied_tree_sha256 is not None:
            raise ValueError("rejected patch cannot carry an applied tree digest")
        if self.status not in {VerificationStatus.PATCH_APPLIED, VerificationStatus.PATCH_REJECTED}:
            raise ValueError("patch receipt requires an application outcome")

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PATCH_APPLIED


def classify_command_result(
    identity: VerificationIdentity,
    *,
    stage: str,
    returncode: int,
    timed_out: bool,
    environment_available: bool,
    output: bytes = b"",
) -> VerificationEvidence:
    if timed_out:
        status = VerificationStatus.TIMED_OUT
    elif not environment_available:
        status = VerificationStatus.ENVIRONMENT_MISSING
    elif returncode < 0:
        status = VerificationStatus.CRASHED
    elif returncode != 0:
        status = VerificationStatus.FAILED
    else:
        status = {
            "compile": VerificationStatus.COMPILE_PASSED,
            "tests": VerificationStatus.TESTS_PASSED,
            "patch": VerificationStatus.PATCH_APPLIED,
        }.get(stage, VerificationStatus.UNSUPPORTED)
    return VerificationEvidence(
        identity,
        stage,
        status,
        exit_code=returncode,
        timed_out=timed_out,
        environment_available=environment_available,
        observed_output_sha256=sha256_bytes(output),
    )


__all__ = [
    "SUCCESS_STATUSES",
    "PatchApplicationReceipt",
    "VerificationEvidence",
    "VerificationIdentity",
    "VerificationStatus",
    "canonical_sha256",
    "classify_command_result",
    "sha256_bytes",
]


RUNNER_STAGES = ("patch", "compile", "tests", "numerical", "performance", "triton")


def preflight_runner_manifest(manifest: dict[str, Any], *, gpu_gate_passed: bool = False) -> dict[str, Any]:
    """Validate ordering and resume identities without starting any executor."""
    errors: list[str] = []
    rows = manifest.get("cases")
    if manifest.get("schema_version") != "nlp-engineering-runner-v1":
        errors.append("unsupported runner manifest schema_version")
    if not isinstance(rows, list) or not rows:
        errors.append("runner manifest needs a non-empty cases list")
        rows = []
    seen: set[str] = set()
    gpu_cases = 0
    for index, row in enumerate(rows):
        prefix = f"cases[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} must be an object")
            continue
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{prefix}.case_id must be non-empty")
        elif case_id in seen:
            errors.append(f"duplicate case_id: {case_id}")
        else:
            seen.add(case_id)
        try:
            identity = VerificationIdentity(
                case_id=case_id if isinstance(case_id, str) else "invalid",
                case_sha256=row["case_sha256"],
                candidate_sha256=row["candidate_sha256"],
                verifier_sha256=row["verifier_sha256"],
                environment_sha256=row["environment_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{prefix} has invalid run identity: {error}")
            identity = None
        stages = row.get("stages")
        if not isinstance(stages, list) or not stages or any(stage not in RUNNER_STAGES for stage in stages):
            errors.append(f"{prefix}.stages contains an unknown or missing stage")
            stages = []
        elif stages != sorted(stages, key=RUNNER_STAGES.index) or len(set(stages)) != len(stages):
            errors.append(f"{prefix}.stages must be unique and ordered")
        if "performance" in stages and "numerical" not in stages:
            errors.append(f"{prefix} performance stage requires numerical verification first")
        if "triton" in stages and "numerical" not in stages:
            errors.append(f"{prefix} Triton stage requires numerical verification first")
        environment = row.get("environment_kind")
        if environment not in {"cpu", "gpu"}:
            errors.append(f"{prefix}.environment_kind must be cpu or gpu")
        elif environment == "gpu":
            gpu_cases += 1
        resume = row.get("resume_identity")
        if resume is not None and identity is not None and resume != identity.__dict__:
            errors.append(f"{prefix} resume identity is stale")
    if errors:
        status = "invalid"
    elif gpu_cases and not gpu_gate_passed:
        status = "blocked_gpu_gate"
    else:
        status = "preflight_passed_no_execution"
    return {
        "schema_version": "nlp-engineering-preflight-v1",
        "status": status,
        "launch_performed": False,
        "gpu_cases": gpu_cases,
        "errors": errors,
        "cases_checked": len(rows),
    }
