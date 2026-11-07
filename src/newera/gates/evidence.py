"""Validate declared evidence presence; this cannot verify human judgment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

RESEARCH_FIELDS = ("run_id", "config", "data_version", "data_fingerprint", "code_revision", "environment", "started_at", "seed", "budget", "stop_conditions", "evaluation_protocol")
RELEASE_FIELDS = ("model_card", "evaluation_report", "security_review", "fairness_review", "legal_review")
ARTIFACT_FIELDS = ("config", "evaluation_protocol", *RELEASE_FIELDS)


def check_evidence(root: Path, stage: str, evidence_path: Path | None) -> list[dict[str, str]]:
    if stage == "explore":
        return []
    findings = []
    payload: object = {}
    if evidence_path is None:
        findings.append({"level": "warning", "check": "evidence", "message": "no evidence manifest supplied; stage is not qualified"})
    else:
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("manifest must be a JSON object")
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            findings.append({"level": "warning", "check": "evidence", "message": f"cannot read evidence manifest: {exc}"})
            payload = {}
    required = RESEARCH_FIELDS + (RELEASE_FIELDS if stage == "release" else ())
    assert isinstance(payload, dict)
    for field in required:
        value = payload.get(field)
        valid = (isinstance(value, int) and not isinstance(value, bool)) if field == "seed" else isinstance(value, str) and bool(value.strip())
        if not valid:
            findings.append({"level": "warning", "check": "evidence", "message": f"missing or invalid field: {field}"})
        elif field in ARTIFACT_FIELDS:
            artifact = Path(value)
            artifact = artifact if artifact.is_absolute() else root / artifact
            if not artifact.is_file():
                findings.append({"level": "warning", "check": "evidence", "message": f"{field} is not an existing file: {value}"})
                continue
            expected = payload.get(f"{field}_sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                findings.append({"level": "warning", "check": "integrity", "message": f"{field}_sha256 is missing or invalid"})
                continue
            digest = hashlib.sha256()
            try:
                with artifact.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError as exc:
                findings.append({"level": "warning", "check": "integrity", "message": f"cannot read {field}: {exc}"})
            else:
                if digest.hexdigest() != expected.lower():
                    findings.append({"level": "warning", "check": "integrity", "message": f"{field} SHA-256 mismatch"})
    findings.append({"level": "warning", "check": "evidence", "message": "presence checks do not verify evidence contents, policy compliance, or authorize a run/release"})
    return findings
