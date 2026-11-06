#!/usr/bin/env python3
"""Build the final evidence package checklist and freeze available hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes() -> dict[str, str]:
    artifacts = (
        "nlptodo.json",
        "configs/nlp_architecture_v1.json",
        "configs/nlp_corpus_plan_v1.json",
        "configs/nlp_benchmark_manifest_v1.json",
        "configs/nlp_benchmark_runners_v1.json",
        "configs/nlp_executor_contracts_v1.json",
        "schema/nlp_json_output_v1.schema.json",
        "configs/nlp_json_schemas_v1.json",
        ".data/nlp/tokenizer_v1/tokenizer.json",
        ".data/nlp/p5m_v2/manifest.json",
        ".data/nlp/p12m_v2/manifest.json",
        "outputs/reports/nlp_tokenizer_budget_v1.json",
        "outputs/reports/nlp_tokenizer_budget_p12m_v1.json",
        "outputs/reports/nlp_json_claim_gate_v1.json",
        "outputs/reports/nlp_ast_roundtrip_v1.json",
        "outputs/reports/nlp_ast_cfm_nfe_curve_v1.json",
        "outputs/reports/nlp_result_cfm_execution_probe_v1.json",
        "outputs/reports/nlp_p5m_screen_v1.json",
        "outputs/reports/nlp_p12m_screen_v1.json",
        "outputs/reports/nlp_scale_decision_v1.json",
        "outputs/reports/nlp_growth_matched_screen_v1.json",
        "outputs/reports/nlp_growth_frontier_v1.json",
        "outputs/reports/nlp_specialization_ablation_v1.json",
        "outputs/reports/nlp_external_eval_preflight_v1.json",
        "outputs/reports/nlp_benchmark_runner_preflight_v1.json",
    )
    return {
        path: _sha256(ROOT / path)
        for path in artifacts
        if (ROOT / path).exists()
    }


def _environment() -> dict[str, object]:
    packages = {}
    for package in ("torch", "tokenizers", "tree-sitter", "numpy"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalist-checkpoint", type=Path, default=None)
    parser.add_argument("--external-eval", type=Path, default=None)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_reproducibility_manifest_v1.json",
    )
    parser.add_argument(
        "--checklist-output",
        type=Path,
        default=ROOT / "outputs/nlp_package_v1/submission_checklist.json",
    )
    parser.add_argument(
        "--claim-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_final_claim_v1.json",
    )
    args = parser.parse_args()
    artifacts = _artifact_hashes()
    json_gate = json.loads(
        (ROOT / "outputs/reports/nlp_json_claim_gate_v1.json").read_text()
    )
    checks = {
        "json_gate_passed": json_gate["status"] == "pass",
        "architecture_contract_frozen": "configs/nlp_architecture_v1.json" in artifacts,
        "corpus_plan_frozen": "configs/nlp_corpus_plan_v1.json" in artifacts,
        "benchmark_manifest_frozen": "configs/nlp_benchmark_manifest_v1.json" in artifacts,
        "runner_manifest_frozen": "configs/nlp_benchmark_runners_v1.json" in artifacts,
        "executor_contract_frozen": "configs/nlp_executor_contracts_v1.json" in artifacts,
        "tokenizer_frozen": ".data/nlp/tokenizer_v1/tokenizer.json" in artifacts,
        "p5m_manifest_frozen": ".data/nlp/p5m_v2/manifest.json" in artifacts,
        "finalist_checkpoint_present": (
            args.finalist_checkpoint is not None
            and args.finalist_checkpoint.is_file()
        ),
        "external_eval_present": (
            args.external_eval is not None and args.external_eval.is_file()
        ),
        "three_seed_evidence_present": False,
        "independent_reproduction_present": False,
    }
    manifest = {
        "version": "nlp-reproducibility-manifest-v1",
        "status": "ready" if all(checks.values()) else "blocked",
        "artifact_hashes": artifacts,
        "environment": _environment(),
        "checks": checks,
        "finalist_checkpoint": (
            None
            if args.finalist_checkpoint is None
            else str(args.finalist_checkpoint)
        ),
        "external_eval": (
            None if args.external_eval is None else str(args.external_eval)
        ),
        "evidence_boundary": "Hash and checklist package only until finalist evidence exists.",
    }
    checklist = {
        "version": "nlp-submission-checklist-v1",
        "status": manifest["status"],
        "checks": checks,
        "required_missing": [
            name for name, passed in checks.items() if not passed
        ],
        "evidence_boundary": manifest["evidence_boundary"],
    }
    claim = {
        "version": "nlp-final-claim-v1",
        "status": "blocked" if manifest["status"] != "ready" else "pending_final_audit",
        "claim_level": None,
        "reason": (
            "Finalist, external evaluation, three-seed reproduction, or independent "
            "reproduction is missing."
        ),
        "evidence_boundary": "No C-level is awarded by this checklist.",
    }
    for path, payload in (
        (args.manifest_output, manifest),
        (args.checklist_output, checklist),
        (args.claim_output, claim),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
