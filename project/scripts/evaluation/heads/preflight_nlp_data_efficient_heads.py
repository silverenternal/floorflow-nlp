#!/usr/bin/env python3
"""CPU-only contract preflight for the factorized grammar-masked node-kind head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.components.research.architecture.experts.efficient_output_heads import (
    FactorizedClassHead,
    apply_legal_candidate_mask,
    rebase_residual_budget_logits,
)
from newera.nlp_flow.components.research.experiments.studies.structure_ablation_runner import _load_inputs

ROOT = Path(__file__).resolve().parents[4]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors = []
    inputs = config["evidence_inputs"]
    for path_key, hash_key in (
        ("support_audit", "support_audit_sha256"),
        ("pilot_decision", "pilot_decision_sha256"),
    ):
        path = ROOT / inputs[path_key]
        if _sha(path) != inputs[hash_key]:
            errors.append(f"{path_key} evidence hash changed")

    support = json.loads((ROOT / inputs["support_audit"]).read_text(encoding="utf-8"))
    decision = json.loads((ROOT / inputs["pilot_decision"]).read_text(encoding="utf-8"))
    vocabulary_source = config["vocabulary_source"]
    data_config_path = ROOT / vocabulary_source["data_config"]
    if _sha(data_config_path) != vocabulary_source["data_config_sha256"]:
        errors.append("frozen vocabulary data configuration hash changed")
    data_config = json.loads(data_config_path.read_text(encoding="utf-8"))
    fitted_inputs = _load_inputs(data_config, ROOT)
    vocabulary = fitted_inputs["vocabulary"]
    node_support = support["per_head"]["node_kind"]["teacher_forced_metrics"]["train"]
    class_count = len(vocabulary["node_ids"]) + 1
    supported_class_count = len(node_support["classes"])
    hidden_size = int(config["scope"]["hidden_size"])
    rank = int(config["candidate_variant"]["rank"])
    control_count = hidden_size * class_count + class_count
    variant = FactorizedClassHead(hidden_size, class_count, rank).cpu()
    variant_count = variant.parameter_count
    if class_count != int(config["scope"]["candidate_count_node_kind_including_unknown"]):
        errors.append("configured class count differs from frozen train vocabulary")
    if supported_class_count > class_count:
        errors.append("support audit contains more node kinds than the frozen output vocabulary")
    if control_count != int(config["control"]["parameters"]):
        errors.append("configured legacy head parameter count is incorrect")
    if variant_count != int(config["candidate_variant"]["parameters"]):
        errors.append("configured factorized head parameter count is incorrect")
    if variant_count >= control_count:
        errors.append("factorized head does not reduce output parameters")
    if decision.get("decision", "").startswith("positive_") is False:
        errors.append("registered pilot decision does not open the head branch")

    logits = torch.randn(2, 3, class_count, requires_grad=True)
    legal = torch.zeros_like(logits, dtype=torch.bool)
    legal[..., :2] = True
    active = torch.tensor([[True, False, True], [False, True, False]])
    targets = torch.zeros((2, 3), dtype=torch.long)
    masked = apply_legal_candidate_mask(logits, legal, active, targets)
    if not torch.isneginf(masked[active][..., 2:]).all():
        errors.append("illegal candidates are not masked")
    torch.nn.functional.cross_entropy(
        masked.reshape(-1, class_count), targets.reshape(-1)
    ).backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        errors.append("masked candidate path did not produce finite gradients")
    budget_minimum = torch.tensor([1, 3])
    budget_targets = torch.tensor([2, 4])
    budget_logits = rebase_residual_budget_logits(
        torch.randn(2, 8), budget_minimum, budget_targets
    )
    if torch.isfinite(budget_logits[0, 0]) or torch.isfinite(budget_logits[1, :3]).any():
        errors.append("residual budget head leaves infeasible totals selectable")

    source_paths = (
        "src/newera/nlp_flow/components/research/architecture/experts/efficient_output_heads.py",
        "src/newera/nlp_flow/components/generation/policies/prefix/prefix_policy_data.py",
        "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_policy.py",
        "src/newera/nlp_flow/components/generation/policies/prefix/prefix_tree_training.py",
        "src/newera/nlp_flow/components/generation/trees/runtime/tree_completion_runtime.py",
        "project/scripts/evaluation/heads/preflight_nlp_data_efficient_heads.py",
        "project/tests/models_training/channels_dynamic_efficient/test_nlp_data_efficient_heads.py",
    )
    report = {
        "version": "nlp-data-efficient-heads-preflight-v1",
        "status": "ready" if not errors else "blocked",
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha(config_path)},
        "evidence": {
            "support_audit_sha256": _sha(ROOT / inputs["support_audit"]),
            "pilot_decision_sha256": _sha(ROOT / inputs["pilot_decision"]),
            "positive_pilot_signal": decision.get("decision"),
            "vocabulary_data_config_sha256": _sha(data_config_path),
            "train_vocabulary_node_kind_count": len(vocabulary["node_ids"]),
            "supported_train_node_kind_count": supported_class_count,
            "train_rows_read": len(fitted_inputs["train_rows"]),
            "validation_rows_read_for_vocabulary_contract": len(fitted_inputs["validation_rows"]),
            "test_content_read": False,
        },
        "candidate_variant": {
            "id": config["candidate_variant"]["id"],
            "hidden_size": hidden_size,
            "class_count_including_unknown": class_count,
            "rank": rank,
            "legacy_parameters": control_count,
            "factorized_parameters": variant_count,
            "parameter_reduction": 1.0 - variant_count / control_count,
            "combined_node_kind_and_budget_parameters": {
                "legacy": control_count + int(config["budget_target"]["parameters"]),
                "candidate": variant_count + int(config["budget_target"]["parameters"]),
            },
            "legal_mask_shape": list(legal.shape),
            "masked_gradient_finite": logits.grad is not None and bool(torch.isfinite(logits.grad).all()),
        },
        "implementation_hashes": {path: _sha(ROOT / path) for path in source_paths},
        "gpu_used": False,
        "errors": errors,
        "evidence_boundary": "CPU head-shape, parameter, mask, and gradient contract only; no quality evidence.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/models_training/learning/heads/nlp_data_efficient_heads.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_data_efficient_heads_preflight_v1.json")
    args = parser.parse_args()
    report = preflight(args.config.resolve(), args.output.resolve())
    print(json.dumps({"status": report["status"], "errors": report["errors"]}, indent=2))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
