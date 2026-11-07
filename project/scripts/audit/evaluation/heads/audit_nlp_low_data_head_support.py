#!/usr/bin/env python3
"""Produce row-aware head support and confusion diagnostics for the frozen NLP policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[5] / "project/scripts/audit/evaluation/heads"))
from audit_nlp_structured_head_support import HEADS, _metric_rows

from newera.nlp_flow.components.generation.policies.prefix.prefix_policy_data import encode_prefix_rows
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import PrefixTreePolicy
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import REAL_ACTIONS
from newera.nlp_flow.components.research.experiments.studies.structure_ablation_runner import _load_inputs

ROOT = Path(__file__).resolve().parents[5]
ACTION_INDEX = {name: index for index, name in enumerate(REAL_ACTIONS)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_class_support(batch: Any, head: str) -> dict[int, int]:
    row_support: Counter[int] = Counter()
    action = batch.action
    for row_index in range(action.shape[0]):
        mask = batch.step_mask[row_index]
        if head == "action":
            labels = action[row_index, mask]
        elif head == "node_kind":
            labels = batch.node_kind[row_index, mask & action[row_index].eq(ACTION_INDEX["open_node"])]
        elif head in {"edge_field", "parent"}:
            edge_mask = mask & action[row_index].eq(ACTION_INDEX["attach_edge"])
            labels = batch.edge_field[row_index, edge_mask] if head == "edge_field" else batch.candidate_target[row_index, edge_mask]
        elif head == "leaf_class":
            labels = batch.leaf_class[row_index, mask & action[row_index].eq(ACTION_INDEX["declare_leaf"])]
        elif head == "node_budget":
            labels = batch.node_budget[row_index : row_index + 1]
        else:
            raise ValueError(f"unknown head {head}")
        row_support.update({int(label) for label in labels.tolist()})
    return dict(sorted(row_support.items()))


def _confusion(observations: list[tuple[int, int, float]]) -> dict[str, Any]:
    counts: Counter[tuple[int, int]] = Counter((gold, prediction) for gold, prediction, _ in observations)
    labels = sorted({label for pair in counts for label in pair})
    return {
        "labels": labels,
        "rows": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in sorted(counts.items())
        ],
    }


def audit(config_path: Path, checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = _load_inputs(config, ROOT)
    train = encode_prefix_rows(inputs["train_rows"], inputs["grammars"], inputs["vocabulary"])
    validation = encode_prefix_rows(inputs["validation_rows"], inputs["grammars"], inputs["vocabulary"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("config_sha256") != _sha(config_path):
        raise ValueError("checkpoint and config identities differ")
    if payload.get("seed") != 71 or payload.get("arm", {}).get("id") != "flat":
        raise ValueError("audit checkpoint must be the frozen flat seed-71 reference")
    metric_report_path = ROOT / "outputs/reports/nlp_structured_head_support_v1.json"
    metric_report = json.loads(metric_report_path.read_text(encoding="utf-8"))
    if (
        metric_report.get("configuration", {}).get("sha256") != _sha(config_path)
        or metric_report.get("checkpoint", {}).get("sha256") != _sha(checkpoint_path)
    ):
        raise ValueError("existing per-class metrics do not match this frozen audit identity")
    model = PrefixTreePolicy(
        inputs["vocabulary"],
        hidden_size=int(config["shared"]["hidden_size"]),
        layers=int(config["shared"]["decoder_layers"]),
        heads=int(config["shared"]["attention_heads"]),
        max_steps=train.action.shape[1],
        blueprint_conditioning=False,
    ).cpu()
    model.load_state_dict(payload["model"])
    model.eval()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    with torch.inference_mode():
        train_predictions, train_candidate_sizes = _metric_rows(
            model, train, batch_rows=int(config["shared"]["batch_rows"])
        )
        validation_predictions, validation_candidate_sizes = _metric_rows(
            model, validation, batch_rows=int(config["shared"]["batch_rows"])
        )

    per_head = {}
    for head in HEADS:
        per_head[head] = {
            "teacher_forced_metrics": metric_report["per_head"][head],
            "train_independent_rows_with_class": _row_class_support(train, head),
            "validation_independent_rows_with_class": _row_class_support(validation, head),
            "train_confusion": _confusion(train_predictions[head]),
            "validation_confusion": _confusion(validation_predictions[head]),
        }
    report = {
        "version": "nlp-low-data-head-audit-v1",
        "status": "complete_diagnostic",
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha(config_path)},
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": _sha(checkpoint_path),
            "arm": "flat",
            "seed": 71,
        },
        "existing_metric_report": {
            "path": str(metric_report_path.relative_to(ROOT)),
            "sha256": _sha(metric_report_path),
        },
        "independent_units": {"train_rows": len(inputs["train_rows"]), "validation_fragments": len(inputs["validation_rows"])},
        "support_unit_warning": "Decision examples within a row are correlated; report both raw decision support and independent row coverage.",
        "parent_candidate_set": {
            "train_decisions": len(train_candidate_sizes),
            "validation_decisions": len(validation_candidate_sizes),
            "train_mean_size": sum(train_candidate_sizes) / max(len(train_candidate_sizes), 1),
            "validation_mean_size": sum(validation_candidate_sizes) / max(len(validation_candidate_sizes), 1),
            "train_max_size": max(train_candidate_sizes, default=0),
            "validation_max_size": max(validation_candidate_sizes, default=0),
            "legality_mask_used": True,
        },
        "per_head": per_head,
        "gpu_used": False,
        "test_content_read": False,
        "target_ast_input": False,
        "selection_on_validation": False,
        "evidence_boundary": "Frozen-checkpoint teacher-forced diagnosis only; no architecture change or capability claim is justified by this report alone.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/structured_generation/structure_matched/matched_structure/nlp_structure_matched_first_clean_attempt.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".data/nlp/structure_ablation_v15/flat/seed_71.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/nlp_low_data_head_audit_v1.json")
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.checkpoint.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "report": str(args.output), "gpu_used": result["gpu_used"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
