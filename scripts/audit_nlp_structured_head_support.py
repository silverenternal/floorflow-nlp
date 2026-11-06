#!/usr/bin/env python3
"""Audit per-head sample support and teacher-forced metrics on frozen data/checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from floorflow.nlp_flow.prefix_policy_data import encode_prefix_rows
from floorflow.nlp_flow.prefix_tree_policy import PrefixTreePolicy
from floorflow.nlp_flow.prefix_tree_training import _slice_batch
from floorflow.nlp_flow.real_corpus_policy_data import REAL_ACTIONS
from floorflow.nlp_flow.structure_ablation_runner import _load_inputs

ROOT = Path(__file__).resolve().parents[1]
HEADS = ("action", "node_kind", "edge_field", "parent", "leaf_class", "node_budget")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_rows(
    model: PrefixTreePolicy,
    batch: Any,
    *,
    batch_rows: int,
) -> tuple[dict[str, list[tuple[int, int, float]]], list[int]]:
    rows = {name: [] for name in HEADS}
    parent_candidate_sizes: list[int] = []
    for start in range(0, batch.action.shape[0], batch_rows):
        index = torch.arange(start, min(start + batch_rows, batch.action.shape[0]))
        local = _slice_batch(batch, index)
        output = model(local)
        action = local.action
        masks = {
            "action": local.step_mask,
            "node_kind": local.step_mask & action.eq(REAL_ACTIONS.index("open_node")),
            "edge_field": local.step_mask & action.eq(REAL_ACTIONS.index("attach_edge")),
            "parent": local.step_mask & action.eq(REAL_ACTIONS.index("attach_edge")),
            "leaf_class": local.step_mask & action.eq(REAL_ACTIONS.index("declare_leaf")),
        }
        targets = {
            "action": local.action,
            "node_kind": local.node_kind,
            "edge_field": local.edge_field,
            "parent": local.candidate_target,
            "leaf_class": local.leaf_class,
        }
        for name, mask in masks.items():
            if not mask.any():
                continue
            logits = output[name][mask].float()
            confidence, prediction = logits.softmax(dim=-1).max(dim=-1)
            for gold, predicted, score in zip(
                targets[name][mask].tolist(), prediction.tolist(), confidence.tolist(), strict=True
            ):
                rows[name].append((int(gold), int(predicted), float(score)))
        edge_mask = masks["parent"]
        if edge_mask.any():
            parent_candidate_sizes.extend(
                local.candidate_mask[edge_mask].sum(dim=-1).tolist()
            )
        logits = output["node_budget"].float()
        confidence, prediction = logits.softmax(dim=-1).max(dim=-1)
        rows["node_budget"].extend(
            (int(gold), int(predicted), float(score))
            for gold, predicted, score in zip(
                local.node_budget.tolist(), prediction.tolist(), confidence.tolist(), strict=True
            )
        )
    return rows, [int(size) for size in parent_candidate_sizes]


def _summarize(
    observations: list[tuple[int, int, float]],
    *,
    names: list[str] | None = None,
) -> dict[str, Any]:
    support = Counter(gold for gold, _, _ in observations)
    predicted = Counter(prediction for _, prediction, _ in observations)
    correct = Counter(gold for gold, prediction, _ in observations if gold == prediction)
    total = len(observations)
    labels = sorted(set(support) | set(predicted))
    per_class = {}
    f1_values = []
    for label in labels:
        tp = correct[label]
        precision = tp / predicted[label] if predicted[label] else 0.0
        recall = tp / support[label] if support[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support[label]:
            f1_values.append(f1)
        label_name = names[label] if names and label < len(names) else str(label)
        per_class[label_name] = {
            "class_id": label,
            "support": support[label],
            "predicted": predicted[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    calibration_bins = []
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        bucket = [
            item for item in observations
            if lower <= item[2] < upper or (bin_index == 9 and item[2] == 1.0)
        ]
        if not bucket:
            continue
        accuracy = sum(gold == prediction for gold, prediction, _ in bucket) / len(bucket)
        confidence = sum(score for _, _, score in bucket) / len(bucket)
        ece += len(bucket) / max(total, 1) * abs(accuracy - confidence)
        calibration_bins.append({
            "lower": lower,
            "upper": upper,
            "count": len(bucket),
            "accuracy": accuracy,
            "mean_confidence": confidence,
        })
    support_counts = list(support.values())
    support_ess = (
        (sum(support_counts) ** 2 / sum(count**2 for count in support_counts))
        if support_counts else 0.0
    )
    return {
        "examples": total,
        "accuracy": sum(gold == prediction for gold, prediction, _ in observations) / max(total, 1),
        "macro_f1_supported_classes": sum(f1_values) / max(len(f1_values), 1),
        "expected_calibration_error_10_bins": ece,
        "support_effective_class_count": support_ess,
        "supported_classes": len(support),
        "minimum_class_support": min(support_counts, default=0),
        "maximum_class_support": max(support_counts, default=0),
        "classes": per_class,
        "calibration_bins": calibration_bins,
    }


def audit(config_path: Path, checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = _load_inputs(config, ROOT)
    train = encode_prefix_rows(inputs["train_rows"], inputs["grammars"], inputs["vocabulary"])
    validation = encode_prefix_rows(
        inputs["validation_rows"], inputs["grammars"], inputs["vocabulary"]
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_sha = _sha(config_path)
    if payload.get("config_sha256") != config_sha:
        raise ValueError("checkpoint and config identities differ")
    if payload.get("seed") != 71 or payload.get("arm", {}).get("id") != "flat":
        raise ValueError("audit checkpoint must be the registered flat seed-71 reference")
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
        train_rows, train_parent_sizes = _metric_rows(
            model, train, batch_rows=int(config["shared"]["batch_rows"])
        )
        validation_rows, validation_parent_sizes = _metric_rows(
            model, validation, batch_rows=int(config["shared"]["batch_rows"])
        )
    label_names = {
        "action": list(REAL_ACTIONS),
        "node_kind": list(inputs["vocabulary"]["node_names"]),
        "edge_field": list(inputs["vocabulary"]["edge_names"]),
        "leaf_class": list(inputs["vocabulary"]["class_names"]),
    }
    heads = {}
    for name in HEADS:
        names = label_names.get(name)
        heads[name] = {
            "train": _summarize(train_rows[name], names=names),
            "validation": _summarize(validation_rows[name], names=names),
        }
    candidate_summary = {
        "train_edge_decisions": len(train_parent_sizes),
        "validation_edge_decisions": len(validation_parent_sizes),
        "train_candidate_count_mean": sum(train_parent_sizes) / max(len(train_parent_sizes), 1),
        "validation_candidate_count_mean": sum(validation_parent_sizes) / max(len(validation_parent_sizes), 1),
        "train_candidate_count_max": max(train_parent_sizes, default=0),
        "validation_candidate_count_max": max(validation_parent_sizes, default=0),
        "parent_logits_use_legal_candidate_mask": True,
    }
    report = {
        "version": "nlp-structured-head-support-audit-v1",
        "status": "complete_diagnostic",
        "configuration": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": _sha(checkpoint_path),
            "arm": "flat",
            "seed": 71,
        },
        "train_rows": len(inputs["train_rows"]),
        "validation_rows": len(inputs["validation_rows"]),
        "per_head": heads,
        "candidate_set": candidate_summary,
        "validation_to_training_accuracy_gap": {
            name: heads[name]["validation"]["accuracy"] - heads[name]["train"]["accuracy"]
            for name in HEADS
        },
        "gpu_used": False,
        "test_content_read": False,
        "target_ast_input": False,
        "selection_on_validation": False,
        "evidence_boundary": (
            "Teacher-forced class support and calibration diagnostic from a frozen checkpoint; "
            "not free-running quality evidence or a capability claim."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nlp_structure_matched_v15.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / ".data/nlp/structure_ablation_v15/flat/seed_71.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_structured_head_support_v1.json",
    )
    args = parser.parse_args()
    result = audit(args.config.resolve(), args.checkpoint.resolve(), args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "report": str(args.output),
        "head_accuracy": {
            name: {
                split: values["accuracy"]
                for split, values in result["per_head"][name].items()
            }
            for name in HEADS
        },
        "gpu_used": result["gpu_used"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
