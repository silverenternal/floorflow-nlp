#!/usr/bin/env python3
"""Audit supervised compositional-emergence components on a frozen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from newera.nlp_flow.components.generation.outputs.dialogue.chat import model_config_from_experiment
from newera.nlp_flow.engineering import load_nlp_checkpoint, restore_nlp_checkpoint
from newera.nlp_flow.model import CompactNLPDecoder
from newera.nlp_flow.components.research.training.data.training_data import PackedTokenDataset

ROOT = Path(__file__).resolve().parents[5]


def _binary_accuracy(scores: torch.Tensor, target: torch.Tensor) -> float:
    if target.numel() == 0:
        return 0.0
    return float(scores.gt(0.0).eq(target.bool()).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/nlp_emergent_chat_validity_v2/best.pt",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--objective-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_emergent_objective_audit_v1.json",
    )
    parser.add_argument(
        "--holdout-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_emergent_composition_holdout_v1.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = load_nlp_checkpoint(
        args.checkpoint.resolve(),
        map_location=device,
    )
    model = CompactNLPDecoder(model_config_from_experiment(checkpoint["config"]))
    restore_nlp_checkpoint(checkpoint, model=model, device=device)
    model = model.to(device).eval()
    if model.emergence is None:
        raise RuntimeError("checkpoint does not enable the emergence objective")
    tokenizer_path = ROOT / checkpoint["config"]["model"]["tokenizer"]
    validation_path = args.checkpoint.parent / "report.json"
    validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
    validation_shards = tuple(
        ROOT / path for path in validation_report["data"]["validation"]
    )
    dataset = PackedTokenDataset(
        validation_shards,
        tokenizer_path,
        context_length=int(checkpoint["config"]["model"]["max_context_tokens"]),
        max_sequences=64,
        text_mode="chat",
        families=frozenset({"chat_instruction"}),
    )
    batch = next(iter(DataLoader(dataset, batch_size=16)))
    input_ids = batch["input_ids"].to(device)
    surface = batch["surface_valid"].to(device)
    execution = batch["execution_valid"].to(device)
    with torch.no_grad():
        output = model(input_ids)
        loss = model.emergence(
            output.hidden_states,
            surface_targets=surface,
            execution_targets=execution,
        )
        positive_logits = model.emergence.validity_head(output.hidden_states.float())
        negative_logits = model.emergence.validity_head(
            output.hidden_states.float().flip(0)
        )
        execution_known = execution.ge(0)
        surface_positive = surface.float()
        surface_negative = torch.zeros_like(surface_positive)
        execution_labels = execution[execution_known].float()
    execution_accuracy = (
        _binary_accuracy(
            positive_logits[..., 1][execution_known],
            execution_labels,
        )
        if execution_known.any()
        else None
    )
    checks = {
        "surface_positive_accuracy": _binary_accuracy(
            positive_logits[..., 0],
            surface_positive,
        )
        >= 0.8,
        "surface_negative_accuracy": _binary_accuracy(
            negative_logits[..., 0],
            surface_negative,
        )
        >= 0.8,
        "execution_accuracy": (
            execution_accuracy is not None
            and execution_accuracy >= 0.8
        ),
        "finite_objectives": all(
            torch.isfinite(value)
            for value in loss.as_dict().values()
        ),
    }
    metrics = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "surface_positive_accuracy": _binary_accuracy(
            positive_logits[..., 0],
            surface_positive,
        ),
        "surface_negative_accuracy": _binary_accuracy(
            negative_logits[..., 0],
            surface_negative,
        ),
        "execution_accuracy": execution_accuracy,
        "execution_label_positive_rate": (
            float(execution_labels.mean()) if execution_labels.numel() else None
        ),
        "execution_coverage": float(execution_known.float().mean()),
        "losses": {
            name: float(value) for name, value in loss.as_dict().items()
        },
        "heldout_combinations_used_in_loss": False,
        "claim_level": "none",
        "evidence_boundary": (
            "Objective and validity-head audit only; held-out executable composition "
            "and C0-C3 performance remain unmeasured."
        ),
    }
    holdout = {
        "version": "nlp-emergent-composition-holdout-v1",
        "status": "blocked",
        "reason": "The frozen atom/operator holdout has not yet been materialized and executed.",
        "require": [
            "atoms seen in training",
            "complete combinations absent from loss and checkpoint selection",
            "grammar/compile/runtime verifier",
            "matched C0-C3 parameters, tokens, and NFE",
        ],
        "claim_level": "none",
    }
    for path, payload in (
        (args.objective_output, metrics),
        (args.holdout_output, holdout),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
