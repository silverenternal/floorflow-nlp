#!/usr/bin/env python3
"""Joint calibration and A0-A5 compact architecture visibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.nlp_flow.engineering import count_unique_trainable_parameters
from floorflow.nlp_flow.model import (
    CompactNLPDecoder,
    NLPModelConfig,
    ScopeController,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(
    *,
    channels: bool = False,
    fusion: bool = False,
    cfm: bool = False,
    json_head: bool = False,
    factory: bool = False,
) -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        channel_size=8,
        enable_typed_channels=channels,
        enable_typed_fusion=fusion,
        enable_two_stage_cfm=cfm,
        enable_json_action_head=json_head,
        enable_expert_factory=factory,
    )


def _inputs():
    torch.manual_seed(41)
    input_ids = torch.randint(0, 64, (8, 8))
    masks = {
        name: torch.randint(0, 2, (8, 8), dtype=torch.bool)
        for name in ("syntax", "symbol", "discourse", "execution", "engineering")
    }
    for mask in masks.values():
        mask[~mask.any(dim=1), 0] = True
    return input_ids, masks


def _joint_calibration(input_ids, masks) -> dict[str, float | bool]:
    torch.manual_seed(3)
    model = CompactNLPDecoder(_config(channels=True, fusion=True, cfm=True, json_head=True))
    ScopeController.from_name("S4-joint").apply(model)
    torch.manual_seed(11)
    readout = torch.nn.Linear(32, 2, bias=False)
    with torch.no_grad():
        initial_output = model(input_ids, channel_masks=masks)
        labels = initial_output.typed_channels["syntax"][:, -1, 0].gt(0.0).long()
        initial_loss = F.cross_entropy(
            readout(initial_output.hidden_states[:, -1]),
            labels,
        )
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *readout.parameters()],
        lr=0.03,
    )
    for _step in range(250):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, channel_masks=masks)
        loss = F.cross_entropy(readout(output.hidden_states[:, -1]), labels)
        loss.backward()
        optimizer.step()
    return {
        "initial_loss": float(initial_loss.detach()),
        "final_loss": float(loss.detach()),
        "improved": float(loss.detach()) < float(initial_loss.detach()),
        "all_branches_retained": all(
            module is not None
            for module in (
                model.typed_memory,
                model.typed_fusion,
                model.cfm,
                model.json_head,
            )
        ),
    }


def _arms() -> dict[str, object]:
    configurations = {
        "A0": _config(),
        "A1": _config(channels=True),
        "A2": _config(channels=True, fusion=True),
        "A3": _config(channels=True, fusion=True, cfm=True),
        "A4": _config(channels=True, fusion=True, cfm=True, json_head=True),
        "A5": _config(
            channels=True,
            fusion=True,
            cfm=True,
            json_head=True,
            factory=True,
        ),
    }
    arms = {}
    for name, config in configurations.items():
        model = CompactNLPDecoder(config)
        arms[name] = {
            "parameters": count_unique_trainable_parameters(model),
            "branches": {
                "channels": model.typed_memory is not None,
                "fusion": model.typed_fusion is not None,
                "cfm": model.cfm is not None,
                "json": model.json_head is not None,
                "factory": model.expert_factory is not None,
            },
            "under_23m_cap": count_unique_trainable_parameters(model) <= 23_000_000,
        }
    return arms


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_joint_calibration_v1.json",
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_architecture_ablation_v1.json",
    )
    args = parser.parse_args()
    input_ids, masks = _inputs()
    joint = _joint_calibration(input_ids, masks)
    joint_report = {
        "version": "nlp-joint-calibration-v1",
        "status": "pass" if joint["improved"] and joint["all_branches_retained"] else "fail",
        **joint,
        "evidence_boundary": "Tiny mechanism probe only; no public-corpus quality claim.",
    }
    arms = _arms()
    ablation_report = {
        "version": "nlp-architecture-ablation-v1",
        "status": (
            "pass"
            if all(arm["under_23m_cap"] for arm in arms.values())
            else "fail"
        ),
        "tokenizer_sha256": _sha256(ROOT / ".data/nlp/tokenizer_v1/tokenizer.json"),
        "data_manifest_sha256": _sha256(
            ROOT / ".data/nlp/mixed_v1/corpus_v1/manifest.json"
        ),
        "declared_token_budget": 46_000_000,
        "primary_inference_nfe": 8,
        "arms": arms,
        "evidence_boundary": (
            "Branch visibility and compact parameter cap only; performance ablation "
            "requires matched public-corpus runs."
        ),
    }
    for path, report in (
        (args.joint_output, joint_report),
        (args.ablation_output, ablation_report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)
        print(json.dumps(report, indent=2))
    if joint_report["status"] != "pass" or ablation_report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
