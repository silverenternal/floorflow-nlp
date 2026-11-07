#!/usr/bin/env python3
"""Run S1/S2/S3 staged parameter-isolation probes on deterministic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from newera.nlp_flow.model import (
    CompactNLPDecoder,
    NLPModelConfig,
    ScopeController,
    frozen_parameter_delta,
    parameter_snapshot,
)
from newera.nlp_flow.model.components.training.objectives.cfm import flow_matching_stage_loss

ROOT = Path(__file__).resolve().parents[5]


def _model() -> CompactNLPDecoder:
    return CompactNLPDecoder(
        NLPModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
            channel_size=8,
            enable_typed_channels=True,
            enable_typed_fusion=True,
            enable_two_stage_cfm=True,
        )
    ).eval()


def _inputs():
    torch.manual_seed(3)
    input_ids = torch.randint(0, 64, (8, 8))
    channel_masks = {
        name: torch.randint(0, 2, (8, 8), dtype=torch.bool)
        for name in ("syntax", "symbol", "discourse", "execution", "engineering")
    }
    for mask in channel_masks.values():
        mask[~mask.any(dim=1), 0] = True
    return input_ids, channel_masks


def _labels(model, input_ids, channel_masks) -> torch.Tensor:
    with torch.no_grad():
        output = model(input_ids, channel_masks=channel_masks)
    return output.typed_channels["syntax"][:, -1, 0].gt(0.0).long()


def _s1_probe(model, input_ids, channel_masks, labels) -> dict[str, float | bool]:
    torch.manual_seed(9)
    ScopeController.from_name("S1-channels").apply(model)
    probe = torch.nn.Linear(model.config.channel_size, 2)
    optimizer = torch.optim.AdamW(
        [*model.typed_memory.parameters(), *probe.parameters()],
        lr=0.03,
    )
    initial = None
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, channel_masks=channel_masks)
        logits = probe(output.typed_channels["syntax"].mean(dim=1))
        loss = F.cross_entropy(logits, labels)
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    return {
        "initial_loss": float(initial),
        "final_loss": float(loss.detach()),
        "channel_gradients": model.typed_memory.down["syntax"].weight.grad is not None,
        "improved": float(loss.detach()) < float(initial),
    }


def _s2_probe(base_model, input_ids, channel_masks, labels) -> dict[str, object]:
    model = base_model
    ScopeController.from_name("S2-cross-attention").apply(model)
    frozen = parameter_snapshot(model)
    torch.manual_seed(11)
    readout = torch.nn.Linear(model.config.hidden_size, 2, bias=False)
    with torch.no_grad():
        frozen_output = model(input_ids, channel_masks=channel_masks)
        frozen_loss = F.cross_entropy(
            readout(frozen_output.hidden_states[:, -1]),
            labels,
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
    )
    for _step in range(300):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, channel_masks=channel_masks)
        loss = F.cross_entropy(readout(output.hidden_states[:, -1]), labels)
        loss.backward()
        optimizer.step()
    deltas = frozen_parameter_delta(model, frozen)
    return {
        "frozen_base_loss": float(frozen_loss.detach()),
        "trained_loss": float(loss.detach()),
        "frozen_parameter_max_delta": max(deltas.values(), default=0.0),
        "fusion_gradient": model.typed_fusion.cross_attention.out.weight.grad is not None,
        "improved": float(loss.detach()) < float(frozen_loss.detach()),
    }


def _s3_probe() -> dict[str, float | bool]:
    torch.manual_seed(13)
    model = _model()
    ScopeController.from_name("S3-cfm").apply(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.03,
    )
    target = torch.randn(2, 5, 32)
    noise = torch.randn_like(target)
    context = torch.randn_like(target)
    time = torch.rand(2)
    no_cfm_loss = 2.0 * flow_matching_stage_loss(
        torch.zeros_like(target),
        target,
        noise,
    )
    initial = None
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        output = model.cfm.predict(target, target, time, time, context)
        loss = flow_matching_stage_loss(
            output.ast_velocity,
            target,
            noise,
        ) + flow_matching_stage_loss(
            output.result_velocity,
            target,
            noise,
        )
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    return {
        "initial_loss": float(initial),
        "final_loss": float(loss.detach()),
        "no_cfm_control_loss": float(no_cfm_loss.detach()),
        "cfm_gradient": model.cfm.stage_a.down.weight.grad is not None,
        "improved": float(loss.detach()) < float(initial),
        "beats_no_cfm_control": float(loss.detach()) < float(no_cfm_loss.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_staged_training_probe_v1.json",
    )
    args = parser.parse_args()
    input_ids, channel_masks = _inputs()
    torch.manual_seed(3)
    s1_model = _model()
    labels = _labels(s1_model, input_ids, channel_masks)
    torch.manual_seed(3)
    s2_model = _model()
    s1 = _s1_probe(s1_model, input_ids, channel_masks, labels)
    s2 = _s2_probe(s2_model, input_ids, channel_masks, labels)
    s3 = _s3_probe()
    report = {
        "version": "nlp-staged-training-probe-v1",
        "status": (
            "pass"
            if s1["improved"]
            and s2["improved"]
            and s2["frozen_parameter_max_delta"] == 0.0
            and s3["improved"]
            and s3["beats_no_cfm_control"]
            else "fail"
        ),
        "S1-channels": s1,
        "S2-cross-attention": s2,
        "S3-cfm": s3,
        "controls": {
            "same_data": True,
            "same_model_initialization": True,
            "same_readout_for_s2_comparison": True,
        },
        "evidence_boundary": (
            "Tiny mechanism probes only; no public-corpus capability or ablation claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
