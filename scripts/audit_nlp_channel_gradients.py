#!/usr/bin/env python3
"""Audit frozen-decoder channel gradients and write a local feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from floorflow.nlp_flow.model.channel_objectives import (
    ChannelObjectiveSpec,
    TypedChannelObjectiveBank,
    channel_objective_loss,
)
from floorflow.nlp_flow.model.feature_cache import save_channel_feature_cache

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "syntax": ChannelObjectiveSpec("multiclass", 4),
    "symbol": ChannelObjectiveSpec("binary", 1),
    "discourse": ChannelObjectiveSpec("multiclass", 4),
    "execution": ChannelObjectiveSpec("multiclass", 3),
    "engineering": ChannelObjectiveSpec("multiclass", 4),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_channel_gradient_audit_v1.json",
    )
    args = parser.parse_args()
    torch.manual_seed(7)
    config = NLPModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        channel_size=8,
        enable_typed_channels=True,
    )
    model = CompactNLPDecoder(config).train()
    bank = TypedChannelObjectiveBank(8, SPECS)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.typed_memory.parameters():
        parameter.requires_grad_(True)
    output = model(torch.randint(0, 128, (2, 6)))
    targets = {
        "syntax": torch.randint(0, 4, (2, 6)),
        "symbol": torch.randint(0, 2, (2, 6)),
        "discourse": torch.randint(0, 4, (2, 6)),
        "execution": torch.randint(0, 3, (2, 6)),
        "engineering": torch.randint(0, 4, (2, 6)),
    }
    logits = bank(output.typed_channels)
    loss, components = channel_objective_loss(
        logits,
        targets,
        None,
        SPECS,
        {name: 1.0 for name in SPECS},
    )
    loss.backward()
    channel_gradients = {
        name: float(model.typed_memory.down[name].weight.grad.norm())
        for name in output.typed_channels
    }
    decoder_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad is False and parameter.grad is not None
    ]
    cache_path = ROOT / ".data/nlp/mixed_v1/probes/channel_features.pt"
    save_channel_feature_cache(
        cache_path,
        hidden_states=output.hidden_states,
        channel_states=output.typed_channels,
        config_hash="channel-audit-v1",
    )
    errors = []
    if any(value <= 0.0 for value in channel_gradients.values()):
        errors.append("one or more channels have zero gradient")
    if decoder_gradients:
        errors.append(f"frozen decoder parameters received gradients: {decoder_gradients}")
    report = {
        "version": "nlp-channel-gradient-audit-v1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "loss": float(loss.detach()),
        "components": {name: float(value) for name, value in components.items()},
        "channel_gradient_norms": channel_gradients,
        "frozen_decoder_gradients": decoder_gradients,
        "feature_cache": str(cache_path),
        "evidence_boundary": "Synthetic gradient probe only; no quality result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
