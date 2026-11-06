#!/usr/bin/env python3
"""Run parameter-isolated CPU probes for S1/S2/S3 and frozen feature caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.nlp_flow.engineering import count_unique_trainable_parameters
from floorflow.nlp_flow.model import (
    CompactNLPDecoder,
    NLPModelConfig,
    ScopeController,
    frozen_parameter_delta,
    parameter_snapshot,
)
from floorflow.nlp_flow.model.cfm import flow_matching_stage_loss
from floorflow.nlp_flow.model.feature_cache import (
    load_channel_feature_cache,
    save_channel_feature_cache,
)

ROOT = Path(__file__).resolve().parents[1]


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def _config() -> NLPModelConfig:
    return NLPModelConfig(
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
        enable_json_action_head=True,
        enable_expert_factory=True,
    )


def _lm_loss(
    model: CompactNLPDecoder,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids).token_logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
    )


def _cross_attention_probe(device: str) -> dict[str, object]:
    torch.manual_seed(107)
    model = CompactNLPDecoder(_config()).to(device)
    input_ids = torch.randint(1, 64, (4, 8), device=device)
    labels = torch.randint(1, 64, (4, 8), device=device)
    scope = ScopeController.from_name("S2-cross-attention").apply(model)
    frozen = parameter_snapshot(model)
    with torch.no_grad():
        before = float(_lm_loss(model, input_ids, labels))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.02,
    )
    for _step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = _lm_loss(model, input_ids, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = float(_lm_loss(model, input_ids, labels))
    frozen_delta = frozen_parameter_delta(model, frozen)
    return {
        "status": "pass" if after < before and max(frozen_delta.values(), default=0.0) == 0.0 else "fail",
        "scope": scope,
        "loss_before": before,
        "loss_after": after,
        "loss_delta": after - before,
        "frozen_tensor_count": len(frozen_delta),
        "maximum_frozen_delta": max(frozen_delta.values(), default=0.0),
    }


def _channel_and_cfm_probes(
    device: str,
) -> tuple[dict[str, object], dict[str, object]]:
    torch.manual_seed(109)
    model = CompactNLPDecoder(_config()).to(device)
    input_ids = torch.randint(1, 64, (2, 8), device=device)
    channel_scope = ScopeController.from_name("S1-channels").apply(model)
    output = model(input_ids)
    channel_loss = (
        output.typed_memory.global_memory.square().mean()
        + sum(value.square().mean() for value in output.typed_channels.values())
    )
    channel_loss.backward()
    channel_gradients = {
        name: parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.typed_memory.named_parameters()
    }
    channel = {
        "status": "pass" if channel_gradients and all(channel_gradients.values()) else "fail",
        "scope": channel_scope,
        "loss": float(channel_loss.detach()),
        "gradients": channel_gradients,
    }

    model.zero_grad(set_to_none=True)
    cfm_scope = ScopeController.from_name("S3-cfm").apply(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.cfm.parameters() if parameter.requires_grad],
        lr=0.02,
    )
    context = torch.randn(4, 6, 32, device=device)
    noise = torch.randn_like(context)
    target = torch.randn_like(context)
    time = torch.rand(4, device=device)
    state = noise + time[:, None, None] * (target - noise)
    losses = []
    for _step in range(100):
        optimizer.zero_grad(set_to_none=True)
        velocity = model.cfm.stage_a(state, time, context)
        loss = flow_matching_stage_loss(velocity, target, noise)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    cfm = {
        "status": "pass" if losses[-1] < losses[0] else "fail",
        "scope": cfm_scope,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    return channel, cfm


def _feature_cache_probe(device: str) -> dict[str, object]:
    torch.manual_seed(113)
    model = CompactNLPDecoder(_config()).to(device).eval()
    input_ids = torch.randint(1, 64, (2, 8), device=device)
    with torch.no_grad():
        output = model(input_ids)
    config_hash = hashlib.sha256(
        json.dumps(model.config.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "features.pt"
        save_channel_feature_cache(
            path,
            hidden_states=output.hidden_states,
            channel_states=output.typed_channels,
            config_hash=config_hash,
        )
        restored = load_channel_feature_cache(path, expected_config_hash=config_hash)
    hidden_equal = torch.equal(
        restored["hidden_states"],
        output.hidden_states.cpu(),
    )
    channels_equal = all(
        torch.equal(restored["channel_states"][name], value.cpu())
        for name, value in output.typed_channels.items()
    )
    return {
        "status": "pass" if hidden_equal and channels_equal else "fail",
        "hidden_equal": hidden_equal,
        "channels_equal": channels_equal,
        "channel_count": len(output.typed_channels),
        "config_hash": config_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_staged_training_probe_v1.json",
    )
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    cross_attention = _cross_attention_probe(args.device)
    channel, cfm = _channel_and_cfm_probes(args.device)
    cache = _feature_cache_probe(args.device)
    model = CompactNLPDecoder(_config())
    scope_rows = {}
    for scope_name in ("S0-base", "S1-channels", "S2-cross-attention", "S3-cfm", "S4-joint", "S5-factory"):
        scoped = CompactNLPDecoder(_config())
        report = ScopeController.from_name(scope_name).apply(scoped)
        scope_rows[scope_name] = {
            **report,
            "total_unique_parameters": count_unique_trainable_parameters(scoped),
        }
    checks = {
        "cross_attention_only_improves": cross_attention["status"] == "pass",
        "channels_receive_gradient": channel["status"] == "pass",
        "cfm_loss_decreases": cfm["status"] == "pass",
        "feature_cache_is_exact": cache["status"] == "pass",
        "joint_excludes_factory": scope_rows["S4-joint"]["trainable_parameters"]
        < scope_rows["S5-factory"]["trainable_parameters"],
    }
    report = {
        "version": "nlp-staged-training-probe-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "device": args.device,
        "checks": checks,
        "cross_attention": cross_attention,
        "channels": channel,
        "cfm": cfm,
        "feature_cache": cache,
        "scopes": scope_rows,
        "model_parameters": count_unique_trainable_parameters(model),
        "evidence_boundary": (
            "GPU parameter-isolation and optimization probes only; no corpus-scale capability claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
