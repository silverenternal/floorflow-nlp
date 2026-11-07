#!/usr/bin/env python3
"""Deterministic contract audit for parameter-shared dynamic stacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from newera.nlp_flow.model.components.architecture.state.dynamic_stack import (
    DynamicStackConfig,
    DynamicStackEncoder,
)

ROOT = Path(__file__).resolve().parents[5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "project/configs/models_training/architecture/definitions/nlp_dynamic_stack.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_dynamic_stack_audit_v1.json",
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda:1"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    return parser.parse_args()


def _model_config(payload: dict, max_steps: int) -> NLPModelConfig:
    model = payload["model"]
    return NLPModelConfig(
        vocab_size=int(model["vocab_size"]),
        hidden_size=int(model["hidden_size"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        num_kv_heads=int(model["num_kv_heads"]),
        intermediate_size=int(model["intermediate_size"]),
        max_seq_len=int(model["max_seq_len"]),
        parameter_cap=int(model["parameter_cap"]),
        tie_embeddings=bool(model["tie_embeddings"]),
        enable_dynamic_stack=True,
        dynamic_min_steps=int(model["dynamic_min_steps"]),
        dynamic_max_steps=max_steps,
        dynamic_parallel_lanes=int(model["dynamic_parallel_lanes"]),
        dynamic_parallel_top_k=int(model["dynamic_parallel_top_k"]),
        dynamic_lane_rank=int(model["dynamic_lane_rank"]),
    )


def _stack_config(payload: dict, max_steps: int) -> DynamicStackConfig:
    model = payload["model"]
    return DynamicStackConfig(
        hidden_size=int(model["hidden_size"]),
        num_heads=int(model["num_heads"]),
        num_kv_heads=int(model["num_kv_heads"]),
        intermediate_size=int(model["intermediate_size"]),
        max_seq_len=int(model["max_seq_len"]),
        max_serial_steps=max_steps,
        min_serial_steps=int(model["dynamic_min_steps"]),
        parallel_lanes=int(model["dynamic_parallel_lanes"]),
        parallel_top_k=int(model["dynamic_parallel_top_k"]),
        lane_rank=int(model["dynamic_lane_rank"]),
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    seed = int(payload["runtime"]["seed"])
    audit = payload["audit"]
    depths = [int(value) for value in audit["serial_depths"]]
    batch = int(audit["batch_size"])
    sequence = int(audit["sequence_length"])
    padding = int(audit["padding_length"])
    valid = sequence - padding
    delayed_halt_bias = float(audit["delayed_halt_bias"])
    torch.manual_seed(seed)

    models = {
        depth: CompactNLPDecoder(_model_config(payload, depth))
        .to(args.device)
        .eval()
        for depth in depths
    }
    parameter_counts = {
        str(depth): count_unique_trainable_parameters(model)
        for depth, model in models.items()
    }
    parameters_depth_independent = len(set(parameter_counts.values())) == 1

    input_ids = torch.randint(
        0,
        int(payload["model"]["vocab_size"]),
        (batch, sequence),
        generator=torch.Generator().manual_seed(seed),
    ).to(args.device)
    attention_mask = torch.zeros(
        batch,
        sequence,
        dtype=torch.bool,
        device=args.device,
    )
    attention_mask[:, :valid] = True
    target_model = models[max(depths)]
    with torch.no_grad():
        target_model.dynamic_stack.halting_head[-1].bias.fill_(
            delayed_halt_bias
        )
        depth_runs = {}
        for max_steps in depths:
            output = target_model.dynamic_stack(
                target_model.embedding(input_ids),
                attention_mask=attention_mask,
                max_steps=max_steps,
            )
            depth_runs[str(max_steps)] = {
                "executed_steps": int(output.executed_steps),
                "mean_depth": float(output.mean_depth),
                "max_depth": float(output.max_depth),
                "expected_depth": float(output.ponder_cost),
                "serial_token_steps": int(
                    output.serial_token_steps
                ),
                "parallel_token_lane_calls": int(
                    output.parallel_token_lane_calls
                ),
                "dense_serial_token_calls": int(
                    output.dense_serial_token_calls
                ),
            }

        causal_first = torch.randint(
            0,
            8192,
            (1, 16),
            device=args.device,
        )
        causal_second = causal_first.clone()
        causal_second[:, 8:] += 7
        causal_left = target_model(causal_first).token_logits
        causal_right = target_model(causal_second).token_logits
        decoder_causal = bool(
            torch.allclose(
                causal_left[:, :8],
                causal_right[:, :8],
                atol=1.0e-6,
            )
        )

        padded_first = torch.randint(
            0,
            8192,
            (1, sequence),
            device=args.device,
        )
        padded_second = padded_first.clone()
        padded_second[:, valid:] = 8191
        padded_left = target_model(
            padded_first,
            attention_mask=attention_mask[:1],
        ).token_logits
        padded_right = target_model(
            padded_second,
            attention_mask=attention_mask[:1],
        ).token_logits
        padding_invariant = bool(
            torch.allclose(
                padded_left[:, :valid],
                padded_right[:, :valid],
                atol=1.0e-6,
            )
        )

        encoder = DynamicStackEncoder(
            _stack_config(payload, max(depths))
        ).to(args.device).eval()
        encoder_first = torch.randn(
            1,
            16,
            int(payload["model"]["hidden_size"]),
            device=args.device,
        )
        encoder_second = encoder_first.clone()
        encoder_second[:, 8:] += 5
        encoder_left = encoder(encoder_first).hidden_states
        encoder_right = encoder(encoder_second).hidden_states
        encoder_bidirectional = not bool(
            torch.allclose(
                encoder_left[:, :8],
                encoder_right[:, :8],
                atol=1.0e-6,
            )
        )

    requested_depths_observed = {
        str(depth): depth_runs[str(depth)]["executed_steps"] == depth
        for depth in depths
    }
    lane_calls_match = {
        str(depth): (
            depth_runs[str(depth)]["parallel_token_lane_calls"]
            == depth_runs[str(depth)]["serial_token_steps"]
            * int(payload["model"]["dynamic_parallel_top_k"])
        )
        for depth in depths
    }
    checks = {
        "parameters_depth_independent": parameters_depth_independent,
        "decoder_causal": decoder_causal,
        "encoder_bidirectional": encoder_bidirectional,
        "padding_invariant": padding_invariant,
        "requested_depths_observed": all(requested_depths_observed.values()),
        "routed_lane_calls_match": all(lane_calls_match.values()),
    }
    report = {
        "version": "nlp-dynamic-stack-audit-v1",
        "status": (
            "pass"
            if all(checks.values())
            else "fail"
        ),
        "config": str(args.config.relative_to(ROOT)),
        "device": args.device,
        "parameter_counts": parameter_counts,
        "parameter_fraction_of_p5m_core_reference": {
            str(depth): count / int(payload["model"]["parameter_cap"])
            for depth, count in parameter_counts.items()
        },
        "depth_runs": depth_runs,
        "checks": checks,
        "claim_gate": payload["claim_gate"],
        "evidence_boundary": (
            "Parameter sharing, routing, attention direction, padding, and "
            "compute accounting only; no model-quality evidence."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
