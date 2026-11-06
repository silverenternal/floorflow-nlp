#!/usr/bin/env python3
"""Run deterministic control/candidate mechanism and verifier probes."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from floorflow.nlp_flow.executors import run_command
from floorflow.nlp_flow.json_tree import strict_json_loads
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from floorflow.nlp_flow.model.json_grammar import JSONGrammarState

ROOT = Path(__file__).resolve().parents[1]


def _config(*, candidate: bool) -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        channel_size=8,
        enable_typed_channels=candidate,
        enable_typed_fusion=candidate,
        enable_two_stage_cfm=candidate,
        enable_json_action_head=candidate,
        enable_expert_factory=candidate,
    )


def _prefix_gradients(model: torch.nn.Module) -> dict[str, bool]:
    prefixes = {
        name.split(".", 1)[0]
        for name, _parameter in model.named_parameters()
    }
    return {
        prefix: any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith(f"{prefix}.")
        )
        for prefix in sorted(prefixes)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_synthetic_probe_v1.json",
    )
    args = parser.parse_args()
    torch.manual_seed(29)
    control = CompactNLPDecoder(_config(candidate=False)).eval()
    torch.manual_seed(29)
    candidate = CompactNLPDecoder(_config(candidate=True))
    input_ids = torch.randint(0, 128, (2, 8))
    with torch.no_grad():
        control_logits = control(input_ids).token_logits
        candidate_logits = candidate(input_ids).token_logits
    zero_gate_identity = torch.equal(control_logits, candidate_logits)

    with torch.no_grad():
        candidate.typed_fusion.cross_attention.gate.fill_(0.1)
        candidate.typed_fusion.router.gate.fill_(0.1)
        candidate.typed_fusion.slot_mixer.gate.fill_(0.1)
        candidate.expert_factory.birth_gate.fill_(1.0)
    output = candidate(input_ids)
    token_loss = F.cross_entropy(
        output.token_logits.reshape(-1, 128),
        torch.randint(0, 128, (16,)),
    )
    channel_loss = sum(state.square().mean() for state in output.typed_channels.values())
    action_logits = candidate.json_action_logits(
        output.hidden_states[:, -1],
        [JSONGrammarState.initial(), JSONGrammarState.initial()],
    )
    action_loss = F.cross_entropy(
        action_logits,
        torch.zeros(2, dtype=torch.long),
    )
    typed_json_loss = sum(
        logits.square().mean()
        for logits in candidate.json_typed_logits(
            output.hidden_states[:, -1]
        ).values()
    )
    ast_state = torch.randn_like(output.hidden_states)
    result_state = torch.randn_like(output.hidden_states)
    time = torch.rand(2)
    cfm_output = candidate.cfm.predict(
        ast_state,
        result_state,
        time,
        time,
        output.hidden_states,
    )
    cfm_loss = (
        cfm_output.ast_velocity.square().mean()
        + cfm_output.result_velocity.square().mean()
    )
    generated = candidate.synthesize_expert(
        output.hidden_states,
        output.hidden_states.mean(dim=1),
    )
    factory_loss = generated.output.square().mean()
    (
        token_loss
        + channel_loss
        + action_loss
        + typed_json_loss
        + cfm_loss
        + factory_loss
    ).backward()
    gradients = _prefix_gradients(candidate)

    masks = {
        "syntax": torch.ones(2, 8, dtype=torch.bool),
        "symbol": torch.tensor(
            [
                [True, True, False, False, False, False, False, False],
                [False, False, True, True, False, False, False, False],
            ]
        ),
    }
    with torch.no_grad():
        first = candidate(input_ids, channel_masks=masks).token_logits
        shifted = dict(masks)
        shifted["symbol"] = ~masks["symbol"]
        second = candidate(input_ids, channel_masks=shifted).token_logits
    state_sensitivity = float((first - second).abs().mean())

    json_rejected = False
    try:
        strict_json_loads('{"a": 1, "a": 2}')
    except ValueError:
        json_rejected = True
    with tempfile.TemporaryDirectory(prefix="nlp-synthetic-probe-") as temporary:
        failed_command = run_command(
            ("python", "-c", "raise SystemExit(1)"),
            cwd=Path(temporary),
            timeout_seconds=2.0,
        )
    verifier_rejected = json_rejected and not failed_command.passed
    required_branches = (
        "typed_memory",
        "typed_fusion",
        "cfm",
        "json_head",
        "json_typed_heads",
        "expert_factory",
    )
    branches_reached = all(gradients.get(name, False) for name in required_branches)
    report = {
        "version": "nlp-synthetic-probe-v1",
        "status": (
            "pass"
            if zero_gate_identity
            and branches_reached
            and state_sensitivity > 0.0
            and verifier_rejected
            else "fail"
        ),
        "zero_gate_identity": zero_gate_identity,
        "branches_reached": branches_reached,
        "gradient_prefixes": gradients,
        "state_sensitivity": state_sensitivity,
        "verifier": {
            "duplicate_key_rejected": json_rejected,
            "deterministic_failure_rejected": not failed_command.passed,
            "combined_pass": verifier_rejected,
        },
        "losses": {
            "token": float(token_loss.detach()),
            "channels": float(channel_loss.detach()),
            "json_action": float(action_loss.detach()),
            "json_typed": float(typed_json_loss.detach()),
            "cfm": float(cfm_loss.detach()),
            "factory": float(factory_loss.detach()),
        },
        "oracle_targets_used_at_inference": False,
        "evidence_boundary": "Tiny deterministic mechanism probe only; no capability claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
