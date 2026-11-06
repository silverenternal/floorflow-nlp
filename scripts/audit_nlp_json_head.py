#!/usr/bin/env python3
"""Audit native JSON action/schema/typed-head contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.model.json_grammar import (
    ACTIONS,
    JSONActionHead,
    JSONGrammarState,
    JSONSchemaKeyHead,
    JSONTypedHeads,
    SchemaGrammarState,
    SchemaNode,
    json_action_loss,
    schema_key_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_json_head_contract_v1.json",
    )
    args = parser.parse_args()
    torch.manual_seed(7)
    hidden = torch.randn(1, 16)
    state = JSONGrammarState.initial()
    action_logits = JSONActionHead(16)(hidden, [state])
    typed = JSONTypedHeads(16)(hidden)
    schema = SchemaNode(
        "object",
        required=("answer",),
        properties={
            "answer": SchemaNode("string"),
            "explanation": SchemaNode("string"),
        },
    )
    schema_state = SchemaGrammarState.initial(schema).transition("start_object")
    schema_actions = JSONActionHead(16)(hidden, [schema_state])
    key_head = JSONSchemaKeyHead(16)
    key_logits = key_head(hidden, [schema_state])
    key_loss = schema_key_loss(key_logits, [schema_state], ["answer"])
    sequence = ["start_object", "key", "colon", "string", "end_object"]
    targets = torch.tensor([[ACTIONS.index(action) for action in sequence]])
    training_logits = torch.randn(1, len(sequence), len(ACTIONS), requires_grad=True)
    loss = json_action_loss(training_logits, targets)
    loss.backward()
    invalid_comma = bool(torch.isneginf(action_logits[0, ACTIONS.index("comma")]))
    invalid_early_close = bool(
        torch.isneginf(schema_actions[0, ACTIONS.index("end_object")])
    )
    report = {
        "version": "nlp-json-head-contract-v1",
        "status": (
            "pass"
            if (
                invalid_comma
                and invalid_early_close
                and torch.isfinite(key_loss)
                and training_logits.grad is not None
            )
            else "fail"
        ),
        "initial_allowed_actions": list(state.allowed_actions()),
        "invalid_comma_probability_zero": invalid_comma,
        "required_field_early_close_masked": invalid_early_close,
        "allowed_schema_keys": list(schema_state.allowed_keys()),
        "schema_key_shapes": [list(value.shape) for value in key_logits],
        "schema_key_loss": float(key_loss.detach()),
        "typed_heads": list(typed),
        "typed_head_shapes": {name: list(value.shape) for name, value in typed.items()},
        "training_loss": float(loss.detach()),
        "training_gradient": training_logits.grad is not None,
        "repair_path": False,
        "retry_path": False,
        "evidence_boundary": "Head contract only; no trained JSON quality result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
