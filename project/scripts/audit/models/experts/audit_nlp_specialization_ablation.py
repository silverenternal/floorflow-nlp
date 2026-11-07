#!/usr/bin/env python3
"""Audit matched specialization objectives without claiming capability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from newera.nlp_flow.components.research.training.objectives.code_objectives import (
    FeedbackBudget,
    VerifierSignal,
    encode_verifier_feedback,
    make_diff_example,
    make_fim_example,
)
from newera.nlp_flow.engineering import count_unique_trainable_parameters
from newera.nlp_flow.model import CompactNLPDecoder, NLPModelConfig

ROOT = Path(__file__).resolve().parents[5]


def _model() -> tuple[CompactNLPDecoder, torch.nn.Linear]:
    config = NLPModelConfig(
        vocab_size=8192,
        hidden_size=224,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        max_seq_len=128,
        enable_typed_channels=True,
        enable_typed_fusion=True,
        enable_two_stage_cfm=True,
        enable_json_action_head=True,
    )
    model = CompactNLPDecoder(config)
    verifier_head = torch.nn.Linear(config.hidden_size, 4)
    return model, verifier_head


def _first_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(next(handle))


def _code_row(tokenizer: Tokenizer) -> list[int]:
    for path in (
        ROOT / ".data/nlp/mixed_v1/corpus_v1/train.jsonl",
        ROOT / ".data/nlp/p5m_v2/delta_train.jsonl",
    ):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("family") == "code":
                    return tokenizer.encode(
                        str(row["text"]),
                        add_special_tokens=False,
                    ).ids[:96]
    raise RuntimeError("no code row found")


def _language_proxy() -> dict:
    path = ROOT / "outputs/reports/nlp_growth_frontier_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("language_retention_relative_family_loss", {})


def _fim_loss(model, tokenizer, ids: list[int]) -> torch.Tensor:
    example = make_fim_example(
        ids,
        seed=20261101,
        prefix_sentinel=1,
        middle_sentinel=2,
        suffix_sentinel=3,
    )
    logits = model(example.input_ids[None]).token_logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, 8192),
        example.labels[1:].reshape(-1),
        ignore_index=-100,
    )


def _diff_loss(model, tokenizer, ids: list[int]) -> torch.Tensor:
    original = tokenizer.decode(ids)
    patched = original.replace("return", "return ", 1)
    if original == patched:
        patched = original + "\n"
    example = make_diff_example(original, patched)
    patch_ids = tokenizer.encode(example.patch_text, add_special_tokens=False).ids[:31]
    sequence = [*ids[:96], *patch_ids]
    labels = [-100] * len(ids[:96]) + patch_ids
    logits = model(torch.tensor([sequence])).token_logits
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, 8192),
        torch.tensor(labels[1:]),
        ignore_index=-100,
    )


def _verifier_loss(model, verifier_head, ids: list[int]) -> torch.Tensor:
    feedback = encode_verifier_feedback(
        VerifierSignal(
            exit_code=1,
            failed_test_count=2,
            timed_out=False,
            stdout_tail="AssertionError",
        ),
        buckets=4,
    )
    hidden = model(torch.tensor([ids[:64]])).hidden_states[:, -1]
    logits = verifier_head(hidden)
    target = (feedback[:3].sum() % 4).reshape(1)
    return F.cross_entropy(logits, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reports/nlp_specialization_ablation_v1.json",
    )
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(
        str(ROOT / ".data/nlp/tokenizer_v1/tokenizer.json")
    )
    ids = _code_row(tokenizer)
    torch.manual_seed(73)
    model, verifier_head = _model()
    parameters = count_unique_trainable_parameters(model) + count_unique_trainable_parameters(
        verifier_head
    )
    lm_loss = F.cross_entropy(
        model(torch.tensor([ids])).token_logits[:, :-1].reshape(-1, 8192),
        torch.tensor(ids[1:]),
    )
    objectives = {
        "C0-lm": lm_loss,
        "C1-fim": _fim_loss(model, tokenizer, ids),
        "C2-diff": _diff_loss(model, tokenizer, ids),
        "C3-verifier": _verifier_loss(model, verifier_head, ids),
    }
    objectives["C4-all"] = sum(objectives.values())
    results = {}
    budgets = {}
    for name, loss in objectives.items():
        model.zero_grad(set_to_none=True)
        verifier_head.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        budget = FeedbackBudget(max_attempts=1)
        budget.record_attempt()
        results[name] = {
            "loss": float(loss.detach()),
            "finite": bool(torch.isfinite(loss)),
            "parameters": parameters,
            "model_gradient": any(
                parameter.grad is not None
                for parameter in model.parameters()
            ),
            "verifier_head_gradient": verifier_head.weight.grad is not None,
        }
        budgets[name] = {
            "max_attempts": budget.max_attempts,
            "attempts": budget.attempts,
            "primary_retry_count": budget.primary_retry_count,
        }
    checks = {
        "all_parameters_matched": len({row["parameters"] for row in results.values()}) == 1,
        "all_losses_finite": all(row["finite"] for row in results.values()),
        "all_objectives_reach_model": all(
            row["model_gradient"] for row in results.values()
        ),
        "verifier_head_reached": results["C3-verifier"]["verifier_head_gradient"],
        "zero_primary_retries": all(
            row["primary_retry_count"] == 0 for row in budgets.values()
        ),
        "language_retention_visible": bool(_language_proxy()),
    }
    report = {
        "version": "nlp-specialization-ablation-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "parameters": parameters,
        "objectives": results,
        "retry_budgets": budgets,
        "language_retention_proxy": _language_proxy(),
        "best_of_n_primary": False,
        "capability_measured": False,
        "evidence_boundary": (
            "Objective isolation and leakage/retry contracts only; no specialization "
            "capability or language-retention improvement is claimed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
