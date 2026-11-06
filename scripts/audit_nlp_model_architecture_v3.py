#!/usr/bin/env python3
"""CPU contract audit for the unified NLP model architecture v3.

Legacy v2 modules remain available as matched control arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from floorflow.nlp_flow.engineering import (
    count_unique_trainable_parameters,
    parameter_breakdown,
)
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from floorflow.nlp_flow.model.benchmark_proxy import (
    REGRESSION_PROXY_TARGETS,
    SCALAR_PROXY_TARGETS,
    TOKEN_PROXY_TARGETS,
)
from floorflow.nlp_flow.structured_generation import TypedTreeGateway

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_model_architecture_v3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/reports/nlp_model_architecture_v3_contract_v1.json",
    )
    args = parser.parse_args()
    args.config = (
        args.config
        if args.config.is_absolute()
        else ROOT / args.config
    )
    args.output = (
        args.output
        if args.output.is_absolute()
        else ROOT / args.output
    )
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = NLPModelConfig(**payload["model"])
    torch.manual_seed(419)
    model = CompactNLPDecoder(config).eval()
    modules = {
        "dynamic_stack": model.dynamic_stack is not None,
        "typed_memory": model.typed_memory is not None,
        "typed_fusion": model.typed_fusion is not None,
        "benchmark_proxy": model.benchmark_proxy is not None,
        "turn_embedding": model.turn_embedding is not None,
        "conversation_head": model.conversation_head is not None,
        "cfm": model.cfm is not None,
        "json_head": model.json_head is not None,
        "json_schema_key_head": (
            model.json_schema_key_head is not None
            if config.enable_json_schema_key_head
            else True
        ),
        "unified_competence": model.unified_competence is not None,
        "legacy_gap_ensemble_disabled": model.gap_ensemble is None,
        "legacy_expert_bank_disabled": model.expert_bank is None,
        "legacy_neuron_disabled": model.neuron is None,
    }
    ids = torch.randint(0, config.vocab_size, (2, 12))
    turns = torch.tensor(
        [[0] * 6 + [1] * 6, [0] * 5 + [1] * 7]
    )
    with torch.no_grad():
        output = model(ids, turn_ids=turns)
    materialized = model.materialize_expert(
        torch.randn(2, config.hidden_size)
    )
    materialized_output = model.apply_materialized_expert(
        output.hidden_states,
        materialized,
    )
    typed_tree = TypedTreeGateway.from_ast(
        "def add(a, b):\n    return a + b\n",
        "python",
    )
    emergence = model.emergence(output.hidden_states)
    checks = {
        **modules,
        "finite_token_logits": bool(torch.isfinite(output.token_logits).all()),
        "benchmark_proxy_emitted": output.benchmark_proxy is not None,
        "benchmark_proxy_heads_complete": (
            output.benchmark_proxy is not None
            and set(output.benchmark_proxy.scalar_logits)
            == set(SCALAR_PROXY_TARGETS)
            and set(output.benchmark_proxy.token_logits)
            == set(TOKEN_PROXY_TARGETS)
            and set(output.benchmark_proxy.regression_logits)
            == set(REGRESSION_PROXY_TARGETS)
        ),
        "conversation_controls_emitted": output.conversation_logits is not None,
        "growth_diagnostics_emitted": output.growth_diagnostics is not None,
        "gap_logits_emitted": output.gap_logits is not None,
        "dynamic_diagnostics_emitted": output.dynamic_diagnostics is not None,
        "tree_mode_emitted": output.tree_mode_logits is not None,
        "sparse_competence_compute": (
            output.gap_diagnostics is not None
            and float(output.gap_diagnostics["competence_token_calls"])
            < float(output.gap_diagnostics["dense_competence_token_calls"])
        ),
        "unified_compute_budget_emitted": (
            output.gap_diagnostics is not None
            and "mean_compute_budget" in output.gap_diagnostics
        ),
        "generated_e_materialization": materialized_output.shape
        == output.hidden_states.shape,
        "runtime_synthesis_signal": output.runtime_synthesis_score is not None,
        "typed_tree_gateway": typed_tree.tree.domain == "ast",
        "emergence_score_emitted": (
            torch.isfinite(emergence.emergence_score)
            and 0.0 <= float(emergence.emergence_score.detach()) <= 1.0
        ),
    }
    report = {
        "version": f"{payload.get('version', 'nlp-model-architecture')}-contract-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "config": str(args.config.relative_to(ROOT)),
        "device": "cpu_contract_only",
        "checks": checks,
        "parameters": count_unique_trainable_parameters(model),
        "parameter_breakdown": parameter_breakdown(model),
        "gpu_experiments": "deferred",
        "evidence_boundary": (
            "Network construction, forward wiring, and shape contract only; "
            "no capability, convergence, or benchmark evidence."
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
