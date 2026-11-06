#!/usr/bin/env python3
"""Prepare deterministic external benchmark samples from a frozen finalist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from floorflow.nlp_flow.engineering import load_nlp_checkpoint, restore_nlp_checkpoint
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_config(payload: dict) -> NLPModelConfig:
    model = payload["model"]
    candidate = payload["arm"] == "candidate"
    return NLPModelConfig(
        vocab_size=int(model["vocab_size"]),
        hidden_size=int(model.get("hidden_size", 160)),
        num_layers=int(model.get("num_layers", 3)),
        num_heads=int(model.get("num_heads", 4)),
        num_kv_heads=int(model.get("num_kv_heads", 2)),
        intermediate_size=int(model.get("intermediate_size", 400)),
        max_seq_len=int(model["max_context_tokens"]),
        channel_size=int(model.get("channel_size", 32)),
        enable_typed_channels=candidate,
        enable_typed_fusion=candidate,
        enable_two_stage_cfm=candidate,
        enable_json_action_head=candidate,
        enable_expert_factory=candidate,
        enable_emergence_objective=bool(
            model.get("enable_emergence_objective", False)
        ),
        emergence_basis_count=int(model.get("emergence_basis_count", 32)),
        emergence_active_basis=int(model.get("emergence_active_basis", 4)),
        enable_dynamic_stack=bool(model.get("enable_dynamic_stack", False)),
        dynamic_max_steps=int(model.get("dynamic_max_steps", 4)),
        dynamic_parallel_lanes=int(model.get("dynamic_parallel_lanes", 2)),
        dynamic_lane_rank=int(model.get("dynamic_lane_rank", 8)),
    )


@torch.no_grad()
def _generate(
    model: CompactNLPDecoder,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("tokenizer needs <bos> and <eos>")
    context_limit = model.config.max_seq_len - 1
    sequence = [bos, *ids[-context_limit:]]
    generated = []
    for _step in range(max_new_tokens):
        if len(sequence) >= context_limit:
            sequence = [bos, *sequence[-(context_limit - 1):]]
        logits = model(torch.tensor([sequence], dtype=torch.long)).token_logits[:, -1]
        token = int(logits.argmax(dim=-1))
        if token == eos:
            break
        sequence.append(token)
        generated.append(token)
    return tokenizer.decode(generated)


def _case_fields(payload: dict) -> dict:
    return payload.get("fields", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/nlp_external_eval_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/external_eval/nlp_finalist_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/reports/nlp_external_eval_preflight_v1.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    args.report = args.report.resolve()
    eval_config = json.loads(args.config.read_text(encoding="utf-8"))
    benchmark_manifest = json.loads(
        (ROOT / eval_config["benchmark_manifest"]).read_text(encoding="utf-8")
    )
    checkpoint = load_nlp_checkpoint(args.checkpoint, map_location="cpu")
    spec_payload = checkpoint["config"]
    model = CompactNLPDecoder(_model_config(spec_payload))
    restore_nlp_checkpoint(checkpoint, model=model)
    model.eval()
    tokenizer = Tokenizer.from_file(str(ROOT / eval_config["tokenizer"]))
    limit = (
        int(eval_config["smoke_cases_per_suite"])
        if args.limit is None
        else args.limit
    )
    max_new_tokens = (
        int(eval_config["primary_sampling"]["max_new_tokens"])
        if args.max_new_tokens is None
        else args.max_new_tokens
    )
    if limit <= 0 or max_new_tokens <= 0:
        raise ValueError("limit and max-new-tokens must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    suites = {}
    for benchmark in benchmark_manifest["benchmarks"]:
        benchmark_id = benchmark["id"]
        source = ROOT / eval_config["benchmark_root"] / f"{benchmark_id}.jsonl"
        if not source.exists():
            suites[benchmark_id] = {"status": "missing_source", "rows": 0}
            continue
        rows = []
        with source.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                payload = json.loads(line)
                fields = _case_fields(payload)
                prompt = str(fields.get("prompt", ""))
                case_id = str(payload.get("id", fields.get("task_id", index)))
                completion = _generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                )
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "case_id": case_id,
                        "prompt": prompt,
                        "completion": completion,
                        "samples": 1,
                        "retries": 0,
                    }
                )
        samples_path = args.output_root / f"{benchmark_id}.jsonl"
        samples_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        suites[benchmark_id] = {
            "status": "samples_generated",
            "rows": len(rows),
            "samples_path": str(samples_path.relative_to(ROOT)),
            "samples_sha256": _sha256(samples_path),
            "correctness_executed": False,
        }
    all_generated = all(
        row["status"] == "samples_generated"
        for row in suites.values()
    )
    report = {
        "version": "nlp-external-eval-preflight-v1",
        "status": "samples_ready" if all_generated else "blocked",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer_sha256": _sha256(ROOT / eval_config["tokenizer"]),
        "primary_sampling": eval_config["primary_sampling"],
        "smoke_limit": limit,
        "max_new_tokens_used": max_new_tokens,
        "best_of_n_used": False,
        "retry_count": 0,
        "suites": suites,
        "correctness_executed": False,
        "runner_manifest": eval_config["runner_manifest"],
        "evidence_boundary": (
            "Generated samples only; benchmark correctness metrics are absent and "
            "no coding or SOTA claim is allowed."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
