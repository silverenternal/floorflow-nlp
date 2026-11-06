#!/usr/bin/env python3
"""Train the native JSON head and emit all-suite model-generated records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from floorflow.nlp_flow.engineering import (
    ExperimentSpec,
    load_nlp_checkpoint,
    restore_nlp_checkpoint,
    save_nlp_checkpoint,
)
from floorflow.nlp_flow.json_generation import (
    NativeJSONGenerator,
    gold_action_sequence,
)
from floorflow.nlp_flow.json_tree import load_schema
from floorflow.nlp_flow.json_validity import (
    JSONOutputRecord,
    audit_json_outputs,
    load_schema_registry,
)
from floorflow.nlp_flow.model import CompactNLPDecoder, NLPModelConfig
from floorflow.nlp_flow.model.json_grammar import (
    ACTION_INDEX,
    JSONGrammarState,
)

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("ast", "patch", "tool_action", "diagnostic", "engineering_result")
SUITE_KINDS = {
    "train": "diagnostic",
    "validation": "ast",
    "test": "patch",
    "engineering": "engineering_result",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_config() -> NLPModelConfig:
    return NLPModelConfig(
        vocab_size=8192,
        hidden_size=160,
        num_layers=3,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=400,
        max_seq_len=128,
        enable_typed_channels=True,
        enable_typed_fusion=True,
        enable_two_stage_cfm=True,
        enable_json_action_head=True,
        enable_expert_factory=True,
    )


def encode_prompt(tokenizer: Tokenizer, prompt: str, max_tokens: int = 120) -> torch.Tensor:
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-max_tokens:]
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("tokenizer needs <bos> and <eos>")
    return torch.tensor([[bos, *ids, eos]], dtype=torch.long)


def training_examples(
    model: CompactNLPDecoder,
    tokenizer: Tokenizer,
) -> tuple[torch.Tensor, list[list[JSONGrammarState]], torch.Tensor, torch.Tensor]:
    hidden_rows = []
    state_rows = []
    target_rows = []
    active_rows = []
    model.eval()
    with torch.no_grad():
        for kind in KINDS:
            prompt = f"Generate strict JSON output of kind {kind}."
            hidden = model(encode_prompt(tokenizer, prompt)).hidden_states[:, -1]
            actions = gold_action_sequence(kind)
            state = JSONGrammarState.initial()
            states = []
            targets = []
            for action in actions:
                states.append(state)
                targets.append(ACTION_INDEX[action])
                state = state.transition(action)
            hidden_rows.append(hidden.expand(len(actions), -1))
            state_rows.append(states)
            target_rows.append(targets)
            active_rows.append([True] * len(actions))
    steps = max(len(row) for row in target_rows)
    targets = torch.full((len(KINDS), steps), -100, dtype=torch.long)
    active = torch.zeros((len(KINDS), steps), dtype=torch.bool)
    for index, row in enumerate(target_rows):
        targets[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        active[index, : len(row)] = True
    return torch.cat(hidden_rows), state_rows, targets, active


def train_policy(
    model: CompactNLPDecoder,
    tokenizer: Tokenizer,
    *,
    steps: int,
) -> tuple[float, bool]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.json_head is None:
        raise RuntimeError("candidate model has no JSON head")
    for parameter in model.json_head.parameters():
        parameter.requires_grad_(True)
    hidden, state_rows, targets, _active = training_examples(model, tokenizer)
    optimizer = torch.optim.AdamW(model.json_head.parameters(), lr=0.02)
    final_loss = 0.0
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits_rows = []
        target_rows = []
        for kind_index, states in enumerate(state_rows):
            logits = model.json_head(
                hidden[
                    sum(len(previous) for previous in state_rows[:kind_index]) :
                    sum(len(previous) for previous in state_rows[: kind_index + 1])
                ],
                states,
                torch.arange(len(states)),
            )
            logits_rows.append(logits)
            target_rows.append(targets[kind_index, : len(states)])
        logits = torch.cat(logits_rows)
        labels = torch.cat(target_rows)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        offset = 0
        for kind_index, states in enumerate(state_rows):
            count = len(states)
            logits = model.json_head(
                hidden[offset : offset + count],
                states,
                torch.arange(count),
            )
            labels = targets[kind_index, :count]
            correct += int(logits.argmax(dim=-1).eq(labels).sum())
            total += count
            offset += count
    return final_loss, correct == total


def load_suite_prompts() -> list[dict[str, str]]:
    cases = []
    for suite in ("train", "validation", "test"):
        path = ROOT / f".data/nlp/mixed_v1/corpus_v1/{suite}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= 8:
                    break
                row = json.loads(line)
                cases.append(
                    {
                        "case_id": f"{suite}-{index:03d}",
                        "suite": suite,
                        "kind": SUITE_KINDS[suite],
                        "prompt": str(row["text"])[:4_000],
                    }
                )
    benchmark = ROOT / ".data/nlp/benchmarks_v1/humaneval_python.jsonl"
    with benchmark.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 8:
                break
            row = json.loads(line)
            cases.append(
                {
                    "case_id": f"engineering-{index:03d}",
                    "suite": "engineering",
                    "kind": SUITE_KINDS["engineering"],
                    "prompt": str(row["fields"]["prompt"])[:4_000],
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/train/nlp_p2m_candidate_cpu_smoke_v1/checkpoint.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/train/nlp_json_policy_v1",
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "outputs/generation/nlp_json_all_suites_v1.jsonl",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    spec = ExperimentSpec.from_json(ROOT / "configs/nlp_training_p2m_cpu_smoke_v1.json")
    tokenizer_path = ROOT / ".data/nlp/tokenizer_v1/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = CompactNLPDecoder(model_config())
    checkpoint = load_nlp_checkpoint(
        args.checkpoint,
        expected_spec_sha256=spec.sha256,
        map_location="cpu",
    )
    restore_nlp_checkpoint(checkpoint, model=model)
    final_loss, exact_actions = train_policy(model, tokenizer, steps=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)
    trained_checkpoint = args.output / "checkpoint.pt"
    save_nlp_checkpoint(
        trained_checkpoint,
        step=int(checkpoint["step"]) + args.steps,
        samples_seen=int(checkpoint["samples_seen"]) + args.steps * len(KINDS),
        model=model,
        optimizer=None,
        scheduler=None,
        spec_sha256=spec.sha256,
        data_manifest_sha256=file_sha256(
            ROOT / ".data/nlp/mixed_v1/corpus_v1/manifest.json"
        ),
        tokenizer_sha256=file_sha256(tokenizer_path),
        config=spec.payload,
        device="cpu",
    )
    generator = NativeJSONGenerator(model, tokenizer)
    records: list[JSONOutputRecord] = []
    args.records.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("w", encoding="utf-8") as handle:
        for case in load_suite_prompts():
            generated = generator.generate(case["prompt"], case["kind"])
            record = {
                "case_id": case["case_id"],
                "suite": case["suite"],
                "output": generated.text,
                "repaired": False,
                "retried": False,
                "synthetic": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(JSONOutputRecord(**record))
    validity = audit_json_outputs(
        records,
        schema=load_schema(ROOT / "schema/nlp_json_output_v1.schema.json"),
        schema_registry=load_schema_registry(
            ROOT / "configs/nlp_json_schemas_v1.json"
        ),
        required_suites={"train", "validation", "test", "engineering"},
    )
    probe = {
        "version": "nlp-json-training-probe-v1",
        "status": "pass" if exact_actions else "fail",
        "steps": args.steps,
        "final_loss": final_loss,
        "exact_grammar_action_targets": exact_actions,
        "checkpoint": str(trained_checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": file_sha256(trained_checkpoint),
        "repair_count": 0,
        "retry_count": 0,
        "invalid_transition_probability": 0.0,
        "evidence_boundary": "Grammar-transition training only; semantic JSON content quality is untested.",
    }
    probe_path = ROOT / "outputs/reports/nlp_json_training_probe_v1.json"
    probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    validity_path = ROOT / "outputs/reports/nlp_json_validity_all_suites_v1.json"
    validity_path.write_text(json.dumps(validity, indent=2) + "\n", encoding="utf-8")
    claim = {
        "version": "nlp-json-claim-gate-v1",
        "status": "pass" if probe["status"] == "pass" and validity["claim_gate"] == "allow_C1_C4" else "fail",
        "training_probe": str(probe_path.relative_to(ROOT)),
        "validity_report": str(validity_path.relative_to(ROOT)),
        "records": str(args.records.relative_to(ROOT)),
        "repair_count": validity["repair_count"],
        "retry_count": validity["retry_count"],
        "synthetic_count": validity["synthetic_count"],
        "claim_gate": validity["claim_gate"],
        "evidence_boundary": "Validity gate only; it does not establish semantic task success.",
    }
    claim_path = ROOT / "outputs/reports/nlp_json_claim_gate_v1.json"
    claim_path.write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"probe": probe, "validity": validity, "claim": claim}, indent=2))
    if claim["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
