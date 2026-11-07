"""Evaluate registered tree-policy checkpoints against the frozen legacy60 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from newera.nlp_flow.components.ast.representation.codec.ast_codec import parse_ast_graph
from newera.nlp_flow.components.ast.representation.codec.ast_codec_io import serialize_ast_graph
from newera.nlp_flow.components.ast.grammar.core.ast_grammar import load_ast_grammar
from newera.nlp_flow.components.research.evaluation.manifests.legacy60_adapter import evaluate_legacy60_prediction
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_policy import PrefixTreePolicy
from newera.nlp_flow.components.generation.policies.prefix.prefix_tree_runtime import generate_prefix_tree
from newera.nlp_flow.components.generation.policies.corpus.real_corpus_policy_data import action_from_record
from newera.nlp_flow.components.generation.trees.actions.tree_action_decoder import decode_actions, render_tree_structure

ROOT = Path(__file__).resolve().parents[4]


def load_validation_rows(path: Path, ordered_case_ids: list[str]) -> list[dict[str, Any]]:
    marker = re.compile(r'"split"\s*:\s*"([^"]+)"')
    wanted = set(ordered_case_ids)
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = marker.search(line)
            if match is None or match.group(1) != "validation":
                continue
            row = json.loads(line)
            if row.get("split") != "validation":
                raise ValueError("split metadata mismatch while loading strict validation data")
            case_id = str(row["case_id"])
            if case_id in wanted:
                rows[case_id] = row
    if set(rows) != wanted:
        raise ValueError("validation corpus does not match the frozen case identity set")
    return [rows[case_id] for case_id in ordered_case_ids]


def _trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = []
    for event in trace:
        if "step" not in event:
            continue
        action = event.get("action")
        decisions.append(
            {
                "step": event["step"],
                "action": None if not isinstance(action, dict) else action.get("type"),
                "error": event.get("error"),
                "heads": [
                    {
                        "head": choice.get("head"),
                        "selected_id": choice.get("selected_id"),
                        "allowed_ids": choice.get("allowed_ids"),
                    }
                    for choice in event.get("choices", [])
                ],
            }
        )
    terminal = next((event for event in trace if event.get("event") == "generation_end"), {})
    return {
        "decision_count": len(decisions),
        "decisions": decisions,
        "terminal_error": terminal.get("error"),
    }


def run_validation(
    *,
    manifest: dict[str, Any],
    checkpoints: dict[int, Path],
    corpus: Path,
    grammar_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    case_ids = list(manifest["ordered_case_ids"])
    rows = load_validation_rows(corpus, case_ids)
    if manifest.get("split") != "validation" or manifest.get("test_content_evaluated") is not False:
        raise ValueError("strict manifest is not validation-only")
    rows_by_seed: list[dict[str, Any]] = []
    evidence = []
    implementation_identity = hashlib.sha256(
        json.dumps(manifest["implementation_sha256"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for seed in sorted(checkpoints):
        checkpoint_path = checkpoints[seed]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("seed", -1)) != seed:
            raise ValueError("checkpoint seed differs from registered seed")
        vocabulary = checkpoint["vocabulary"]
        config = checkpoint["config"]
        model_state = checkpoint["model"]
        maximum_steps = int(model_state["step_position_embedding.weight"].shape[0])
        model = PrefixTreePolicy(
            vocabulary,
            hidden_size=int(config["hidden_size"]),
            layers=int(config["decoder_layers"]),
            heads=int(config["decoder_heads"]),
            max_steps=maximum_steps,
        ).to(device)
        model.load_state_dict(model_state)
        model.eval()
        cases = []
        for row, frozen in zip(rows, manifest["cases"], strict=True):
            if row["case_id"] != frozen["case_id"]:
                raise ValueError("frozen validation order changed")
            if row["target_source_sha256"] != frozen["gold_source_sha256"]:
                raise ValueError(f"frozen source fingerprint differs: {row['case_id']}")
            language = str(row["language"])
            if language != frozen["language"]:
                raise ValueError("validation language differs from frozen manifest")
            grammar = load_ast_grammar(grammar_path, language)
            gold_actions = [action_from_record(item) for item in row["actions"]]
            gold_source = render_tree_structure(decode_actions(grammar, gold_actions))
            gold_graph = parse_ast_graph(gold_source, language)
            gold_graph_hash = hashlib.sha256(serialize_ast_graph(gold_graph).encode()).hexdigest()
            if gold_graph_hash != frozen["gold_graph_sha256"]:
                raise ValueError(f"frozen graph identity differs: {row['case_id']}")
            trace: list[dict[str, Any]] = []
            generated_source, actions, error = generate_prefix_tree(
                model,
                str(row["prompt"]),
                language,
                grammar,
                vocabulary,
                device=device,
                max_steps=maximum_steps,
                trace=trace,
            )
            if generated_source is None:
                gate = {
                    "case_id": row["case_id"],
                    "status": "fail",
                    "strict_passed": False,
                    "checks": {"parser_valid": False},
                    "metrics": None,
                    "prediction_graph_sha256": None,
                }
            else:
                prediction = serialize_ast_graph(parse_ast_graph(generated_source, language))
                gate = evaluate_legacy60_prediction(
                    prediction,
                    case_id=str(row["case_id"]),
                    reference_graph=gold_graph,
                    manifest=manifest,
                )
            cases.append(
                {
                    **gate,
                    "actions": len(actions),
                    "generation_error": error,
                    "live_trace": _trace_summary(trace),
                }
            )
            evidence.append(
                {
                    "case_id": row["case_id"],
                    "seed": seed,
                    "gold_source_sha256": frozen["gold_source_sha256"],
                    "gold_graph_sha256": frozen["gold_graph_sha256"],
                    "parser_grammar_sha256": frozen["parser_grammar_sha256"],
                    "evaluator_sha256": implementation_identity,
                    "parser_valid": generated_source is not None,
                    "strict_passed": bool(gate["strict_passed"]),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rows_by_seed.append(
            {
                "seed": seed,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "passed": sum(bool(case["strict_passed"]) for case in cases),
                "parser_valid": sum(bool(case["checks"].get("parser_valid")) for case in cases),
                "cases": cases,
            }
        )
    return {
        "version": "nlp-strict-ast-validation-v11",
        "status": "passed" if rows_by_seed and all(row["passed"] == len(case_ids) for row in rows_by_seed) else "failed",
        "split": "validation",
        "test_content_evaluated": False,
        "ordered_case_ids": case_ids,
        "seed_ids": sorted(checkpoints),
        "thresholds": manifest["thresholds"],
        "oracle_inputs": [],
        "evidence": evidence,
        "seeds": rows_by_seed,
        "aggregate": {
            "case_count": len(case_ids),
            "seed_count": len(rows_by_seed),
            "strict_passed_by_seed": {str(row["seed"]): row["passed"] for row in rows_by_seed},
            "parser_valid_by_seed": {str(row["seed"]): row["parser_valid"] for row in rows_by_seed},
        },
        "quality_claim": "internal_reconstruction_only_if_all_registered_strict_thresholds_pass",
        "evidence_boundary": "Target-free generation against frozen validation labels; no test rows are parsed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "project/configs/structured_generation/transition_grammar_ast/cfm_eval_fast/nlp_legacy60_eval_manifest.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / ".data/nlp/tree_action_v1/corpus.jsonl")
    parser.add_argument("--grammar", type=Path, default=ROOT / "project/configs/structured_generation/transition_grammar_ast/ast_grammar_transition/nlp_ast_transition_grammar_base.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs/train/nlp_hierarchical_tree_v11")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoints = {seed: args.checkpoint_dir / f"seed_{seed}.pt" for seed in args.seeds}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing registered checkpoints: " + ", ".join(missing))
    result = run_validation(
        manifest=manifest,
        checkpoints=checkpoints,
        corpus=args.corpus,
        grammar_path=args.grammar,
        device=torch.device(args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
