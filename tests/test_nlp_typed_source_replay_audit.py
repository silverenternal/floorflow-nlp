from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from floorflow.nlp_flow.ast_codec import parse_ast_graph
from floorflow.nlp_flow.ast_codec_io import serialize_ast_graph
from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.real_corpus_action_data import action_record
from floorflow.nlp_flow.tree_action_decoder import trace_teacher_actions

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_nlp_typed_source_replay",
    ROOT / "scripts/audit_nlp_typed_source_replay.py",
)
assert SPEC is not None and SPEC.loader is not None
AUDIT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT_MODULE)
audit_train_replay = AUDIT_MODULE.audit_train_replay


def test_train_only_replay_report_has_explicit_unsupported_and_control_metrics(tmp_path) -> None:
    source = "# π comment\ndef add(left, right):\n    return left + right\n"
    graph = parse_ast_graph(source, "python")
    grammar_source = ROOT / "configs/nlp_ast_transition_grammar_v2.json"
    grammar_path = tmp_path / "grammar.json"
    grammar_path.write_bytes(grammar_source.read_bytes())
    grammar = load_ast_grammar(grammar_path, "python")
    ast_payload = json.loads(serialize_ast_graph(graph))
    actions = [action_record(value) for value in trace_teacher_actions(grammar, ast_payload)]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps(
            {
                "case_id": "python-train-fixture",
                "language": "python",
                "split": "train",
                "target_source_sha256": graph.source_sha256,
                "actions": actions,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "training_action_corpus": "train.jsonl",
        "expected_training_action_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "expected_training_rows": 1,
        "transition_grammar": "grammar.json",
        "expected_transition_grammar_sha256": hashlib.sha256(grammar_path.read_bytes()).hexdigest(),
        "languages": ["python"],
        "arms": [{"id": name} for name in ("oracle_lexical", "flat_source", "typed_projection")],
        "metric_schema": [{"name": "canonical_replay_rate"}, {"name": "trivia_byte_exact_rate"}],
        "evidence_boundary": "train fixture only",
    }
    report = audit_train_replay(config, root=tmp_path)
    assert report["status"] == "pass_train_only_replay"
    assert report["language_audit"]["python"]["canonical_replays"] == 1
    assert report["language_audit"]["python"]["source_byte_exact"] == 1
    assert report["language_audit"]["python"]["trivia_byte_exact"] == 1
    assert report["unsupported_rows"] == []
    assert report["matched_dry_run"]["quality_claim"] is False
    assert [arm["id"] for arm in report["matched_dry_run"]["arms"]] == [
        "oracle_lexical",
        "flat_source",
        "typed_projection",
    ]
    assert all(all(value is None for value in arm["metrics"].values()) for arm in report["matched_dry_run"]["arms"])
