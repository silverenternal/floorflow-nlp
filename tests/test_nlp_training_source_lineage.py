import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from floorflow.nlp_flow.ast_grammar import load_ast_grammar
from floorflow.nlp_flow.real_corpus_action_data import action_record, iter_declaration_fragments
from floorflow.nlp_flow.training_source_lineage import audit_training_sources
from floorflow.nlp_flow.tree_action_decoder import trace_teacher_actions

ROOT = Path(__file__).resolve().parents[1]


def test_train_metadata_filter_accepts_compact_and_spaced_json(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "audit_train_lineage", ROOT / "scripts/audit_nlp_training_source_lineage.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "rows.jsonl"
    path.write_text('{"split":"train","id":1}\n{"split": "train", "id": 2}\n{"split":"test","id":3}\n')
    assert [row["id"] for row in module.load_train(path)] == [1, 2]


def test_training_lineage_verifies_file_bytes_and_action_replay(tmp_path: Path) -> None:
    source = "def f(x):\n    return x + 1\n"
    path = tmp_path / "sample.py"
    path.write_text(source)
    parent = {
        "case_id": "parent",
        "language": "python",
        "split": "train",
        "source_path": "sample.py",
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    config = json.loads((ROOT / "configs/nlp_tree_action_corpus_v1.json").read_text())["fragments"]
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", "python")
    fragment = iter_declaration_fragments([parent], config)[0]
    records = [action_record(action) for action in trace_teacher_actions(grammar, json.loads(fragment["ast"]))]
    action_row = {
        "case_id": fragment["case_id"],
        "language": "python",
        "split": "train",
        "actions": records,
        "target_source_sha256": fragment["source_sha256"],
    }
    kwargs = {"root": tmp_path, "source_root": tmp_path, "fragment_config": config, "grammars": {"python": grammar}}
    result = audit_training_sources([parent], [action_row], **kwargs)
    assert result["status"] == "pass"
    assert result["verified_parents"] == result["verified_fragments"] == 1
    changed = {**action_row, "target_source_sha256": "0" * 64}
    assert audit_training_sources([parent], [changed], **kwargs)["status"] == "partial_or_failed"
    path.write_text("modified source")
    assert audit_training_sources([parent], [action_row], **kwargs)["verified_parents"] == 0
    with pytest.raises(ValueError, match="only train split"):
        audit_training_sources([{**parent, "split": "test"}], [action_row], **kwargs)
