import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "legacy60_manifest", ROOT / "scripts/freeze_nlp_legacy60_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifest_loader_skips_nonvalidation_rows_before_json_decode(tmp_path: Path) -> None:
    corpus = tmp_path / "actions.jsonl"
    corpus.write_text(
        '{"split":"train","target":"do not decode"}\n'
        "not json test content\n"
        + json.dumps({"split": "validation", "case_id": "valid"})
        + "\n",
        encoding="utf-8",
    )
    assert MODULE.load_validation_rows(corpus) == [{"split": "validation", "case_id": "valid"}]


def test_manifest_freezer_rejects_changed_or_missing_suite_size() -> None:
    with pytest.raises(ValueError, match="exactly 60"):
        MODULE.freeze_manifest([], grammar_path=ROOT / "configs/nlp_ast_transition_grammar_v1.json")
