from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    path = ROOT / "scripts/generate_nlp_conversation_v3.py"
    spec = importlib.util.spec_from_file_location("conversation_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load conversation generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_rows_cover_json_code_and_multiturn() -> None:
    rows = _generator()._rows(0, 12)
    json_rows = [
        row for row in rows if row["response"].startswith("{")
    ]
    python_rows = [
        row
        for row in rows
        if row["response"].startswith("def ")
    ]
    multi_turn = [row for row in rows if row.get("role_formatted")]

    assert json_rows
    assert all(isinstance(json.loads(row["response"]), dict) for row in json_rows)
    assert python_rows
    assert all(
        isinstance(ast.parse(row["response"]), ast.AST)
        for row in python_rows
    )
    assert multi_turn
    assert any("User: Rename" in row["prompt"] for row in multi_turn)


def test_written_rows_have_unique_prompt_response_pairs(tmp_path) -> None:
    rows = _generator()._rows(100, 125)
    output = tmp_path / "rows.jsonl"
    _hash, count = _generator()._write(output, rows)
    written = [
        json.loads(line) for line in output.read_text().splitlines()
    ]
    pairs = [(row["prompt"], row["response"]) for row in written]

    assert len(pairs) == len(set(pairs))
    assert count == len(written)
    assert all(row["surface_valid"] is True for row in written)
