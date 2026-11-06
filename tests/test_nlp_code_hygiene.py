from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _audit_module():
    path = ROOT / "scripts/audit_nlp_code_sizes.py"
    spec = importlib.util.spec_from_file_location("nlp_code_size_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load code-size audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_party_code_files_stay_below_six_hundred_lines() -> None:
    module = _audit_module()
    violations = [
        row
        for row in module.code_file_sizes(ROOT)
        if int(row["lines"]) > module.MAX_LINES
    ]

    assert violations == []
