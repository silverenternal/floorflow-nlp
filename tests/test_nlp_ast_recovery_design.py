import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nlp_recovery_design", ROOT / "scripts/build_nlp_ast_recovery_design.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build_decision = module.build_decision


def _evidence(tmp_path: Path) -> dict[str, Path]:
    cases = [
        {"case_id": str(index), "mask_exclusions": [], "replay_error": None, "terminated": True} for index in range(60)
    ]
    cases[0]["mask_exclusions"] = [{"mask": {"reasons": ["parent_not_in_active_scope"]}}]
    rows = [{"case_id": str(index), "observed_category": "no_candidate"} for index in range(60)]
    payloads = {
        "reachability": {
            "split": "validation",
            "test_evaluated": False,
            "cases": cases,
            "case_count": 60,
            "reasons": {"parent_not_in_active_scope": 1},
        },
        "failures": {
            "split": "validation",
            "test_read": False,
            "arms": {
                key: {"cases": rows, "observed_categories": {"no_candidate": 60}}
                for key in ("event_mlp", "prefix_transformer")
            },
        },
        "oracle_fixture": {"status": "pass", "split": "synthetic_fixture_only"},
        "comparison": {"status": "rejected", "selected_policy": None},
    }
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        paths[name] = path
    return paths


def test_design_decision_tracks_measured_evidence_and_limits(tmp_path: Path) -> None:
    result = build_decision(_evidence(tmp_path))
    assert result["teacher_prefix"]["occurrences"]["parent_not_in_active_scope"] == 1
    assert result["teacher_prefix"]["affected_cases"]["parent_not_in_active_scope"] == 1
    assert result["status"] == "implementation_decision_only"
    assert all(decision["control"] and decision["limitation"] for decision in result["decisions"])
    assert len(result["evidence_sha256"]) == 4


def test_design_decision_rejects_inconsistent_or_nonvalidation_evidence(tmp_path: Path) -> None:
    paths = _evidence(tmp_path)
    reachability = json.loads(paths["reachability"].read_text())
    reachability["reasons"] = {}
    paths["reachability"].write_text(json.dumps(reachability))
    with pytest.raises(ValueError, match="aggregate"):
        build_decision(paths)
    reachability["reasons"] = {"parent_not_in_active_scope": 1}
    reachability["test_evaluated"] = True
    paths["reachability"].write_text(json.dumps(reachability))
    with pytest.raises(ValueError, match="validation-only"):
        build_decision(paths)
