from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location(
    "preflight_nlp_implementation", ROOT / "project/scripts/experiments/research/preflight_nlp_implementation.py"
)
preflight_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(preflight_module)


def _fixture(root: Path, *, dependency_status: str = "blocked") -> Path:
    files = {
        "ledger.json": json.dumps({"tasks": [{"id": "ready", "status": dependency_status}]}),
        "train.jsonl": '{"split":"train"}\n',
        "validation.json": json.dumps({"test_content_evaluated": False, "training_action_replay_status": "pass"}),
        "grammar.json": "{}\n",
        "queue.json": json.dumps({"training_allowed_now": False}),
    }
    for name, content in files.items():
        (root / name).write_text(content, encoding="utf-8")
    from hashlib import sha256

    config = {
        "preflight": {
            "task_ledger": "ledger.json",
            "required_dependencies": ["ready"],
            "train_actions": "train.jsonl",
            "train_actions_sha256": sha256(files["train.jsonl"].encode()).hexdigest(),
            "validation_manifest": "validation.json",
            "validation_manifest_sha256": sha256(files["validation.json"].encode()).hexdigest(),
            "grammar": "grammar.json",
            "grammar_sha256": sha256(files["grammar.json"].encode()).hexdigest(),
            "seeds": [17, 29],
            "gpu_count": 1,
            "gpu_lock": "run.lock",
            "queue": "queue.json",
        }
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_preflight_checks_hashes_and_keeps_blocked_dependencies_closed(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    result = preflight_module.build_preflight(config_path, root=tmp_path)

    assert result["status"] == "pass_blocked_until_dependencies"
    assert result["launch_allowed"] is False
    assert result["optimizer_started"] is False
    assert result["gpu_allocated"] is False


def test_preflight_detects_busy_gpu_lock_and_changed_hash(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path, dependency_status="passed")
    (tmp_path / "run.lock").write_text("busy", encoding="utf-8")
    (tmp_path / "train.jsonl").write_text("changed", encoding="utf-8")

    result = preflight_module.build_preflight(config_path, root=tmp_path)

    assert result["status"] == "fail"
    assert result["launch_allowed"] is False
    assert any("fingerprint changed" in item for item in result["errors"])
    assert any("lock is already held" in item for item in result["errors"])
