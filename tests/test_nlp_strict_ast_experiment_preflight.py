from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/preflight_nlp_strict_ast_experiment.py"
    spec = importlib.util.spec_from_file_location("strict_ast_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixtures(tmp_path: Path):
    artifact = tmp_path / "frozen.json"
    artifact.write_text("{}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config = {
        "task_id": "NLP-068-STRICT-AST-CLOSURE",
        "device_index": 1,
        "minimum_free_memory_mib": 24000,
        "seed_set": [17, 29, 43],
        "qualification": {"per_case_thresholds": "configs/nlp_legacy60_eval_manifest_v11.json"},
        "execution": {"launch_enabled": False},
    }
    freeze = {
        "status": "passed",
        "artifacts": {"frozen.json": digest},
        "fingerprints": {"corpus": {"training_actions_sha256": "a" * 64}},
    }
    suite = {
        "split": "validation",
        "test_content_evaluated": False,
        "thresholds": {"minimum_node_type_f1": 0.95},
    }
    training = {"preflight": {"seeds": [17, 29, 43], "train_actions_sha256": "a" * 64}}
    queue = {"training_allowed_now": False, "quality_evaluation_allowed_now": False}
    return config, freeze, suite, training, queue


def test_strict_ast_preflight_never_launches_and_blocks_when_device_is_busy(tmp_path: Path) -> None:
    module = _module()
    config, freeze, suite, training, queue = _fixtures(tmp_path)
    report = module.build_preflight(
        config,
        freeze=freeze,
        suite=suite,
        training=training,
        queue=queue,
        device_rows={1: (0, 32607)},
        root=tmp_path,
    )
    assert report["status"] == "blocked"
    assert "registered_gpu_memory_is_in_use" in report["blockers"]
    assert report["launch_performed"] is False
    assert report["training_enabled"] is False


def test_strict_ast_preflight_requires_exact_fingerprints_and_seed_set(tmp_path: Path) -> None:
    module = _module()
    config, freeze, suite, training, queue = _fixtures(tmp_path)
    config["seed_set"] = [17, 29]
    (tmp_path / "frozen.json").write_text('{"changed":true}')
    report = module.build_preflight(
        config,
        freeze=freeze,
        suite=suite,
        training=training,
        queue=queue,
        device_rows={1: (30000, 32607)},
        root=tmp_path,
    )
    assert "seed_set_differs_from_frozen_training_contract" in report["blockers"]
    assert any(item.startswith("implementation_fingerprint_changed") for item in report["blockers"])
    assert "gpu_queue_policy_is_not_closed" not in report["blockers"]


def test_strict_ast_preflight_allows_only_registered_cpu_side_checks(tmp_path: Path) -> None:
    module = _module()
    report = module.build_preflight(
        _fixtures(tmp_path)[0],
        freeze=_fixtures(tmp_path)[1],
        suite=_fixtures(tmp_path)[2],
        training=_fixtures(tmp_path)[3],
        queue=_fixtures(tmp_path)[4],
        device_rows={1: (30000, 32607)},
        root=tmp_path,
    )
    assert report["status"] == "preflight_passed_no_execution"
    assert report["launch_performed"] is False
