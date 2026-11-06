from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_nlp_hierarchical_tree as runner


def test_corrective_experiment_is_preregistered_without_reusing_failed_attempt(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_device_rows", lambda: {1: (32_000, 32_768)})
    monkeypatch.setattr(runner, "_active_compute_devices", lambda: set())

    report = runner.build_launch_preflight(ROOT)

    assert report["status"] == "ready"
    assert report["blockers"] == []
    assert "gpu_queue_registration_differs_from_preregistered_experiment" not in report["blockers"]
    assert "experiment_not_registered_in_gpu_queue" not in report["blockers"]
    assert report["preregistered_budget"]["max_attempts_per_seed"] == 1
    assert report["launch_performed"] is False
    assert report["exact_source_overlap"]["status"] == "pass"
    assert report["budget_estimate"]["training_rows"] == 1_914
    assert report["budget_estimate"]["expected_examples"] == 57_420
    assert report["budget_estimate"]["expected_optimizer_steps"] == 7_200


def test_exact_overlap_audit_rejects_any_validation_source_reuse() -> None:
    report = runner.audit_exact_training_validation_overlap(
        [
            {"case_id": "train-1", "split": "train", "source_sha256": "same"},
            {"case_id": "train-2", "split": "train", "source_sha256": "train-only"},
        ],
        {
            "cases": [
                {"case_id": "validation-1", "gold_source_sha256": "same"},
            ]
        },
    )

    assert report["status"] == "blocked"
    assert report["exact_source_hash_matches"] == [
        {"train_case_id": "train-1", "validation_case_id": "validation-1"}
    ]


def test_action_loader_skips_test_lines_before_json_decode(tmp_path) -> None:
    path = tmp_path / "actions.jsonl"
    path.write_text(
        '{"split":"train","case_id":"train-row"}\n'
        '{"split":"test","test_payload": not-json}\n',
        encoding="utf-8",
    )

    rows = runner._read_action_splits(path, {"train"})

    assert rows == [{"split": "train", "case_id": "train-row"}]


def test_experiment_rows_use_only_train_and_frozen_validation_splits() -> None:
    from floorflow.nlp_flow.ast_grammar import load_ast_grammar
    from floorflow.nlp_flow.real_corpus_vocabulary import build_vocabulary

    training = runner._read(ROOT / "configs/nlp_tree_training_v11.json")

    train, validation, filter_report = runner._load_rows(ROOT, training)

    assert train
    assert all(row["split"] == "train" for row in train)
    assert len(validation) == 60
    assert all(row["split"] == "validation" for row in validation)
    assert filter_report["candidate_train_rows"] == 1_981
    assert filter_report["used_train_rows"] == 1_914
    assert len(filter_report["excluded_oversize_rows"]) == 67
    assert filter_report["policy"] == "exclude_and_report"
    grammar_path = ROOT / training["preflight"]["grammar"]
    grammars = {
        language: load_ast_grammar(grammar_path, language)
        for language in ("python", "rust", "cpp")
    }
    assert build_vocabulary(train, grammars)["node_names"]


def test_rollout_batch_uses_generated_train_states_and_failure_labels(monkeypatch) -> None:
    from floorflow.nlp_flow.ast_grammar import load_ast_grammar
    from floorflow.nlp_flow.prefix_tree_training import _build_rollout_batch
    from floorflow.nlp_flow.real_corpus_policy_data import action_from_record
    from floorflow.nlp_flow.real_corpus_vocabulary import build_vocabulary

    training = runner._read(ROOT / "configs/nlp_tree_training_v11.json")
    train, _validation, _filter_report = runner._load_rows(ROOT, training)
    row = train[0]
    grammar = load_ast_grammar(ROOT / "configs/nlp_ast_transition_grammar_v1.json", row["language"])
    grammars = {row["language"]: grammar}
    vocabulary = build_vocabulary([row], grammars)
    actions = [action_from_record(item) for item in row["actions"]]

    for generated, expected_success in ((None, 0), ("fixture_source", 1)):
        monkeypatch.setattr(
            "floorflow.nlp_flow.prefix_tree_runtime.generate_prefix_tree",
            lambda *_args, _generated=generated, **_kwargs: (
                _generated,
                actions,
                None if _generated is not None else "fixture_dead_end",
            ),
        )
        rollout, target, counts = _build_rollout_batch(
            model=None,
            rows=[row],
            grammars=grammars,
            vocabulary=vocabulary,
            device=torch.device("cpu"),
            max_steps=len(row["actions"]),
        )

        assert rollout.step_mask.any()
        assert target.shape == rollout.step_mask.shape
        assert torch.all(target[rollout.step_mask] == expected_success)
        assert counts == {
            "rollout_rows": 1,
            "successful_rollouts": expected_success,
            "failed_rollouts": 1 - expected_success,
        }



