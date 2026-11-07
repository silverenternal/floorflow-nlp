from __future__ import annotations

import json

import pytest
import torch

import newera.nlp_flow.components.research.experiments.config.experiment_deadline as experiment_deadline
import newera.nlp_flow.components.research.experiments.studies.structure_ablation_runner as runner
from newera.nlp_flow.components.research.experiments.config.experiment_deadline import (
    BudgetDeadlineExceeded,
    ExperimentDeadline,
)


def test_deadline_reports_total_budget_and_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_deadline.time, "monotonic", lambda: 21.0)
    deadline = ExperimentDeadline(started_at=10.0, total_deadline=20.0, seed_deadline=30.0)

    with pytest.raises(BudgetDeadlineExceeded) as raised:
        deadline.check("free_running_action_generation")

    assert raised.value.scope == "total"
    assert raised.value.phase == "free_running_action_generation"
    assert raised.value.elapsed_seconds == 11.0


def test_deadline_reports_per_seed_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_deadline.time, "monotonic", lambda: 21.0)
    deadline = ExperimentDeadline(started_at=10.0, total_deadline=30.0, seed_deadline=20.0)

    with pytest.raises(BudgetDeadlineExceeded) as raised:
        deadline.check("optimizer_batch")

    assert raised.value.scope == "per_seed"
    assert raised.value.phase == "optimizer_batch"


def test_run_experiment_returns_partial_report_at_budget_stop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_contract = tmp_path / "prompt.json"
    prompt_contract.write_text('{"fragments": []}', encoding="utf-8")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "nlp-structure-matched-v13",
                "shared": {"seeds": [17, 29], "prompt_contract": "prompt.json"},
                "arms": [{"id": "hierarchical"}],
                "rollout_curriculum": {},
                "evidence_boundary": "test fixture",
            }
        ),
        encoding="utf-8",
    )
    budget = {"max_gpu_hours_total": 1, "max_gpu_hours_per_seed": 1}
    monkeypatch.setattr(
        runner,
        "build_preflight",
        lambda *_args, **_kwargs: {"gpu_queue": {"budget": budget}},
    )
    monkeypatch.setattr(
        runner,
        "_load_inputs",
        lambda *_args, **_kwargs: {
            "train_rows": [],
            "validation_rows": [],
            "grammars": {},
            "vocabulary": {},
            "validation_sources": [],
            "train_path": tmp_path / "train.jsonl",
        },
    )
    monkeypatch.setattr(runner, "encode_prefix_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_completion_budgets", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "iter_declaration_fragments", lambda *_args, **_kwargs: [])

    def stop_seed(*_args, **_kwargs):
        error = BudgetDeadlineExceeded(
            scope="per_seed",
            phase="free_running_action_generation",
            elapsed_seconds=3600.0,
        )
        error.partial_checkpoint = ".data/hierarchical/seed_17.partial.pt"
        raise error

    monkeypatch.setattr(runner, "_execute_arm_seed", stop_seed)
    report = runner.run_experiment(config_path, root=tmp_path, device=torch.device("cpu"))

    assert report["status"] == "budget_stopped_incomplete"
    assert report["stop_reason"] == "per_seed_gpu_hour_budget_reached:hierarchical:17"
    assert report["arms"][0]["runs"][0]["status"] == "budget_stopped"
    assert report["arms"][0]["runs"][0]["phase"] == "free_running_action_generation"
    assert report["arms"][0]["runs"][0]["partial_checkpoint"].endswith("seed_17.partial.pt")
    assert report["arms"][0]["runs"][1]["status"] == "not_attempted_after_stop"
