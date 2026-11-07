from __future__ import annotations

import json
from pathlib import Path

import pytest

from newera.nlp_flow.components.research.architecture.search.architecture_search import (
    ArchitectureCandidate,
    ArchitectureCandidateFactory,
    CandidateEvaluation,
    EfficientArchitectureSearch,
    SearchSpace,
    estimate_inference_flops,
    freeze_selected_architecture,
    pareto_front,
)
from newera.nlp_flow.components.research.architecture.search.architecture_search_trainer import (
    TorchValidationArchitectureEvaluator,
)
from newera.nlp_flow.model import NLPModelConfig


def _space() -> SearchSpace:
    return SearchSpace(
        base_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 64,
            "max_seq_len": 32,
            "parameter_cap": 2_000_000,
            "enable_dynamic_stack": True,
            "dynamic_min_steps": 1,
            "dynamic_max_steps": 2,
            "dynamic_parallel_lanes": 2,
            "dynamic_parallel_top_k": 1,
            "dynamic_lane_rank": 4,
            "enable_unified_competence": True,
            "unified_cell_rank": 4,
            "unified_top_k": 2,
            "unified_planner_dim": 8,
            "unified_generated_rank": 4,
            "unified_generated_basis_count": 2,
            "enable_runtime_competence": False,
        },
        axes={
            "hidden_size": (32, 36),
            "dynamic_max_steps": (2, 3),
            "unified_planner_dim": (8, 12),
        },
        max_parameters=2_000_000,
        max_training_flops_per_token=12_000_000,
        max_inference_flops=200_000_000,
        candidate_limit=8,
        sequence_length=16,
        batch_size=2,
        expected_depth_fraction=0.5,
        seed=17,
    )


def _candidate(
    candidate_id: str,
    parameters: int,
    inference_flops: int,
) -> ArchitectureCandidate:
    return ArchitectureCandidate(
        candidate_id=candidate_id,
        config={"hidden_size": 32},
        parameters=parameters,
        estimated_training_flops_per_token=6 * parameters,
        estimated_inference_flops=inference_flops,
    )


def _evaluation(
    candidate_id: str,
    *,
    quality: float,
    budget: int = 16,
    split: str = "validation",
    latency: float = 1.0,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate_id,
        budget_tokens=budget,
        training_tokens=budget,
        quality_score=quality,
        metrics={"validation_loss": -quality},
        evaluation_split=split,
        measured_latency_ms=latency,
    )


def test_candidate_factory_is_bounded_deterministic_and_materialized() -> None:
    factory = ArchitectureCandidateFactory(_space())

    first = factory.generate()
    second = factory.generate()

    assert first
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]
    assert any("base" in item.tags for item in first)
    assert len({item.candidate_id for item in first}) == len(first)
    assert all(item.parameters <= 2_000_000 for item in first)
    assert all(NLPModelConfig(**item.config) for item in first)


def test_explicit_controls_preserve_paired_e_ablation() -> None:
    space = _space()
    controlled = SearchSpace(
        **{
            **space.__dict__,
            "controls": (
                {
                    "id": "e-disabled",
                    "unified_generated_expert_mode": "disabled",
                    "enable_runtime_competence": False,
                },
            ),
        }
    )
    candidates = ArchitectureCandidateFactory(controlled).generate()
    control = next(
        candidate
        for candidate in candidates
        if "control:e-disabled" in candidate.tags
    )
    baseline = next(
        candidate
        for candidate in candidates
        if "base" in candidate.tags
    )

    assert control.config["unified_generated_expert_mode"] == "disabled"
    assert control.config["enable_runtime_competence"] is False
    assert control.parameters < baseline.parameters


def test_inference_flops_track_architecture_capacity() -> None:
    small = NLPModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        parameter_cap=2_000_000,
    )
    large = NLPModelConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=32,
        parameter_cap=2_000_000,
    )

    kwargs = {
        "batch_size": 2,
        "sequence_length": 16,
        "expected_depth_fraction": 0.5,
    }
    assert estimate_inference_flops(large, **kwargs) > estimate_inference_flops(
        small,
        **kwargs,
    )


def test_pareto_and_plateau_selection_preserve_small_efficient_model() -> None:
    tiny = _candidate("tiny", 10, 100)
    best = _candidate("best", 20, 120)
    dominated = _candidate("dominated", 30, 130)
    evaluations = {
        "tiny": _evaluation("tiny", quality=0.99),
        "best": _evaluation("best", quality=1.0),
        "dominated": _evaluation("dominated", quality=0.8),
    }

    front = pareto_front([tiny, best, dominated], evaluations)
    search = EfficientArchitectureSearch(
        candidates=[tiny, best],
        token_budgets=[16],
        keep_fraction=1.0,
        quality_plateau=0.02,
    )
    result = search.run(_MapEvaluator(evaluations))

    assert {item.candidate_id for item in front} == {"tiny", "best"}
    assert result["selected_candidate_id"] == "tiny"


def test_successive_halving_uses_matched_validation_budgets() -> None:
    candidates = [
        _candidate("a", 10, 100),
        _candidate("b", 12, 110),
        _candidate("c", 14, 120),
    ]
    evaluator = _RungEvaluator(
        {
            8: {"a": 0.8, "b": 0.9, "c": 0.3},
            16: {"a": 0.95, "b": 0.94, "c": 0.5},
        }
    )
    result = EfficientArchitectureSearch(
        candidates=candidates,
        token_budgets=[8, 16],
        keep_fraction=0.5,
        quality_plateau=0.02,
    ).run(evaluator)

    assert result["selected_candidate_id"] == "a"
    assert all(record["budget_tokens"] == 8 for record in result["rounds"][0]["results"])
    assert all(record["budget_tokens"] == 16 for record in result["rounds"][1]["results"])
    assert set(evaluator.calls[16]) == {"a", "b"}


def test_test_split_cannot_enter_architecture_selection() -> None:
    with pytest.raises(ValueError, match="validation splits"):
        _evaluation("a", quality=1.0, split="test")


def test_freeze_writes_selected_config_and_ledger(tmp_path: Path) -> None:
    candidate = _candidate("a", 10, 100)
    result = EfficientArchitectureSearch(
        candidates=[candidate],
        token_budgets=[16],
        required_metrics=("executable_success",),
    ).run(_MapEvaluator({"a": _evaluation("a", quality=1.0)}))

    config_path = tmp_path / "selected.json"
    report_path = tmp_path / "search.json"
    frozen = freeze_selected_architecture(
        result,
        config_path=config_path,
        report_path=report_path,
    )

    assert frozen["candidate_id"] == "a"
    assert result["promotion_eligible"] is False
    assert result["promotion_blockers"] == ["executable_success"]
    assert json.loads(config_path.read_text())["model"]["hidden_size"] == 32
    assert json.loads(report_path.read_text())["selected_candidate_id"] == "a"


@pytest.mark.skip(reason="Requires the external legacy tokenizer artifact under .data/nlp/tokenizer_v1/.")
def test_training_evaluator_continues_matched_budgets(tmp_path: Path) -> None:
    rows = [
        json.dumps(
            {
                "text": (
                    f"User: Write function {index}.\\n"
                    f"Assistant: def f{index}(x):\\n    return x + {index}"
                )
            }
        )
        for index in range(24)
    ]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    validation_path.write_text("\n".join(rows[:8]) + "\n", encoding="utf-8")
    candidate = ArchitectureCandidate(
        candidate_id="tiny-training",
        config={
            "vocab_size": 8192,
            "hidden_size": 32,
            "num_layers": 1,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 64,
            "max_seq_len": 16,
            "parameter_cap": 2_000_000,
        },
        parameters=100_000,
        estimated_training_flops_per_token=600_000,
        estimated_inference_flops=1_000_000,
    )
    evaluator = TorchValidationArchitectureEvaluator(
        train_paths=[train_path],
        validation_paths=[validation_path],
        tokenizer_path=Path(".data/nlp/tokenizer_v1/tokenizer.json"),
        device="cpu",
        batch_size=1,
        sequence_length=16,
        train_batches=4,
        validation_batches=1,
        learning_rate=1.0e-3,
        seed=3,
        precision="fp32",
        text_mode="raw",
        emergence_weight=0.01,
        dynamic_depth_weight=0.01,
    )

    first = evaluator.evaluate(candidate, 16)
    second = evaluator.evaluate(candidate, 32)

    assert first.training_tokens == 16
    assert second.training_tokens == 32
    assert second.evaluation_split == "validation"
    assert second.metrics["parameters"] == 100_000
    assert second.measured_latency_ms is not None


class _MapEvaluator:
    def __init__(self, evaluations: dict[str, CandidateEvaluation]) -> None:
        self.evaluations = evaluations

    def evaluate(
        self,
        candidate: ArchitectureCandidate,
        budget_tokens: int,
    ) -> CandidateEvaluation:
        evaluation = self.evaluations[candidate.candidate_id]
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            budget_tokens=budget_tokens,
            training_tokens=budget_tokens,
            quality_score=evaluation.quality_score,
            metrics=evaluation.metrics,
            evaluation_split=evaluation.evaluation_split,
            measured_latency_ms=evaluation.measured_latency_ms,
        )


class _RungEvaluator:
    def __init__(self, qualities: dict[int, dict[str, float]]) -> None:
        self.qualities = qualities
        self.calls: dict[int, list[str]] = {}

    def evaluate(
        self,
        candidate: ArchitectureCandidate,
        budget_tokens: int,
    ) -> CandidateEvaluation:
        self.calls.setdefault(budget_tokens, []).append(candidate.candidate_id)
        quality = self.qualities[budget_tokens][candidate.candidate_id]
        return _evaluation(
            candidate.candidate_id,
            quality=quality,
            budget=budget_tokens,
        )
